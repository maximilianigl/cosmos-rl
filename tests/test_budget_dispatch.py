# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Budget dispatch: rollouts carry an integer weight, data ranks a budget.

The controller fills every data rank to ``train_units_per_data_rank`` with whole
rollouts before publishing one real training command, keeps drawing prompts
while any rank is deficient, retires zero-weight rollouts without training, and
settles in-flight accounting at completion rather than at the training ACK.
"""

import asyncio
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgpack
import pytest

from cosmos_rl.dispatcher.command import (
    BuildMeshCommand,
    DataFetchCommand,
    TrainingCompleteCommand,
)
from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.data.schema import RLPayload, Rollout
from cosmos_rl.dispatcher.status import (
    CollectionStalledError,
    PolicyStatus,
    PolicyStatusManager,
)
from cosmos_rl.policy.config import Config, GrpoConfig, TrainingConfig
from cosmos_rl.policy.worker.rl_worker import RLPolicyWorker
from cosmos_rl.utils.payload import extract_rollouts

N_GENERATION = 2


def _config(
    *,
    budget=4,
    key="units",
    max_num_steps=10,
    timeout=None,
    ceiling=None,
    dispatch_incomplete=False,
    logger=("console",),
):
    return SimpleNamespace(
        mode="disaggregated",
        policy=SimpleNamespace(parallelism=SimpleNamespace(n_init_replicas=1)),
        validation=SimpleNamespace(enable=False, freq=1, val_before_train=False),
        logging=SimpleNamespace(logger=list(logger)),
        rollout=SimpleNamespace(
            n_generation=N_GENERATION, include_stop_str_in_output=False
        ),
        train=SimpleNamespace(
            max_num_steps=max_num_steps,
            epoch=1,
            non_text=True,
            sync_weight_interval=1,
            coalesce_weight_sync=False,
            ckpt=SimpleNamespace(
                enable_checkpoint=False, save_freq=1, save_freq_in_epoch=0
            ),
            # Unused under budget dispatch; a wrong value here must not matter.
            train_batch_per_replica=99,
            train_policy=SimpleNamespace(
                type="grpo",
                variant="grpo",
                on_policy=False,
                allowed_outdated_steps=10**9,
                data_dispatch_as_rank_in_mesh=False,
                rollout_as_token_ids=False,
                outdated_rollout_fetch_batch_size=0,
                max_retry_for_on_policy=0,
                max_inflight_steps=None,
                train_units_per_data_rank=budget,
                train_units_key=key,
                max_inflight_rollouts=ceiling,
                collection_no_progress_timeout_s=timeout,
                dispatch_incomplete_collections=dispatch_incomplete,
            ),
        ),
    )


def _replica(name, start_time, data_ranks=1):
    return SimpleNamespace(
        name=name,
        start_time=start_time,
        data_rank_count=lambda: data_ranks,
        put_rollout=MagicMock(),
        sub_profiler_config=SimpleNamespace(
            do_profile=False,
            active_steps=None,
            rank_filter=None,
            record_shape=None,
            profile_memory=None,
            with_stack=None,
            with_modules=None,
        ),
    )


def _manager(config, replicas, *, samples_on_the_fly=0):
    manager = PolicyStatusManager()
    manager.config = config
    manager.redis_handler = object()
    manager.data_fetcher = SimpleNamespace(activated_val_iter=None)
    manager.samples_per_epoch = 100
    manager.remain_samples_num = 1000
    manager.samples_on_the_fly = samples_on_the_fly
    manager.policy_replicas = {replica.name: replica for replica in replicas}
    manager.status = {replica.name: PolicyStatus.READY for replica in replicas}
    manager.get_all_atoms_arrived_replicas = lambda: list(replicas)
    manager._publish_payload_transport_cleanup = MagicMock()
    manager.recompute_total_steps()
    manager._reset_collection()
    return manager


def _rollout(prompt_idx, units, key="units"):
    return Rollout(
        prompt_idx=prompt_idx,
        completion=f"completion-{prompt_idx}",
        extra_info={key: units},
        weight_version=0,
    )


def _rollout_status(*, ended=False):
    return SimpleNamespace(
        all_rollouts_ended=lambda: ended,
        get_safe_weight_sync_replicas=lambda validation_enabled: [],
    )


def _ack_report(step):
    return {
        "train_step": step,
        "train/loss_avg": 0.1,
        "train/loss_max": 0.2,
        "train/learning_rate": 1e-6,
        "train/iteration_time": 1.0,
    }


class _DispatchRecorder:
    """Capture every DataFetchCommand the manager publishes."""

    def __init__(self):
        self.commands = []

    def __enter__(self):
        self._patch = patch.object(
            DataFetchCommand,
            "trigger",
            side_effect=lambda **kwargs: self.commands.append(kwargs),
        )
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


def test_collection_fills_every_rank_to_budget_and_keeps_overshoot():
    replicas = [_replica("p0", 0), _replica("p1", 1)]
    manager = _manager(_config(budget=4), replicas, samples_on_the_fly=10)
    rollouts = [_rollout(i, units) for i, units in enumerate([3, 3, 2, 2, 5])]

    with _DispatchRecorder() as dispatch:
        manager.put_rollouts(rollouts)

    # 3 -> rank 0, 3 -> rank 1, 2 -> rank 0 (tie broken by index), 2 -> rank 1:
    # both ranks hold 5 >= 4 and the fifth rollout waits for the next collection.
    assert [rollout.train_units for rollout in rollouts] == [3, 3, 2, 2, 5]
    assert manager.current_step == 1
    assert manager.dispatched_rollouts_by_step == {1: 4}
    assert manager.dispatched_train_units_by_step == {1: [5, 5]}
    assert [call.args[0] for call in replicas[0].put_rollout.call_args_list] == [
        rollouts[0],
        rollouts[2],
    ]
    assert [call.args[0] for call in replicas[1].put_rollout.call_args_list] == [
        rollouts[1],
        rollouts[3],
    ]
    assert len(dispatch.commands) == 2
    for command in dispatch.commands:
        assert command["items_count"] == 2
        assert command["items_per_data_rank"] == [2]
        assert command["global_step"] == 1
        assert command["total_steps"] == 10
    assert manager.remain_samples_num == 996
    # Completions settle in-flight accounting; the training ACK does not.
    assert manager.samples_on_the_fly == 5
    # The fifth rollout opens the next collection instead of joining the dispatched one.
    assert manager.total_pending_rollouts() == 1
    assert manager.collection_filled_units == [5, 0]
    report = manager.train_report_data[1]
    assert report["dispatch/rollouts"] == 4
    assert report["dispatch/train_units_min"] == 5
    assert report["dispatch/train_units_max"] == 5
    assert report["dispatch/pending_rollouts"] == 0


def test_sharded_replica_receives_one_count_per_data_rank():
    replicas = [_replica("p0", 0, data_ranks=2), _replica("p1", 1, data_ranks=1)]
    manager = _manager(_config(budget=2), replicas)
    rollouts = [_rollout(i, 1) for i in range(6)]

    with _DispatchRecorder() as dispatch:
        manager.put_rollouts(rollouts)

    by_name = {command["replica"].name: command for command in dispatch.commands}
    assert by_name["p0"]["items_per_data_rank"] == [2, 2]
    assert by_name["p0"]["items_count"] == 4
    assert by_name["p1"]["items_per_data_rank"] == [2]
    # Rank order within the replica: data rank 0's rollouts first, then rank 1's.
    published = [call.args[0] for call in replicas[0].put_rollout.call_args_list]
    assert published == [rollouts[0], rollouts[3], rollouts[1], rollouts[4]]


def test_zero_unit_rollouts_are_retired_without_training():
    replicas = [_replica("p0", 0)]
    manager = _manager(_config(budget=4), replicas)
    rollouts = [_rollout(i, units) for i, units in enumerate([0, 4, 0])]

    with _DispatchRecorder() as dispatch:
        manager.put_rollouts(rollouts)

    assert [call.args[0] for call in replicas[0].put_rollout.call_args_list] == [
        rollouts[1]
    ]
    assert dispatch.commands[0]["items_per_data_rank"] == [1]
    cleaned = [
        call.args[0][0]
        for call in manager._publish_payload_transport_cleanup.call_args_list
    ]
    assert cleaned == [rollouts[0], rollouts[2]]
    assert manager.zero_unit_rollouts_total == 2
    # The zero-unit rollout that arrived after the dispatch counts for the next one.
    assert manager.collection_zero_unit_rollouts == 1
    assert manager.train_report_data[1]["dispatch/zero_unit_rollouts"] == 1


def test_missing_or_invalid_weight_fails_admission():
    manager = _manager(_config(budget=4), [_replica("p0", 0)])

    with pytest.raises(ValueError, match="carries no extra_info"):
        manager.put_rollouts([Rollout(prompt_idx=0, completion="c", extra_info={})])
    with pytest.raises(ValueError, match="non-negative integer"):
        manager.put_rollouts([_rollout(1, -1)])
    with pytest.raises(ValueError, match="non-negative integer"):
        manager.put_rollouts([_rollout(2, True)])


def test_second_collection_assembles_during_training_and_ack_settles_nothing():
    replicas = [_replica("p0", 0), _replica("p1", 1)]
    manager = _manager(_config(budget=4), replicas, samples_on_the_fly=20)
    first = [_rollout(i, 4) for i in range(2)]
    second = [_rollout(10 + i, 4) for i in range(2)]

    with _DispatchRecorder() as dispatch:
        manager.put_rollouts(first)
        assert manager.current_step == 1
        assert manager.all_with_status([PolicyStatus.RUNNING])
        # Arrives while step 1 trains: assigned to the assembling collection,
        # which cannot dispatch until the trainers are ready again.
        manager.put_rollouts(second)
        assert manager.collection_complete()
        assert manager.current_step == 1
        in_flight_before_ack = manager.samples_on_the_fly

        for replica in replicas:
            manager.train_ack(
                replica.name, 1, 10, False, _ack_report(1), _rollout_status()
            )

    assert manager.samples_on_the_fly == in_flight_before_ack
    assert manager.current_step == 2
    assert manager.dispatched_rollouts_by_step == {2: 2}
    assert [call.args[0] for call in replicas[0].put_rollout.call_args_list] == [
        first[0],
        second[0],
    ]
    assert [command["global_step"] for command in dispatch.commands] == [1, 1, 2, 2]


def test_prompt_admission_pauses_while_the_collection_is_full():
    replicas = [_replica("p0", 0)]
    config = _config(budget=4)
    manager = _manager(config, replicas)
    controller = object.__new__(Controller)
    controller.config = config
    controller.policy_status_manager = manager
    controller.rollout_status_manager = SimpleNamespace(replica_scaling_log=[])
    controller.data_fetcher = SimpleNamespace(
        get_batched_prompt=MagicMock(
            side_effect=lambda n, *args, **kwargs: (
                [RLPayload(prompt_idx=index) for index in range(n)],
                False,
            )
        )
    )

    payloads, is_end = asyncio.run(controller._get_batched_prompt_impl(3))
    assert len(payloads) == 3
    assert not is_end
    assert all(payload.weight_version == 0 for payload in payloads)
    assert manager.samples_on_the_fly == 3 * N_GENERATION
    assert manager.prompts_dispatched_total == 3

    # Hold the trainer so the full collection cannot dispatch yet.
    manager.status["p0"] = PolicyStatus.RUNNING
    with _DispatchRecorder():
        manager.put_rollouts([_rollout(0, 4)])
    assert manager.collection_complete()
    payloads, is_end = asyncio.run(controller._get_batched_prompt_impl(3))
    assert payloads == [] and not is_end

    # The trainer takes the collection: the next one assembles and draws resume.
    manager.status["p0"] = PolicyStatus.READY
    with _DispatchRecorder():
        manager.try_trigger_data_fetch_and_training()
    assert manager.current_step == 1
    payloads, _ = asyncio.run(controller._get_batched_prompt_impl(2))
    assert len(payloads) == 2
    assert all(payload.weight_version == 1 for payload in payloads)


def test_deficient_collection_outlives_the_inflight_ceiling():
    """A heavily filtered collection needs more rollouts than the fleet holds at once.

    The ceiling bounds only dispatched-but-uncompleted rollouts and completions
    release it, so draws continue until every rank reaches its budget instead of
    wedging against a ceiling that counts completed-but-untrained work.
    """
    replicas = [_replica("p0", 0), _replica("p1", 1)]
    fleet_hold = 3 * N_GENERATION
    config = _config(budget=8, ceiling=fleet_hold)
    manager = _manager(config, replicas)
    controller = object.__new__(Controller)
    controller.config = config
    controller.policy_status_manager = manager
    controller.rollout_status_manager = SimpleNamespace(replica_scaling_log=[])
    next_prompt = iter(range(10_000))
    controller.data_fetcher = SimpleNamespace(
        get_batched_prompt=MagicMock(
            side_effect=lambda n, *args, **kwargs: (
                [RLPayload(prompt_idx=next(next_prompt)) for _ in range(n)],
                False,
            )
        )
    )

    with _DispatchRecorder() as dispatch:
        rounds = 0
        while manager.current_step == 0:
            rounds += 1
            assert rounds < 100
            payloads, _ = asyncio.run(controller._get_batched_prompt_impl(1))
            if not payloads:
                # Ceiling reached: complete what is in flight (one unit each).
                assert manager.samples_on_the_fly == fleet_hold
                assert not manager.collection_complete()
                completions = [
                    _rollout(index, 1) for index in range(manager.samples_on_the_fly)
                ]
                manager.put_rollouts(completions)
                assert manager.samples_on_the_fly == 0

    assert manager.current_step == 1
    assert manager.dispatched_rollouts_by_step[1] == 16
    assert manager.prompts_dispatched_total * N_GENERATION >= 16
    assert manager.prompts_dispatched_total > fleet_hold // N_GENERATION
    assert len(dispatch.commands) == 2


def test_no_progress_timeout_reports_deficits_and_resets_on_positive_weight():
    manager = _manager(_config(budget=4, timeout=10.0), [_replica("p0", 0)])
    manager.status["p0"] = PolicyStatus.RUNNING

    manager.check_collection_progress(now=1_000.0)  # no completion yet: silent
    with patch("cosmos_rl.dispatcher.status.time.time", return_value=1_000.0):
        manager.put_rollouts([_rollout(0, 0)])
    manager.check_collection_progress(now=1_009.0)
    with pytest.raises(CollectionStalledError, match=r"deficits=\[4\]"):
        manager.check_collection_progress(now=1_011.0)

    with patch("cosmos_rl.dispatcher.status.time.time", return_value=1_012.0):
        manager.put_rollouts([_rollout(1, 2)])
    manager.check_collection_progress(now=1_021.0)
    with pytest.raises(CollectionStalledError, match=r"deficits=\[2\]"):
        manager.check_collection_progress(now=1_023.0)

    with patch("cosmos_rl.dispatcher.status.time.time", return_value=1_024.0):
        manager.put_rollouts([_rollout(2, 2)])
    assert manager.collection_complete()
    manager.check_collection_progress(now=10_000.0)  # complete collections never stall


def test_replica_layout_change_redistributes_assigned_rollouts():
    replicas = [_replica("p0", 0)]
    manager = _manager(_config(budget=6), replicas)
    rollouts = [_rollout(i, 2) for i in range(2)]

    with _DispatchRecorder():
        manager.put_rollouts(rollouts)
    assert manager.collection_filled_units == [4]

    replicas.append(_replica("p1", 1))
    manager.policy_replicas["p1"] = replicas[1]
    manager.status["p1"] = PolicyStatus.READY
    manager.rearrange_rollout_buffer_after_mesh_rebuild(replicas)

    assert manager.collection_filled_units == [2, 2]
    assert sorted(
        rollout.prompt_idx
        for assigned in manager.collection_assignments
        for rollout in assigned
    ) == [0, 1]
    assert manager.total_pending_rollouts() == 2


def test_terminal_cleanup_releases_assigned_collection_members():
    manager = _manager(_config(budget=6), [_replica("p0", 0)])
    rollouts = [_rollout(i, 2) for i in range(2)]
    with _DispatchRecorder():
        manager.put_rollouts(rollouts)

    assert manager.cleanup_buffered_rollouts() == 2
    assert manager.collection_assignments == [[]]
    assert manager.collection_filled_units == [0]
    cleaned = manager._publish_payload_transport_cleanup.call_args.args[0]
    assert cleaned == rollouts


@pytest.mark.parametrize("dispatch_incomplete", [True, False])
def test_ready_trainers_are_dispatched_to_at_registration_only_under_the_flag(
    dispatch_incomplete,
):
    """Trainers holding their own replay pool are never woken by a completion.

    Registration is the one transition that makes every replica READY without a
    rollout in sight, so it must publish the first command itself.
    """
    replicas = [_replica("p0", 0, data_ranks=2)]
    config = _config(budget=4, dispatch_incomplete=dispatch_incomplete)
    manager = _manager(config, replicas)
    manager.data_fetcher.set_policy_global_mesh_size = MagicMock()

    with _DispatchRecorder() as dispatch, patch.object(BuildMeshCommand, "trigger"):
        manager.post_register_hook(replicas, replicas[0], config, _rollout_status())

    if not dispatch_incomplete:
        assert dispatch.commands == []
        assert manager.current_step == 0
        return

    assert manager.current_step == 1
    assert len(dispatch.commands) == 1
    assert dispatch.commands[0]["items_count"] == 0
    assert dispatch.commands[0]["items_per_data_rank"] == [0, 0]
    assert manager.dispatched_rollouts_by_step == {1: 0}
    assert manager.dispatched_train_units_by_step == {1: [0, 0]}
    assert manager.all_with_status([PolicyStatus.RUNNING])
    # Nothing was trained on, so there is no rollout report for the step.
    assert 1 not in manager.train_report_data


def test_losing_a_peer_dispatches_to_the_ready_survivors():
    """Reaping a replica is the other transition no completion follows."""
    replicas = [_replica("p0", 0), _replica("p1", 1)]
    for replica in replicas:
        replica.in_mesh = True
    manager = _manager(_config(budget=4, dispatch_incomplete=True), replicas)
    manager.get_all_atoms_arrived_replicas = lambda: [
        replica for replica in replicas if replica.name in manager.policy_replicas
    ]
    manager.data_fetcher.set_policy_global_mesh_size = MagicMock()

    with _DispatchRecorder() as dispatch, patch.object(BuildMeshCommand, "trigger"):
        manager.unregister("p1")

    assert [command["replica"].name for command in dispatch.commands] == ["p0"]
    assert dispatch.commands[0]["items_per_data_rank"] == [0]
    assert manager.current_step == 1


def test_incomplete_collection_is_dispatched_and_the_surplus_opens_the_next_one():
    replicas = [_replica("p0", 0), _replica("p1", 1)]
    manager = _manager(
        _config(budget=4, dispatch_incomplete=True), replicas, samples_on_the_fly=10
    )
    for replica in replicas:
        manager.status[replica.name] = PolicyStatus.RUNNING

    with _DispatchRecorder() as dispatch:
        # Trainers busy: four 3-unit rollouts fill both ranks past the budget and
        # the fifth waits for the next collection.
        manager.put_rollouts([_rollout(index, 3) for index in range(5)])
        assert manager.current_step == 0
        assert manager.collection_filled_units == [6, 6]

        for replica in replicas:
            manager.status[replica.name] = PolicyStatus.READY
        manager.try_trigger_data_fetch_and_training()
        assert manager.current_step == 1
        # The surplus opens the next collection, which is deficient but still
        # dispatched as soon as the trainers come back.
        assert manager.collection_filled_units == [3, 0]

        for replica in replicas:
            manager.status[replica.name] = PolicyStatus.READY
        manager.try_trigger_data_fetch_and_training()

    assert manager.current_step == 2
    assert manager.dispatched_rollouts_by_step == {1: 4, 2: 1}
    assert manager.dispatched_train_units_by_step[2] == [3, 0]
    by_name = {command["replica"].name: command for command in dispatch.commands[-2:]}
    assert by_name["p0"]["items_per_data_rank"] == [1]
    assert by_name["p1"]["items_per_data_rank"] == [0]
    assert by_name["p1"]["items_count"] == 0
    assert manager.total_pending_rollouts() == 0


def test_prompt_issue_still_pauses_while_the_collection_is_full():
    """The budget keeps gating producers even when it no longer gates training."""
    replicas = [_replica("p0", 0)]
    config = _config(budget=4, dispatch_incomplete=True)
    manager = _manager(config, replicas)
    controller = object.__new__(Controller)
    controller.config = config
    controller.policy_status_manager = manager
    controller.rollout_status_manager = SimpleNamespace(replica_scaling_log=[])
    controller.data_fetcher = SimpleNamespace(
        get_batched_prompt=MagicMock(
            side_effect=lambda n, *args, **kwargs: (
                [RLPayload(prompt_idx=index) for index in range(n)],
                False,
            )
        )
    )

    manager.status["p0"] = PolicyStatus.RUNNING
    with _DispatchRecorder():
        manager.put_rollouts([_rollout(0, 4)])
    assert manager.collection_complete()
    assert asyncio.run(controller._get_batched_prompt_impl(3)) == ([], False)

    manager.status["p0"] = PolicyStatus.READY
    with _DispatchRecorder() as dispatch:
        manager.try_trigger_data_fetch_and_training()
    assert dispatch.commands[0]["items_count"] == 1
    payloads, _ = asyncio.run(controller._get_batched_prompt_impl(2))
    assert len(payloads) == 2


def test_draining_keeps_stepping_to_the_frozen_horizon_without_rollouts():
    replicas = [_replica("p0", 0)]
    horizon = 3
    manager = _manager(
        _config(budget=4, max_num_steps=horizon, dispatch_incomplete=True), replicas
    )
    ended = _rollout_status(ended=True)

    with (
        _DispatchRecorder() as dispatch,
        patch.object(TrainingCompleteCommand, "trigger") as completion,
    ):
        manager.finish_draining_phase(ended)
        assert manager.current_step == 1
        for expected_step in (2, 3):
            manager.train_ack(
                "p0",
                manager.current_step,
                horizon,
                False,
                _ack_report(manager.current_step),
                ended,
            )
            assert manager.current_step == expected_step
        assert not completion.called
        manager.train_ack("p0", horizon, horizon, False, _ack_report(horizon), ended)

    assert [command["global_step"] for command in dispatch.commands] == [1, 2, 3]
    assert all(command["items_count"] == 0 for command in dispatch.commands)
    assert manager.training_finished()
    assert manager.real_terminal_command_acked()
    assert manager.terminal_complete
    assert not completion.called
    assert not manager.completion_recipients
    assert DataFetchCommand.replica_should_stop(
        SimpleNamespace(global_step=horizon, total_steps=horizon)
    )


@pytest.mark.parametrize("capture", [False, True])
def test_reports_without_training_or_rollout_statistics_reach_custom_loggers(capture):
    """Consecutive capture/replay reports keep their values and never invent losses/rewards."""
    manager = _manager(
        _config(budget=4, max_num_steps=3, dispatch_incomplete=True),
        [_replica("p0", 0)],
    )
    reports = []
    manager.custom_logger_fns = [
        lambda report, step: reports.append((step, dict(report)))
    ]
    with _DispatchRecorder():
        manager.try_trigger_data_fetch_and_training()
        for step in (1, 2):
            report = (
                {"train_step": step, "train/seed_capture/groups": step * 4}
                if capture
                else {**_ack_report(step), "train/loss_avg": step * 0.1}
            )
            manager.train_ack("p0", step, 3, False, report, _rollout_status())
            assert not manager.report_data_list

    assert [step for step, _ in reports] == [1, 2]
    for step, report in reports:
        assert "train/reward_mean" not in report
        if capture:
            assert report["train/seed_capture/groups"] == step * 4
            assert "train/loss_avg" not in report
        else:
            assert report["train/loss_avg"] == pytest.approx(step * 0.1)


def test_policy_worker_takes_nothing_when_every_per_rank_count_is_zero(monkeypatch):
    worker = object.__new__(RLPolicyWorker)
    worker.config = SimpleNamespace(
        train=SimpleNamespace(
            local_dataset=False,
            train_policy=SimpleNamespace(
                uncentralized_training=False, data_dispatch_as_rank_in_mesh=False
            ),
        )
    )
    worker.global_rank = 0
    worker.parallel_dims = SimpleNamespace(get_rank_in_dim=lambda dim, rank: rank)
    worker.replica_batch_for_this_step = 0
    worker.data_queue = Queue()
    scattered = {}

    def fake_scatter(output, scatter_list, src):
        scattered["list"] = scatter_list
        output[0] = scatter_list[0]

    monkeypatch.setattr(
        "cosmos_rl.policy.worker.rl_worker.dist.scatter_object_list", fake_scatter
    )

    worker.world_size = 2
    worker.dp_world_size = 2
    worker.replica_items_per_data_rank = [0, 0]
    assert worker.dispatch_rollouts() == []
    assert scattered["list"] == [[], []]

    # A single-rank replica never scatters; the empty stream must survive that too.
    worker.world_size = 1
    worker.dp_world_size = 1
    worker.replica_items_per_data_rank = [0]
    assert worker.dispatch_rollouts() == []


def test_data_fetch_command_round_trips_per_rank_counts_and_omits_them_natively():
    replica = _replica("p0", 0)
    redis_handler = SimpleNamespace(publish_command=MagicMock())

    DataFetchCommand.trigger(
        replica=replica,
        items_count=5,
        global_step=3,
        total_steps=10,
        remain_samples_num=7,
        do_save=False,
        redis_handler=redis_handler,
        items_per_data_rank=[3, 2],
    )
    packed = redis_handler.publish_command.call_args.args[0]
    command = DataFetchCommand.depack(packed)
    assert command.items_per_data_rank == [3, 2]
    assert command.items_count == 5

    # A command that hands over nothing still carries one count per data rank.
    DataFetchCommand.trigger(
        replica=replica,
        items_count=0,
        global_step=3,
        total_steps=10,
        remain_samples_num=7,
        do_save=False,
        redis_handler=redis_handler,
        items_per_data_rank=[0, 0],
    )
    empty = DataFetchCommand.depack(redis_handler.publish_command.call_args.args[0])
    assert empty.items_per_data_rank == [0, 0]
    assert empty.items_count == 0

    DataFetchCommand.trigger(
        replica=replica,
        items_count=4,
        global_step=3,
        total_steps=10,
        remain_samples_num=7,
        do_save=False,
        redis_handler=redis_handler,
    )
    native = msgpack.unpackb(redis_handler.publish_command.call_args.args[0])
    assert "items_per_data_rank" not in native
    assert DataFetchCommand.depack(msgpack.packb(native)).items_per_data_rank is None


def test_policy_worker_splits_the_stream_by_per_rank_counts(monkeypatch):
    worker = object.__new__(RLPolicyWorker)
    worker.config = SimpleNamespace(
        train=SimpleNamespace(
            local_dataset=False,
            train_policy=SimpleNamespace(
                uncentralized_training=False, data_dispatch_as_rank_in_mesh=False
            ),
        )
    )
    worker.global_rank = 0
    worker.world_size = 2
    worker.dp_world_size = 2
    worker.parallel_dims = SimpleNamespace(get_rank_in_dim=lambda dim, rank: rank)
    worker.replica_batch_for_this_step = 3
    worker.replica_items_per_data_rank = [2, 1]
    worker.data_queue = Queue()
    stream = [_rollout(index, 1) for index in range(3)]
    for rollout in stream:
        worker.data_queue.put(rollout)
    scattered = {}

    def fake_scatter(output, scatter_list, src):
        scattered["list"] = scatter_list
        output[0] = scatter_list[0]

    monkeypatch.setattr(
        "cosmos_rl.policy.worker.rl_worker.dist.scatter_object_list", fake_scatter
    )

    received = worker.dispatch_rollouts()

    assert received == stream[:2]
    assert scattered["list"] == [stream[:2], stream[2:]]
    assert worker.data_queue.empty()


def test_extract_rollouts_fans_per_completion_weights_out_to_each_rollout():
    payload = RLPayload(
        prompt_idx=4,
        completions=["a", "b"],
        rewards=[1.0, 0.0],
        advantages=[0.5, -0.5],
        extra_info={"units": [3, 0], "random_seed": 7},
    )

    [rollouts] = extract_rollouts([payload], False)

    assert [rollout.extra_info["units"] for rollout in rollouts] == [3, 0]
    assert [rollout.extra_info["random_seed"] for rollout in rollouts] == [7, 7]
    assert all(rollout.train_units == 1 for rollout in rollouts)


def test_grpo_config_rejects_incompatible_budget_dispatch_settings():
    GrpoConfig(type="grpo", train_units_per_data_rank=4, train_units_key="units")
    with pytest.raises(ValueError, match="dapo"):
        GrpoConfig(type="grpo", variant="dapo", train_units_per_data_rank=4)
    with pytest.raises(ValueError, match="data_dispatch_as_rank_in_mesh"):
        GrpoConfig(
            type="grpo", train_units_per_data_rank=4, data_dispatch_as_rank_in_mesh=True
        )
    with pytest.raises(ValueError, match="positive"):
        GrpoConfig(type="grpo", train_units_per_data_rank=0)
    with pytest.raises(ValueError, match="require train_units_per_data_rank"):
        GrpoConfig(type="grpo", train_units_key="units")
    with pytest.raises(ValueError, match="require train_units_per_data_rank"):
        GrpoConfig(type="grpo", dispatch_incomplete_collections=True)


def test_config_requires_disaggregated_step_bounded_budget_dispatch():
    with pytest.raises(ValueError, match="max_num_steps"):
        Config(
            mode="disaggregated",
            train=TrainingConfig(
                train_policy=GrpoConfig(type="grpo", train_units_per_data_rank=4)
            ),
        )
    with pytest.raises(ValueError, match="disaggregated"):
        Config(
            mode="colocated",
            train=TrainingConfig(
                train_policy=GrpoConfig(type="grpo", train_units_per_data_rank=4),
                max_num_steps=5,
            ),
        )
    Config(
        mode="disaggregated",
        train=TrainingConfig(
            train_policy=GrpoConfig(type="grpo", train_units_per_data_rank=4),
            max_num_steps=5,
        ),
    )
