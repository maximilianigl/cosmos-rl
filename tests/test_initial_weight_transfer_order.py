# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.dispatcher.command import (
    BuildMeshCommand,
    Command,
    PolicyToRolloutUnicastCommand,
    RolloutToRolloutBroadcastCommand,
)
from cosmos_rl.dispatcher.protocol import MESH_NAMES
from cosmos_rl.dispatcher.replica import Atom
from cosmos_rl.dispatcher.status import PolicyStatusManager, RolloutStatusManager


def atom(name):
    return Atom(
        global_rank=0,
        host_ip="127.0.0.1",
        host_name="localhost",
        trace_path="",
        ranks=[0] * len(MESH_NAMES),
        group_size=[1] * len(MESH_NAMES),
        replica_name=name,
    )


def _managers(policy_replicas, rollout_replicas):
    policy, rollouts = PolicyStatusManager(), RolloutStatusManager()
    config = SimpleNamespace(
        mode="disaggregated",
        train=SimpleNamespace(train_policy=SimpleNamespace(type="grpo")),
        policy=SimpleNamespace(
            parallelism=SimpleNamespace(n_init_replicas=policy_replicas)
        ),
        rollout=SimpleNamespace(
            parallelism=SimpleNamespace(n_init_replicas=rollout_replicas)
        ),
    )
    redis = Mock()
    policy.current_step, policy.total_steps = 0, 20
    policy.redis_handler = rollouts.redis_handler = redis
    rollouts.policy_status_manager = policy
    # Keep real registration and command serialization; omit mesh/training work.
    policy.post_register_hook = Mock()
    rollouts.trigger_rebuild_mesh = lambda replicas: BuildMeshCommand.trigger(
        replicas, redis_handler=redis
    )
    return policy, rollouts, config, redis


def _published(redis):
    return [
        (Command.depack(call.args[0]), call.args[1])
        for call in redis.publish_command.call_args_list
    ]


@pytest.mark.parametrize("policy_position", [0, 1, 2, 3])
def test_initial_weight_transfer_follows_the_rollout_mesh_build(policy_position):
    # Workers execute commands in order: a first P2R queued ahead of BuildMesh
    # holds its receiver in the transfer while the other rollouts wait in the
    # mesh collective (job 8145682). Every registration order must queue
    # BuildMesh first and send the first transfer exactly once.
    policy, rollouts, config, redis = _managers(1, 3)

    order = ["rollout-0", "rollout-1", "rollout-2"]
    order.insert(policy_position, "policy")
    for name in order:
        if name == "policy":
            policy.register(atom(name), config, rollouts)
        else:
            rollouts.register(atom(name), config, policy)

    published = _published(redis)
    receivers = [
        target
        for cmd, target in published
        if isinstance(cmd, PolicyToRolloutUnicastCommand) and target != "policy"
    ]
    assert len(receivers) == 1
    for name in order[:policy_position] + order[policy_position + 1 :]:
        kinds = [type(cmd) for cmd, target in published if target == name]
        assert BuildMeshCommand in kinds
        first_weights = [
            index
            for index, kind in enumerate(kinds)
            if kind in (PolicyToRolloutUnicastCommand, RolloutToRolloutBroadcastCommand)
        ]
        assert first_weights and kinds.index(BuildMeshCommand) < first_weights[0]


@pytest.mark.parametrize("rollouts_first", [False, True])
def test_loaded_policy_does_not_transfer_before_its_cohort_is_complete(
    rollouts_first,
):
    # Job 8191055: the first of six policy replicas got a P2R at registration,
    # blocked on rollout readiness for 13 min, and the other five timed out in
    # the policy-to-policy broadcast it was meant to send. Until the policy
    # cohort is complete, no registration may queue a transfer.
    policy, rollouts, config, redis = _managers(2, 2)

    registrations = [
        lambda: policy.register(atom("policy-0"), config, rollouts),
        lambda: rollouts.register(atom("rollout-0"), config, policy),
        lambda: rollouts.register(atom("rollout-1"), config, policy),
    ]
    if rollouts_first:
        registrations = registrations[1:] + registrations[:1]
    for register in registrations:
        register()

    assert not any(
        isinstance(cmd, PolicyToRolloutUnicastCommand)
        for cmd, _ in _published(redis)
    )
