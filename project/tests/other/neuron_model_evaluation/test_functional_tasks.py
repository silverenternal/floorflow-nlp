import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(
    0,
    str(ROOT / "project/scripts/experiments/training/neurons/capabilities/functional/memory"),
)
from run_memory_capability_suite import StatefulMemoryModel

sys.path.insert(
    0,
    str(ROOT / "project/scripts/experiments/training/neurons/capabilities/aggregation"),
)
from compare_message_aggregators import SharedAggregationModel, _batch, _make_sample, _select_task_output

from newera.capabilities.evaluation.functional_tasks import (
    grouped_binding_oracle,
    interference_recall_oracle,
    make_conditional_route,
    make_grouped_binding,
    make_interference_recall,
    make_rule_composition,
    make_set_sum,
    rule_composition_oracle,
    set_sum_oracle,
)
from newera.capabilities.evaluation.memory_tasks import (
    make_delayed_recall,
    make_interference_recovery,
    make_order_recognition,
)


def test_set_sum_is_seeded_and_oracle_reads_raw_values():
    first = make_set_sum(size=64, seed=10)
    second = make_set_sum(size=64, seed=10)
    assert torch.equal(first.values, second.values)
    assert torch.equal(first.target, set_sum_oracle(first.values))
    assert first.target == first.values.sum()


def test_shared_neuron_output_uses_observable_task_channel():
    values = torch.tensor([[3.0, 11.0], [5.0, 13.0]])
    family_ids = torch.tensor([0.0, 1.0])
    torch.testing.assert_close(_select_task_output(values, family_ids), torch.tensor([3.0, 13.0]))


def test_aggregator_comparisons_keep_shared_neuron_initialization_identical():
    torch.manual_seed(82)
    summed = SharedAggregationModel(hidden_size=32, aggregator_kind="global_sum")
    torch.manual_seed(82)
    grouped = SharedAggregationModel(hidden_size=32, aggregator_kind="group_role_deep_sets")
    summed_core = {
        name: parameter for name, parameter in summed.neuron.named_parameters()
        if not name.startswith("message_aggregator.")
    }
    grouped_core = {
        name: parameter for name, parameter in grouped.neuron.named_parameters()
        if not name.startswith("message_aggregator.")
    }
    assert summed_core.keys() == grouped_core.keys()
    for name, value in summed_core.items():
        torch.testing.assert_close(value, grouped_core[name], atol=0, rtol=0)


def test_aggregation_count_context_uses_configured_capacity():
    sample64 = _make_sample("set_sum", 83, 64)
    sample128 = _make_sample("set_sum", 84, 128)
    context64 = _batch([sample64], 64, torch.device("cpu"))["context"][0, 1]
    context128 = _batch([sample128], 128, torch.device("cpu"))["context"][0, 1]
    assert context64 == 1
    assert context128 == 1


def test_group_binding_oracle_is_permutation_invariant():
    sample = make_grouped_binding(group_count=8, seed=11)
    order = torch.randperm(sample.values.numel(), generator=torch.Generator().manual_seed(12))
    permuted = grouped_binding_oracle(
        sample.groups[order], sample.roles[order], sample.values[order]
    )
    assert torch.equal(sample.target, permuted)
    assert torch.equal(sample.target, make_grouped_binding(group_count=8, seed=11).target)


def test_group_binding_rejects_missing_role():
    try:
        grouped_binding_oracle(torch.tensor([0]), torch.tensor([0]), torch.tensor([1.0]))
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("invalid groups must be rejected")


def test_interference_recall_returns_last_write_for_query():
    keys = torch.tensor([0, 1, 0, 1, 0])
    values = torch.tensor([3.0, 5.0, 7.0, 9.0, 11.0])
    assert interference_recall_oracle(keys, values, torch.tensor(1)) == 9.0
    sample = make_interference_recall(length=32, key_count=4, seed=20)
    assert torch.equal(
        sample.target,
        interference_recall_oracle(sample.keys, sample.values, sample.query_key),
    )


def test_conditional_route_is_keyed_memory_and_order_independent():
    sample = make_conditional_route(candidate_count=8, seed=21)
    assert sample.keys.unique().numel() == sample.keys.numel()
    assert torch.equal(
        sample.target,
        interference_recall_oracle(sample.keys, sample.values, sample.selector),
    )
    order = torch.randperm(sample.keys.numel(), generator=torch.Generator().manual_seed(22))
    assert torch.equal(
        sample.target,
        interference_recall_oracle(sample.keys[order], sample.values[order], sample.selector),
    )


def test_rule_composition_generator_has_independent_lookup_oracle():
    sample = make_rule_composition(domain_size=16, seed=23)
    assert torch.equal(
        sample.target,
        rule_composition_oracle(sample.first_rule, sample.second_rule, sample.value),
    )
    replay = make_rule_composition(domain_size=16, seed=23)
    assert torch.equal(sample.first_rule, replay.first_rule)
    assert torch.equal(sample.second_rule, replay.second_rule)
    assert torch.equal(sample.target, replay.target)


def test_delayed_recall_places_query_after_distractors_and_is_reproducible():
    first = make_delayed_recall(length=16, seed=91)
    second = make_delayed_recall(length=16, seed=91)
    assert torch.equal(first.events, second.events)
    assert torch.equal(first.events[-1, 4], torch.tensor(1.0))
    assert torch.equal(first.events[0, 3], torch.tensor(1.0))
    assert torch.equal(first.events[0, 5], first.target)
    assert torch.count_nonzero(first.events[:, 3]) == 1


def test_interference_recovery_returns_latest_marked_write():
    sample = make_interference_recovery(length=16, seed=73)
    writes = sample.events[:, 3] == 1
    assert int(writes.sum()) == 2
    assert torch.equal(sample.events[writes][-1, 5], sample.target)
    assert torch.equal(sample.events[-1, 4], torch.tensor(1.0))


def test_order_recognition_target_matches_symbol_order():
    sample = make_order_recognition(length=16, seed=17)
    position_a = (sample.events[:, 6] == 1).nonzero(as_tuple=True)[0].item()
    position_b = (sample.events[:, 7] == 1).nonzero(as_tuple=True)[0].item()
    assert torch.equal(sample.target, torch.tensor(float(position_a < position_b)))


def test_shared_multichannel_memory_model_decodes_tasks_without_copying_core():
    torch.manual_seed(11)
    model = StatefulMemoryModel(16, cell_kind="gru_multi_output", output_channels=3).eval()
    events = torch.zeros(2, 6, 8)
    events[0, :, 0] = 1
    events[0, 0, 3] = 1
    events[0, 0, 5] = 0.75
    events[0, -1, 4] = 1
    events[1, :, 1] = 1
    events[1, 2, 6] = 1
    events[1, 4, 7] = 1
    events[1, -1, 4] = 1
    outputs, state = model.run(events)
    assert outputs.shape == (2, 3)
    assert state.shape == (2, 16)
    selected = model.forward_for_families(events, ["delayed_recall", "order_recognition"])
    torch.testing.assert_close(selected, torch.stack((outputs[0, 0], outputs[1, 1])))
    assert model.neuron.output_projection.weight.shape == (3, 16)
    assert len({name.split(".")[0] for name, _ in model.named_parameters()}) == 1
    standalone, _ = model.run(events[1:2])
    torch.testing.assert_close(standalone, outputs[1:2])
