from __future__ import annotations

import pytest
import torch
from torch import nn

from newera.capabilities import (
    Capability,
    RegressionCapability,
    RegressionExample,
    RegressionTaskAdapter,
    SequenceExample,
    SequenceSumCapability,
    SequenceTaskAdapter,
    TaskAdapter,
)
from newera.neuron import (
    MLPNeuron,
    Neuron,
    NeuronResult,
    SequentialNeuron,
    StatefulMLPNeuron,
)


class LinearNeuron(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def step(
        self,
        inputs: torch.Tensor,
        *,
        state: None = None,
    ) -> NeuronResult[torch.Tensor, None]:
        return NeuronResult(self.linear(inputs))


class AccumulatingNeuron:
    def step(
        self,
        inputs: torch.Tensor,
        *,
        state: torch.Tensor | None = None,
    ) -> NeuronResult[torch.Tensor, torch.Tensor]:
        previous = torch.zeros_like(inputs) if state is None else state
        next_state = previous + inputs
        return NeuronResult(next_state, next_state)


def apply(neuron: Neuron[torch.Tensor, torch.Tensor, object], inputs: torch.Tensor):
    return neuron.step(inputs)


def test_stateless_and_stateful_implementations_share_a_structural_caller() -> None:
    inputs = torch.ones(2, 3)
    stateless = LinearNeuron()
    stateful = AccumulatingNeuron()

    assert isinstance(stateless, Neuron)
    assert isinstance(stateful, Neuron)
    assert apply(stateless, inputs).value.shape == (2, 2)
    result = apply(stateful, inputs)
    assert torch.equal(result.value, inputs)
    assert torch.equal(result.state, inputs)


def test_state_reset_is_controlled_by_the_caller() -> None:
    neuron = AccumulatingNeuron()
    inputs = torch.ones(1)
    first = neuron.step(inputs)
    continued = neuron.step(inputs, state=first.state)
    reset = neuron.step(inputs, state=None)

    assert torch.equal(first.value, torch.ones(1))
    assert torch.equal(continued.value, torch.full((1,), 2.0))
    assert torch.equal(reset.value, first.value)


def test_protocol_does_not_break_autograd_for_trainable_implementations() -> None:
    neuron = LinearNeuron()
    inputs = torch.randn(4, 3)
    result = neuron.step(inputs)
    result.value.square().mean().backward()

    assert result.state is None
    assert neuron.linear.weight.grad is not None
    assert torch.isfinite(neuron.linear.weight.grad).all()


def test_stateful_candidate_composes_and_resets_through_explicit_state() -> None:
    torch.manual_seed(19)
    neuron = StatefulMLPNeuron(3, 8, 2)
    inputs = torch.randn(4, 5, 3)
    baseline_result = apply(MLPNeuron(3, 8, 2), inputs[:, 0])
    candidate_result = apply(neuron, inputs[:, 0])
    assert baseline_result.value.shape == candidate_result.value.shape == (4, 2)
    state = None
    outputs = []
    for step in range(inputs.shape[1]):
        result = neuron.step(inputs[:, step], state=state)
        outputs.append(result.value)
        state = result.state

    continued = neuron.step(inputs[:, 0], state=state)
    reset = neuron.step(inputs[:, 0], state=None)
    assert continued.state is not None
    assert reset.state is not None
    assert not torch.allclose(continued.value, reset.value)
    assert torch.allclose(reset.value, outputs[0])

    torch.stack(outputs, dim=1).square().mean().backward()
    assert all(parameter.grad is not None for parameter in neuron.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in neuron.parameters())


def test_stateful_candidate_rejects_state_shape_or_dtype_mismatches() -> None:
    neuron = StatefulMLPNeuron(3, 8, 2)
    inputs = torch.ones(4, 3)

    with pytest.raises(ValueError, match="state must have shape"):
        neuron.step(inputs, state=torch.zeros(4, 7))
    with pytest.raises(ValueError, match="device and dtype"):
        neuron.step(inputs, state=torch.zeros(4, 8, dtype=torch.float64))


def test_sequential_composition_keeps_per_neuron_state_and_batch_isolation() -> None:
    torch.manual_seed(43)
    composition = SequentialNeuron(
        [MLPNeuron(3, 8, 3), StatefulMLPNeuron(3, 8, 2)]
    )
    inputs = torch.randn(4, 3)
    first = composition.step(inputs)
    continued = composition.step(inputs, state=first.state)
    reset = composition.step(inputs, state=None)

    assert isinstance(composition, Neuron)
    assert first.value.shape == (4, 2)
    assert first.state[0] is None
    assert first.state[1] is not None and first.state[1].shape == (4, 8)
    assert not torch.allclose(first.value, continued.value)
    assert torch.allclose(first.value, reset.value)

    separate = torch.cat(
        [composition.step(inputs[index : index + 1]).value for index in range(4)],
        dim=0,
    )
    assert torch.allclose(first.value, separate)
    first.value.square().mean().backward()
    assert all(parameter.grad is not None for parameter in composition.parameters())


def test_sequential_composition_rejects_misaligned_state_bundles() -> None:
    composition = SequentialNeuron([MLPNeuron(3, 8, 2)])

    with pytest.raises(ValueError, match="one entry per neuron"):
        composition.step(torch.ones(2, 3), state=())


def test_capabilities_and_task_adapters_are_independently_replaceable() -> None:
    class ScalarAdapter:
        def encode(self, task_input: float) -> torch.Tensor:
            return torch.tensor([[task_input]])

        def decode(self, capability_output: torch.Tensor) -> float:
            return float(capability_output.item())

    class DoubleCapability:
        def execute(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs * 2

    class SquareCapability:
        def execute(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs.square()

    def run(adapter: TaskAdapter, capability: Capability, task_input: float) -> float:
        return adapter.decode(capability.execute(adapter.encode(task_input)))

    adapter = ScalarAdapter()
    double = DoubleCapability()
    square = SquareCapability()
    assert isinstance(adapter, TaskAdapter)
    assert isinstance(double, Capability)
    assert isinstance(square, Capability)
    assert run(adapter, double, 3.0) == 6.0
    assert run(adapter, square, 3.0) == 9.0


def test_regression_task_adapter_and_capability_preserve_their_boundary() -> None:
    adapter = RegressionTaskAdapter()
    capability = RegressionCapability(MLPNeuron(2, 8, 1))
    examples = (
        RegressionExample((1.0, 2.0), 0.5),
        RegressionExample((-1.0, 3.0), -4.0),
    )

    inputs = adapter.encode(examples)
    predictions = capability.execute(inputs)

    assert isinstance(capability, Capability)
    assert isinstance(adapter, TaskAdapter)
    assert inputs.shape == (2, 2)
    assert predictions.shape == (2, 1)
    assert len(adapter.decode(predictions)) == 2
    with pytest.raises(ValueError, match="must not be empty"):
        adapter.encode(())


def test_stateful_sequence_capability_preserves_order_and_task_boundary() -> None:
    adapter = SequenceTaskAdapter()
    capability = SequenceSumCapability(StatefulMLPNeuron(1, 8, 1))
    examples = (
        SequenceExample((1.0, 2.0, 3.0), 6.0),
        SequenceExample((-1.0, 0.5, 2.0), 1.5),
    )

    inputs = adapter.encode(examples)
    predictions = capability.execute(inputs)

    assert isinstance(capability, Capability)
    assert isinstance(adapter, TaskAdapter)
    assert inputs.shape == (2, 3, 1)
    assert predictions.shape == (2, 1)
    assert len(adapter.decode(predictions)) == 2
    with pytest.raises(ValueError, match="same non-empty"):
        adapter.encode((examples[0], SequenceExample((), 0.0)))
