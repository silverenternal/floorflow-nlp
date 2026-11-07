from __future__ import annotations

import torch
from torch import nn

from newera.capabilities.evaluation import load_checkpoint, save_checkpoint, train_with_accumulation
from newera.neuron.composition import ModelNeuron, ModelNeuronGraph, ModelNeuronGraphState


class Samples:
    total_samples = 12

    def __init__(self, generator: torch.Generator, count: int) -> None:
        self.generator = generator
        self.count = count

    def __iter__(self):
        for _ in range(self.count):
            index = torch.randint(0, 8, (1,), generator=self.generator)
            yield index.float()


def _loss(model: nn.Module, sample: torch.Tensor, _step: int) -> torch.Tensor:
    return (model(sample) - (2 * sample + 1)).square().mean()


def _train_segment(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    *,
    steps: int,
    start_step: int = 0,
    on_step=None,
) -> None:
    train_with_accumulation(
        model,
        Samples(generator, steps * 3),
        optimizer=optimizer,
        loss_for=_loss,
        accumulation_steps=3,
        start_step=start_step,
        on_step=on_step,
    )


def test_accumulation_matches_mean_batch_gradient() -> None:
    model = nn.Linear(1, 1)
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    batch = torch.arange(1, 5, dtype=torch.float32).reshape(-1, 1)
    labels = 2 * batch + 1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    nn.functional.mse_loss(model(batch), labels).backward()
    optimizer.step()
    expected_state = {key: value.clone() for key, value in model.state_dict().items()}

    model.load_state_dict(initial)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    samples = [batch[index : index + 1] for index in range(batch.shape[0])]
    train_with_accumulation(
        model,
        samples,
        optimizer=optimizer,
        loss_for=lambda current, sample, _step: nn.functional.mse_loss(
            current(sample), 2 * sample + 1
        ),
        accumulation_steps=4,
    )
    for key, expected in expected_state.items():
        torch.testing.assert_close(model.state_dict()[key], expected)


def test_checkpoint_resume_replays_same_sample_sequence(tmp_path) -> None:
    torch.manual_seed(71)
    full_model = nn.Linear(1, 1)
    resumed_model = nn.Linear(1, 1)
    resumed_model.load_state_dict(full_model.state_dict())
    full_optimizer = torch.optim.AdamW(full_model.parameters(), lr=0.01)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.01)
    full_generator = torch.Generator(device="cpu").manual_seed(901)
    resumed_generator = torch.Generator(device="cpu").manual_seed(901)

    checkpoint_path = tmp_path / "checkpoint.pt"

    def save_at(step: int, samples_seen: int) -> None:
        if step == 2:
            save_checkpoint(
                checkpoint_path,
                model=resumed_model,
                optimizer=resumed_optimizer,
                step=step,
                samples_seen=samples_seen,
                sample_generator=resumed_generator,
                caller_state={"neuron_states": torch.tensor([[1.0, 2.0]])},
                metadata={"task": "linear-recall"},
            )

    _train_segment(
        resumed_model,
        resumed_optimizer,
        resumed_generator,
        steps=2,
        on_step=save_at,
    )
    _train_segment(full_model, full_optimizer, full_generator, steps=4)

    state = load_checkpoint(
        checkpoint_path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        sample_generator=resumed_generator,
        expected_metadata={"task": "linear-recall"},
    )
    assert state["step"] == 2
    torch.testing.assert_close(
        state["caller_state"]["neuron_states"], torch.tensor([[1.0, 2.0]])
    )
    _train_segment(
        resumed_model,
        resumed_optimizer,
        resumed_generator,
        steps=2,
        start_step=state["step"],
    )
    for key, expected in full_model.state_dict().items():
        torch.testing.assert_close(resumed_model.state_dict()[key], expected)
    assert torch.equal(resumed_generator.get_state(), full_generator.get_state())


def test_graph_state_remaps_by_identity_and_new_instances_start_empty():
    graph = ModelNeuronGraph(ModelNeuron(input_size=2, hidden_size=4, message_size=3))
    state = ModelNeuronGraphState(
        neuron_states=torch.arange(12, dtype=torch.float32).reshape(1, 3, 4),
        previous_values=torch.arange(9, dtype=torch.float32).reshape(1, 3, 3),
        node_ids=torch.tensor([10, 20, 30]),
    )
    remapped = graph.remap_state(state, torch.tensor([[30, 10, 40]]))
    assert remapped.node_ids.tolist() == [30, 10, 40]
    torch.testing.assert_close(remapped.neuron_states[:, 0], state.neuron_states[:, 2])
    torch.testing.assert_close(remapped.neuron_states[:, 1], state.neuron_states[:, 0])
    assert torch.count_nonzero(remapped.neuron_states[:, 2]) == 0
    assert torch.count_nonzero(remapped.previous_values[:, 2]) == 0


def test_graph_weights_and_caller_owned_state_round_trip_separately(tmp_path):
    model = ModelNeuronGraph(ModelNeuron(input_size=2, hidden_size=4, message_size=3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    generator = torch.Generator().manual_seed(17)
    state = ModelNeuronGraphState(
        neuron_states=torch.randn(1, 2, 4),
        previous_values=torch.randn(1, 2, 3),
        node_ids=torch.tensor([51, 73]),
    )
    checkpoint = tmp_path / "graph_state.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        step=3,
        samples_seen=9,
        sample_generator=generator,
        caller_state={"graph": state},
    )
    restored = ModelNeuronGraph(ModelNeuron(input_size=2, hidden_size=4, message_size=3))
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01)
    payload = load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        sample_generator=torch.Generator().manual_seed(0),
    )
    restored_state = payload["caller_state"]["graph"]
    assert torch.equal(restored_state.node_ids, state.node_ids)
    torch.testing.assert_close(restored_state.neuron_states, state.neuron_states)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
