# Modelized Neuron Graph Validation v9

## Question

Can the sparse graph composition execute one shared `ModelNeuron` architecture and one neuron weight set at every neuron identity, learn useful links from task loss, and carry topology and state across reordered graph frames?

This is the first experiment in this series that runs `ModelNeuronGraph`; v5–v8 used shared graph operators over node vectors and are not evidence for modelized neurons.

## Architecture contract

- One `ModelNeuron` instance and one neuron parameter set are applied at every graph position. Each stable identity has separate recurrent state, remapped by identity after slot permutations.
- Incoming messages of any represented degree are aggregated inside `ModelNeuron`; aggregation is not another model.
- The experiment also has a link proposal policy and scalar task readout. They are graph/task components, not per-neuron weights.
- The report's 11,102 total parameters includes the neuron, link policy, and readout. The neuron parameter set is instantiated once regardless of graph size.

## Protocol

The MPS run used six fresh seeds for single-link retrieval and two-link composition, 1,200 training steps per model, validation-selected action thresholds, and held-out task/structure checks. It compared the graph model with parameter-matched and Linear-MAC-matched flat MLP controls. Separate evaluations used 4, 16, 32, and 64 memory neurons. Identity slot order changed independently per frame; each rollout carried topology and state for 32 frames across two sequences per seed/task. The identity registry was saved, reloaded, and extended after restart.

The sealed test outcomes did not tune thresholds. The same selected threshold was used for the pool generalization checks.

## Results

| Task | Graph test MSE | Relative MSE delta vs parameter-matched MLP (95% CI) | Relative delta vs MAC-matched MLP (95% CI) | Edge F1 | Operation recall | Strength MAE | Structure gate |
|---|---:|---:|---:|---:|---|---:|---|
| Single-link retrieval | 0.1191 | -4.3% (-9.7%, +1.1%) | -21.9% (-32.5%, -11.2%) | 0.128 | remove 0.643; retype/set-strength 0.000 | 0.350 | Fail |
| Two-link composition | 0.1659 | -6.4% (-16.3%, +3.5%) | -18.8% (-29.4%, -8.2%) | 0.189 | remove 0.290; set-strength 0.179; add/retype 0.000 | 0.377 | Fail |

Negative MSE deltas favor the graph model. It beats the MAC-matched control on single-link retrieval under the preset CI rule, but neither task beats the parameter-matched control with a CI wholly below zero. The edge and action metrics remain far below their preset thresholds. The single-link task has no expected add operation; zero add recall there is not evidence of a missed positive. In two-link composition, add recall is zero despite positive examples.

Permutation-relative MSE drift was 0.000 in both tasks, but identity-edge Jaccard was only 0.374 and 0.737. Task outputs are slot invariant in this check; learned hard topology is not reliably identity invariant. Mean rollout edge F1 was 0.012 for single-link and 0.030 for two-link. Some seeds removed all edges, so sustained task quality and useful topology evolution are not established.

At 4/16/32/64 memory neurons, total parameter count stayed at 11,102. MPS median forward latency was 8.04/8.21/7.88/8.39 ms; observed maxima (reported as p95 for these nine-call samples) were 8.36/8.44/8.00/8.50 ms. Mean training duration was 12.42 seconds for the graph model, 1.46 seconds for the parameter-matched MLP, and 2.01 seconds for the MAC-matched MLP. This is a small synthetic microbenchmark, not a broad performance claim.

Registry disk round-trip and post-restart ID allocation passed. The test used a local JSON registry and does not cover concurrent writers or remote persistence.

## Decision

The shared-weight modelized-neuron execution and state/identity remapping contract is demonstrated on MPS. The claim that this model can learn useful dynamic connectivity is **not** demonstrated. The hard policy often collapses to removing all edges; structure, operation, and rollout gates fail. Do not promote this as a completed neuron capability or proceed to broader model claims.

Keep one shared neuron weight set as the default. This run provides no evidence that per-identity weight sets would fix the link-policy collapse; adding them would increase storage and confound the next diagnosis. The next iteration should isolate link-policy/action learning and calibration, test whether its gradients and action distributions are healthy, and preserve the shared neuron weights while doing so.

## Artifacts and reproducibility

- Config: `project/configs/models_training/learning/neuron_comparisons/topology_evolution/modelized_neuron_shared_weights.json`
- Runner: `project/scripts/experiments/training/neurons/run_modelized_neuron_graph_validation.py`
- Raw results: `results.json` (includes config/source SHA-256 values and per-seed records)
- Registry after restart check: `identity_registry.json`
- Device: Apple MPS; PyTorch profiler reported CPU activity only. FLOPs are reported only as a Linear/GRU MAC subtotal; total MPS FLOPs are not claimed.
