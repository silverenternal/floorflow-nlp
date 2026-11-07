"""Neuron implementations used as reusable local computation units."""

from newera.neuron.cells.baselines import MLPNeuron
from newera.neuron.cells.candidates import StatefulMLPNeuron

__all__ = ["MLPNeuron", "StatefulMLPNeuron"]
