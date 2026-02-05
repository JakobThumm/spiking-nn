"""
Hebbian Learning & STDP Demo
============================

Demonstrates "neurons that fire together, wire together" - the principle
that the timing between pre- and post-synaptic spikes determines whether
a connection strengthens or weakens.

STDP Rule:
- Pre fires BEFORE post → weight increases (causation)
- Pre fires AFTER post  → weight decreases (no causation)

This demo uses two INDEPENDENT neurons driven by phase-shifted sine waves.
By adjusting the phase shift, you control which neuron fires first.
"""

import numpy as np
import nengo

# ===========================================
# SINE WAVE PARAMETERS
# ===========================================
FREQUENCY = 0.5        # Hz - how fast the neurons cycle
AMPLITUDE = 1.01        # Above 1.0 to ensure firing at peaks
PHASE_SHIFT = 0.005      # Fraction of period (0.1 = pre leads by 10%)
                       # Positive = pre fires first (potentiation)
                       # Negative = post fires first (depression)
                       # Try: -0.01, -0.005, 0, 0.005, 0.01

# ===========================================
# STDP PARAMETERS
# ===========================================
A_PLUS = 1       # Learning rate for potentiation
A_MINUS = 1      # Learning rate for depression
TAU_PLUS = 0.01     # Time constant (20ms)
TAU_MINUS = 0.01
W_MIN = 0.0
W_MAX = 1.0
W_INIT = 0.5


def pre_sine(t):
    """Pre-synaptic input: sine wave (leading)."""
    return AMPLITUDE * (0.5 + 0.5 * np.sin(2 * np.pi * FREQUENCY * t))


def post_sine(t):
    """Post-synaptic input: sine wave (phase shifted)."""
    phase = t - PHASE_SHIFT / FREQUENCY  # Shift in time
    return AMPLITUDE * (0.5 + 0.5 * np.sin(2 * np.pi * FREQUENCY * phase))


class STDPTracker:
    """Tracks spike timing and computes STDP weight."""

    def __init__(self):
        self.weight = W_INIT
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self.last_t = 0.0

    def __call__(self, t, x):
        pre_spike, post_spike = x[0], x[1]

        # Time step
        dt = max(t - self.last_t, 0.001)
        self.last_t = t

        # Decay traces
        self.pre_trace *= np.exp(-dt / TAU_PLUS)
        self.post_trace *= np.exp(-dt / TAU_MINUS)

        # STDP updates
        if post_spike > 0:
            # Post spike: potentiate based on recent pre activity
            self.weight += A_PLUS * self.pre_trace * (W_MAX - self.weight)
            self.post_trace += 1.0

        if pre_spike > 0:
            # Pre spike: depress based on recent post activity
            self.weight -= A_MINUS * self.post_trace * (self.weight - W_MIN)
            self.pre_trace += 1.0

        self.weight = np.clip(self.weight, W_MIN, W_MAX)
        return self.weight


stdp_tracker = STDPTracker()

model = nengo.Network(label="STDP Demo")

with model:
    # ===========================================
    # INPUTS (phase-shifted sine waves)
    # ===========================================
    pre_input = nengo.Node(output=pre_sine, label="Pre Input")
    post_input = nengo.Node(output=post_sine, label="Post Input")

    # ===========================================
    # INDEPENDENT NEURONS (not connected to each other)
    # ===========================================
    pre_neuron = nengo.Ensemble(
        n_neurons=1, dimensions=1,
        neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002),
        gain=[1.0], bias=[0.0], encoders=[[1]],
        label="Pre Neuron"
    )

    post_neuron = nengo.Ensemble(
        n_neurons=1, dimensions=1,
        neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002),
        gain=[1.0], bias=[0.0], encoders=[[1]],
        label="Post Neuron"
    )

    nengo.Connection(pre_input, pre_neuron.neurons, synapse=None)
    nengo.Connection(post_input, post_neuron.neurons, synapse=None)

    # ===========================================
    # STDP WEIGHT COMPUTATION
    # ===========================================
    weight_node = nengo.Node(stdp_tracker, size_in=2, size_out=1,
                             label="STDP Weight")

    nengo.Connection(pre_neuron.neurons, weight_node[0], synapse=None)
    nengo.Connection(post_neuron.neurons, weight_node[1], synapse=None)

    # ===========================================
    # PROBES
    # ===========================================
    pre_spikes = nengo.Probe(pre_neuron.neurons, label="Pre Spikes")
    post_spikes = nengo.Probe(post_neuron.neurons, label="Post Spikes")
    weight_probe = nengo.Probe(weight_node, label="Weight")


# ===========================================
# EXPERIMENTS
# ===========================================
"""
Adjust PHASE_SHIFT at the top of the file and reload:

PHASE_SHIFT = 0.1  → Pre fires before Post → Weight INCREASES
PHASE_SHIFT = 0.0  → Neurons fire together → Weight stable
PHASE_SHIFT = -0.1 → Post fires before Pre → Weight DECREASES

In nengo-gui:
1. Add Value plots for "Pre Input" and "Post Input"
   (see the phase difference)
2. Add Spike rasters for both neurons
   (see timing of spikes)
3. Add Value plot for "STDP Weight"
   (watch it rise or fall based on timing)

The key insight: causality matters!
- If pre consistently fires before post, the brain interprets this as
  "pre causes post to fire" and strengthens the connection.
- If post fires before pre, there's no causal relationship, so the
  connection weakens.
"""
