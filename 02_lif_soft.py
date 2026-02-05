"""
Soft LIF Neuron Demo
====================

Compares a standard LIF neuron with a Soft LIF neuron.

Standard LIF: Hard threshold - neuron either fires or doesn't.
Soft LIF: Smooth approximation - the threshold is "softened" using a
          sigma parameter, making the response differentiable.

The sigma parameter controls how smooth the transition is:
- sigma → 0: Approaches hard LIF behavior
- sigma large: Very smooth, gradual response

Soft LIF is used in deep learning with spiking networks because
gradient-based training (backpropagation) requires smooth,
differentiable activation functions.
"""

import nengo
from nengo_dl import SoftLIFRate

model = nengo.Network(label="Soft LIF Demo")

with model:
    # ===========================================
    # INPUT
    # ===========================================
    input_current = nengo.Node(output=0.5, label="Input Current")

    # ===========================================
    # STANDARD LIF NEURON (hard threshold)
    # ===========================================
    lif_neuron = nengo.Ensemble(
        n_neurons=1, dimensions=1,
        neuron_type=nengo.LIF(tau_rc=0.02, tau_ref=0.002),
        gain=[1.0], bias=[0.0], encoders=[[1]],
        label="LIF Neuron"
    )

    # ===========================================
    # SOFT LIF NEURON - RATE (smooth, continuous output)
    # ===========================================
    # sigma controls smoothness of the threshold
    # Try: 0.01 (nearly hard), 0.1 (moderate), 0.5 (very smooth)
    soft_neuron = nengo.Ensemble(
        n_neurons=1, dimensions=1,
        neuron_type=SoftLIFRate(sigma=0.1, tau_rc=0.02, tau_ref=0.002),
        gain=[1.0], bias=[0.0], encoders=[[1]],
        label="Soft LIF (Rate)"
    )

    # ===========================================
    # SOFT LIF NEURON - SPIKING (soft rate + spike generation)
    # ===========================================
    # Wrapping SoftLIFRate with RegularSpiking generates actual spikes
    # at intervals determined by the soft rate output
    soft_spiking = nengo.Ensemble(
        n_neurons=1, dimensions=1,
        neuron_type=nengo.RegularSpiking(
            SoftLIFRate(sigma=0.1, tau_rc=0.02, tau_ref=0.002)
        ),
        gain=[1.0], bias=[0.0], encoders=[[1]],
        label="Soft LIF (Spiking)"
    )

    # ===========================================
    # CONNECTIONS
    # ===========================================
    nengo.Connection(input_current, lif_neuron.neurons, synapse=None)
    nengo.Connection(input_current, soft_neuron.neurons, synapse=None)
    nengo.Connection(input_current, soft_spiking.neurons, synapse=None)

    # ===========================================
    # POST-SYNAPTIC SIGNALS
    # ===========================================
    SYNAPSE_TAU = 0.05

    lif_post = nengo.Node(size_in=1, label="LIF Post-Synaptic")
    nengo.Connection(lif_neuron.neurons, lif_post,
                     synapse=nengo.Lowpass(SYNAPSE_TAU))

    soft_post = nengo.Node(size_in=1, label="Soft LIF Output")
    nengo.Connection(soft_neuron.neurons, soft_post,
                     synapse=nengo.Lowpass(SYNAPSE_TAU))

    soft_spk_post = nengo.Node(size_in=1, label="Soft LIF Post-Synaptic")
    nengo.Connection(soft_spiking.neurons, soft_spk_post,
                     synapse=nengo.Lowpass(SYNAPSE_TAU))

    # ===========================================
    # PROBES
    # ===========================================
    input_probe = nengo.Probe(input_current, label="Input")

    lif_voltage = nengo.Probe(lif_neuron.neurons, 'voltage',
                              label="LIF Voltage")
    lif_spikes = nengo.Probe(lif_neuron.neurons, label="LIF Spikes")
    lif_post_probe = nengo.Probe(lif_post, label="LIF Post-Syn")

    soft_output = nengo.Probe(soft_neuron.neurons, label="Soft LIF Rate")
    soft_post_probe = nengo.Probe(soft_post, label="Soft LIF Post-Syn")

    soft_spk_spikes = nengo.Probe(soft_spiking.neurons,
                                  label="Soft LIF Spikes")
    soft_spk_voltage = nengo.Probe(soft_spiking.neurons, 'voltage',
                                   label="Soft LIF Voltage")
    soft_spk_post_probe = nengo.Probe(soft_spk_post,
                                      label="Soft LIF Spk Post-Syn")


# ===========================================
# EXPERIMENTS
# ===========================================
"""
In nengo-gui:
1. Add slider to "Input Current" (range 0 to 2)
2. Add plots for both neurons

COMPARE:
- LIF: Discrete spikes, all-or-nothing
  -> Add Voltage and Spike plots
- Soft LIF: Continuous rate output, smooth response
  -> Add Value plot to "Soft LIF Output"

KEY OBSERVATIONS:
1. Below threshold (input < 1.0):
   - LIF: No spikes at all
   - Soft LIF: Small but non-zero output (soft threshold)

2. Near threshold (input ~ 1.0):
   - LIF: Abrupt onset of spiking
   - Soft LIF: Gradual increase in output rate

3. Above threshold (input > 1.0):
   - Both produce similar output levels
   - LIF through spike rate, Soft LIF through continuous rate

WHY SOFT LIF MATTERS:
- Standard LIF has a non-differentiable threshold
- This makes gradient-based learning impossible
- Soft LIF provides a smooth approximation
- Used in training SNNs with backpropagation (e.g., nengo-dl)
"""
