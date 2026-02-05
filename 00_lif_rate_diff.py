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

model = nengo.Network(label="LIF Demo")

with model:
    # ===========================================
    # INPUT
    # ===========================================
    input_current = nengo.Node(output=0.5, label="Input Current")

    # Common parameters for both neurons
    tau_rc = 0.02
    tau_ref = 0.002

    # ===========================================
    # STANDARD LIF NEURON (Spiking)
    # ===========================================
    lif_neuron = nengo.Ensemble(
        n_neurons=1, dimensions=1,
        neuron_type=nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref),
        gain=[1.0], bias=[0.0], encoders=[[1]],
        label="LIF Neuron"
    )

    # ===========================================
    # RATE NEURON (Non-Spiking Comparison)
    # ===========================================
    rate_neuron = nengo.Ensemble(
        n_neurons=1, dimensions=1,
        neuron_type=nengo.LIFRate(tau_rc=tau_rc, tau_ref=tau_ref),
        encoders=[[1]],
        label="Rate Neuron"
    )
    
    # ===========================================
    # CONNECTIONS
    # ===========================================
    nengo.Connection(input_current, lif_neuron.neurons, synapse=None)
    nengo.Connection(input_current, rate_neuron.neurons, synapse=None)

    # ===========================================
    # POST-SYNAPTIC SIGNALS
    # ===========================================
    SYNAPSE_TAU = 0.05

    # Probing the spiking neuron's output via a synapse
    lif_post = nengo.Node(size_in=1, label="LIF Post-Synaptic")
    nengo.Connection(lif_neuron.neurons, lif_post,
                     synapse=nengo.Lowpass(SYNAPSE_TAU))

    # ===========================================
    # PROBES
    # ===========================================
    input_probe = nengo.Probe(input_current, label="Input")

    # Spiking Probes
    lif_voltage = nengo.Probe(lif_neuron.neurons, 'voltage')
    lif_spikes = nengo.Probe(lif_neuron.neurons)
    lif_post_probe = nengo.Probe(lif_post)

    # Rate Probes
    # Note: rate_neuron.neurons outputs the rate directly
    rate_probe = nengo.Probe(rate_neuron.neurons, label="Rate Output")



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
