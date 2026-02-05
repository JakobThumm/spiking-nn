"""
LIF Direct Current Demo - Textbook-Style Behavior
==================================================

This demo gives you direct control over the input current to a LIF neuron,
matching the textbook equation more closely:

    v'(t) = (1/τ)I(t) - (1/τ)v(t)

In standard Nengo ensembles, gain and bias are automatically computed
from max_rates and intercepts, which can be confusing for teaching.

Here we inject current directly into the neuron, so:
- Input = 0 → no current → voltage stays at 0
- Input > 0 → positive current → voltage rises
- Input = 1 → voltage rises to threshold → neuron fires

Instructions:
1. Add voltage and spike plots to the neuron
2. Watch voltage rise during "on" phase and decay during "off" phase
3. Edit STEP_AMPLITUDE, STEP_PERIOD, STEP_DUTY at the top to experiment
4. Switch to manual slider by uncommenting Option 1 and commenting Option 2
"""

import nengo

model = nengo.Network(label="LIF Direct Current Demo")

with model:
    # ===========================================
    # INPUT OPTIONS (comment/uncomment to switch)
    # ===========================================

    # Option 1: Manual slider control
    input_current = nengo.Node(output=0.5, label="Input Current")

    # ===========================================
    # SINGLE LIF NEURON (direct mode)
    # ===========================================
    # We create a neuron with gain=1 and bias=0, so input maps
    # directly to current without Nengo's automatic scaling.
    #
    # The neuron fires when total current >= 1 (normalized threshold)
    
    # Start with tau_rc=0.1, tau_ref=0.001
    # Show different tau_rc=0.01 -> faster response
    # Show spiking
    # Show different tau_ref=0.1 -> Refractory period
    # Show different bias=0.2 -> Constant offset
    # Show different gain=2.0 -> Scaling

    neuron = nengo.Ensemble(
        n_neurons=1,
        dimensions=1,
        neuron_type=nengo.LIF(
            tau_rc=0.01,    # Membrane time constant (20ms)
            tau_ref=0.01,  # Refractory period (2ms)
        ),
        # These settings make input map directly to current:
        gain=[1.0],         # No scaling: current = input
        bias=[0.0],         # No bias: zero input = zero current
        encoders=[[1]],
        label="LIF Neuron"
    )

    # Connect input directly to neuron's current
    # (bypass the normal encoding by connecting to .neurons)
    nengo.Connection(input_current, neuron.neurons, synapse=None)

    # ===========================================
    # PROBES
    # ===========================================
    input_probe = nengo.Probe(input_current, label="Input")
    voltage_probe = nengo.Probe(neuron.neurons, 'voltage', label="Membrane Voltage")
    spike_probe = nengo.Probe(neuron.neurons, label="Spikes")
