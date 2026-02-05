"""
LIF Sine Wave Demo - Continuous Input Response
==============================================

This demo shows how a LIF neuron responds to a smoothly varying sine wave input.
Useful for demonstrating:
- Continuous integration/leak dynamics
- How firing rate varies with input amplitude
- Phase relationship between input and voltage

The LIF equation: v'(t) = (1/τ)I(t) - (1/τ)v(t)
"""

import numpy as np
import nengo

# ===========================================
# SINE WAVE PARAMETERS (edit these!)
# ===========================================
SINE_AMPLITUDE = 2.0   # Peak amplitude (try 0.4, 0.8, 1.2)
SINE_OFFSET = 0.0      # DC offset (shifts wave up/down)
SINE_FREQUENCY = 5.0   # Frequency in Hz (try 0.5, 1.0, 2.0)


def sine_function(t):
    """Sine wave input: offset + amplitude * sin(2*pi*freq*t)"""
    return SINE_OFFSET + SINE_AMPLITUDE * np.sin(2 * np.pi * SINE_FREQUENCY * t)


model = nengo.Network(label="LIF Sine Wave Demo")

with model:
    input_current = nengo.Node(output=sine_function, label="Sine Input")

    neuron = nengo.Ensemble(
        n_neurons=1,
        dimensions=1,
        neuron_type=nengo.LIF(
            tau_rc=0.01,
            tau_ref=0.002,
        ),
        gain=[1.0],
        bias=[0.0],
        encoders=[[1]],
        label="LIF Neuron"
    )

    nengo.Connection(input_current, neuron.neurons, synapse=None)

    # ===========================================
    # PROBES
    # ===========================================
    input_probe = nengo.Probe(input_current, label="Input")
    voltage_probe = nengo.Probe(neuron.neurons, 'voltage', label="Membrane Voltage")
    spike_probe = nengo.Probe(neuron.neurons, label="Spikes")


# ===========================================
# EXPERIMENTS
# ===========================================
"""
1. SUBTHRESHOLD (AMPLITUDE=0.4, OFFSET=0.5):
   - Input oscillates between 0.1 and 0.9
   - Voltage follows smoothly, never fires

2. CROSSING THRESHOLD (AMPLITUDE=0.5, OFFSET=0.7):
   - Input peaks above 1.0
   - Neuron fires only during high phase

3. FREQUENCY EFFECTS:
   - Low freq (0.5 Hz): voltage tracks input closely
   - High freq (3.0 Hz): voltage can't keep up (filtering effect)
   - This demonstrates the neuron as a low-pass filter!

4. ALWAYS FIRING (AMPLITUDE=0.3, OFFSET=1.2):
   - Input always above threshold
   - Firing rate modulates with input
"""
