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

# ===========================================
# STEP FUNCTION PARAMETERS (edit these!)
# ===========================================
STEP_AMPLITUDE = 0.8   # Height of the step (try 0.5, 0.8, 1.2)
STEP_PERIOD = 0.5      # Time for one full cycle in seconds
STEP_DUTY = 0.5        # Fraction of period that is "on" (0.5 = 50%)


def step_function(t):
    """Repeating step/square wave input."""
    phase = (t % STEP_PERIOD) / STEP_PERIOD
    return STEP_AMPLITUDE if phase < STEP_DUTY else 0.0


model = nengo.Network(label="LIF Direct Current Demo")

with model:
    # ===========================================
    # INPUT OPTIONS (comment/uncomment to switch)
    # ===========================================
    input_current = nengo.Node(output=step_function, label="Step Input")

    # ===========================================
    # SINGLE LIF NEURON (direct mode)
    # ===========================================
    # We create a neuron with gain=1 and bias=0, so input maps
    # directly to current without Nengo's automatic scaling.
    #
    # The neuron fires when total current >= 1 (normalized threshold)

    neuron = nengo.Ensemble(
        n_neurons=1,
        dimensions=1,
        neuron_type=nengo.LIF(
            tau_rc=0.1,    # Membrane time constant (20ms)
            tau_ref=0.1,  # Refractory period (2ms)
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


# ===========================================
# EXPECTED BEHAVIOR
# ===========================================
"""
Now the voltage should behave as expected:

- Input = 0: Voltage stays at 0 (no current, no integration)
- Input = 0.5: Voltage rises to ~0.5 and stabilizes (leak = input)
- Input = 0.8: Voltage rises to ~0.8 and stabilizes
- Input = 1.0: Voltage reaches threshold (1.0) → neuron fires!
- Input > 1.0: Neuron fires repeatedly, faster with more current

The steady-state voltage (when not firing) equals the input current,
because at equilibrium: leak = input → v = I

Try these experiments with the step function:

1. SUBTHRESHOLD PULSES (STEP_AMPLITUDE = 0.8):
   - Voltage rises during "on", decays during "off"
   - Never reaches threshold, no spikes

2. SUPRATHRESHOLD PULSES (STEP_AMPLITUDE = 1.2):
   - Voltage crosses threshold → neuron fires during "on"
   - Voltage decays during "off" phase

3. INTEGRATION OVER TIME (STEP_AMPLITUDE = 0.9, STEP_DUTY = 0.8):
   - Long "on" periods let voltage build up closer to threshold
   - Short "off" periods don't allow full decay

4. EFFECT OF TAU_RC:
   - Larger tau_rc (0.2): slower rise/decay, more "memory"
   - Smaller tau_rc (0.02): faster dynamics, less integration

5. MANUAL MODE:
   - Uncomment Option 1, comment Option 2
   - Use slider for direct control
"""
