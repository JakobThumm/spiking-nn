# Spiking Neural Networks with Nengo

Interactive demonstrations of spiking neural network concepts using [Nengo](https://www.nengo.ai/) and [Nengo GUI](https://github.com/nengo/nengo-gui).

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

All demos run in Nengo GUI, which provides an interactive visualization:

```bash
nengo <script.py>
```

This opens a web browser at `http://localhost:8080` with the network visualization.

## Demos

### 1. LIF Neuron Basics — `01_lif_slider.py`

```bash
nengo 01_lif_slider.py
```

Demonstrates the Leaky Integrate-and-Fire (LIF) neuron model:

- **Voltage integration**: Input current causes membrane voltage to rise
- **Firing threshold**: When voltage exceeds 1, the neuron fires (spikes)
- **Reset**: After firing, voltage resets to zero
- **Refractory period**: Neuron cannot fire again during recovery time

**Try this**: Add a voltage plot to the neuron. Adjust the input slider and watch the voltage ramp up until it crosses threshold, then reset.

### 2. Synaptic Filtering — `02_lif_sine_synapses.py`

```bash
nengo 02_lif_sine_synapses.py
```

Shows how spikes are filtered by synapses:

- Raw spikes are discrete impulses
- Synaptic filtering smooths spikes into continuous signals
- Different time constants (`tau`) control the smoothing amount
- This is how downstream neurons "see" the spiking activity

### 3. LIF Rate Response — `02_lif_soft.py`

```bash
nengo 02_lif_soft.py
```

Compares standard LIF with SoftLIF neurons:

- **LIF firing rate curve**: Shows the non-differentiable discontinuity at I=1
  - Below threshold (I<1): zero firing rate
  - Above threshold (I>1): firing rate increases
  - The sharp corner at I=1 is problematic for gradient-based learning

- **SoftLIF**: A smoothed approximation that is differentiable everywhere
  - Enables backpropagation through spiking networks
  - Key for converting trained ANNs to SNNs

### 4. MNIST Spiking Classifier

Train a spiking neural network on MNIST, then visualize it in real-time.

#### Step 1: Train the network

```bash
./venv/bin/python 05_mnist_train.py
```

This trains a 3-layer spiking network:
- Input (784) → Dense (256) → LIF → Dense (128) → LIF → Dense (10)
- Uses rate-based training (SoftLIF approximation during backprop)
- Achieves ~98% training accuracy
- Saves weights to `mnist_weights.npz`

#### Step 2: Run the GUI visualization

```bash
nengo 05_mnist_gui.py
```

Watch the trained spiking network classify digits in real-time:
- Input cycles through test digits (one per second)
- See spike rasters in each hidden layer
- View classification scores and predictions
- Right-click "MNIST Image" node → "HTML" to see the input digit

## Key Concepts

### Why Spiking Neural Networks?

1. **Biological plausibility**: Brains use spikes, not continuous activations
2. **Energy efficiency**: Neuromorphic hardware can be orders of magnitude more efficient
3. **Temporal coding**: Information can be encoded in spike timing, not just rates

### The ANN → SNN Conversion Problem

Standard deep learning uses continuous activations and gradient descent. SNNs use discrete spikes, which are non-differentiable. The solution:

1. **Train with rate approximation**: Use SoftLIF (differentiable) during training
2. **Deploy with spikes**: Replace with actual LIF neurons for inference
3. **Temporal integration**: Run multiple timesteps to accumulate spike statistics

This is exactly what `05_mnist_train.py` does — it trains using nengo-dl's automatic rate-based approximation, then the weights transfer directly to a standard spiking nengo model.

## File Overview

| File | Description |
|------|-------------|
| `01_lif_slider.py` | Basic LIF neuron with manual input control |
| `01_lif_step.py` | LIF response to step input |
| `01_lif_sine.py` | LIF response to sinusoidal input |
| `02_lif_slider_synapses.py` | Synaptic filtering demonstration |
| `02_lif_sine_synapses.py` | Synapses with sine wave input |
| `02_lif_soft.py` | LIF vs SoftLIF comparison |
| `03_hebbian_stdp.py` | Spike-timing dependent plasticity |
| `05_mnist_train.py` | Train MNIST classifier (standalone script) |
| `05_mnist_gui.py` | Visualize trained MNIST classifier |

## Tips for Nengo GUI

- **Right-click** on any node/ensemble to add visualizations
- **Voltage plot**: Shows membrane potential over time
- **Spike raster**: Shows discrete spike events
- **Value plot**: Shows decoded/filtered output
- **Slider**: Manually control input values
- **HTML**: Custom visualizations (like the MNIST digit display)

## Requirements

- Python 3.8+
- nengo
- nengo-gui
- nengo-dl (for training)
- tensorflow 2.x (for training)
