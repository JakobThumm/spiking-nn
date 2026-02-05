"""
MNIST Spiking Classification - Nengo GUI Visualization
======================================================

Loads pre-trained weights (from 05_mnist_train.py) and builds a
standard nengo spiking network for real-time visualization in nengo-gui.

Architecture:
    Input Node (784) -> Ensemble(256 LIF) -> Ensemble(128 LIF) -> Output Node (10)

The input cycles through test digit images. Watch the spiking activity
in each layer and the classification output in real-time.

Prerequisites:
    Run 05_mnist_train.py first to generate:
    - mnist_weights.npz (trained weights)
    - mnist_test_images.npz (sample test images)

Usage:
    ./venv/bin/nengo 05_mnist_gui.py
"""

import os
import numpy as np
import nengo

# ===========================================
# LOAD TRAINED WEIGHTS AND TEST IMAGES
# ===========================================
script_dir = os.path.dirname(os.path.abspath(__file__))

weights_path = os.path.join(script_dir, "mnist_weights.npz")
images_path = os.path.join(script_dir, "mnist_test_images.npz")

if not os.path.exists(weights_path) or not os.path.exists(images_path):
    raise FileNotFoundError(
        "Missing trained weights! Run 05_mnist_train.py first:\n"
        "  ./venv/bin/python 05_mnist_train.py"
    )

data = np.load(weights_path)
w1, b1 = data["w1"], data["b1"]           # (784, 256), (256,)
w2, b2 = data["w2"], data["b2"]           # (256, 128), (128,)
w_out, b_out = data["w_out"], data["b_out"]  # (128, 10), (10,)

ens1_gain = data["ens1_gain"]       # (256,)
ens1_bias = data["ens1_bias"]       # (256,)
ens1_encoders = data["ens1_encoders"]  # (256, 1)
ens2_gain = data["ens2_gain"]       # (128,)
ens2_bias = data["ens2_bias"]       # (128,)
ens2_encoders = data["ens2_encoders"]  # (128, 1)

img_data = np.load(images_path)
test_images = img_data["images"]  # (30, 784) - 3 per digit
test_labels = img_data["labels"]  # (30,)

# ===========================================
# PARAMETERS
# ===========================================
IMAGE_DISPLAY_TIME = 0.2  # Show each image for 1 second
LIF_AMPLITUDE = 0.01     # Must match training amplitude

# ===========================================
# INPUT FUNCTION
# ===========================================
def mnist_input(t):
    """Return current MNIST image pixels as a flat 784-vector.

    Also sets _nengo_html_ to display the image in nengo-gui.
    Right-click the node and select "HTML" to see it.
    """
    # noqa: this function has _nengo_html_ initialized below
    idx = int(t / IMAGE_DISPLAY_TIME) % len(test_images)
    pixels = test_images[idx]
    label = int(test_labels[idx])

    # Render 28x28 image as SVG for nengo-gui HTML view
    rows = []
    for y in range(28):
        for x in range(28):
            v = int(pixels[y * 28 + x] * 255)
            if v > 0:
                rows.append(
                    f'<rect x="{x*4}" y="{y*4}" width="4" height="4" '
                    f'fill="rgb({v},{v},{v})"/>'
                )
    svg = (
        '<svg width="100%" height="100%" viewBox="0 0 140 140" '
        'style="background:black">'
        + "".join(rows)
        + f'<text x="70" y="135" text-anchor="middle" fill="white" '
        f'font-size="12">Label: {label}</text>'
        + "</svg>"
    )
    mnist_input._nengo_html_ = svg

    return pixels

# Initialize _nengo_html_ so nengo-gui detects it at load time
mnist_input._nengo_html_ = ""


def current_label(t):
    """Return the true label of the current image (for display)."""
    idx = int(t / IMAGE_DISPLAY_TIME) % len(test_labels)
    return test_labels[idx]


# ===========================================
# BUILD THE NENGO MODEL
# ===========================================
#
# In nengo-dl training, Layer(Dense(N)) creates a TensorNode computing
# W@x + b_dense, and Layer(LIF) creates an Ensemble with .neurons.
# The connection from TensorNode to .neurons has no transform, so:
#
#   neuron_input = bias_ens + (W @ x + b_dense)
#
# To replicate in standard nengo, we connect to .neurons with
# transform=W^T, and fold b_dense into the ensemble bias.
# Connections to .neurons bypass encoders/gain entirely.
#
model = nengo.Network(label="MNIST Spiking Classifier", seed=0)

with model:
    # --- Input ---
    input_node = nengo.Node(output=mnist_input, size_out=784, label="MNIST Image")
    label_node = nengo.Node(output=current_label, size_out=1, label="True Label")

    # --- Layer 1: 256 LIF neurons ---
    with nengo.Network(label="Layer 1 (256 LIF)"):
        # Fold the Dense bias (b1) into the ensemble bias
        # neuron_input = bias_ens + b1 + W1 @ image
        combined_bias1 = ens1_bias + b1

        layer1 = nengo.Ensemble(
            n_neurons=256,
            dimensions=1,
            neuron_type=nengo.LIF(amplitude=LIF_AMPLITUDE),
            gain=ens1_gain,
            bias=combined_bias1,
            encoders=ens1_encoders,
            label="Hidden 1",
        )
        # W1 is (784, 256) -> transpose to (256, 784) for the connection
        nengo.Connection(input_node, layer1.neurons,
                         transform=w1.T, synapse=None)

    # --- Layer 2: 128 LIF neurons ---
    with nengo.Network(label="Layer 2 (128 LIF)"):
        combined_bias2 = ens2_bias + b2

        layer2 = nengo.Ensemble(
            n_neurons=128,
            dimensions=1,
            neuron_type=nengo.LIF(amplitude=LIF_AMPLITUDE),
            gain=ens2_gain,
            bias=combined_bias2,
            encoders=ens2_encoders,
            label="Hidden 2",
        )
        # W2 is (256, 128) -> transpose to (128, 256) for the connection
        nengo.Connection(layer1.neurons, layer2.neurons,
                         transform=w2.T, synapse=0.01)

    # --- Output layer ---
    output_node = nengo.Node(size_in=10, label="Class Scores")

    # W_out is (128, 10) -> transpose to (10, 128) for the connection
    nengo.Connection(layer2.neurons, output_node,
                     transform=w_out.T, synapse=0.01)

    # Add the output bias as a constant input
    bias_node = nengo.Node(output=b_out, label="Output Bias")
    nengo.Connection(bias_node, output_node, synapse=None)

    # --- Classification result ---
    def argmax_func(t, x):
        return np.argmax(x)

    prediction_node = nengo.Node(argmax_func, size_in=10, size_out=1,
                                 label="Prediction")
    nengo.Connection(output_node, prediction_node, synapse=0.05)

    # ===========================================
    # PROBES
    # ===========================================
    input_probe = nengo.Probe(input_node, label="Image Pixels")
    label_probe = nengo.Probe(label_node, label="True Label")
    l1_spikes = nengo.Probe(layer1.neurons, label="Layer 1 Spikes")
    l2_spikes = nengo.Probe(layer2.neurons, label="Layer 2 Spikes")
    output_probe = nengo.Probe(output_node, synapse=0.05, label="Class Scores")
    pred_probe = nengo.Probe(prediction_node, label="Prediction")
