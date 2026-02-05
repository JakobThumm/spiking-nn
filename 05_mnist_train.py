"""
MNIST Spiking Neural Network Training
======================================

Trains a dense spiking neural network on MNIST using nengo-dl.

Architecture:
    Input(784) -> Dense(256) -> LIF -> Dense(128) -> LIF -> Dense(10)

During training, LIF neurons are automatically swapped for their
rate-based equivalents (smooth approximation). After training,
we evaluate with actual spiking neurons over multiple timesteps.

Saves trained weights to mnist_weights.npz and sample test images
to mnist_test_images.npz for use with the nengo-gui visualization.

Usage:
    ./venv/bin/python 05_mnist_train.py
"""

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf

# Fix numpy 1.20+ / TF 2.5 compatibility: _layers renamed, symbolic tensor issue
if not hasattr(tf.keras.layers.Layer, "_layers"):
    tf.keras.layers.Layer._layers = property(
        lambda self: self._self_tracked_trackables
    )
import tensorflow.python.ops.array_ops as _array_ops
_orig_constant_if_small = _array_ops._constant_if_small
def _patched_constant_if_small(value, shape, dtype, name):
    try:
        resolved = [int(s) for s in shape]
        if np.prod(resolved) < 1000:
            return _array_ops.constant(value, shape=resolved, dtype=dtype, name=name)
    except (TypeError, ValueError):
        pass
    return None
_array_ops._constant_if_small = _patched_constant_if_small

import nengo
import nengo_dl


def build_network():
    """Build a dense spiking MNIST classification network.

    Returns the network and references to the Dense layers
    so we can extract trained weights later.
    """
    with nengo.Network(seed=0) as net:
        nengo_dl.configure_settings(stateful=False)

        # Input node: 784 pixels, flattened 28x28 image
        net.inp = nengo.Node(np.zeros(28 * 28))

        # Layer 1: Dense(256) + LIF
        net.dense1 = tf.keras.layers.Dense(units=256, name="dense1")
        x = nengo_dl.Layer(net.dense1)(net.inp)
        x = nengo_dl.Layer(nengo.LIF(amplitude=0.01))(x)

        # Layer 2: Dense(128) + LIF
        net.dense2 = tf.keras.layers.Dense(units=128, name="dense2")
        x = nengo_dl.Layer(net.dense2)(x)
        x = nengo_dl.Layer(nengo.LIF(amplitude=0.01))(x)

        # Output layer: Dense(10) - no activation (logits)
        net.dense_out = tf.keras.layers.Dense(units=10, name="dense_out")
        x = nengo_dl.Layer(net.dense_out)(x)

        # Probe the output
        net.out_p = nengo.Probe(x, label="output")

    return net


def load_mnist():
    """Load and preprocess MNIST dataset."""
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Flatten and normalize to [0, 1]
    train_images = train_images.reshape(-1, 784).astype(np.float32) / 255.0
    test_images = test_images.reshape(-1, 784).astype(np.float32) / 255.0

    return train_images, train_labels, test_images, test_labels


def train_and_evaluate():
    """Train the network and save weights."""
    print("=" * 60)
    print("MNIST Spiking Neural Network Training")
    print("=" * 60)

    # Load data
    print("\nLoading MNIST dataset...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    print(f"  Train: {train_images.shape[0]} images")
    print(f"  Test:  {test_images.shape[0]} images")

    # Build network
    print("\nBuilding network...")
    net = build_network()

    # Training uses 1 timestep (rate-based approximation)
    n_train_steps = 1
    # Evaluation uses 30 timesteps (actual spiking)
    n_eval_steps = 30

    # Reshape data for nengo-dl: (batch, n_steps, features)
    train_x = {net.inp: train_images[:, None, :].repeat(n_train_steps, axis=1)}
    train_y = {net.out_p: train_labels[:, None, None].repeat(n_train_steps, axis=1)}

    test_x = {net.inp: test_images[:, None, :].repeat(n_eval_steps, axis=1)}
    test_y = {net.out_p: test_labels[:, None, None].repeat(n_eval_steps, axis=1)}

    # Train with nengo-dl simulator
    print("\nTraining (rate-based approximation, 1 timestep)...")
    with nengo_dl.Simulator(net, minibatch_size=200, seed=0) as sim:
        sim.compile(
            optimizer=tf.optimizers.RMSprop(0.001),
            loss={net.out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)},
            metrics=["accuracy"],
        )

        # Train
        sim.fit(train_x, train_y, epochs=10)

        # Evaluate with rate neurons (1 timestep, same as training)
        print("\n" + "=" * 60)
        print("Evaluating (rate-based, 1 timestep)...")
        rate_x = {net.inp: test_images[:, None, :]}
        rate_y = {net.out_p: test_labels[:, None, None]}
        results = sim.evaluate(rate_x, rate_y, verbose=0)
        print(f"  Rate accuracy: {results['output_accuracy']:.4f}")

        # Evaluate with spiking neurons (30 timesteps)
        print(f"\nEvaluating (spiking, {n_eval_steps} timesteps)...")
        results = sim.evaluate(test_x, test_y, verbose=0)
        print(f"  Spiking accuracy: {results['output_accuracy']:.4f}")

        # Extract trained weights from keras layers
        print("\nExtracting trained weights...")
        w1, b1 = [tf.keras.backend.get_value(w) for w in net.dense1.weights]
        w2, b2 = [tf.keras.backend.get_value(w) for w in net.dense2.weights]
        w_out, b_out = [tf.keras.backend.get_value(w) for w in net.dense_out.weights]

        # Also extract ensemble parameters (gain, bias, encoders)
        ens_list = list(net.all_ensembles)
        ens_params = sim.get_nengo_params(ens_list)

        weights_path = os.path.join(os.path.dirname(__file__), "mnist_weights.npz")
        np.savez(
            weights_path,
            # Dense layer weights and biases
            w1=w1, b1=b1,       # (784, 256), (256,)
            w2=w2, b2=b2,       # (256, 128), (128,)
            w_out=w_out, b_out=b_out,  # (128, 10), (10,)
            # Ensemble parameters (gain, bias, encoders)
            ens1_gain=ens_params[0]["gain"],
            ens1_bias=ens_params[0]["bias"],
            ens1_encoders=ens_params[0]["encoders"],
            ens2_gain=ens_params[1]["gain"],
            ens2_bias=ens_params[1]["bias"],
            ens2_encoders=ens_params[1]["encoders"],
        )
        print(f"  Saved weights to {weights_path}")
        print(f"    w1: {w1.shape}, b1: {b1.shape}")
        print(f"    w2: {w2.shape}, b2: {b2.shape}")
        print(f"    w_out: {w_out.shape}, b_out: {b_out.shape}")

    # Save sample test images for GUI
    # Pick 3 samples per digit (0-9) for variety
    sample_indices = []
    for digit in range(10):
        idx = np.where(test_labels == digit)[0][:3]
        sample_indices.extend(idx)
    sample_indices = np.array(sample_indices)

    images_path = os.path.join(os.path.dirname(__file__), "mnist_test_images.npz")
    np.savez(
        images_path,
        images=test_images[sample_indices],
        labels=test_labels[sample_indices],
    )
    print(f"  Saved {len(sample_indices)} test images to {images_path}")

    print("\n" + "=" * 60)
    print("Done! Files saved:")
    print(f"  - mnist_weights.npz  (trained weights for GUI)")
    print(f"  - mnist_test_images.npz (sample images for GUI)")
    print("=" * 60)


if __name__ == "__main__":
    train_and_evaluate()
