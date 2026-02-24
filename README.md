Project Overview
This project implements three fundamental deep learning architectures using Python and NumPy with minimal reliance on external libraries. The goal is to demonstrate a ground-up understanding of gradient-based learning, dimensionality reduction, and generative feature extraction.

Implemented Models
Multi-Layer Perceptron (MLP): A supervised 2-layer classifier using ReLU and Softmax.

Sparse Autoencoder: An unsupervised bottleneck architecture for reconstruction and anomaly detection.

Restricted Boltzmann Machine (RBM): An energy-based generative model using Contrastive Divergence (CD-1).

Prerequisites
To run the code, ensure you have a Python 3 environment (Python 3.8+ recommended). The project requires the following libraries:

numpy: For all mathematical operations and matrix manipulations.

matplotlib: For generating training curves and visualizations.

tensorflow: Only used for the tf.keras.datasets.mnist utility to download the dataset.

How to Run
Option 1: Google Colab (Recommended)
Open Google Colab.

Create a New Notebook.

Copy the combined source code provided in the implementation phase into a code cell.

Press Shift + Enter or click the Play button.

The script will automatically:

Download the MNIST dataset.

Train the MLP, Autoencoder, and RBM sequentially.

Generate a summary of outliers detected.

Display all required plots and reconstructions.

Option 2: Local Machine
Ensure you have the required libraries installed via terminal/command prompt:

Bash
pip install numpy matplotlib tensorflow
Save the provided code into a file named main.py.

Execute the script using:

Bash
python main.py
Code Structure
class MLP: Handles forward pass (ReLU/Softmax), backpropagation, and weight updates.

class Autoencoder: Includes an L1 sparsity penalty in the hidden layer and reconstruction logic.

class RBM: Implements the positive/negative phases of Contrastive Divergence and Gibbs sampling.

get_data(): Downloads, flattens, and normalizes MNIST images to the [0, 1] range.

Visualization Block: Generates three distinct figures:

Loss and Accuracy curves.

Original vs. Reconstructed images from the Autoencoder.

Learned feature filters from the RBM hidden units.

Expected Outputs
Upon successful execution, the console will print the Loss and Accuracy for each epoch. Following the training, the script will output:

Reconstruction Plots: A side-by-side comparison of handwritten digits.

Outlier Count: A report of how many images were flagged as anomalies based on the reconstruction error threshold.

Feature Maps: A grid showing the "strokes" the RBM has learned to identify.

Troubleshooting
Dataset Download Error: If the dataset fails to download, ensure your internet connection is active.

Memory Issues: If running locally on a low-spec machine, reduce the hidden_size in the MLP or Autoencoder classes to save RAM.

Plotting: If plots do not appear locally, ensure you have a GUI backend installed or use plt.savefig('output.png') instead of plt.show().
