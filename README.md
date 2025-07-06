# Geometric-Learning-for-Molecules-Representation

Geometric learning is a field of machine learning that focuses on data with inherent geometric structures. In this study, the objects of interest are molecules, and the goal is to predict their formation energy, an important property related to molecular stability.

Molecules can naturally be represented as graphs, where atoms are nodes and chemical bonds are edges. This makes them well-suited for **graph neural networks (GNNs)**, which are designed to handle such structured data.

In this study, we aim to build, train, and compare two regression models. One model uses a dataset in which molecules are represented as strings using the SMILES format, while the other uses a dataset that preserves the geometric structure of the molecules. The objective is to assess the strengths and capabilities of geometric learning when appropriately applied to problems involving structured data.

During the course of the study, we also conduct experiments using a restricted number of data points to train the networks under more challenging conditions. The results of these experiments are shown below.

![training_size_comparison_with_R2](https://github.com/user-attachments/assets/b7f42b54-a15c-4b21-aa16-dc84f74c6bea)

The folders named checkpoints_s[number] contain the checkpoints for the networks trained on only [number] data points. In particular, we take into account the Mean Squared Error (MSE) and the R-squared (R2) to check the performances of the developed networks.
