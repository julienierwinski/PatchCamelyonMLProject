# PatchCamelyonMLProject
Theory of Machine Learning - Final Project

Uses PatchCamelyon (PCam) medical imaging dataset to implement the following machine learning methods:

  1. Transfer Learning - uses Path Foundation Model to extract embeddings from dataset and implements 3 classifiers including Multi-Layer Perceptron, Logistic Regression, and the HistGradientBoost ensemble               classifier to train, validate, and test
  2. Convolutional Neural Networks - creates basic CNN using PCam images and compare test accuracy to rotation-equivariant CNN
  3. Generative Adversarial Networks - generates high-fidelity pathological image samples and investigates performance   
  4. Autoencoders - trains autoencoder neural network, updates encoder and decoder parameters

Main branch - includes transfer learning (Path Foundation), classifiers, and standard CNN implementation to PCam
Master branch - includes GAN and authoencoders implementation
