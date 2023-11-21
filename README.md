# NN-projects
## This repository contains jupyter notebook files from my class on Neural Networks
- The class is Mike X Cohen's "A Deep Understanding of Deep Learning"
- The projects in here are my unique solutions to projects from the class.
- Much of the code has been adapted from the class.

### Heart_disease_pred
- The goal of this project is to create a neural network to predict if a patient has heart disease, based on some clinical features.
- Here is a link to the dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
- I use the pytorch library for the implementation of this project.
- My NN performs with around 80% accuracy after training on the dev and test sets.
- I believe that accuracy could be higher if I simplified some of the metaparamaters of this network (remove node dropout and L2 regularization),
but I wanted practice implementing both of these features

### Missing_data_pred
- The goal of this project is to create a deep learning network that predicts values for missing data.
- I randomly remove values from the residual sugar column of the UCI machine learning repository wine data set, and attempt to use deep learning to predict these values.
- I use the pytorch library for the implementation of this project.
- This is the link to the data I will be using: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
- My NN usually achieves an R value of at least .9, but it depends on the data that is being predicted and how the weights are initialized.

### Cifar10autoencoder
- The goal of this project is to create a convolutional neural network autoencoder for the CIFAR10 dataset, with a pre-specified architecture.
- Here is a link to the dataset: https://www.cs.toronto.edu/~kriz/cifar.html.
- I use the pytorch library for the implementation of this project.
- This is my unique solution to a project created for Mike X Cohen's "A Deep Understanding of Deep Learning" class
- Much of the code is adapted from this course.
- Images that have been pushed through the autoencoder can be seen at the bottom of the file.
