# Traffic-Sign-Recognition
This repository contains a Jupyter Notebook that implements a deep learning model to classify traffic signs. The project uses a Convolutional Neural Network (CNN) to achieve high accuracy on the German Traffic Sign Recognition Benchmark (GTSRB) dataset.

The notebook demonstrates a complete workflow, including data exploration, image preprocessing, model architecture design, training, and detailed evaluation.

## 📋 Table of Contents

- Project Overview

- Dataset

- Methodology

- Performance Metrics

- How to Run

- Requirements

## 🚀 Project Overview <a name="project-overview"></a>

The goal of this project is to build a robust image classification model capable of identifying 43 different classes of traffic signs. This is a crucial task for autonomous driving systems and driver assistance technologies.

The notebook covers the following key stages:

1- Data Exploration (EDA): Visualizing the distribution of traffic sign classes and inspecting sample images to understand the dataset's characteristics.

2- Image Preprocessing: Preparing the image data for the neural network by resizing all images to a uniform size (32x32 pixels), normalizing pixel values to a [0, 1] range, and one-hot encoding the labels.

3- Model Building: Designing a CNN architecture with multiple convolutional, max-pooling, and dropout layers to effectively learn features from the images.

4- Training and Evaluation: Training the model on the preprocessed data and evaluating its performance on an unseen test set using accuracy and a confusion matrix.

## 💾 Dataset <a name="dataset"></a>

This project uses the German Traffic Sign Recognition Benchmark (GTSRB) dataset, a multi-class, single-image classification challenge.

- Classes: 43 different traffic sign categories.

- Training Set: Over 39,000 images.

- Test Set: Over 12,000 images.

- Image Size: Varies from 15x15 to 250x250 pixels.

You can find the dataset on [Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)

## 🛠️ Methodology <a name="methodology"></a>

1. Image Preprocessing

- Resizing: All images from the dataset are resized to 32x32 pixels to ensure a consistent input size for the CNN.

- Normalization: Pixel values are scaled from the original [0, 255] range to a [0, 1] range by dividing by 255.0. This helps stabilize the training process.

- One-Hot Encoding: The integer labels (0-42) are converted into one-hot encoded vectors, which is necessary for the categorical_crossentropy loss function.

2. CNN Model Architecture
The model is built using a Sequential Keras model with the following structure:

- Two Convolutional Blocks:

  - Each block consists of two Conv2D layers (with 32 and 64 filters respectively) followed by a MaxPooling2D layer to downsample the feature maps.

  - A Dropout layer (rate=0.25) is added after each block to prevent overfitting.

- Flatten Layer: Converts the 2D feature maps into a 1D vector.

- Dense Layers:

  - A fully-connected Dense layer with 256 neurons and a relu activation function.

  - A final Dropout layer (rate=0.5) for further regularization.

- Output Layer: A Dense layer with 43 neurons (one for each class) and a softmax activation function to output class probabilities.

The model is compiled with the Adam optimizer and categorical cross-entropy as the loss function.

## 📊 Performance Metrics <a name="performance-metrics"></a>

The model's performance was evaluated on the unseen test set.

- Test Accuracy: The model achieved an impressive accuracy of 97.40%.

Confusion Matrix
The confusion matrix provides a detailed breakdown of the model's predictions versus the actual labels. The strong diagonal line indicates that the model correctly classified the vast majority of the test images, with very few misclassifications between different sign types.

## 💻 How to Run <a name="how-to-run"></a>

1- Clone the repository:
<div class="command-block">
<pre>git clone https://github.com/your-username/traffic-sign-recognition.git</pre>
</div>
<div class="command-block">
<pre>cd traffic-sign-recognition</pre>
</div>
2- Create a virtual environment (recommended):
<div class="command-block">
<pre>python -m venv venv</pre>
</div>
<div class="command-block">
<pre>source venv/bin/activate  # On Windows, use venv\Scripts\activate</pre>
</div>
3- Install the required packages:
<div class="command-block">
<pre>pip install -r requirements.txt</pre>
</div>
4- Launch Jupyter Notebook:
<div class="command-block">
<pre>jupyter notebook</pre>
</div>
5- Open the traffic-sign-recognition.ipynb notebook and run the cells.

## 📦 Requirements <a name="requirements"></a>
The project requires the following Python libraries. You can install them using the requirements.txt file.

<div class="command-block">
<pre>pandas
numpy
matplotlib
seaborn
tensorflow
Pillow
scikit-learn</pre>
</div>
