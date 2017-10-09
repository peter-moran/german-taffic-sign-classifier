# German Traffic Sign Classifier [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<a href="http://petermoran.org/"><img src="https://img.shields.io/badge/Peter%20Moran's%20Blog-Find_more_projects_at_my_website-blue.svg?style=social"></a>

# Overview

This project was completed as part of the Udacity Self Driving Car program. This project served as a warm up for further applications of Deep Learning, and so we used TensorFlow with networks of our own design (rather than re-training well tested networks like GoogLenet).

The goals of the project were to:

* Explore, summarize, and visualize the [INI German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset),
* Design, train, and test a model architecture to classify the signs, and to 
* Use the model to make predictions on new images and inspect their softmax probabilities.

## How it Works

In its broad strokes, the classifier is remarkably simple:

1. Separate the training, validation, and testing sets.
2. Apply a center crop on every image, removing 3 pixels from each edge.
3. Normalize and center-mean the image pixel intensities to the [-1, 1] range.
4. Train the images in a convolutional neural network inspired by the simplicity of VGG, with:
   * A 32x32x3 RGB image input.
   * Four consecutive 3x3 kernel convolutions followed by a RELU, 50% dropout, and max pooling on all but the first.
   * Three consecutive fully connected layers (including the final output).

To make this work at peak performance and to determine the exact structure and parameterization of the neural network, plenty of tuning and experimentation was needed. This was supported by the `conv()` function, which makes it easy to apply TensorFlow convolution without manually entering input sizes and provides simple flags for applying RELUs, pooling, and change the padding type.

## Results

The final testing accuracy was 97.3%, which is only 1.5% below human accuracy for the images in this data set.

For a deeper dive into the top 5 softmax scores for a selection of example images, read the last section of `Traffic_Sign_Classifier.ipynb`.

---

# Installation

## This Repository

Download this repository by running:

```sh
git clone https://github.com/peter-moran/german-taffic-sign-classifier.git
cd german-taffic-sign-classifier
```

## Software Dependencies

This project utilizes the following, easy to obtain software:

* Python 3
* TensorFlow
* OpenCV
* Jupyter
* Matplotlib
* Numpy
* Scikit-learn

An easy way to obtain these is with the [Udacity CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit) and Anaconda. To install the software dependencies this way, see the [full instructions](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/doc/configure_via_anaconda.md) or simply run the following:

```
git clone https://github.com/udacity/CarND-Term1-Starter-Kit.git
cd CarND-Term1-Starter-Kit
conda env create -f environment.yml
activate carnd-term1
```

**You will also need the latest version of TensorFlow** (the page above uses an outdated version) and it is also recommended you install TensorFlow with GPU support. To get the latest version, use the [TensorFlow installation guide](https://www.tensorflow.org/install/#optional_install_cuda_gpus_on_linux).

## Data Dependencies

This project uses the [INI German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Udacity has provided this in a pickled form and [can be downloaded here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip), who's contents should then be extracted to the `data/signs` folder.

---

# Usage

This project is entirely contained in a Jupyter Notebook. Simply open `Traffic_Sign_Classifier.ipynb` in Jupyter and start running the cells.

```
jupyter notebook Traffic_Sign_Classifier.ipynb
```

