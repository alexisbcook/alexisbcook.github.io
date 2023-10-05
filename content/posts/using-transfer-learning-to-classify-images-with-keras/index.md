---
title: "Using transfer learning to classify images with Keras"
draft: false
date: 2017-04-08T09:16:45.000Z
description: "Also use t-SNE to visualize bottleneck features"
categories:
- computer-vision
tags:
- keras
- image-classification
- transfer-learning
---

In this blog post, I will describe my [GitHub repository](https://github.com/alexisbcook/keras_transfer_cifar10) that performs object classification with transfer learning.  This blog post is inspired by a [Medium article](https://medium.com/@st553/using-transfer-learning-to-classify-images-with-tensorflow-b0f3142b9366) that made use of Tensorflow.  The code is written in Keras (version 2.0.2) and Python 3.5.  

If you need to learn more about CNNs, I recommend reading the notes for the [CS231n](http://cs231n.github.io/convolutional-networks/) course at Stanford.  All lectures are also available [online](https://www.youtube.com/watch?v=LxfUGhug-iQ&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=7).  You are also encouraged to check out Term 2 of Udacity's [Artificial Intelligence Nanodegree](https://www.udacity.com/course/artificial-intelligence-nanodegree--nd889), where you can find a comprehensive introduction to neural networks (NNs), CNNs (including transfer learning), and recurrent neural networks (RNNs).

# The Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) is a popular dataset composed of 60,000 tiny color images that each depict an object from one of ten different categories.

{{< img src="cifar10.png" alt="cifar10" size-method="Fit" size-format="600x400 webp" position="center" >}}

The [dataset](https://keras.io/datasets/) is simple to load in Keras.
```python
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

# Extracting the InceptionV3 Bottleneck Features

Instead of building a CNN from scratch, I used __transfer learning__ to leverage a pre-trained CNN that has demonstrated state-of-the-art performance in object classification tasks. 

Keras makes it very easy to access several pre-trained [CNN architectures](https://keras.io/applications/).  I decided to use the InceptionV3 architecture. 

{{< img src="inception.png" alt="inception" size-method="Fit" size-format="600x400 webp" position="center" >}}

After importing the necessary Python class, it's only one line of code to get the model, along with the pre-trained weights.

```
from keras.applications.inception_v3 import InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=True)
```

The pre-trained InceptionV3 architecture is stored in the variable `base_model`.  The final layer of the network is a fully connected layer designed to distinguish between the [1000 different object categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) in the ImageNet database.  Using the line of code below, I remove this final layer and save the resultant network in a new model.  

```python
model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)
```

This new model will no longer return a predicted image class, since the classification layer has been removed; however, the CNN now stored in `model` still provides us with a useful way to extract features from images.  By passing each of the CIFAR-10 images through this model, we can convert each image from its 32x32x3 array of raw image pixels to a vector with 2048 entries.  In practice, we refer to this dataset of 2048-dimensional points as InceptionV3 bottleneck features.  

# Using t-SNE to Visualize Bottleneck Features

Towards visualizing the bottleneck features, I used a dimensionality reduction technique called [t-SNE](http://distill.pub/2016/misread-tsne/) (aka t-Distributed Stochastic Neighbor Embedding).  t-SNE reduces the dimensionality of each point, in a way where the points in the lower-dimensional space preserve the pointwise distances from the original, higher-dimensional space.  Scikit-learn [has an implementation](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) of t-SNE, but it does not scale well to large datasets.  Instead, I worked with an implementation that can be found [on GitHub](https://github.com/alexisbcook/tsne); it can be installed by running in the terminal:

```
pip install git+https://github.com/alexisbcook/tsne.git
```

Visualizing the resulting 2-dimensional points yields the plot below, where points are color-coded according to the object class contained in the corresponding image.

{{< img src="tsne.png" alt="tsne" size-method="Fit" size-format="600x400 webp" position="center" >}}

InceptionV3 does an amazing job with teasing out the content in the image, where points containing objects from the same class are mostly confined to nearby regions in the 2D plot.  Thus, training a classifier on the bottleneck features should yield good performance.

# Performing Classification with Transfer Learning

In the Jupyter notebook in the repository, I trained a very shallow CNN on the bottleneck features.  It yields a test accuracy of 82.68%! :)

# Play with the Code!

Can you do better with other architectures?  Feel free to download the [repository](https://github.com/alexisbcook/keras_transfer_cifar10) on GitHub and try your own hand at transfer learning! 