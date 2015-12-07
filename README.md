# Handwritten-Recognition
Handwritten Recognition using SVM, KNN and AdaBoosting

1.Reading the MNIST data
We have given a helper file "mnist.py" (thanks to http://g.sweyla.com/blog/2012/mnist-numpy/) that you can use to read in the data as follows:

import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist
%matplotlib inline

images, labels = load_mnist(digits=[9], path='.')
#Displaying the mean image for digit 9.
plt.imshow(images.mean(axis=0), cmap = 'gray')
plt.show()

That is what 9, when handwritten, looks like on average. Changing the digits argument to a list would give you all the images that match the labels in the list (e.g. digits = [0, 1, 2] would give you all the 0s, 1s, and 2s in MNIST). Setting path = '.' makes it look for the MNIST data in the current directory.

2.Exploring data
There are 6,0000 images in total. 
There 5923 images of “0”. 
There 6742 images of “1”.
There 5958 images of “2”.
There 6131 images of “3”.
There 5842 images of “4”.
There 5421 images of “5”. 
There 5918 images of “6”. 
There 6265 images of “7”.
There 5851 images of “8”. 
There 5949 images of “9”.

We randomly draw 54,000 data from the dataset as training set, and 6,000 data from the rest data in dataset as testing set. We selected the data randomly both for training and testing because our classifier can be generative and our testing set are selected to test whether our classifier generalizes (challenging digit cases are not thrown out in testing set because those digits do exist in real life)

3.Data encoding:
For each data in the training set and testing set, we preprocess each image pixel and set the pixel value to 1 if origin value is bigger than 0.1, otherwise set it to -1. Then we flatten 28x28 matrix into 1x784 vector.

4.Classification using SVM
Classification using SVM: SVM performs kernel trick, implicitly mapping the data inputs into high-dimensional feature spaces. SVM optimizes the classification by maximizing the margin, while also allowing misclassifications by using penalty terms (slack parameter). We choose the kernel and slack parameter for the SVM classifier, and build the classifier based on our training set and use this classifier to predict the labels of our testing set.
Classifier output interpretation: The kernel parameter in the classifier specifies the kernel function used by this classifier. The slack parameter in the classifier specifies the penalty terms this classifier assigns for misclassification.

5.Classification using KNN
Classification using KNN: KNN get examples with known outputs from the training set and assign the output of the testing set to be the same as the most K similar known cases. We choose the K and the distance measure for the KNN classifier, and build the classifier based on our training set and use this classifier to predict the labels of our testing set.
Classifier output interpretation: The K parameter in the classifier specifies how many neighbors will be considered when labeling the unknown data. The distance measure parameter specifies the way distance between data will be calculated.