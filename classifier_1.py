import pickle
import sklearn
from sklearn import svm # this is an example of using SVM
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist

def preprocess(images):
    for image in images:
        for i in range(0,28):
            for j in range(0, 28):
                if(image[i][j] > 0.1):
                    image[i][j] = 1.0;
                else:
                    image[i][j] = -1.0;
    return [i.flatten() for i in images]

def build_classifier(images, labels):
    classifier = svm.SVC(kernel='poly')
    classifier.fit(images, labels)
    return classifier

##the functions below are required
def save_classifier(classifier, training_set, training_labels):
    #this saves the classifier to a file "classifier" that we will
    #load from. It also saves the data that the classifier was trained on.
    import pickle
    pickle.dump(classifier, open('classifier_1.p', 'w'))
    pickle.dump(training_set, open('training_set_1.p', 'w'))
    pickle.dump(training_labels, open('training_labels_1.p', 'w'))


def classify(images, classifier):
    #runs the classifier on a set of images. 
    return classifier.predict(images)

def error_measure(predicted, actual):
    return np.count_nonzero(abs(predicted - actual))/float(len(predicted))

if __name__ == "__main__":

    # Code for loading data
    imagesData, labelsData = load_mnist(digits=range(0, 10), path='.')
    print "1 Finish loading data"
    # preprocessing
    images = preprocess(imagesData)
    labels = labelsData
    
    print "2 Finish preprocessing"
    # pick training and testing set
    # YOU HAVE TO CHANGE THIS TO PICK DIFFERENT SET OF DATA
    training_set = images[0:54000]
    training_labels = labels[0:54000]
    testing_set = images[-6000:]
    testing_labels = labels[-6000:]
    print "3 Finish training testing split"
    #build_classifier is a function that takes in training data and outputs an sklearn classifier.
    classifier = build_classifier(training_set, training_labels)
    print classifier
    print "4 Finish build_classifier"
    save_classifier(classifier, training_set, training_labels)
    print "5 Finish save_classifier"
    classifier = pickle.load(open('classifier_1.p'))
    print "6 Finish loading classifier"
    predicted = classify(testing_set, classifier)
    print "7 Finish classify"
    print error_measure(predicted, testing_labels)
    # plt.imshow(imagesData[54000+5719], cmap = 'gray')
    # print predicted[5719]
    # print testing_labels[5719]
    # plt.show()
    print "8 confusion_matrix"
    print confusion_matrix(predicted, testing_labels)
