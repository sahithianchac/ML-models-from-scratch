import pandas as pd
import numpy as np
#KNN from Scratch 

#Calculating the euclidean distance for the test and train samples 
def euclidean_distance(train,test):
    d=np.sum(np.square(train - test), axis=1)
    distance=np.sqrt(d)
    return distance

#Let us calculate the nearest neighbours and classes 
def neighbours_classes(k,xtrain,ytrain,xtest):
    df = np.column_stack([ytrain,euclidean_distance(xtrain,xtest)])
    neighb = list(df[df[:, 1].argsort()][:k:, 0]) #Sorts and stores in K classes 
    classes=max(set(neighb), key=neighb.count)#Gives the class
    return classes

#Predicts the classes and calculates the accuracy
def prediction_accuracy(k,xtrain,ytrain, xtest,ytest):
    
    for test in xtest:
        prediction = neighbours_classes(k,xtrain,ytrain, test)
    
    correct = (np.array(ytest) == np.array(prediction))
    no_of_correct=correct.sum()
    accuracy = round((no_of_correct / len(ytest))*100,2)
    return accuracy

#The neighbours_classes gives the "K" number of classes for the data and we can calculate accuracy using prediction_accuracy function.