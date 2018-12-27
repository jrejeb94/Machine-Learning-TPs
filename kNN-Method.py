#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 23:35:46 2018

@author: intissare
"""

import csv
import random
import math
import operator


def loadDataset (filename, split, trainingSet=[], testSet=[]):
    """
    Loads data from a csv file and splits it into a trainning set and a test set of data
    """
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) -1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance (instance1, instance2):
    """
    Computes the euclidean distance between two instances.
    """
    distance = 0
    if len(instance1) != len(instance2):
        print('Different vectors length, please check your data set.')
        exit(1)
    else:
        length = len(instance1)
    for x in range(length-1):
        distance += pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)


def getNeighbors (trainingSet, testInstance, k):
    """
    computes the distance between a given testInstance and all the instance of a training set
    :param trainingSet: (array of the data instance)
    :param testInstance: (array) a point of the data
    :param k: (int) number of neighbors
    """
    distances = []
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x])
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getMatch(neighbors):
    """
    Decides the label of the vector depending on the labels of the neighbors.
    """
    classVotes={}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    """
    Gives the rate of the accurate (correct) answers among the predicted labels.
    """
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def main():
    trainingSet = []
    testSet = []
    split = 0.66
    loadDataset('iris.data', split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getMatch(neighbors)
        predictions.append(result)
        print('predicted=' + repr(result) + '/ actual= ' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


if __name__ == '__main__':
    main()