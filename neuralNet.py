import numpy as np
import random
import simpleMath
import pickle
from math import *


class NeuralNet:
    def __init__(self, inputs, hidden, outputs):
        self.iNodes = inputs
        self.hNodes = hidden
        self.oNodes = outputs

        self.whi = np.random.random_integers(-1, 1, (self.hNodes, self.iNodes + 1))  # first layer
        self.whh = np.random.random_integers(-1, 1, (self.hNodes, self.hNodes + 1))  # second layer
        self.woh = np.random.random_integers(-1, 1, (self.oNodes, self.hNodes + 1))  # third layer

    def __repr__(self):
        return '%s\n%s\n%s' % (self.whi, self.whh, self.woh)

    def mutate(self, mr):
        '''Mutates each layer of the neural net.'''

        mutateMatrix(self.whi, mr)
        mutateMatrix(self.whh, mr)
        mutateMatrix(self.woh, mr)

    def output(self, inputArray):
        '''Runs an input through the neural net.'''

        # prepares input array
        inputs = singleColumnMatrixFromArray(inputArray)
        inputs = addBias(inputs)

        # run through first hidden layer
        hiddenInputs = self.whi.dot(inputs)
        hiddenOutputs = activate(hiddenInputs)
        hiddenOutputs = addBias(hiddenOutputs)

        # run through second hidden layer
        hiddenInputs2 = self.whh.dot(hiddenOutputs)
        hiddenOutputs2 = activate(hiddenInputs2)
        hiddenOutputs2 = addBias(hiddenOutputs2)

        # run through output layer
        outputInputs = self.woh.dot(hiddenOutputs2)
        outputs = activate(outputInputs)

        return activate(outputInputs).tolist()

    def crossover(self, partner):
        '''Crosses over each layer.'''
        child = NeuralNet(self.iNodes, self.hNodes, self.oNodes)

        child.whi = crossoverMatrix(self.whi, partner.whi)
        child.whh = crossoverMatrix(self.whh, partner.whh)
        child.woh = crossoverMatrix(self.woh, partner.woh)

        return child

    def clone(self):
        '''Copies this neural net.'''
        clone = NeuralNet(self.iNodes, self.hNodes, self.oNodes)

        clone.whi = self.whi.copy()
        clone.whh = self.whh.copy()
        clone.woh = self.woh.copy()

        return clone;

    def export(self, path):
        '''Saves this neural net into a file.'''

        with open(path, 'wb') as f:
            pickle.dump(self, f)


def mutateMatrix(m, rate):
    '''Mutates a matrix for machine learning.'''
    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            if random.random() < rate:
                m[x][y] += random.gauss(0, 1) / 5

                m[x][y] = simpleMath.clamp(m[x][y], -1, 1)


def crossoverMatrix(m1, m2):
    '''randomly combines two matrices for machine learning'''
    child = np.zeros(shape=(m1.shape[0], m2.shape[1]))

    # choose a random element in the matrix to set crossover point
    randC = random.randint(0, m1.shape[1])
    randR = random.randint(0, m2.shape[0])

    for x in range(m1.shape[0]):
        for y in range(m2.shape[1]):
            if (x < randR) or (y == randR and y <= randC):
                child[x][y] = m1[x][y]
            else:
                child[x][y] = m2[x][y]
    return child


def singleColumnMatrixFromArray(array):
    '''Converts an array to a column vector.'''
    n = np.zeros(shape=(len(array), 1))
    for i in range(len(array)):
        n[i][0] = array[i]
    return n


def activate(m):
    '''passes matrix through sigmoid function'''
    n = np.zeros(shape=(m.shape[0], m.shape[1]))
    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            n[x][y] = sigmoid(m[x][y])
    return n


def sigmoid(x):
    '''the logistic function'''
    return 1 / (1 + e ** (-x))


def relu(x):
    '''rectified linear unit'''
    if x < 0:
        return 0
    return x


def addBias(m):
    '''Adds 1 to the bottom.'''

    n = np.zeros(shape=(m.shape[0] + 1, 1))
    for i in range(m.shape[0]):
        n[i][0] = m[i][0]
    n[m.shape[0]][0] = 1
    return n
