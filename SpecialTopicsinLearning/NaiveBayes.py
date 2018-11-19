# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math

class NaiveBayes():
    def __init__(self):
        self.probabilities = []

    def help(self):
        print('\n')
        print('*'*50)
        print("Code to implement Naive Bayes Algorithm")
        print('*'*50)
        print('\n')

    def fit(self, data, targets_name='target'):

        self.mean_classes = []
        self.variance_classes = []
        self.classes = max(data[targets_name]) + 1
        dim = len(data.keys()) - 1

        for x in range(self.classes):
            self.mean_classes.append(np.mean(data[data[targets_name] == x].iloc[:,0:dim].values))
            self.variance_classes.append(np.var(data[data[targets_name] == x].iloc[:,0:dim].values))

    def predict(self, value):
        probabilities = []
        self.probabilities = []
        for x in range(self.classes):
            # exponent = math.exp(-(math.pow(value-self.mean_classes[x],2)/(2*self.variance_classes[x])))
            exponent = (-(np.power(value-self.mean_classes[x],2)/(2*self.variance_classes[x])))
            # probabilities.append((1 / (math.sqrt(2*math.pi* self.variance_classes[x]))) * exponent)
            probabilities.append((np.power((1 / (np.sqrt(2*math.pi* self.variance_classes[x]))),exponent)[0]))

        print(probabilities)
        for x in probabilities:
            prob = x/sum(probabilities)
            p = 1
            for x in prob:
                p = p * x
            self.probabilities.append(p)
        print(self.probabilities)
        return self.probabilities.index(max(self.probabilities)), max(self.probabilities)
