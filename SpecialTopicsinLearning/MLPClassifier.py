# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

class MLPClassifier():
    def __init__(self, mlp_shape=[2,4,1], iteractions=100, alpha=0.2):
        self.iteractions = iteractions
        self.learning_rate = alpha
        self.mlp_shape = mlp_shape
        # Initializing the weights
        self.__input_weights = np.random.rand(1,self.mlp_shape[0])
        self.__layer1_weights = np.random.rand(self.mlp_shape[0],self.mlp_shape[1])
        self.__layer2_weights = np.random.rand(self.mlp_shape[1],self.mlp_shape[2])

    def help(self):
        print('\n')
        print('*'*50)
        print("Code to implement MLP Classifier Algorithm")
        print('*'*50)
        print('\n')

    def __sigmoid(x):
        return np.tanh(x)

    def __deltaSigmoid(x):
        return 1.0-x**2

    def predict(self, inputs):
        w1 = np.full((self.mlp_shape[1],1), self.__sigmoid(np.dot(inputs,self.__input_weights.T)))

        w2 = self.__sigmoid(np.dot(w1,self.__layer1_weights.T))
        w3 = self.__sigmoid(np.dot(w2,self.__layer2_weights.T))
        return w3
