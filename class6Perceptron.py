# -*- coding: utf-8 -*-
import pandas as pd
from SpecialTopicsinLearning.Perceptron import Perceptron
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def main():
    '''Instanciando a classe do método'''
    p = Perceptron(4)

    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=['A', 'B', 'C', 'D'])
    df['target'] = iris.target

    train, test = train_test_split(df[df['target'] != 2], test_size=0.2)

    # p.fit(X=np.array([[0,0],[0,1],[1,0],[1,1]]), Y=np.array([0,1,1,0]))

    p.fit(X=train.iloc[:,0:4].values, Y=train.iloc[:,4:].values)

    for x in range(len(test)):

        print("\n\t" + "#"*10 + " Predicted " + "#"*9)
        print("\t" + "#"*30)
        input = test.iloc[x,0:4].values
        # input = np.array([[0,x]])
        print("\tInputs: {}".format(input))
        print("\tPredicted Class: {}".format(p.predict(input)))
        print("\tY: {}".format(test.iloc[x,4:].values))
        # print("\tY: {}".format(x))
        print("\tWeights: {}".format(p.weights))
        print("\t" + "#"*30 + "\n")

    # plt.scatter(train.iloc[:,0].values,train.iloc[:,1].values)
    # plt.show()

if __name__ == '__main__':
    main()
