# -*- coding: utf-8 -*-
import pandas as pd
from SpecialTopicsinLearning.MLPClassifier import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    '''Instanciando a classe do método'''
    net = MLPClassifier()

    # iris = datasets.load_iris()
    # df = pd.DataFrame(data=iris.data, columns=['A', 'B', 'C', 'D'])
    # df['target'] = iris.target
    #
    # train, test = train_test_split(df[df['target'] != 2], test_size=0.2)
    print("\nSem treino\n")
    print(net.predict(np.array([[0,1]]))[2])

    net.fit(np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([[0, 1, 1, 0]]).T)
    print("\nCom treino\n")
    print(net.predict(np.array([[1,1]]))[2])


if __name__ == '__main__':
    main()
