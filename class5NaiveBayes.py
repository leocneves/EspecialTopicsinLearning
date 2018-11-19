# -*- coding: utf-8 -*-
import pandas as pd
from SpecialTopicsinLearning.NaiveBayes import NaiveBayes
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def main():
    '''Instanciando a classe do método'''
    nb = NaiveBayes()
    # #Wine database
    # df = pd.read_csv("winequality-red.csv", delimiter=";")
    # data_wine = df[df.keys()[0:11]].values
    #
    # #Motor database
    # df = pd.read_csv("Sensorless_drive_diagnosis.txt", delimiter=" ",names=['data' + str(x) for x in range(49)])
    # data_motor = df[0:10000].values
    data = [[25.2,1],[19.3,1],[18.5,1],[21.7,1],[20.1,1],[24.3,1],[22.8,1],[23.1,1],[19.8,1],[27.3,0],[30.1,0],[17.4,0],[29.5,0],[15.1,0]]
    df = pd.DataFrame(data=data, columns=['Data_YesNo','target'])

    iris = datasets.load_iris()
    df = pd.DataFrame(data=iris.data, columns=['A', 'B', 'C', 'D'])
    df['target'] = iris.target

    train, test = train_test_split(df, test_size=0.2)

    nb.fit(data=train,targets_name='target')

    prediction, probability = nb.predict(value=test.iloc[0:1,0:4].values)
    print(test.iloc[0:1,:].values)

    print('\nClass: ' + str(prediction),'\nProbability: ' + str(probability) + '\n')

if __name__ == '__main__':
    main()
