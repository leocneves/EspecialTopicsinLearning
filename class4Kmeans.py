# -*- coding: utf-8 -*-
import pandas as pd
from sklearn import datasets
from SpecialTopicsinLearning.KMeans import KMeans

def main():
    '''Instanciando a classe do método'''
    kmeans = KMeans()

    #Wine database
    df = pd.read_csv("winequality-red.csv", delimiter=";")
    data_wine = df[df.keys()[0:11]].values

    #Motor database
    df = pd.read_csv("Sensorless_drive_diagnosis.txt", delimiter=" ",names=['data' + str(x) for x in range(49)])
    data_motor = df[0:10000].values

    #Slide database
    data_1 = [[1.9,7.3],[3.4,7.5],[2.5,6.8],[1.5,6.5],[3.5,6.4],[2.2,5.8],[3.4,5.2],[3.6,4],[5,3.2]\
                ,[4.5,2.4],[6,2.6],[1.9,3],[1,2.7],[1.9,2.4],[0.8,2],[1.6,1.8],[1,1]]

    #iris database
    iris = datasets.load_iris()
    dataframe = iris.data
    data_targets = iris.target
    data_iris = dataframe.tolist()

    kmeans.fit(data=data_motor,k_number=11)
    print('Centroids: ' + str(kmeans.centroids))
    print(kmeans.targets)


if __name__ == '__main__':
    main()
