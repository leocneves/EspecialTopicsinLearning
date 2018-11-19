# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn import datasets
from SpecialTopicsinLearning.LinearDiscriminantAnalysis import LinearDiscriminantAnalysis
from SpecialTopicsinLearning.PrincipalComponentAnalysis import PrincipalComponentAnalysis

def main():
    # '''Instanciando a classe do método'''
    lda = LinearDiscriminantAnalysis()
    pca = PrincipalComponentAnalysis()

    iris = datasets.load_iris()

    dataframe = iris.data
    data_targets = iris.target
    feature_names = list(iris.feature_names)
    df = pd.DataFrame(np.append(dataframe,np.array([data_targets]).T,axis=1),columns=feature_names + ['target'])

    # LDA
    lda.fit(data=df,n_components=2, plot=True)

    # # PCA + LDA
    # data_pca = pca.fit(df.iloc[:,:4].values,n_components=3)
    # df_mod_pca = df = pd.DataFrame(np.hstack((data_pca,np.array([data_targets]).T)).real,columns=feature_names + ['target'])
    # print(df_mod_pca)
    # lda.fit(data=df_mod_pca,n_components=2, plot=True)

    # print(lda.lda_data)

if __name__ == '__main__':
    main()
