import numpy as np
import matplotlib.pyplot as plt
import SpecialTopicsinLearning.Matrix as mt

class PrincipalComponentAnalysis():
    def __init__(self):
        super(PrincipalComponentAnalysis, self).__init__()

    def help(self):
        print('\n')
        print('*'*50)
        print("Code to implement Principal Component Analysis Algorithm")
        print('*'*50)
        print('\n')
        print("\t - Enter with a list of Xn vectors")

    def fit(self, vectors, n_components = 1):

        # vectors = np.array([vectors])
        self.vectors = vectors

        self.vect_med = [np.array([x]).mean() for x in vectors]

        vectors = (np.array(vectors).T - self.vect_med).T

        #Covariance Matrix
        covarMatrix = np.ones([len(vectors),len(vectors)])
        for i in range(len(vectors)):
            for j in range(len(vectors)):
                covarMatrix[i][j] = self.getCovariance(vectors[i], vectors[j])


        #Computing the eigenvalues and right eigenvectors of a square array.
        self.W, self.v = np.linalg.eig(covarMatrix)

        feature_vector = self.v[:,0:n_components]

        final_data = feature_vector.T.dot(np.array(vectors))

        original_data = np.add(feature_vector.dot(final_data), np.array([self.vect_med]).T)

        plt.scatter(original_data[0],original_data[1],color='red')
        # print(self.vect_med)
        # data = (np.add(original_data.T, )
        #
        # print(self.vectors)


    def plot(self):

        ax = plt.axes()
        ax.arrow(self.vect_med[0],self.vect_med[1] , self.v[:,0][0], self.v[:,0][1], head_width=0.15, head_length=0.5)
        ax.arrow(self.vect_med[0],self.vect_med[1] , self.v[:,1][0], self.v[:,1][1], head_width=0.15, head_length=0.5)

        plt.plot(self.vect_med[0],self.vect_med[1],'ok')

        plt.scatter(self.vectors[0],self.vectors[1])
        plt.show()

    def getCovariance(self, x, y):
        if len(x) == len(y):
            #subtracting the mean
            cov = 0
            for i in range(len(x)):
                cov = cov + (x[i] - np.array([x]).mean()) * (y[i] - np.array([y]).mean())
            return (cov/(len(x)-1))
        else:
            print("There are vectors with different lenghts!!")
            return "error"
