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

        #Covariance Matrix
        covarMatrix = np.ones([len(vectors),len(vectors)])
        for i in range(len(vectors)):
            for j in range(len(vectors)):
                covarMatrix[i][j] = self.getCovariance(vectors[i], vectors[j])

        # print(covarMatrix)
        self.vect_med = [np.array([x]).mean() for x in vectors]

        #Computing the eigenvalues and right eigenvectors of a square array.
        self.W, self.v = np.linalg.eig(covarMatrix)

        # e = self.vectors

        plt.scatter()

        print(self.vect_med)
        print(self.W)
        print(self.v)


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
