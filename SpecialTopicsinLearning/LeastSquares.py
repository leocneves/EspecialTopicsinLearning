import numpy as np
import matplotlib.pyplot as plt

class LeastSquares(object):
    def __init__(self):
        super(LeastSquares, self).__init__()

    def help(self):
        print('\n')
        print('*'*50)
        print("Code to implement Least Squares Algorithm")
        print('*'*50)
        print('\n')
        print("\t - Enter X,Y (array) data and use fit(X,Y) function")
        print("\t - Enter X (array) data and use predict(X) function\n")

    def fit(self, X, Y, bias=0):
        #Converting to vectors
        if bias != 0:
            self.X = np.append(np.ones([len(np.array(X).T),1])*bias, np.array(X).T, axis=1)
        else:
            self.X = np.array(X).T

        self.Y = np.array(Y).T
        #transposing X ...
        self.Xt = (self.X).T
        #doing Xt * X
        linesT = (self.Xt).shape[0]
        lines = (self.X).shape[0]
        columns = (self.X).shape[1]
        XtX = np.zeros([linesT,columns])

        for i in range(linesT):
           # iterate through columns of Y
           for j in range(columns):
               # iterate through rows of Y
               for k in range(lines):
                   XtX[i][j] += self.Xt[i][k] * self.X[k][j]
        #doing Xt * y
        linesT = (self.Xt).shape[0]
        lines = (self.Y).shape[0]
        columns = (self.Y).shape[1]
        XtY = np.zeros([linesT,columns])
        for i in range(linesT):
           # iterate through columns of Y
           for j in range(columns):
               # iterate through rows of Y
               for k in range(lines):
                   XtY[i][j] += self.Xt[i][k] * self.Y[k][j]
        #Inverting (Xt * X)
        determinant = (XtX[0][0] * XtX[1][1]) - (XtX[0][1] * XtX[1][0])
        invM = (1/determinant) * np.array([[XtX[1][1],-XtX[0][1]],[-XtX[1][0], XtX[0][0]]])
        #print(invM)
        #doing (XtX)-1 * XtY
        linesT = (invM).shape[0]
        lines = (XtY).shape[0]
        columns = (XtY).shape[1]
        beta = np.zeros([linesT,columns])
        for i in range(linesT):
           # iterate through columns of Y
           for j in range(columns):
               # iterate through rows of Y
               for k in range(lines):
                   beta[i][j] += invM[i][k] * XtY[k][j]
        # print(beta)
        self.beta = beta

    def predict(self, X):
        #Predicting values
        Xt = np.array(X)
        #doing Xt * beta
        linesT = (Xt).shape[0]
        lines = (self.beta).shape[0]
        columns = (self.beta).shape[1]
        y = np.zeros([linesT,columns])
        for i in range(linesT):
           for j in range(columns):
               for k in range(lines):
                   y[i][j] += Xt[i][k] * self.beta[k][j]
        print(y)
        return y

    def plot(self):

        if self.beta.size != 0:
            fig = plt.figure()
            X = np.array([self.X[:,1]]).T
            #Plot of dataset
            plt.plot(X,self.Y,'ro')
            # Plot of regression
            X = np.array([list(range(int(min(X.T[0])) ,int(max(X.T[0])), 1))])
            Xmod = np.append(np.ones([len(np.array(X).T),1])*1, np.array(X).T, axis=1)
            y = list()
            for x in Xmod:
                y.append(x[0]*self.beta[0][0] + x[1]*self.beta[1][0])
            plt.plot(X[0],y)

            plt.show()
