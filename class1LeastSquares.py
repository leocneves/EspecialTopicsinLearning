from SpecialTopicsinLearning.LeastSquares import LeastSquares

def main():
    ls = LeastSquares()

    #ls.help()

    X_train = [[69,67,71,65,72,68,74,65,66,72]]
    Y_train = [[9.5,8.5,11.5,10.5,11,7.5,12,7,7.5,13]]

    ls.fit(X=X_train,Y=Y_train, bias=1)

    X_test = [[1, 70]]

    ls.predict(X=X_test)

    ls.plot()

if __name__ == '__main__':
    main()
