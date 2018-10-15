import pandas as pd
from SpecialTopicsinLearning.LeastSquares import LeastSquares

def main():
    '''Instanciando a classe do método'''
    ls = LeastSquares()

    '''Trocando o nome das bases e as variáveis de entrada, podemos gerar a resposta'''

    # file = 'alpswater.xlsx'
    file = 'Books_attend_grade.xls'
    # file = 'USCensus.xls'
    df = pd.read_excel('database/' + file)

    '''Para cada base temos uma certa entrada, descomentar a usada anteriormente'''

    '''Books_attend_grade.xls'''
    X_train = [df.BOOKS.values, df.ATTEND.values]
    Y_train = [df.GRADE.values]

    '''alpswater.xlsx'''
    # Y_train = [df.BPt.values]
    # X_train = [df.Pressure.values]

    '''USCensus.xls'''
    # Y_train = [df.Census.values]
    # X_train = [df.Year.values, [x**2 for x in df.Year.values]]


    # ls.fit(X=X_train,Y=Y_train,W=[[1,1,1,0,1,0,1,1,1,0]], bias=1)
    ls.fit(X=X_train,Y=Y_train, bias=1)

    '''Para cada tipo de entrada, temos que usar no vetor abaixo o primeiro valor como 1
    que representa a dimensão do bias e os proximos valores são o conjunto de
    predição Xo, X1, Xn'''

    X_test = [[1, 1, 20]]

    #função de predição
    ls.predict(X=X_test)

    # ls.plot()

if __name__ == '__main__':
    main()
