import pandas as pd
from SpecialTopicsinLearning.PrincipalComponentAnalysis import PrincipalComponentAnalysis
def main():
    '''Instanciando a classe do método'''
    pca = PrincipalComponentAnalysis()

    '''Trocando o nome das bases e as variáveis de entrada, podemos gerar a resposta'''

    file = 'alpswater.xlsx'
    # file = 'Books_attend_grade.xls'
    # file = 'USCensus.xls'
    df = pd.read_excel('database/' + file)

    '''Para cada base temos uma certa entrada, descomentar a usada anteriormente'''

    # '''Books_attend_grade.xls'''
    # X_train = [df.BOOKS.values, df.ATTEND.values]
    # Y_train = [df.GRADE.values]

    '''alpswater.xlsx'''
    Y_train = df.BPt.values
    X_train = df.Pressure.values

    '''USCensus.xls'''
    # Y_train = df.Census.values
    # X_train = [df.Year.values, [x**2 for x in df.Year.values]]


    pca.fit([Y_train, X_train])

    pca.plot()

if __name__ == '__main__':
    main()
