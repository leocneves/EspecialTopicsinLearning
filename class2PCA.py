import pandas as pd
from SpecialTopicsinLearning.PrincipalComponentAnalysis import PrincipalComponentAnalysis
def main():
    '''Instanciando a classe do método'''
    pca = PrincipalComponentAnalysis()

    '''Trocando o nome das bases e as variáveis de entrada, podemos gerar a resposta'''

    # file = 'alpswater.xlsx'
    file = 'Books_attend_grade.xls'
    # file = 'USCensus.xls'
    df = pd.read_excel('database/' + file)

    '''Para cada base temos uma certa entrada, descomentar a usada anteriormente'''

    # '''Books_attend_grade.xls'''
    X_1 = df.BOOKS.values
    Y_1 = df.ATTEND.values
    Z_1 = df.GRADE.values

    '''alpswater.xlsx'''
    # Y_1 = df.BPt.values
    # X_1 = df.Pressure.values

    '''USCensus.xls'''
    # Y_1 = df.Census.values
    # X_1 = df.Year.values
    # X_2 = [x**2 for x in df.Year.values]


    data_pca = pca.fit([X_1, Y_1, Z_1],n_components=2)

    pca.plot()

if __name__ == '__main__':
    main()
