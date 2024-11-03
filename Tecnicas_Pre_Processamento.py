import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Lê e carrega os dados em memória
url_dados = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
colunas = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'],
dados = pd.read_csv(url_dados, header=None, sep=',', na_values=[' ?', '?', '? '], names=colunas)

# Exibe as primeiras linhas do conjunto
dados.head()

# Exibe informações sobre o conjunto de dados carregado
# Observamos: volume de dados, nomes das colunas, tipos de dados e dados faltantes
dados.info()

