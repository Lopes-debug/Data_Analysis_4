import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

## Campanha de Marketing para Instituição Financeira

## Start

pd.set_option('display.max_columns', 25)
dataset = pd.read_csv(r'C:\Users\leand\OneDrive\Documentos\FormacaoDSA\f_projeto4\Scripts-P4\dados\dataset.csv', na_values= 'undefined')


## Exploratory Analysis

# print(dataset.head())
# print(dataset.info())
# print(dataset.shape)
# print(dataset.describe())
# for x in dataset.columns:
#     print(dataset[x].value_counts(), '\n ------')
# print(dataset.isna().sum())
    

## Data cleaning

# Repart one column in two columns
datacopy = dataset.copy()
# list_job = []
# list_edu = []
# for i in datacopy['jobedu']:
#     pos_v = i.find(',')
#     job = i[:pos_v]
#     edu = i[pos_v +1:]
#     list_job.append(job)
#     list_edu.append(edu)

# datacopy.insert(4, 'job', list_job, True)
# datacopy.insert(5, 'education', list_edu, True)
# datacopy = datacopy.drop(columns=['customerid','jobedu'])

# print(datacopy.info())
# print(datacopy.head())

## Answer key exercise 1 - Gabarito Exercicio 1
datacopy['job'] = datacopy['jobedu'].apply(lambda x: x.split(',')[0])
datacopy['education'] = datacopy['jobedu'].apply(lambda x: x.split(',')[1])
datacopy = datacopy.drop(columns=['customerid','jobedu'])

# Repart one column in two columns
# list_year = []
# list_month = []
# for i in datacopy['month']:
#     i = str(i)
#     v = i.find(',')
#     year = i[v+1:]
#     month = i[:v]
#     list_month.append(month)
#     list_year.append(year)

# datacopy = datacopy.drop(columns='month')
# datacopy.insert(12, 'month', list_month, True)
# datacopy.insert(13, 'year', list_year, True)
# print(datacopy.head())

##Create a new column from values of three columns, concatenate
# datacopy['datetime'] = datacopy.apply(lambda x: '{}/{}/{}'.format(x['day'],x['month'],x['year']), axis=1)

## Formating
# datacopy['datetime'] = datacopy['datetime'].replace(' ', '', regex=True)
# datacopy = datacopy.drop(columns=['day', 'month','year'])

## Move column position 
# dfdatetime = datacopy.pop('datetime')
# datacopy.insert(10, 'datetime', dfdatetime )


## Missing Values

import sys
sys.path.append(r'C:\Users\leand\OneDrive\Documentos\FormacaoDSA\f_project3\Scripts-P3\modulos')
from estrategia1 import*

# func_calc_percentual_valores_ausentes(datacopy)
# print("O dataset tem {} % de valores ausentes do total de {} valores".format(round((datacopy.isnull().sum().sum()/(datacopy.shape[0]*datacopy.shape[1]))*100, 2), datacopy.shape[0]))

# func_calc_percentual_valores_ausentes_linha(datacopy)
# print('{:.2} % das linhas contém ao menos um valor ausente'.format((sum([True for idx, row in datacopy.iterrows() if any(row.isna())])/datacopy.shape[0])*100))

# func_calc_percentual_valores_ausentes_coluna(datacopy)
# miss_col = pd.concat([datacopy.isnull().sum(), 100 * datacopy.isnull().sum()/len(datacopy), datacopy.dtypes], axis=1).rename(columns={0:'Missing Values', 1:'% Missing Values', 2: 'Dtypes'})
# miss_col = miss_col[miss_col.iloc[:,0]!= 0].sort_values('% Missing Values', ascending=False).round(2)
# print('O dataset tem ' + str(datacopy.shape[1]) + ' colunas.\nEncontrado: ' + str(miss_col.shape[0]) + ' colunas que têm valores ausentes.')
# print(miss_col)


## Change types
#* to change types is necessary that the columns not have any missing value
# print(datacopy.info())
# datacopy['age'] = datacopy['age'].astype('int64')
# for col in datacopy['datetime']:
#     datacopy[col] = pd.to_datetime(dataset[col])

# print(datacopy.info())


