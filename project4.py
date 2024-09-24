import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Campanha de Marketing para Instituição Financeira


## Handling Missing Values

# import sys
# sys.path.append(r'C:\Users\leand\OneDrive\Documentos\FormacaoDSA\f_project3\Scripts-P3\modulos')
# from estrategia1 import*
# from estrategia2 import *

# func_calc_percentual_valores_ausentes(datacopy)
# print("O dataset tem {} % de valores ausentes do total de {} valores".format(round((datacopy.isnull().sum().sum()/(datacopy.shape[0]*datacopy.shape[1]))*100, 2), datacopy.shape[0]))

# func_calc_percentual_valores_ausentes_linha(datacopy)
# print('{:.2} % das linhas contém ao menos um valor ausente'.format((sum([True for idx, row in datacopy.iterrows() if any(row.isna())])/datacopy.shape[0])*100))

# func_calc_percentual_valores_ausentes_coluna(datacopy)
# miss_col = pd.concat([datacopy.isnull().sum(), 100 * datacopy.isnull().sum()/len(datacopy), datacopy.dtypes], axis=1).rename(columns={0:'Missing Values', 1:'% Missing Values', 2: 'Dtypes'})
# miss_col = miss_col[miss_col.iloc[:,0]!= 0].sort_values('% Missing Values', ascending=False).round(2)
# print('O dataset tem ' + str(datacopy.shape[1]) + ' colunas.\nEncontrado: ' + str(miss_col.shape[0]) + ' colunas que têm valores ausentes.')
# print(miss_col)

# fix_missing_ffill(datacopy, 'column')
# datacopy['column'] = datacopy['column'].fillna(method='ffill')
# print('Method progressive fill is completed')

# fix_missing_bfill(datacopy, 'column')
# datacopy['column'] = datacopy['column'].fillna(method='bfill')
# print('Method regressive fill is completed')

# fix_missing_median(datacopy, 'column')
# median = datacopy['column'].median()
# datacopy['column'] = datacopy['column'].fillna(median)
# print('Method fill by median is completed')

# fix_missing_value(datacopy, 'column', valor)
# datacopy['column'] = datacopy['column'].fillna(valor)

# drop_duplicates(datacopy)
# datacopy.drop_duplicates(inplace=True)

# drop_rows_with_missing_values(datacopy)
# datacopy.dropna(inplace=True)

# drop_columns(datacopy, 'column')
# datacopy.drop('column', axis=1, inplace=True)


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
datacopy = dataset.copy()

# Repart one column in two columns 
datacopy['job'] = datacopy['jobedu'].apply(lambda x: x.split(',')[0])
datacopy['education'] = datacopy['jobedu'].apply(lambda x: x.split(',')[1])
datacopy = datacopy.drop(columns=['customerid','jobedu'])

# Repart one column in two columns
datacopy['month'] = datacopy['month'].astype('str').str.replace(' ', '')
datacopy['M'] = datacopy['month'].apply(lambda x:x.split(',')[0] if ',' in x else x)
datacopy['Y'] = datacopy['month'].apply(lambda x:x.split(',')[1] if ',' in x else None)
datacopy['datetime'] = datacopy['day'].astype('str') + '/' + datacopy['M'].astype('str') + '/' + datacopy['Y'].astype('str')
#or this way: datacopy['datetime'] = datacopy.apply(lambda x: '{}/{}/{}'.format(x['day'],x['month'],x['year']), axis=1)
datacopy = datacopy.drop(columns=['day', 'M', 'Y', 'month'])
# print(datacopy.head(2))

# Move column position 
# dfdatetime = datacopy.pop('datetime')
# datacopy.insert(10, 'datetime', dfdatetime )

# # better way to apply a simple replace name to number(str) in column
# month_map = {
#     'jan': '1', 'feb': '2', 'mar': '3', 'apr': '4', 'may': '5', 'jun': '6',
#     'jul': '7', 'ago': '8', 'sep': '9', 'oct': '10', 'nov': '11', 'dec': '12'}

# def replace_month(df_column_str):
#     for month, number in month_map.items():
#         if month in df_column_str:
#             df_column_str = df_column_str.replace(month, number)
#     return df_column_str

#convert sec to min 
# def replaces(df):
#     if 'min' in df:
#         return float(df.replace(' min', ''))
#     elif 'sec' in df:
#         return float(df.replace(' sec', ''))/60
#     else:
#         return np.nan


## Fix Missing Values

# df = func_calc_percentual_valores_ausentes_coluna(datacopy)
# print(df)

## Age
# datacopy['age'].plot(kind='hist',title='Histogram of Age \n')
# plt.show()
# sns.boxplot(datacopy['age'])
# plt.title('Box Graphs of Age')
# plt.show()

# print(datacopy.age.describe())
# print(datacopy.age.mode())

datacopy.fillna({'age':32}, inplace=True)
# datacopy['age'].fillna(value=32, inplace=True) #future warning
# print(datacopy['age'].isnull().mean()*100)


## Salary
# datacopy['salary'].plot(kind='hist')
# plt.title('Histogram of Salary')
# plt.show()
# sns.boxplot(datacopy['salary'])
# plt.title('Histogram of Salary')
# plt.show()

# print(datacopy['salary'].describe())
# print(datacopy['salary'].mode())

datacopy.fillna({'salary':60000}, inplace=True)
# datacopy['salary'].fillna(60000, inplace=True)  #future warning
datacopy['salary'] = datacopy['salary'].replace(0, datacopy['salary'].median())
# print(datacopy['salary'].value_counts())

# print(datacopy.isna().any())


## Response
datacopy.dropna(subset=['response'], inplace=True)
# print(datacopy.isnull().sum())


## Pdays
# print(datacopy['pdays'].describe())
datacopy['pdays'] = datacopy['pdays'].replace({-1.0:np.NaN})
# print(datacopy['pdays'].isnull().mean()*100)
datacopy.drop(columns=['pdays'],inplace=True)

# print(datacopy.isnull().sum())

## Change types
#* to change types is necessary that the columns not have any missing value

datacopy['age'] = datacopy['age'].astype('int64')

# datacopy['datetime'] = pd.to_datetime(datacopy['datetime'], format='%d/%m/%Y', errors='coerce')

# datacopy['duration'] = datacopy['duration'].apply(replaces).astype(float).round(2)
# datacopy= datacopy.rename(columns={'duration': 'duration (min)'})

# print(datacopy.info())
# print(datacopy.head())


## Univariate Analysis
# Marital
# print(datacopy['marital'].value_counts(normalize=True))
# datacopy['marital'].value_counts(normalize=True).plot(kind='barh')
# plt.show()

# Job
# print(datacopy['job'].value_counts(normalize=True))
# datacopy['job'].value_counts(normalize=True).plot(kind='barh')
# plt.legend()
# plt.show()

# Education
# print(datacopy['education'].value_counts(normalize=True))
# datacopy['education'].value_counts(normalize=True).plot(kind='pie')
# plt.legend()
# plt.show()

# Response
# print(datacopy['response'].value_counts(normalize=True))
# datacopy['response'].value_counts(normalize=True).plot(kind='pie')
# plt.legend()
# plt.show()


## Multivariate Analysis
# Balance, Salary
# sns.scatterplot([datacopy['balance'], datacopy['salary']])
# plt.legend()
# plt.show()

# Balance, Age
# sns.scatterplot([datacopy['balance'], datacopy['age']])
# plt.legend()
# plt.show()

# Balance, Salary, Age
# sns.pairplot(datacopy[['balance', 'salary', 'age']])
# plt.show()

# Corr Balance, Salary, Age
# cor = datacopy[['balance', 'salary', 'age']].corr()
# plt.figure(figsize=(10,5))
# sns.heatmap(cor, annot=True, cmap='Reds')
# plt.title('Correlation Map \n', fontdict = {'fontsize': 20, 'fontweight':5, 'color': 'Green'})
# plt.show()

## Numerical x Categoric
# Response, Salary
# print(datacopy.groupby(by=['response'])['salary'].mean())
# print(datacopy.groupby(by=['response'])['salary'].median())
# plt.figure(figsize=(10,5))
# sns.boxplot(data=datacopy, x='response', y='salary')
# plt.title('Response X Salary \n', fontdict = {'fontsize': 20, 'fontweight':5, 'color': 'Green'})
# plt.show()

# Education, Salary
# print(datacopy.groupby(by=['education'])['salary'].mean())

#Categorical to 0 or 1
# datacopy['response_flag'] = np.where(datacopy['response']=='yes', 1, 0)
# corr1 = datacopy.pivot_table(index='education', columns='marital', values='response_flag', aggfunc='mean')
# sns.heatmap(corr1 ,annot=True, cmap='Reds')
# plt.title('Education vs Marital vs Response \n', fontdict = {'fontsize': 20, 'fontweight':5, 'color': 'Green'})
# plt.show()


## Final
# Missing values were fix using the median, was the best way for not change data a lot
# Person with tertiary education level and single have more propensity to buy the product/service 
# The correlation between balance, salary and age is minimum
# Investing in education for people is a good way to better the sales. The more education level the person have, the more he will have high salary and more propensity to buy 
#