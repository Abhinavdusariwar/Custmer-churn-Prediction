#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
df = pd.read_csv(r'C:\Users\Hp\Downloads\churn.csv')
df.head()

#checking the shape of the dataset
df.shape

#drop coulumns
df = df.drop(['RowNumber','CustomerId','Surname'], axis=1)

df.isnull().sum()

#column data types
df.dtypes

#dulicate values
df.duplicated().sum()

#rename column
df.rename(columns={'Exited':'Churn'}, inplace=True)
