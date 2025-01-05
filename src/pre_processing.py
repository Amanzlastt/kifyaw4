import pandas as pd 
import numpy as np
import os 

from sklearn.impute import SimpleImputer

train_data = pd.read_csv("C:\\Users\\Aman\\Desktop\\kifyaw4\\data\\raw\\train.csv")
test_data = pd.read_csv("C:\\Users\\Aman\\Desktop\\kifyaw4\\data\\raw\\test.csv")


# changing date column to datetime data type
train_data['Date'] = pd.to_datetime(train_data['Date'])
test_data['Date'] = pd.to_datetime(test_data['Date'])

# Droping data with missing dates
def drop_missing_dates(df):
    cleaned_data = df.dropna(subsets = 'Date')
    return cleaned_data

train_data = drop_missing_dates(train_data)
test_data = drop_missing_dates(test_data)

# Encodin catagorical column ("State Hoiday")
def encoder(df, column):
    df[column] = df[column].astype(str)
    encoded_column = pd.get_dummies(df[column], prefix=column)
    df_encoded = pd.concat([df.drop(column, axis= 1), encoded_column], axis=1)
    return df_encoded

train_data = encoder(train_data, 'StateHliday')
test_data = encoder(test_data, 'StateHliday')


# Filling missing values using simple imputer
imputer = SimpleImputer(strategy='median')
train_imputed = imputer.fit_test(train_data.drop['Date'], axis = 1)
test_imputed = imputer.fit_transform(test_data.drop['Date'], axis = 1)

# merging date column with the imputed ones
train_processed_data = pd.concat([train_data['Date'], train_imputed], axis=1)
test_processed_data = pd.concat([test_data['Date'], test_imputed], axis=1)



data_path = os.path.join("data", 'processed')

os.makedirs(data_path)

train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
