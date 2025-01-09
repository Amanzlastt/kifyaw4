import pandas as pd 
import numpy as np
import os 

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv("C:\\Users\\Aman\\Desktop\\kifyaw4\\data\\raw\\train.csv")
test_data = pd.read_csv("C:\\Users\\Aman\\Desktop\\kifyaw4\\data\\raw\\test.csv")


# changing date column to datetime data type
train_data['Date'] = pd.to_datetime(train_data['Date'])
test_data['Date'] = pd.to_datetime(test_data['Date'])

# Droping data with missing dates
def drop_missing_dates(df):
    cleaned_data = df.dropna(subset = ['Date'])
    return cleaned_data

train_data = drop_missing_dates(train_data)
test_data = drop_missing_dates(test_data)

# Encodin catagorical column ("State Hoiday")
encoder = LabelEncoder()
def encoder_fun(df, column):
    df[column] = df[column].astype(str)
    encoded_column = encoder.fit_transform(df[column])
    df[column] = encoded_column
    # encoded_column_df = pd.DataFrame(encoded_column, columns=encoder.get_feature_names_out([column]))
    # df_encoded = pd.concat([df.drop(column, axis= 1), encoded_column_df], axis=1)
    return df

train_data = encoder_fun(train_data, 'StateHoliday')
test_data = encoder_fun(test_data, 'StateHoliday')


# Filling missing values using simple imputer
imputer = SimpleImputer(strategy='median')
train_imputed = imputer.fit_transform(train_data.drop(['Date'], axis = 1))
test_imputed = imputer.fit_transform(test_data.drop(['Date'], axis = 1))

# changing parrays formed during imputation to dataframe
train_imputed = pd.DataFrame(train_imputed,columns=train_data.drop(['Date'],axis=1).columns)
test_imputed = pd.DataFrame(test_imputed,columns=test_data.drop(['Date'], axis=1).columns)

# merging date column with the imputed ones
train_processed_data = pd.concat([train_data[['Date']], train_imputed], axis=1)
test_processed_data = pd.concat([test_data[['Date']], test_imputed], axis=1)

print (train_processed_data.dtypes)


data_path = os.path.join("data", 'processed')

os.makedirs(data_path)

train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)

