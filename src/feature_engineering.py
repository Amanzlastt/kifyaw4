import pandas as pd 
import numpy as np 
import os
import sys

# laoding the pre processed data for feature extraction
df_train = pd.read_csv('C:\\Users\\Aman\\Desktop\\kifyaw4\\data\\processed\\train_processed.csv')
df_test = pd.read_csv('C:\\Users\\Aman\\Desktop\\kifyaw4\\data\\processed\\test_processed.csv')



def to_holiday(df, holiday_list):
    # Ensure holiday_list is in datetime format
    holiday_list = pd.to_datetime(holiday_list)
    result = []
    
    for date in df['Date']:
        # Calculate the timedelta to each holiday
        future_holidays = holiday_list[holiday_list > date]
        
        # Find the minimum timedelta in days
        if len(future_holidays) > 0:
            min_days = (future_holidays - date).days.min()
        else:
            min_days = 300  # Arbitrary large value if no future holiday
        
        result.append(min_days)
    return np.array(result)

def after_holiday(df, holiday_list):
    # Ensure holiday_list is in datetime format
    holiday_list = pd.to_datetime(holiday_list)
    result = []
    
    for date in df['Date']:
        # Calculate the timedelta to each holiday
        future_holidays = holiday_list[holiday_list < date]
        
        # Find the minimum timedelta in days
        if len(future_holidays) > 0:
            min_days = (date - future_holidays).days.min()
        else:
            min_days = 300  # Arbitrary large value if no future holiday
        
        result.append(min_days)
    return np.array(result)
def create_columns(df):
    # changing to date time datatype
    df_train['Date'] = pd.to_datetime(df_train['Date'])

    df_train['IsWeekday'] = df_train['Date'].dt.weekday < 5  # True for Monday-Friday
    df_train['IsWeekend'] = ~df_train['IsWeekday']          # True for Saturday-Sunday

    holiday_a = df[df['StateHoliday_a']== 1]['Date'].unique()
    holiday_b = df[df['StateHoliday_b']== 1]['Date'].unique()
    holiday_c = df[df['StateHoliday_c']== 1]['Date'].unique()

    from_holiday_a = to_holiday(df, holiday_a)
    from_holiday_b = to_holiday(df, holiday_b)
    from_holiday_c = to_holiday(df, holiday_c)

    after_holiday_a = after_holiday(df, holiday_a)
    after_holiday_b = after_holiday(df, holiday_b)
    after_holiday_c = after_holiday(df, holiday_c)

    df['Days from Holiday_a'] = from_holiday_a
    df['Days from Holiday_b'] = from_holiday_b
    df['Days from Holiday_c'] = from_holiday_c

    df['Days after Holiday_a'] = after_holiday_a
    df['Days after Holiday_b'] = after_holiday_b
    df['Days after Holiday_c'] = after_holiday_c

    return df
train_featured = create_columns(df_train)
test_featured = create_columns(df_test)

data_path = os.path.join("data", 'features')
os.makedirs(data_path)
train_featured.to_csv(os.path.join(data_path, 'added_features.csv'), index=False)
test_featured.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)