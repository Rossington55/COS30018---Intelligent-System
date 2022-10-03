"""
Processing the data
"""
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime as dt



def process_data(train, test, lags):

    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """
    #encoding='utf-8'
    attr = 'Lane 1 Flow (Veh/5 Minutes)'



    df1 = pd.read_excel(train, sheet_name='Data', skiprows=0).fillna(0)

    dfHeaders = pd.read_excel(train, sheet_name='Data', skiprows=1).fillna(0)

    df2 = pd.read_excel(test, sheet_name='Data', skiprows=0).fillna(0)
    
    timeColumnNames = df1
    timeColumnNames = timeColumnNames.loc[:1]
    
    timeColumnNames.drop(columns=timeColumnNames.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], 
        axis=1, 
        inplace=True)

    dateColumn = dfHeaders[['Date', 'SCATS Number']]
    
    timeColumnNames = timeColumnNames.iloc[:1]

    timeColumns = df1

    timeColumns.drop(columns=timeColumns.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], 
    axis=1, 
    inplace=True)

    timeColumns = timeColumns[1:]

    # df_LocationMapping = df1.drop_duplicates('SCATS Number')
    # df_LocationMapping = df_LocationMapping[['SCATS Number', 'Location']]

    new = pd.merge(dateColumn, timeColumns, left_index=True, right_index=True)

    pivotData = pd.melt(new, id_vars=['Date', 'SCATS Number'], value_vars=timeColumnNames)

    # pivotData = pivotData.where(pivotData['SCATS Number'] == locationSearch)

    #removes unneeded columns.
    pivotData.drop(columns=pivotData.columns[[1]], 
    axis=1, 
    inplace=True)
    
    pivotData['# Lane Points']=1
    pivotData['% Observed']=100
    pivotData.rename(columns = {'value':'Lane 1 Flow (Veh/5 Minutes)'}, inplace = True)
    pivotData.rename(columns = {'Date':'5 Minutes'}, inplace = True)

    # print(pivotData)
    df1 = pivotData
    df2 = pivotData

    #filter data framne to reduce columns to only whats required (compared to the supplied working data)
    #Cleaning - remove unwanted columns
    # pivot table into same format as supplied data 
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html

    #obtain uneque road codes and store in an array (so we can filter dataset by road names at some stage maybe)
    
    #shuffle

    scaler = StandardScaler().fit(df1[attr].values)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    # train, test = [], []
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])

    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, scaler

# if __name__ == '__main__':

#     process_data(train, test, lags, locationSearch)