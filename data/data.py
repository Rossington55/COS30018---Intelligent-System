"""
Processing the data
"""
from asyncio.windows_events import NULL
import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import datetime as dt
from pathlib import Path

def get_scats_list(train):
    df = pd.read_excel(train, sheet_name='Data', skiprows=1).fillna(0)
    df = df.drop_duplicates(subset=['SCATS Number'])
    df = df[['SCATS Number']]
    return df.to_numpy()

def process_data(train, test, lags, scat_number):

    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
        scat_number: SCAT Number of the site to be trained
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
   
    dateColumn = dfHeaders[['Date', 'SCATS Number', 'HF VicRoads Internal']]
    timeColumnNames = timeColumnNames.iloc[:1]
    timeColumns = df1
    timeColumns.drop(columns=timeColumns.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], 
    axis=1, 
    inplace=True)
    timeColumns = timeColumns[1:]
    # df_LocationMapping = df1.drop_duplicates('SCATS Number')
    # df_LocationMapping = df_LocationMapping[['SCATS Number', 'Location']]

    new = pd.merge(dateColumn, timeColumns, left_index=True, right_index=True)
   
    pivotData = pd.melt(new, id_vars=['Date', 'SCATS Number', 'HF VicRoads Internal'], value_vars=timeColumnNames)
    
    # pivotData = pivotData.where(pivotData['SCATS Number'] == locationSearch)

    #removes unneeded columns.
    # pivotData.drop(columns=pivotData.columns[[1]], 
    # axis=1, 
    # inplace=True)
    #scatsSites = pivotData['SCATS Number'].tolist()
    pivotData['# Lane Points']=1
    pivotData['% Observed']=100
    pivotData.rename(columns = {'value':'Lane 1 Flow (Veh/5 Minutes)'}, inplace = True)
    pivotData.rename(columns = {'Date':'5 Minutes'}, inplace = True)
    
    df1 = pivotData.where(pivotData['SCATS Number'] == scat_number).dropna()
    df2 = pivotData.where(pivotData['SCATS Number'] == scat_number).dropna()
    df2 = df1.sort_values(by=['5 Minutes', 'HF VicRoads Internal','variable'])

    filepath = Path('data/out.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    pivotData.to_csv(filepath) 
    #filter data framne to reduce columns to only whats required (compared to the supplied working data)
    #Cleaning - remove unwanted columns
    # pivot table into same format as supplied data 
    #https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.pivot_table.html

    #obtain uneque road codes and store in an array (so we can filter dataset by road names at some stage maybe)
    
    #shuffle


    #scaler = StandardScaler().fit(df1[attr].values)

    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    train, test = [], []
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
    # return 0

def get_opposite_direction(direction):
    if (direction == 'N'):
        return 'S'
    elif (direction == 'S'):
        return 'N'
    elif (direction == 'E'):
        return 'W'
    elif (direction == 'W'):
        return 'E'
    elif (direction == 'NE'):
        return 'SW'
    elif (direction == 'NW'):
        return 'SE'
    elif (direction == 'SE'):
        return 'NW'
    elif (direction == 'SW'):
        return 'NE'
    else:
        return 'N'

def sort_streets_by_direction(df_streets, direction, long, lat):
    if (direction == 'N' or direction == 'NE' or direction == 'E'):
        df_streets = df_streets.sort_values(['NB_LONGITUDE', 'NB_LATITUDE'], ascending=[True, True])
        sorted_streets = df_streets.to_numpy()
        #print('Sorted Streets: ' + str(sorted_streets))
        for scat in sorted_streets:
            if (direction == 'N'):
                if (scat[2] > lat):
                    return scat
            elif (direction == 'NE'):
                if ((scat[2] > lat) and (scat[3] > long)):
                    return scat
            elif (direction == 'E'):
                if (scat[3] > long):
                    return scat
    elif (direction == 'S' or direction == 'SW' or direction == 'W'):
        df_streets = df_streets.sort_values(['NB_LONGITUDE', 'NB_LATITUDE'], ascending=[False, False])
        sorted_streets = df_streets.to_numpy()
        #print('Sorted Streets: ' + str(sorted_streets))
        for scat in sorted_streets:
            if (direction == 'S'):
                if (scat[2] < lat):
                    return scat
            elif (direction == 'SW'):
                if ((scat[2] < lat) and (scat[3] < long)):
                    return scat
            elif (direction == 'W'):
                if (scat[3] < long):
                    return scat
    elif (direction == 'NW'):
        df_streets = df_streets.sort_values(['NB_LONGITUDE', 'NB_LATITUDE'], ascending=[True, False])
        sorted_streets = df_streets.to_numpy()
        #print('Sorted Streets: ' + str(sorted_streets))
        for scat in sorted_streets:
            if ((scat[2] > lat) and (scat[3] < long)):
                return scat
    elif (direction == 'SE'):
        df_streets = df_streets.sort_values(['NB_LONGITUDE', 'NB_LATITUDE'], ascending=[False, True])
        sorted_streets = df_streets.to_numpy()
        #print('Sorted Streets: ' + str(sorted_streets))
        for scat in sorted_streets:
            if ((scat[2] < lat) and (scat[3] > long)):
                return scat
    else:
        emp = []
        return emp

def process_map(file, scat):
    df1 = pd.read_excel(file, sheet_name='Data', skiprows=1).fillna(0)
    df1 = df1.drop_duplicates(subset=['Location'])
    df1 = df1[['SCATS Number', 'Location', 'NB_LATITUDE', 'NB_LONGITUDE']] # Get just the SCAT Num, Streets, Long and Lat

    df_val = df1.drop_duplicates(subset=['SCATS Number'])
    for row in df_val.to_numpy():
        if (row[0] == scat):
            long = row[3]
            lat = row[2]

    #print('Long: ' + str(long))
    #print('Lat: ' + str(lat))

    df2 = pd.read_excel(file, sheet_name='Data', skiprows=1).fillna(0)
    df2 = df2.drop_duplicates(subset=['Location'])
    df2.drop(df2.loc[df2['SCATS Number']!=scat].index, inplace=True) # Get only streets of indicated SCAT
    df2 = df2[['SCATS Number', 'Location', 'NB_LATITUDE', 'NB_LONGITUDE']]
    unclean_streets = df2.to_numpy()
    clean_streets = [] # 2D Array, [0]=index, [1]=list ==> [0]=Street Name, [1]=Direction
    for street in unclean_streets:
        split_street = street[1].split()
        street = []
        
        dirs = ['N', 'NW', 'NE', 'E', 'SE', 'S', 'SW', 'W']
        if (split_street[1] in dirs):
            street.append(split_street[0])
            street.append(split_street[1])
        else: # Street has a space in it
            street.append(split_street[0] + ' ' + split_street[1])
            street.append(split_street[2])
        #print('Street 0: ' + street[0])
        #print('Street 1: ' + street[1])
        clean_streets.append(street)

    street_connections = []
    for scat_street in clean_streets:
        #print('Scat Street: ' + str(scat_street))
        street_con = []
        street_con.append(scat_street[0])
        street_con.append(scat_street[1])
        #direction = get_opposite_direction(scat_street[1])
        direction = scat_street[1]

        df_street = pd.read_excel(file, sheet_name='Data', skiprows=1).fillna(0)
        df_street = df_street.drop_duplicates(subset=['Location'])
        df_street = df_street[['SCATS Number', 'Location', 'NB_LATITUDE', 'NB_LONGITUDE']] # Get just the SCAT Num, Streets, Long and Lat
        df_street.drop(df_street.loc[df_street['SCATS Number']==scat].index, inplace=True) # Remove any remaining 'host' scat site entries

        df_street = df_street[df_street['Location'].str.contains(scat_street[0] + ' ') == True]

        #df_street.drop(df1.loc[df1['Location'].str.contains(scat_street[0] + ' ')].index, inplace=True) # Remove where street name is not same
        
        # df_street.drop(df_street.loc[df1['Location'].split()[1]!=direction].index, inplace=True) # Remove where direction is not opposite (pointing towards) SCAT Not necessarily needed
        
        # Drop rows based on direction of travel at SCAT based on opposite direction of origin SCAT
        connected_scat = sort_streets_by_direction(df_street, direction, long, lat)
        #print('Connection SCAT: ' + str(connected_scat))
        if connected_scat is not None:
            street_con.append(connected_scat[0])
            street_connections.append(street_con)

        

    return scat, long, lat, street_connections # Return SCAT Number, SCAT Longitude, SCAT Latitude, All Street Names for SCAT [0] + Direction of Travel of Street [1] + Connected SCAT [2]]

# if __name__ == '__main__':

#     process_data(train, test, lags, locationSearch)