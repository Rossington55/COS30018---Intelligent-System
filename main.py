"""
Traffic Flow Prediction with Neural Networks(SAEs、LSTM、GRU).
"""
import math
import warnings
import numpy as np
import pandas as pd
from data.data import process_data, get_scats_list
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import sklearn.metrics as metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings("ignore")


def MAPE(y_true, y_pred):
    """Mean Absolute Percentage Error
    Calculate the mape.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    # Returns
        mape: Double, result data for train.
    """

    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape


def eva_regress(y_true, y_pred):
    """Evaluation
    evaluate the predicted resul.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
    """
    
    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)



def plot_results(y_true, y_preds, names, scat):
    """Plot
    Plot the true data and predicted data.

    # Arguments
        y_true: List/ndarray, ture data.
        y_pred: List/ndarray, predicted data.
        names: List, Method names.
    """

    d = '2006-1-10 00:00'
    x = pd.date_range(d, periods=96, freq='15min')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, y_true, label='True Data')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name)
    ax.set_title(scat)

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    plt.savefig('foo.png')
    plt.show()
   
def distance_betweeen(long1,lat1,long2,lat2):
    LAT_corrected = lat1 + 0.00155
    LONG_corrected = long1 + 0.00113
    LAT_corrected2 = lat2 + 0.00155
    LONG_corrected2 = long2 + 0.00113
    R = 6371 #Radius of earth in km
    dLat = deg_to_rad(LAT_corrected2-LAT_corrected)
    dLon = deg_to_rad(LONG_corrected2-LONG_corrected)
    a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg_to_rad(LAT_corrected)) * math.cos(deg_to_rad(LAT_corrected2)) * math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c 
    return d

def deg_to_rad(deg):
    return deg*(math.pi/180)

def speed_at_flow(flow):
    x = 8 * (math.sqrt(-10*(flow-1000))+100)
    return x/25

def main():
    lstm = 'model/lstm/'
    gru = 'model/gru/'
    saes = 'model/saes/'
    srnn = 'model/srnn/'
    biDir = 'model/biDir/'
    
    models = [lstm, gru, saes, srnn, biDir]
    names = ['LSTM', 'GRU', 'SAEs', 'SimpleRNN', 'Bidirectional']

    lag = 12
    file1 = 'data/data1.xls'
    file2 = 'data/data1.xls'

    for num in tqdm(get_scats_list(file1)):
        print(num[0])
        _, _, X_test, y_test, scaler = process_data(file1, file2, lag, num[0])
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]
        
        y_preds = []
        for name, model in zip(names, models):
            if name == 'SAEs' or name == "SimpleRNN":
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
            else:
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            file = 'images/' + name + '.png'
            mdl = load_model(model + str(num[0]) + '.h5')
            plot_model(mdl, to_file=file, show_shapes=True)
            predicted = mdl.predict(X_test)
            predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
            y_preds.append(predicted[:96])
            print(name)
            eva_regress(y_test, predicted)
        
        plot_results(y_test[: 96], y_preds, names, num[0])
        


if __name__ == '__main__':
    main()
