"""
Train the NN model.
"""
import sys
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data, get_scats_list
from model import model
from keras.models import Model
from keras.callbacks import EarlyStopping
from tqdm import tqdm
warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, name, config, scat_number):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
        scat_number: SCAT Number of the model trained
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save('model/' + name + '/' + str(scat_number) + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + name + '/' + str(scat_number) + ' loss.csv', encoding='utf-8', index=False)


def train_seas(models, X_train, y_train, name, config, scat_number):
    """train
    train the SAEs model.

    # Arguments
        models: List, list of SAE model.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
        scat_number: SCAT Number of the model trained
    """

    temp = X_train
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')

    for i in range(len(models) - 1):
        if i > 0:
            p = models[i - 1]
            hidden_layer_model = Model(p.input,
                                       p.get_layer('hidden').output)
            temp = hidden_layer_model.predict(temp)

        m = models[i]
        m.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])

        m.fit(temp, y_train, batch_size=config["batch"],
              epochs=config["epochs"],
              validation_split=0.05)

        models[i] = m

    saes = models[-1]
    for i in range(len(models) - 1):
        weights = models[i].get_layer('hidden').get_weights()
        saes.get_layer('hidden%d' % (i + 1)).set_weights(weights)

    train_model(saes, X_train, y_train, name, config, scat_number)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="lstm",
        help="Model to train.")
    args = parser.parse_args()
    lag = 12
    config = {"batch": 256, "epochs": 100}
    file1 = 'data/data1.xls'
    file2 = 'data/data1.xls'

    for num in tqdm(get_scats_list(file1)):
        X_train, y_train, _, _, _ = process_data(file1, file2, lag, num[0])

        if args.model == 'lstm':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_lstm([12, 64, 64, 1])
            train_model(m, X_train, y_train, args.model, config, num[0])
        if args.model == 'gru':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_gru([12, 64, 64, 1])
            train_model(m, X_train, y_train, args.model, config, num[0])
        if args.model == 'saes':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
            m = model.get_saes([12, 400, 400, 400, 1])
            train_seas(m, X_train, y_train, args.model, config, num[0])
        if args.model == 'srnn':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_srnn([12, 64, 64, 1])
            train_model(m, X_train, y_train, args.model, config, num[0])
        if args.model == 'bidir':
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            m = model.get_biDir([12, 64, 64, 1])
            train_model(m, X_train, y_train, args.model, config, num[0])


if __name__ == '__main__':
    main(sys.argv)
