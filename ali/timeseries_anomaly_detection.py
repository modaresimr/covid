import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import json
import os
import compress_pickle
from sklearn.preprocessing import StandardScaler


def selectTrainTestSet(rhr1,info):
    
    if info['covid_test_date']:
        dates_s = rhr1.loc[(rhr1.index < info['covid_test_date']-pd.to_timedelta('21d'))].index.floor('1D').unique()
        dates_e = rhr1.loc[(rhr1.index > info['covid_test_date']+pd.to_timedelta('21d'))].index.floor('1D').unique()
    else:
        dates_s = rhr1.index.floor('1D').unique()
        dates_e = rhr1.index[[]]

    s = min(28,len(dates_s))
    e = -min(28-s,len(dates_e))-1
    train = rhr1.loc[rhr1.index < dates_s[s]|rhr1.index > dates_e[e]]
    test = rhr1.loc[(rhr1.index >= dates_s[s]) & (rhr1.index <= dates_e[e])]

    return train,test
        

def preprocessing(train,test):
    # display(rhr)

    if os.path.isfile(f'ali/an_data/preprocess.pkl.l4z'):
        scaler = compress_pickle.load(f'ali/an_data/preprocess.pkl.l4z')
    else:
        scaler = StandardScaler()
        scaler = scaler.fit(train)
        compress_pickle.load(scaler,f'ali/an_data/preprocess.pkl.l4z')

    train=scaler.transform(train)
    test=scaler.transform(test)

def create_sequences(data,seg='1D'):
    

    output = []
    for w in data.rolling(window=seg):
        output.append(w.values.T[0])


    return np.stack(output)

def anomaly_detection(rhr,info,seg='1T'):
    print(seg)
    TIME_STEPS =int(pd.to_timedelta('7h')/pd.to_timedelta(seg)*3)

    print(info)
    # ali.ui.plot(rhr,alerts=pd.DataFrame(),info=info,show=True)
    rhr1=rhr.resample(seg).mean().dropna()
    train,test=selectTrainTestSet(rhr1,info)
    train, test=preprocessing(train,test)

    
    # Generated training sequences for use in the model.
    def create_sequences(values, time_steps=TIME_STEPS):
        output = []
        for i in range(len(values) - time_steps + 1):
            output.append(values[i : (i + time_steps)])
        return np.stack(output)


    x_train = create_sequences(df_training_value.values)
    # print("Training input shape: ", x_train.shape)


    if os.path.isfile(f'ali/an_data/model-{seg}.h5'):
        model = tf.keras.models.load_model(f'ali/an_data/model-{seg}.h5')
    else:

        model = keras.Sequential(
        [
            layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    # model.summary()
    history = model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=128,
        validation_split=0.1,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )
    model.save(f'ali/an_data/model-{seg}.h5')

    # plt.plot(history.history["loss"], label="Training Loss")
    # plt.plot(history.history["val_loss"], label="Validation Loss")
    # plt.legend()
    # plt.show()
    

    # Get train MAE loss.
    x_train_pred = model.predict(x_train)
    train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

    # plt.hist(train_mae_loss, bins=50)
    # plt.xlabel("Train MAE loss")
    # plt.ylabel("No of samples")
    # plt.show()

    # Get reconstruction loss threshold.
    threshold = np.max(train_mae_loss)/2
    # print("Reconstruction error threshold: ", threshold)








    
    df_test_value = (test - training_mean) / training_std
    # fig, ax = plt.subplots()
    # df_test_value.plot(legend=False, ax=ax)
    # plt.show()

    # Create sequences from test values.
    x_test = create_sequences(df_test_value.values)
    print("Test input shape: ", x_test.shape)

    # Get test MAE loss.
    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
    test_mae_loss = test_mae_loss.reshape((-1))

    # plt.hist(test_mae_loss, bins=50)
    # plt.xlabel("test MAE loss")
    # plt.ylabel("No of samples")
    # plt.show()

    # Detect all the samples which are anomalies.
    anomalies = test_mae_loss > threshold
    # print("Number of anomaly samples: ", np.sum(anomalies))
    # print("Indices of anomaly samples: ", np.where(anomalies))

    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    anomalous_data_indices = []
    for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
        if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
            anomalous_data_indices.append(data_idx)

    df_subset = test.iloc[anomalous_data_indices]


    dates=test.iloc[anomalous_data_indices].resample('1D').count().fillna(0).rename(columns={'heartrate':'count'})

    dates['alarm']=(dates['count']>0)*2
    return dates
    # ali.ui.plot(rhr,alerts=dates,info=info,show=True)
