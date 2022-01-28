from sklearn.preprocessing import SplineTransformer
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import json
import os
import compress_pickle
from sklearn.preprocessing import StandardScaler
from tqdm.keras import TqdmCallback
from keras_tqdm import TQDMNotebookCallback
from tensorflow_addons.callbacks import TQDMProgressBar
import ipywidgets as widgets

from . import ui


def selectTrainTestSet(rhr, info, train_days=100, test_days=14):
    rhr = rhr.sort_index()
    if info['covid_test_date']:
        dates_s = rhr.loc[(rhr.index < info['covid_test_date'] - pd.to_timedelta(f'{test_days}d'))].index.floor('1D').unique()
        dates_e = list(rhr.loc[(rhr.index > info['covid_test_date'] + pd.to_timedelta(f'{test_days}d'))].index.floor('1D').unique())
    else:
        dates_s = rhr.index.floor('1D').unique()
        dates_e = []

    dates_s = dates_s.insert(0, rhr.index[0])
    dates_e.append(rhr.index[-1])

    s = min(train_days, len(dates_s)-1)
    if(s < train_days and len(dates_e) > train_days):
        s = 0
    e = -min(max(0, train_days-s)+1, len(dates_e)-1)
    print(f's={s} e={e} train_days={train_days} test_days={test_days}')
    train = rhr.loc[(rhr.index < dates_s[s]) | (rhr.index > dates_e[e])]
    test = rhr.loc[(rhr.index >= dates_s[s]) & (rhr.index <= dates_e[e])]

    ax = plt.gca()
    train.plot(ax=ax, color='red')
    test.plot(ax=ax)
    plt.show()

    return train, test


def selectTrainTestSetAll(rhr, info, params):
    rhr = rhr.sort_index()
    min_train_days = params['min_train_days']
    max_train_days = params['max_train_days']
    test_days = params['test_days']+params['seg'].days
    future_data_if_not_enough_data = params['future_data_if_not_enough_data']

    dates = rhr.index.floor('1D').unique()
    total = len(dates)-test_days

    def generator():
        for i in range(test_days, len(dates)):
            if i < min_train_days + test_days:
                if future_data_if_not_enough_data <= 0:
                    yield None, None
                    continue

                s = dates[min(len(dates)-1, i+future_data_if_not_enough_data)]
                e = dates[min(len(dates)-1, i+future_data_if_not_enough_data+min_train_days)]
                train = rhr.loc[s:e]

            else:
                e = i-test_days
                s = max(0, e-max_train_days)

                train = rhr.loc[dates[s]:dates[e]]

            test = rhr.loc[dates[i-test_days]:dates[i]]
            if test is None or len(test) == 0:
                yield None, None
                continue
            if train is None or len(train) == 0:
                yield None, None
                continue
            # printInfo(train,test,info)
            yield train, test
    # raise StopIteration
    return total, generator()


def printInfo(train, test, info):
    train_days = train.index.floor('1D').unique()
    test_days = test.index.floor('1D').unique()
    covid_test_date = info['covid_test_date']

    print(f'covid={covid_test_date.date()}\t train: {train_days[0].date()}   {train_days[-1].date()} count={len(train_days)}   test={test_days[0].date()}   {test_days[-1].date()} count={len(test_days)}')


def preprocessing(train, test):
    if os.path.isfile(f'ali/an_data/preprocess.pkl.lz41'):
        scaler = compress_pickle.load(f'ali/an_data/preprocess.pkl.lz4')
    else:
        scaler = StandardScaler()
        scaler = scaler.fit(train[['heartrate']])
        compress_pickle.dump(scaler, f'ali/an_data/preprocess.pkl.lz4')

    train = pd.DataFrame(scaler.transform(train), index=train.index, columns=train.columns)
    if not (test is None):
        test = pd.DataFrame(scaler.transform(test), index=test.index, columns=test.columns)
    return train, test


def create_sequences_bad(data, resolution, seg):
    dates = data.loc[data.index.hour < 7].dropna().resample('1D').count()
    min_acceptable_count = dates.loc[dates['heartrate'] > 0].mean().values[0]/4
    data['date'] = data.index.floor('1D')
    data['time'] = data.index.time
    day_time = data.set_index(['time', 'date']).unstack(level=0).droplevel(0, axis=1).interpolate(limit_direction='both', axis=1)
    out = []
    times = []
    for w in dates.rolling(window=f'{seg}D', closed='right'):
        if len(w) == seg:  # and w.sum().values[0]>min_acceptable_count*seg:
            res = day_time.reindex(w.index).interpolate(limit_direction='both', axis=0).values
            out.append(np.reshape(res, (res.shape[0]*res.shape[1], 1)))
            times.append(w.index[-1])
#             out.append(res)

    return np.stack(times), np.stack(out)


def create_sequences_new(data, params, min_data_count=200):
    seqs = []
    seg = params['seg'].days
    resolution = params['resolution']

    data.loc[data.index[0].floor('1D'), 'heartrate'] = None
    data.loc[data.index[-1].floor('1D')+pd.to_timedelta('23:59:59'), 'heartrate'] = None

    allp2 = data.resample(resolution, origin='start_day',).mean()
    if 'o' in params['flags']:
        allp2 = allp2.loc[allp2.index.hour < 7]

    allp2['date'] = allp2.index.floor('1D')
    allp2['time'] = allp2.index.time
    allp2['id'] = 1
    day_time = allp2.set_index(['id', 'time', 'date']).astype(np.float64).unstack(level=1).droplevel(0, axis=1)
    day_time_info = day_time.notnull().sum(axis=1)
    day_time_info.hist()
    # print(day_time)
    min_data_count = min(min_data_count, day_time_info.mean()/4)
    day_time2 = day_time.loc[day_time.notnull().sum(axis=1) > min_data_count]
    ids = day_time2.index.get_level_values(0).unique()
    # print(day_time2)
    day_time2 = day_time2.interpolate(limit_direction='both', axis=1)  # .astype(np.float16)
    # display(day_time2.isnull().sum().sum())
    # n = 0
    # shape = None
    # for id in ids:
    #     for w in day_time2.loc[id].rolling(window=f'{seg}D', closed='right'):
    #         if len(w) == seg:
    #             n += 1
    #             shape = w.values.shape

    out = []  # np.zeros((n, shape[0]*shape[1], 1), np.float16)
    times = []
    i = 0
    for id in ids:
        for w in day_time2.loc[id].rolling(window=f'{seg}D', closed='right'):
            if w.shape[0] == seg:
                out.append(np.reshape(w.values, (w.shape[0]*w.shape[1], 1)))
                # out[i, :, :] = (np.reshape(w.values, (shape[0]*shape[1], 1)))
                times.append(w.index[-1])
#             print(w.index)
                # i += 1
    # print(f'i={len(out)} ids={len(ids)}')
    if len(out) == 0:
        tmp = day_time.loc[day_time.notnull().sum(axis=1) > min_data_count]
        # print(f'seg={seg} day_time_info={day_time_info} day_time_info2={tmp.notnull().sum(axis=1)}')
        return None, None
    return np.stack(times), np.stack(out)


def periodic_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period
    n_knots = n_splines + 1  # periodic and include_bias is True
    spline = SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )
    spline.fit(np.linspace(0, period, n_knots).reshape(n_knots, 1))
    return spline


time_spline = periodic_spline_transformer(24*60*60, 12)
day_spline = periodic_spline_transformer(7, 7)


def create_time_sequences(data, params, ):
    seqs = []
    seg = params['seg']
    overlap = params['overlap']
    resolution = params['resolution']
    seg_data_count = int(seg/resolution)
    seg_overlap_count = int(overlap/resolution)
    min_data_count = seg_data_count/10+1
    # min_data_count = 5

    data.loc[data.index[0].floor('1D'), 'heartrate'] = None
    data.loc[data.index[-1].floor('1D')+pd.to_timedelta('23:59:59'), 'heartrate'] = None
    allp2 = data.resample(resolution,).mean()
    allp2['hr_inter'] = allp2['heartrate'].interpolate(limit_direction='both')
    allp2['hr_mdeian'] = allp2['heartrate'].interpolate(limit_direction='both')
    if 'n' in params['flags']:
        rhr24median = data.resample('1D').mean().expanding().median().astype(int)
    if 'b' in params['flags']:
        rhr24avg = data.resample('1D').mean().expanding().mean().astype(int)
    out = []
    times = []

    # print(seg_overlap_count)
    wstarts = np.arange(seg_data_count, allp2.shape[0])[::seg_overlap_count]
    # print(wstarts)
    for s in wstarts:
        #     if(s>2000):continue
        w = allp2.iloc[s-seg_data_count:s]
        if w['heartrate'].count()<min_data_count:continue
        vals = w['hr_inter'].values

        if 'n' in params['flags']:
            m=rhr24median.loc[w.index[-1].floor('1D')]
            vals = np.concatenate([vals, m])
        if 'm' in params['flags']:
            m=w.median()
            vals = np.concatenate([vals, m])
        if 'a' in params['flags']:
            m=w.mean()
            vals = np.concatenate([vals, m])
        if 'b' in params['flags']:
            m=rhr24avg.loc[w.index[-1].floor('1D')]
            vals = np.concatenate([vals, m])
        if 't' in params['flags']:
            ts = time_spline.transform([[w.index[-1].hour*60*60+w.index[-1].minute*60+w.index[-1].second]])
            vals = np.concatenate([vals, ts[0]])
        if 'd' in params['flags']:
            ts = day_spline.transform([[w.index[-1].day]])
            vals = np.concatenate([vals, ts[0]])

        if params['model']=='auto-encoder':# and len(vals)%4!=0:
            vals=np.concatenate([vals, np.zeros((4-len(vals))%4)])
        # out.append(vals)
        out.append(vals.reshape(-1, 1))
        times.append(w.index[-1])

    if len(out) == 0:
        # tmp=day_time.loc[day_time.notnull().sum(axis=1) > min_data_count]
        # print(f'seg={seg} day_time_info={day_time_info} day_time_info2={tmp.notnull().sum(axis=1)}')
        return None, None
    return np.stack(times), np.stack(out)


def create_sequences_unk(data, resolution, seg):
    time_steps = int(pd.to_timedelta('7h')/pd.to_timedelta(resolution)*seg)
    data = data.resample(resolution).mean().dropna()
#     data=data.loc[data.index.hour<7]
    values = data.values
    output = []
    times = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i: (i + time_steps)])
        times.append(data.index[(i + time_steps)-1])
    return np.stack(times), np.stack(output)


def create_sequences(data, params):
    #     return create_sequences_bad(data,resolution,seg)
    if 'f' in params['flags']:
        return create_time_sequences(data, params)
    return create_sequences_new(data, params, min_data_count=200)


#     return create_sequences_unk(data,resolution,seg)
# x_train = create_sequences(train)


def createAutoEncoderModel(shape, size=16):
    # print(shape)
    model = keras.Sequential(
        [
            layers.Input(shape=(shape[1], shape[2])),
            layers.Conv1D(
                filters=size*2, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=size, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=size, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=size*2, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    # model.summary()
    return model


def createLSTMModel(shape):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(
        units=16,
        stateful=True,
        input_shape=(shape[1], shape[2]),
        batch_input_shape=(1, shape[1], shape[2])
    ))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=shape[1]))
    model.add(keras.layers.LSTM(units=16, return_sequences=True, stateful=True))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Dense(units=shape[2])
        )
    )
    model.compile(loss='mae', optimizer='adam')
    return model


def anomaly_detection(rhr, info, params):
    # model='lstm'
    
    from tqdm.notebook import tqdm
    # params['seg_day']=params['seg'].day

    # maxIter = len(rhr.index.floor('1D').unique())
    rhr1 = rhr.resample(params['resolution']).mean().dropna()
    model = None
    alarms = pd.DataFrame(columns=['alarm'])
    anomaly_score = pd.DataFrame(columns=['date','error'])
    print(' ', end='', flush=True)

    _, trainp = preprocessing(rhr1[:rhr1.index[0]+pd.to_timedelta(params['min_train_days'])], rhr1)
    x_times, x_all = create_sequences(trainp, params)
    if x_times is None:
        return alarms
    min_train_c=max(params['min_train_days'],len(x_times[x_times<x_times[0]+pd.to_timedelta(params['min_train_days'],unit='d')]))
    max_train_c=len(x_times[x_times<x_times[0]+pd.to_timedelta(params['max_train_days'],unit='d')])
    test_c = max(1,min_train_c//params['min_train_days'])
    onlinePloter=ui.onlinePloter(args=params)
    onlinePloter.plot(info, rhr1, None, alarms, None)
    # fig.show()
    tbar_items = tqdm(range(min_train_c, len(x_times)-test_c), desc=f"{info['id']}-{params['current_method']}", leave=False)
    debug_view = ui.debugView(params['debug'])
    debug_view.print('ali')
    # print('a')
    # threshold
    # with  tqdm(total=100, desc=f'{info["id"]}',leave=False) as pbar:
    for i in tbar_items:
        # print(x_all[i-10:i, :])
        train_c = min(i, max_train_c)
        eval_c = min(i, max_train_c)
        train_times, x_train = x_times[i-train_c:i], x_all[i-train_c:i, :].reshape(train_c, -1, 1)
        test_times, x_test = x_times[i:i+test_c], x_all[i:i+test_c, :].reshape(test_c, -1, 1)
        # print(x_train.shape,x_test.shape)
        train_eval_x = x_all[i-eval_c:i, :].reshape(eval_c, -1, 1)

        # if x_train is None or not (x_train.shape[0] > 10) or x_test is None or not (x_test.shape[0] > 0):
        #     # print('not enough data')
        #     # out = rhr1.resample('1D').mean()
        #     # out['alarm'] = 0
        #     # return out[['alarm']]
        #     continue

        # print("Training input shape: ", x_train.shape)
        model_path=f'ali/an_data/model-{params["model"]}-{params["current_method"]}.tf'
        if model is None:
            if 'l' in params['flags'] and os.path.isfile(model_path):
                model = tf.keras.models.load_model(model_path)
            else:
                if params["model"] == 'auto-encoder':
                    model = createAutoEncoderModel(x_train.shape)
                elif params["model"] == 'lstm':
                    model = createLSTMModel(x_train.shape)
        tbar = TQDMProgressBar(
            update_per_second=1,
            leave_epoch_progress=False,
            leave_overall_progress=False,
            show_epoch_progress=False
        )
        
        history = model.fit(
            x_train,
            x_train,
            epochs=3,
            batch_size=128,
            # validation_split=0.05,
            validation_split=0,
            # validation_data=(x_train[-1,:], x_train[-1,:]),
            verbose=0,
            shuffle=False,
            use_multiprocessing=True,
            workers=8,
            callbacks=[
                # keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
                # TQDMNotebookCallback( leave_inner=False, leave_outer=False)
                # tbar
                # TqdmCallback(verbose=1)

            ],
        )
        

        # plt.figure()
        # plt.plot(history.history["loss"], label="Training Loss")
        # plt.plot(history.history["val_loss"], label="Validation Loss")
        # plt.legend()
        # plt.show()

        # Get train MAE loss.
        x_train_pred = model.predict(x_train)
        train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

        # plt.figure()
        # plt.hist(train_mae_loss, bins=50)
        # plt.xlabel("Train MAE loss")
        # plt.ylabel("No of samples")
        # plt.show()

        # Get reconstruction loss threshold.
        threshold = np.max(train_mae_loss)
        

        # print("Reconstruction error threshold: ", threshold)

        # fig, ax = plt.subplots()
        # df_test_value.plot(legend=False, ax=ax)
        # plt.show()

        # Create sequences from test values.

        # print("Test input shape: ", x_test.shape)

        # Get test MAE loss.
        x_test_pred = model.predict(x_test)
        test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
        test_mae_loss = test_mae_loss.reshape((-1))
        
        # plt.figure()
        # plt.hist(test_mae_loss, bins=50)
        # plt.xlabel("test MAE loss")
        # plt.ylabel("No of samples")
        # plt.show()

        # Detect all the samples which are anomalies.
        anomalies = test_mae_loss > threshold
        # if np.sum(anomalies)>0:
        # print("Number of anomaly samples: ", np.sum(anomalies))
        # print("Indices of anomaly samples: ", np.where(anomalies))
        # print("anomaly samples: ", test_times[np.where(anomalies)].resample('1D').count())
        an_times = test_times[np.where(anomalies)]

        an_times = pd.DataFrame(index=an_times)

        an_times['anomaly'] = 1
        
        # if len(an_times) > 0:
        #     print("anomaly samples: ", an_times.resample('1D').count())
        if an_times.shape[0] == 0:
            dates = an_times
        else:
            dates = an_times.resample('1D').count()
        # display(dates)
        dates['alarm'] = (dates['anomaly'] > 0)*2
        for k, row in dates.iterrows():
            if k in alarms.index:
                alarms.loc[k, 'alarm'] += row['alarm']
            else:
                alarms.loc[k, 'alarm'] = row['alarm']
        # with debug_view:
            # debug_view.clear_output()
            # display(alarms)
        if alarms.shape[0] > 0:
            a2 = alarms.copy().rename(columns={'alarm': info['id']})
            a2.index = a2.index.strftime('%Y %m %d')
            debug_view.set_html(a2.iloc[::-1].T.to_html())
        
        error = (test_mae_loss - threshold) / (threshold)  # Normalize error
        for i in range(len(error)):
            anomaly_score=anomaly_score.append({'date':test_times[i],'error':error[i]},ignore_index=True)

        if dates.shape[0] > 0:
            anomay_score_avg = anomaly_score.set_index('date').resample('1D').max()
            onlinePloter.plot(info, rhr1, None, alarms, anomay_score_avg)
        # if dates.shape[0]>0:
        #     print(alarms.iloc[-5:])
    # with debug_view:
    #     debug_view.clear_output()
    #     display(alarms)
    # if debug_view:
    #     debug_view.value = ''
    if anomaly_score.shape[0]>0:
        anomay_score_avg=anomaly_score.set_index('date').resample('1D').max()
    else:
        anomay_score_avg = None
    onlinePloter.plot(info, rhr1, None, alarms, anomay_score_avg)
    onlinePloter.close()
    if 'l' in params['flags'] and model is not None:
        model.save(model_path)
    return alarms



def anomaly_detection_allpoints(rhr, info, params):
    # model='lstm'
    from tqdm.notebook import tqdm
    # params['seg_day']=params['seg'].day

    # maxIter = len(rhr.index.floor('1D').unique())
    rhr1 = rhr.resample(params['resolution']).mean().dropna()
    model = None
    total, alltt = selectTrainTestSetAll(rhr1, info, params)
    alarms = pd.DataFrame(columns=['alarm'])
    print(' ', end='', flush=True)

    # with  tqdm(total=100, desc=f'{info["id"]}',leave=False) as pbar:
    for train, test in tqdm(alltt, total=total, desc=f'{info["id"]}', leave=False):
        # pbar.n=round(pr*100);pbar.update()
        if train is None or test is None:
            continue
        info['train'] = train
        info['test'] = test
        # print('train')
        # display(train)
        # print('test')
        # display(test)
        trainp, testp = preprocessing(train, test)

        train_times, x_train = create_sequences(trainp, params)
        test_times, x_test = create_sequences(testp, params)

        if x_train is None or not (x_train.shape[0] > 10) or x_test is None or not (x_test.shape[0] > 0):
            # print('not enough data')
            # out = rhr1.resample('1D').mean()
            # out['alarm'] = 0
            # return out[['alarm']]
            continue

        # print("Training input shape: ", x_train.shape)
        if model is None or 1:
            if os.path.isfile(f'ali/an_data/model-{params["model"]}-{params["seg"]}.h51'):
                model = tf.keras.models.load_model(f'ali/an_data/model-{params["model"]}-{params["seg"]}.h5')
            else:
                if params["model"] == 'auto-encoder':
                    model = createAutoEncoderModel(x_train.shape)
                elif params["model"] == 'lstm':
                    model = createLSTMModel(x_train.shape)
        tbar = TQDMProgressBar(
            update_per_second=1,
            leave_epoch_progress=False,
            leave_overall_progress=False,
            show_epoch_progress=False
        )
        # model.summary()
        history = model.fit(
            x_train,
            x_train,
            epochs=50,
            batch_size=128,
            validation_split=0.1,
            verbose=0,

            callbacks=[
                keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
                # TQDMNotebookCallback( leave_inner=False, leave_outer=False)
                # tbar
            ],
        )

        # model.save(f'ali/an_data/model-{params["model"]}-{params["seg"]}.h5')

        # plt.figure()
        # plt.plot(history.history["loss"], label="Training Loss")
        # plt.plot(history.history["val_loss"], label="Validation Loss")
        # plt.legend()
        # plt.show()

        # Get train MAE loss.
        x_train_pred = model.predict(x_train)
        train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

        # plt.figure()
        # plt.hist(train_mae_loss, bins=50)
        # plt.xlabel("Train MAE loss")
        # plt.ylabel("No of samples")
        # plt.show()

        # Get reconstruction loss threshold.
        threshold = np.max(train_mae_loss)
        # print("Reconstruction error threshold: ", threshold)

        # fig, ax = plt.subplots()
        # df_test_value.plot(legend=False, ax=ax)
        # plt.show()

        # Create sequences from test values.

        # print("Test input shape: ", x_test.shape)

        # Get test MAE loss.
        x_test_pred = model.predict(x_test)
        test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
        test_mae_loss = test_mae_loss.reshape((-1))

        # plt.figure()
        # plt.hist(test_mae_loss, bins=50)
        # plt.xlabel("test MAE loss")
        # plt.ylabel("No of samples")
        # plt.show()

        # Detect all the samples which are anomalies.
        anomalies = test_mae_loss > threshold
        # if np.sum(anomalies)>0:
        # print("Number of anomaly samples: ", np.sum(anomalies))
        # print("Indices of anomaly samples: ", np.where(anomalies))
        # print("anomaly samples: ", test_times[np.where(anomalies)].resample('1D').count())
        an_times = test_times[np.where(anomalies)]

        an_times = pd.DataFrame(index=an_times)

        an_times['anomaly'] = 1
        if len(an_times) > 0:
            print("anomaly samples: ", an_times.resample('1D').count())
        if an_times.shape[0] == 0:
            dates = an_times
        else:
            dates = an_times.resample('1D').count()
        # display(dates)
        dates['alarm'] = (dates['anomaly'] > 0)*2
        for k, row in dates.iterrows():
            if k in alarms.index:
                alarms.loc[k, 'alarm'] += row['alarm']
            else:
                alarms.loc[k, 'alarm'] = row['alarm']
        # return dates
        # ui.plot(rhr1, alerts=dates, info=info, show=True)
        # print(f'anomaly dates={sum(dates["alarm"]==2)}/{len(np.unique([p.date() for p in test_times]))}')
        # print(f'from={test_times[0]} to={test_times[-1]}')
    # print(alarms)

    return alarms
    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    # if 1:
    #     anomalous_data_indices = np.where(anomalies)
    # else:
    #     anomalous_data_indices = []

    #     for data_idx in range(TIME_STEPS - 1, len(test) - TIME_STEPS + 1):
    #         if np.all(anomalies[data_idx - TIME_STEPS + 1: data_idx]):
    #             anomalous_data_indices.append(data_idx)

    # # df_subset = test.iloc[anomalous_data_indices]

    # dates = test.iloc[anomalous_data_indices].resample('1D').count().rename(columns={'heartrate': 'count'})

    # dates['alarm'] = (dates['count'] > 0)*2
    # return dates
    # ali.ui.plot(rhr,alerts=dates,info=info,show=True)


def anomaly_detection_single(rhr, info, model='auto-encoder', resolution='1T', seg=3):

    rhr1 = rhr.resample(resolution).mean().dropna()
    train, test = selectTrainTestSet(rhr1, info, train_days=50)
    info['train'] = train
    info['test'] = test
    # print('train')
    # display(train)
    # print('test')
    # display(test)
    trainp, testp = preprocessing(train, test)

    train_times, x_train = create_sequences(trainp, resolution=resolution, seg=seg)
    if x_train is None:
        print('not enough data')
        out = rhr1.resample('1D').mean()
        out['alarm'] = 0
        return out[['alarm']]

    # print("Training input shape: ", x_train.shape)

    if os.path.isfile(f'ali/an_data/model-{model}-{seg}.h51'):
        model = tf.keras.models.load_model(f'ali/an_data/model-{model}-{seg}.h5')
    else:
        if model == 'auto-encoder':
            model = createAutoEncoderModel(x_train.shape)
        elif model == 'lstm':
            model = createLSTMModel(x_train.shape)

    # model.summary()
    history = model.fit(
        x_train,
        x_train,
        epochs=50,
        batch_size=128,
        # validation_split=0.1,
        validation_data=([x_train[-1,:]], [x_train[-1,:]]),
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            TqdmCallback(verbose=1)
        ],
    )
    model.save(f'ali/an_data/model-{model}-{seg}.h5')

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
    threshold = np.max(train_mae_loss)
    # print("Reconstruction error threshold: ", threshold)

    # fig, ax = plt.subplots()
    # df_test_value.plot(legend=False, ax=ax)
    # plt.show()

    # Create sequences from test values.
    test_times, x_test = create_sequences(testp, seg=seg, resolution=resolution)
    # print("Test input shape: ", x_test.shape)

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
    an_times = test_times[np.where(anomalies)]
    # print("anomaly samples: ", an_times)
    an_times = pd.DataFrame(index=an_times)
    an_times['anomaly'] = 1

    if an_times.shape[0] == 0:
        dates = an_times
    else:
        dates = an_times.resample('1D').count()
    # display(dates)
    dates['alarm'] = (dates['anomaly'] > 0)*2

    # return dates
    # ali.ui.plot(rhr, alerts=dates, info=info, show=True)
    # print(f'anomaly dates={sum(dates["alarm"]==2)}/{len(np.unique([p.date() for p in test_times]))}')
    # print(f'from={test_times[0]} to={test_times[-1]}')
    return dates[['alarm']]
    # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
    # if 1:
    #     anomalous_data_indices = np.where(anomalies)
    # else:
    #     anomalous_data_indices = []

    #     for data_idx in range(TIME_STEPS - 1, len(test) - TIME_STEPS + 1):
    #         if np.all(anomalies[data_idx - TIME_STEPS + 1: data_idx]):
    #             anomalous_data_indices.append(data_idx)

    # # df_subset = test.iloc[anomalous_data_indices]

    # dates = test.iloc[anomalous_data_indices].resample('1D').count().rename(columns={'heartrate': 'count'})

    # dates['alarm'] = (dates['count'] > 0)*2
    # return dates
    # ali.ui.plot(rhr,alerts=dates,info=info,show=True)
