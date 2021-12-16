
import pandas as pd
import pandasql as ps
threshold = 3
indection_detation_window = pd.to_timedelta('21D')


def eval_both(rhr, alerts, info):
    return {
        # 'my' : eval_my(rhr, alerts, info),
    'nature' : eval_nature(rhr, alerts,info),
    'new' : eval_new(rhr, alerts,info)}
    


def eval_my(rhr, alerts, info):
    hasData = rhr.resample('1d').count()
    hasData = hasData.index[hasData['heartrate'] > 0]
    
    pwindow = None
    symptom_date=info['symptom_date']
    covid_test_date = info['covid_test_date']
    if(symptom_date != None):
        pwindow = [symptom_date-indection_detation_window, symptom_date]
    elif covid_test_date != None:
        pwindow = [covid_test_date-indection_detation_window, covid_test_date]

    if pwindow == None:
        tp = alerts.loc[alerts.index == None]  # empty df
        fp = alerts  # .loc[alerts['alarm']>0]
    else:
        tp = alerts.loc[(alerts.index >= pwindow[0]) & (alerts.index <= pwindow[1])]
        fp = alerts.loc[(alerts.index < pwindow[0]) | (alerts.index > pwindow[1]+indection_detation_window)]
    tp = tp.loc[tp.index.isin(hasData)]
    fp = fp.loc[fp.index.isin(hasData)]
    # tp['datetime'] = pd.to_datetime(tp['datetime'])
    # fp['datetime'] = pd.to_datetime(fp['datetime'])
    # tp = tp.set_index('datetime')
    # fp = fp.set_index('datetime')
    # print(f'window={window}')
    # print(f'true_alerts={true_alerts}')
    # print(f'false_alerts={false_alerts}')
    # print(f'fsum={false_alerts.sum()}')
    res = {}
    res['tp'] = 1 if sum(tp['alarm'] == 2) > 0 else 0
    res['fn'] = 1 if (pwindow != None) and (sum(tp['alarm'] == 2) == 0) else 0
    res['fp'] = 1 if sum(fp['alarm'] == 2) > 0 else 0
    res['tn'] = 1 if sum(fp['alarm'] == 2) == 0 else 0

    return res


def eval_new(rhr, alerts, info):
    hasData = rhr.resample('1d').count()
    hasData = hasData.loc[hasData['heartrate'] > 0]
    
    alerts=hasData.join(alerts,how='outer').fillna(-4)
    covid_test_date = info['covid_test_date']
    start = rhr.index[0]
    
    if covid_test_date != None:
        start = covid_test_date-pd.Timedelta(days=(info['covid_test_date']-start).days//7*7)
    
    alerts['alarm'] = (alerts['alarm']==2)*1
    
    week_alerts = (alerts.resample('7d', origin=start).sum()).fillna(-4)
    week_alerts.index+=pd.to_timedelta('3d')
    tp=0
    fn=0
    fp=0
    tn=0
    for i,row in week_alerts.iterrows():
        if covid_test_date != None and abs((i-covid_test_date).days) <=7:
            # print(f'i {i} \talarm={row["alarm"]}  \tdays={(i-covid_test_date).days}')
            if row['alarm']>0:tp+=1
            
        elif covid_test_date != None and abs((i-covid_test_date).days) <= 14:
            pass
        else:
            if row['alarm'] > 0:
                fp += 1
            elif row['alarm'] == 0:
                tn += 1
    
    res = {}
    res['tp'] = min(1,tp)
    res['fn'] = 0 if covid_test_date == None else 1- res['tp']
    res['fp'] = fp
    res['tn'] = tn

    return res



def eval_nature(rhr, alerts, info):
    hasData = rhr.resample('1d').count()
    hasData = hasData.index[hasData['heartrate'] > 0]


    symptom_date = info['symptom_date']
    covid_test_date = info['covid_test_date']
    pwindow = None
    if(symptom_date != None):
        pwindow = [symptom_date-indection_detation_window, symptom_date]
    elif covid_test_date != None:
        pwindow = [covid_test_date-indection_detation_window, covid_test_date]



    if pwindow == None:
        tp = alerts.loc[alerts.index == None]  # empty df
        fp = alerts  # .loc[alerts['alarm'] > 0]
    else:
        pwindow[0] += pd.to_timedelta('7d')
        pwindow[1] += pd.to_timedelta('7d')
        tp = alerts.loc[(alerts.index >= pwindow[0]) & (alerts.index <= pwindow[1])]
        fp = alerts.loc[(alerts.index < pwindow[0]) | (alerts.index > pwindow[1]+indection_detation_window)]

    tp = tp.loc[tp.index.isin(hasData)]
    fp = fp.loc[fp.index.isin(hasData)]
    # print(f'alerts={alerts}')
    # print(f'tp={tp}')
    # print(f'fp={fp}')
    # print(f'hasdata={hasData}')

    res = {}
    res['tp'] = 1 if sum(tp['alarm'] == 2) > 0 else 0
    res['fn'] = 1 if (pwindow != None) and sum(tp['alarm'] == 2) == 0 else 0
    res['fp'] = sum(fp['alarm'] == 2)
    if pwindow:
        tn = sum(hasData < pwindow[0]) + sum(hasData > (pwindow[1] + indection_detation_window))
    else:
        tn = len(hasData)
    res['tn'] = max(0, tn-res['fp'])

    return res
