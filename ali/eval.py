
from matplotlib.ticker import FixedLocator
import pandas as pd
import pandasql as ps
import numpy as np
threshold = 3
# indection_detation_window = pd.to_timedelta('21D')
indection_detation_window = pd.to_timedelta('21D')


def eval_both(rhr, alerts, info):
    return {
        # 'my' : eval_my(rhr, alerts, info),
    'cov_score' : eval_cov_score(rhr, alerts, info),
    'nature' : eval_nature(rhr, alerts,info),
    'new' : eval_new(rhr, alerts,info),
    'total': total_info(rhr,alerts,info)
    }
    
def total_info(rhr,alerts,info):
    day_info=rhr.resample('1d').min()['heartrate']
    hour_info=rhr.resample('1h').min()['heartrate']
    Ref = info['covid_test_date']
    if info['symptom_date']:
        Ref = info['symptom_date']
    
    res = {
        'persons': 1,
        'covid': (info['covid_test_date'] != None)*1,
        'sympthom': (info['symptom_date'] != None)*1,
        'days': day_info.dropna().count(),
        'hours': hour_info.dropna().count(),
        # 'missD':day_info.isna().count(),
        # 'missH':hour_info.isna().count(),
    }
    for i in range(22):
        res[i]=0
    res['nd']=0
    if Ref!=None:
        valid = alerts.loc[(alerts.index <= Ref)&(alerts['alarm'] > 1)]
        d='nd'
        if valid.shape[0]>0:
            d=(Ref-valid.index[-1]).days
            if d > 21:
                d = 'nd'

        res[d]=1

    return res

    

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
    # print(tp)
    
    # tp['datetime'] = pd.to_datetime(tp['datetime'])
    # fp['datetime'] = pd.to_datetime(fp['datetime'])
    # tp = tp.set_index('datetime')
    # fp = fp.set_index('datetime')
    # print(f'window={pwindow}')
    # print(f'tp={tp}')
    # print(f'fp={fp}')
    # print(f'false_alerts={false_alerts}')
    # print(f'fsum={false_alerts.sum()}')
    res = {}
    res['tp'] = 1 if sum(tp['alarm'] >= 2) > 0 else 0
    res['fn'] = 1 if (pwindow != None) and (sum(tp['alarm'] >= 2) == 0) else 0
    res['fp'] = 1 if sum(fp['alarm'] >= 2) > 0 else 0
    res['tn'] = 1 if sum(fp['alarm'] >= 2) == 0 else 0

    return res


def eval_new(rhr, alerts, info):
    hasData = rhr.resample('1d').count()
    hasData = hasData.loc[hasData['heartrate'] > 0]
    
    alerts=hasData.join(alerts,how='outer').fillna(-4)
    covid_test_date = info['covid_test_date']
    if info['symptom_date']:
        covid_test_date = info['symptom_date']
    start = rhr.index[0]
    
    if covid_test_date != None:
        start = covid_test_date-pd.Timedelta(days=(info['covid_test_date']-start).days//7*7)
    
    alerts['alarm'] = (alerts['alarm']>=2)*1
    
    week_alerts = (alerts.resample('7d', origin=start).sum()).fillna(-4)
    week_alerts.index+=pd.to_timedelta('3d')
    tp=0
    fn=0
    fp=0
    tn=0
    for i,row in week_alerts.iterrows():
        if covid_test_date != None and abs((i-covid_test_date).days) <= 7 and i<=covid_test_date:
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


def eval_cov_score(rhr, alerts, info):
    rhr24=rhr.resample('1d').mean().dropna()
    m = -14
    avg = (m)/2
    r = 21
    t1 = .5 
    t2 = .5
    W_tp=1
    W_fp= 1/m
    W_tn= 1/(m*(m+1))
    W_fn=-1
    import math
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def score(x):
        # print(x)        
        if x > r:
            return -1
        if x > 0:
            return 0
        if(x >= avg):
            #           return 2*sigmoid(t1*(-x+1))-1
            return sigmoid(t1*(-x))
        if(x < avg):
            return 1-2*sigmoid(t2*(-x+m))

    symptom_date = info['symptom_date']
    covid_test_date = info['covid_test_date']
    ref=None
    if(symptom_date != None):
        ref = symptom_date
    elif covid_test_date != None:
        ref = covid_test_date

    red=alerts.loc[alerts['alarm']>=2].copy()
    if ref is None:
        fp=red['alarm'].count()
        # total_negative=rhr24.count()-fp
        fn=0
        tp=0
        tn=rhr24.shape[0]
        total_negative = rhr24.shape[0]
        
    else:
        
        # print((red.index-ref).days)
        red['d']=(red.index-ref).days
        rhr24['d']=(rhr24.index-ref).days
        fp_period = red.loc[(red['d']<m) | (red['d']>r)]
        tp_period = red.loc[(red['d'] >= m) & (red['d'] <= r)]

        if tp_period.shape[0]==0:
            tp=0
            fn=1
            
        else:
            tp=max([score(x) for x in tp_period['d']])
            fn=0
        fp=sum([-score(x) for x in fp_period['d']])
        total_negative = rhr24.loc[(rhr24['d'] < m) | (rhr24['d'] > r)].shape[0]
        tn = total_negative - fp_period.shape[0]
        
    calcScore = lambda tp, fp, fn,tn: W_tp*tp+W_fp*fp+W_fn*fn+W_tn*tn
    s = calcScore(tp,fp,fn,tn)
    # s_best = calcScore(1,0,0)
    # s_null = calcScore(0, 0, 1)
    # norm_score=(s-s_null)/(s_best-s_null)
    # W_fp=m/total_negative
    # print(f' s={s:.3f} tp={tp:.3f} fp={fp:.3f} fn={fn:.3f}, W_tp={W_tp:.3f} W_fp={W_fp:.3f} W_fn={W_fn:.3f} s_best={s_best:.3f} s_null={s_null:.3f} ')
    res = {}
    res['tp'] = tp
    res['fn'] = fn
    res['fp'] = fp
    res['tn'] = tn
    res['score'] = s
    
    res['s_best'] = calcScore(1 if ref else 0,0,0,total_negative)
    res['s_null'] = calcScore(0, 0, 1 if ref else 0, total_negative)

    
    # print(res)
    return res
    
    



def eval_nature(rhr, alerts, info):
    
    hasData = rhr.resample('1d').count()
    hasData = hasData.index[hasData['heartrate'] > 0]


    symptom_date = info['symptom_date']
    covid_test_date = info['covid_test_date']
    pwindow = None
    # if symptom_date != None and covid_test_date != None:
    #     if (covid_test_date-symptom_date).days>7:
    #         pwindow=[covid_test_date-indection_detation_window, covid_test_date]
    #     else:
    #         pwindow = [symptom_date-indection_detation_window, symptom_date]
    # el
    if(symptom_date != None):
        pwindow = [symptom_date-indection_detation_window, symptom_date]
    elif covid_test_date != None:
        pwindow = [covid_test_date-indection_detation_window, covid_test_date]
        


    if pwindow == None:
        tp = alerts.loc[alerts.index == None]  # empty df
        fp = alerts  # .loc[alerts['alarm'] > 0]
    else:
        # pwindow[0] += pd.to_timedelta('7d')
        # pwindow[1] += pd.to_timedelta('7d')
        tp = alerts.loc[(alerts.index >= pwindow[0]) & (alerts.index <= pwindow[1])]
        fp = alerts.loc[(alerts.index < pwindow[0]) | (alerts.index > pwindow[1]+indection_detation_window)]
        # fp = alerts.loc[(alerts.index < pwindow[0])]
        
    # tp = tp.loc[tp.index.isin(hasData)]
    # fp = fp.loc[fp.index.isin(hasData)]
    # print(f'pwindow={pwindow} , nwindow={nwindow}')
    # print(f'alerts={alerts}')
    # print(f'tp={tp}')
    # print(f'fp={fp}')
    # print(f'hasdata={hasData}')

    res = {}
    res['tp'] = 1 if sum(tp['alarm'] >= 2) > 0 else 0
    res['fn'] = 1 if (pwindow != None) and sum(tp['alarm'] >= 2) == 0 else 0
    res['fp'] = sum(fp['alarm'] >= 2)
    if pwindow:
        tn = sum(hasData < pwindow[0]) + sum(hasData > (pwindow[1] + indection_detation_window))
        # tn = sum(hasData < pwindow[0])
    else:
        tn = len(hasData)
    res['tn'] = max(0, tn-sum(fp['alarm'] >= 1))

    return res
