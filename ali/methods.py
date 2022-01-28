import pandas as pd
import os
from IPython.display import display
import subprocess
from . import ui
from . import eval
from . import timeseries_anomaly_detection
import os,shutil
import traceback
import json
def read(device,hr_file,step_file):
  
    hr = pd.read_csv(hr_file)
    # display( hr.loc[pd.to_datetime(hr['datetime'], errors='coerce').isna()])
    hr['datetime'] = pd.to_datetime(hr['datetime'],errors='coerce')
    error = hr.loc[hr['datetime'].isna()]
    if len(error)>0:
        hr=hr.iloc[0:error.index[0]] #for resolving P678649 
        hr['heartrate'] = hr['heartrate'].astype(int)
    hr=hr.set_index('datetime')
    # if len(error) > 0:
    #     display(hr)
    try:
        step = pd.read_csv(step_file)
    except:
        raise Exception(f"Invalid Format ")
    if device=='Fitbit':
        step['datetime']=pd.to_datetime(step['datetime'])
        step=step.set_index('datetime')
    else: #Apple Watch
        if step.columns[0] == 'BigQuery error in query operation: Error processing job':
            raise Exception(f"Invalid Format {step}")
        if len(step['end_datetime'].unique()) < 2:
            device = 'Fitbit' #step['device'][0]

            step=step.drop(columns=['device']).rename(columns={'start_datetime':'datetime'})
            step['datetime']=pd.to_datetime(step['datetime'])
            step=step.set_index('datetime')
        else:
            error = step.loc[pd.to_datetime(step['start_datetime'], errors='coerce').isna()]
            
            if len(error) > 0:
                step = step.iloc[0:error.index[0]]  # for resolving P678649
                step['steps']=step['steps'].astype(int)
            step['start_datetime']=pd.to_datetime(step['start_datetime']).dt.floor('1T')
            step['end_datetime'] = pd.to_datetime(step['end_datetime']).dt.ceil('1T')
            step=step.loc[step['end_datetime'] > step['start_datetime']]
            step=step.loc[step['end_datetime'] - step['start_datetime']<pd.to_timedelta('10H')]
    return device,hr,step

def createRestingHR_new(device,hr,step):
    
    if device=='Fitbit':
        # Cacluating Resting HR
        step_n=step.resample('1T').mean()
    #     hr['datetime_m'] = hr.index.floor('1T')
    #     hr_step = hr.join(step, on='datetime_m', how='outer')
    else: #Apple Watch
        step_n=pd.DataFrame(columns=['steps'])
    #         print(step)
        all=[]
        for k,row in step.iterrows():
            newRange=pd.date_range(row['start_datetime'].floor('1T'),row['end_datetime'].floor('1T'),freq='1T')
            a=pd.DataFrame(index=newRange)
            a['steps']=row['steps']/len(newRange)
            all.append(a)

        step_n=pd.concat(all)

    hr = hr.resample('1T').mean()
    hr_step = hr.join(step_n, how='outer')

    rhr_hint=hr_step.rolling('12T',min_periods=0).count().loc[hr.index].dropna()
    rhr=hr.loc[rhr_hint[(rhr_hint['heartrate']>0) &(rhr_hint['steps']==0)].index].dropna()
    return rhr
    

def createRestingHR(device,hr,step):
    nighthr = hr[(hr.index.hour >= 0) & (hr.index.hour <= 6)].copy()
    
    if device=='Fitbit':
        # Cacluating Resting HR
        nighthr['datetime_m'] = nighthr.index.floor('T')
        hr_step = nighthr.join(step, on='datetime_m', how='left')
        rhr = hr_step[hr_step['steps'] != hr_step['steps']][['heartrate']]
    else: #Apple Watch
        import pandasql as ps
        sqlcode = '''
        select datetime,heartrate from nighthr where datetime not in (
            select nighthr.datetime
            from nighthr,step
            where nighthr.datetime >= step.start_datetime and nighthr.datetime <= step.end_datetime
            )
        '''
        rhr = ps.sqldf(sqlcode, locals())
        rhr['datetime'] = pd.to_datetime(rhr['datetime'])
        rhr = rhr.set_index('datetime')
    return rhr

def isolationforest(device,rhr):
    # rhr=pd.read_csv('RHR.csv')
    from sklearn.ensemble import IsolationForest
    df = rhr.resample("24H").mean().reset_index()
    df['heartrate'] = df['heartrate'].interpolate()
    model = IsolationForest(contamination="auto")
    model.fit(df[['heartrate']])
    df['alarm'] = pd.Series(model.predict(df[['heartrate']])).apply(lambda x: 1 if (x == -1) else 0)
    # df.to_csv('if_anomalies.csv', columns=['datetime', 'alarm'], index=False)
    
    
    allalarms = df.set_index('datetime')[['alarm']]
    allalarms['cum'] = allalarms.rolling(2).sum()
    allalarms['alarm']=allalarms['alarm']*allalarms['cum']
    # display(allalarms)
    return allalarms[['alarm']].fillna(0)


    
def isolationforest2(device,rhr):
    # rhr=pd.read_csv('RHR.csv')
    from sklearn.ensemble import IsolationForest
    df = rhr.resample("24H").mean().reset_index()
    df['median'] = df.expanding().median().astype(int)['heartrate']
    df['heartrate'] = df['heartrate'].dropna()
    model = IsolationForest(contamination="auto")
    model.fit(df[['heartrate']])
    df['alarm'] = pd.Series(model.predict(df[['heartrate']])).apply(lambda x: 1 if (x == -1) else 0)
    # df.to_csv('if_anomalies.csv', columns=['datetime', 'alarm'], index=False)
    df=df.set_index('datetime')
    # df.join(rhr24median.rename(columns={'heartrate':'median'}))
    
    allalarms = df[['alarm']][df['heartrate']>df['median']]
    allalarms['cum'] = allalarms.rolling(2).sum()
    allalarms['alarm']=allalarms['alarm']*allalarms['cum']
    # display(allalarms)
    return allalarms[['alarm']].fillna(0)

def nightsignal(device, rhr):
    rhr24 = rhr.resample('24H').mean()
    rhr24median = rhr24.expanding().median().astype(int)

    missings = rhr24.index[rhr24['heartrate'].isnull()]
    rhr24 = rhr24.interpolate(limit=1)
    
    # newJoin = rhr24.join(rhr24median.rename(columns={'heartrate': 'hr_median'})).sort_index()
    # red_and_yellow_alert_dates = rhr24[rhr24['heartrate'] >= rhr24median['heartrate']+3].index
    # display(rhr24['heartrate'])
    # display(rhr24median['heartrate'])
    red_alert_dates = rhr24.index[rhr24['heartrate'] >= rhr24median['heartrate']+4]
    yellow_alert_dates = rhr24.index[rhr24['heartrate'] >= rhr24median['heartrate']+3]
    # red_alert_dates = newJoin[newJoin['heartrate'] > newJoin['hr_median']+4].index
    # yellow_alert_dates = newJoin[(newJoin['heartrate'] > newJoin['hr_median']+3)].index

    allalarms = rhr24.copy()
    allalarms['red_alarm'] = 0
    allalarms['yellow_alarm'] = 0
    allalarms.loc[red_alert_dates, 'red_alarm'] = 1
    allalarms.loc[yellow_alert_dates, 'yellow_alarm'] = 1
    allalarms = allalarms.rolling(2).sum()
    allalarms.loc[allalarms['red_alarm'] >= 2,'red'] = 1
    allalarms.loc[allalarms['yellow_alarm'] >= 2, 'yellow'] = 1
    allalarms=allalarms.fillna(0)
    allalarms['alarm'] = allalarms['red']+allalarms['yellow']
    # print('doing original one')
    # nightsignal2(device,hr_file,step_file)
    return allalarms[['alarm']]


def alisignal(device, rhr,seg):
    # display(rhr)
    rhr24 = rhr.resample('1D').mean()
    # display(rhr24)
    rhr24median = rhr24.expanding().median().astype(int)

    rhr1 = rhr.resample(f'{seg}T').mean()
    rhr1 = rhr1.loc[rhr1.index.hour<=6]
    

    missings = rhr24.index[rhr24['heartrate'].isnull()]
    rhr24 = rhr24.interpolate(limit=1)
    # rhr1 = rhr1.interpolate(limit=1)
    rhr1['date']=rhr1.index.floor('1D')
    newJoin = rhr1.join(rhr24median.rename(columns={'heartrate': 'hr_median'}),on='date').sort_index()
    n = pd.DataFrame(index=newJoin.index[newJoin['heartrate'] >= newJoin['hr_median']])
    n['alarm'] = 1
    n = n.resample('1D').sum()
    # red_and_yellow_alert_dates = rhr24[rhr24['heartrate'] >= rhr24median['heartrate']+3].index
    # display(rhr24['heartrate'])
    # display(rhr24median['heartrate'])
    total = rhr1[['heartrate']].rename(columns={'heartrate': 'count'}).resample('1D').count().join(n).fillna(0)
    total=total.loc[total['count']>(7*60/seg)/5]
    # red_alert_dates = n.index[n['alarm']>total['heartrate']/1.5]
    # yellow_alert_dates = n.index[n['alarm'] > total['heartrate']/2]
    red_alert_dates = total.index[total['alarm']/total['count']>0.75]
    yellow_alert_dates = total.index[total['alarm']/total['count']>0.5]

    # red_alert_dates = newJoin[newJoin['heartrate'] > newJoin['hr_median']+4].index
    # yellow_alert_dates = newJoin[(newJoin['heartrate'] > newJoin['hr_median']+3)].index

    allalarms = rhr24.copy()
    allalarms['red_alarm'] = 0
    allalarms['yellow_alarm'] = 0
    allalarms.loc[red_alert_dates, 'red_alarm'] = 1
    allalarms.loc[yellow_alert_dates, 'yellow_alarm'] = 1
    allalarms = allalarms.rolling(2).sum()
    allalarms.loc[allalarms['red_alarm'] >= 2,'red'] = 1
    allalarms.loc[allalarms['yellow_alarm'] >= 2, 'yellow'] = 1
    allalarms=allalarms.fillna(0)
    allalarms['alarm'] = allalarms['red']+allalarms['yellow']
    # print('doing original one')
    # nightsignal2(device,hr_file,step_file)
    return allalarms[['alarm']]


def anomaly_detection(rhr,rhrf,info,params):
    
    return timeseries_anomaly_detection.anomaly_detection(rhrf,info,params)


def randomSignal(device, rhr,rate):
    rhr24 = rhr.resample('1D').max().dropna()
    # rhr24median = rhr24.expanding().median().astype(int)
    red_alert_dates = rhr24.sample(frac=rate, replace=False, random_state=1).index

    # red_alert_dates = rhr24.sample(frac=0.13, replace=True, random_state=1).index
    # red_alert_dates=rhr24.dropna().iloc[::int(1/rate),:].index
    
    # rhr24 = rhr24.interpolate(limit=1)
    # yellow_alert_dates = red_alert_dates

    # red_alert_dates = newJoin[newJoin['heartrate'] > newJoin['hr_median']+4].index
    # yellow_alert_dates = newJoin[(newJoin['heartrate'] > newJoin['hr_median']+3)].index

    allalarms = rhr24.copy()
    allalarms['red_alarm'] = 0
    allalarms['yellow_alarm'] = 0
    allalarms.loc[red_alert_dates, 'alarm'] = 2 
    allalarms = allalarms.fillna(0)
    # print('doing original one')
    # nightsignal2(device,hr_file,step_file)
    return allalarms[['alarm']]

def CuSum(device,hr,step,info):
    id=info["id"]
    root = f'/tmp/cusum/{info["id"]}'
    subprocess.check_output(f'rm -rf {root}; mkdir -p {root}', shell=True)
    # import os
    # os.makedirs(root)
    hrf=f'{root}/hr.csv'
    stepf=f'{root}/step.csv'
    stepf2=f'{root}/step2.csv'
    hr.to_csv(hrf)
    if info["device"]=='AppleWatch':
        import pandasql as ps
        sqlcode = '''
                select hr.datetime,step.steps
                from hr,step
                where hr.datetime >= step.start_datetime and hr.datetime <= step.end_datetime            
            '''
        step2 = ps.sqldf(sqlcode, locals())
        step2['datetime'] = pd.to_datetime(step2['datetime'])
        step2 = step2.set_index('datetime')
        step2.to_csv(stepf)
    else:
        step.to_csv(stepf)
    # step.to_csv(stepf2)
    # display(step2)
    pos_date= info["covid_test_date"] if info["symptom_date"]==None else info["symptom_date"]
    # dev = 'apple' if device == 'AppleWatch' else 'fitbit'

    dev='fitbit'
    try:
        # print(f'{id} {hrf} {stepf} {dev} {pos_date}')
        out = subprocess.check_output(f"/usr/bin/Rscript --vanilla ali/online_cusum_alarm_fn.R {id} {hrf} {stepf} {dev} {pos_date}", shell=True, stderr=subprocess.STDOUT)
        # print(out)
        
        if os.path.isdir(f'output/figure/{id}_figure_under_par0.95/'):
            for f in os.listdir(f'output/figure/{id}_figure_under_par0.95/'):
                shutil.copyfile(f'output/figure/{id}_figure_under_par0.95/{f}',f'output/my/{id}/CuSum-{f}')

    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    subprocess.check_output(f'rm -rf {root}', shell=True)
    if os.path.isfile(f'output/table/{id}_table_under_par_0.95/{id}_eval_under0.95.csv'):
        eval_under=pd.read_csv(f'output/table/{id}_table_under_par_0.95/{id}_eval_under0.95.csv')
        offline=pd.read_csv(f'output/table/{id}_table_under_par_0.95/{id}_offline_result_under_par0.95.csv')
        online = pd.read_csv(f'output/table/{id}_table_under_par_0.95/{id}_online_result_under_par0.95.csv')
        eval_under = eval_under.rename(columns={'evaluation time': 'datetime', 'alarm for previous 24 hours':'alarm'})
        eval_under['datetime'] = pd.to_datetime(eval_under['datetime']).dt.floor('1D')
        eval_under=eval_under.set_index('datetime')
        eval_under['alarm'] = eval_under['alarm'].replace({'Green': 0, 'Yellow': 1, 'Red': 2, 'N/A': -1, 'Baseline': -2})
    else:
        eval_under=pd.DataFrame(columns=['alarm']);
    # display(eval_under)
    # display(offline)
    # display(online)
    return eval_under
    
def nightsignal_orig(device, hr, step,info,params):
    import sys
    sys.path.append("..")
    from wearable_infection import nightsignal  
    if device=='Fitbit':
        nighthr = hr[(hr.index.hour >= 0) & (hr.index.hour <= 6)][['heartrate']]
        
        rhr = nighthr.resample("1T").mean().dropna().round().join(step).fillna(0)[['heartrate','steps']].astype(int)
        
        # df.drop(['row_num','start_date','end_date','symbol'], axis=1, errors='ignore').astype(int)
        # display(rhr.loc[rhr['end_datetime'] == ' '])
        rhr.to_csv(f'/tmp/rhr{info["id"]}.csv')
        out=nightsignal.run(heartrate_file=None,
                        step_file=None,
                        device=device,
                        restinghr_file=f'/tmp/rhr{info["id"]}.csv',
                        id=info['id'],
                            red_threshold=params['red_threshold'])
    else:
        out=nightsignal.run(heartrate_file=hr,
                        step_file=step,
                        device=device,
                        id=info['id'],
                         red_threshold=params['red_threshold'])
        # out.index=out.index.rename('datetime') 
    # print(out)
    os.system(f'rm /tmp/rhr{info["id"]}.csv')
    if out.columns.shape[0]==0:
        allalarms = hr.resample('1D').max()
        allalarms['alarm'] = 0
        return allalarms[['alarm']]
    out = out.rename(columns={'date':'datetime','val': 'alarm'})
    out['datetime'] = pd.to_datetime(out['datetime'])
    out=out.set_index('datetime').astype(int)
    # print(out)
    # out=out.loc[out['alarm']>0]
    out.index=out.index.floor('1D')
    return out

def rhrad(info,args,hr,step,rhr,params):
    import sys
    sys.path.append("..")
    from rhrad.scripts import main
    if info["device"] == 'AppleWatch':
        # step_n = pd.DataFrame(columns=['steps'])
    #         print(step)
        all=[]
        for k,row in step.iterrows():
            newRange=pd.date_range(row['start_datetime'].floor('1T'),row['end_datetime'].floor('1T'),freq='1T')
            a=pd.DataFrame(index=newRange)
            a['steps']=row['steps']/len(newRange)
            all.append(a)

        step=pd.concat(all)
    
    return main.run(hr,step,info,params)

def laad(info,args,hr,step):
    import sys
    sys.path.append("..")
    from LAAD.scripts import laad_covid19 


    
    if info["device"] == 'AppleWatch':
        import pandasql as ps
        sqlcode = '''
                select hr.datetime,step.steps
                from hr,step
                where hr.datetime >= step.start_datetime and hr.datetime <= step.end_datetime
            '''
        step2 = ps.sqldf(sqlcode, locals())
        step2['datetime'] = pd.to_datetime(step2['datetime'])
        step2 = step2.set_index('datetime')
        step=step2['steps']
    
    return laad_covid19.run(info,args,hr,step)

def usingAll(allAlarms):
    alarms=allAlarms.clip(lower=0)
    targets = ['if', 'nightsignal', 'CuSum']
    for c in targets:
        if c not in allAlarms.columns:
            return

    su = alarms[targets].sum(axis=1)
    alarms['alarm'] = (su >= len(targets))*1+(su >= len(targets)*1.5)*1
    return alarms[['alarm']]




def load(id):
    hr = pd.read_hdf(f'output/my/{id}/hr.h5', 'hr',mode='r')
    rhr = pd.read_hdf(f'output/my/{id}/rhr.h5', 'rhr',mode='r')
    rhrf = pd.read_hdf(f'output/my/{id}/rhrf.h5', 'rhr',mode='r')
    step = pd.read_hdf(f'output/my/{id}/step.h5', 'step',mode='r')
    with open(f'output/my/{id}/data.json', 'r') as f:
        info = json.load(f)
    info['covid_test_date'] = pd.to_datetime(info['covid_test_date']) if info['covid_test_date'] != 'None' else None
    info['symptom_date'] = pd.to_datetime(info['symptom_date'])if info['symptom_date'] != 'None' else None
    return hr,step,rhr,rhrf,info

def convert(hr_file, step_file, info,force=False):
    id=info['id']
    os.makedirs(f'output/my/{id}', exist_ok=True)
    files = [f'output/my/{id}/hr.h5', f'output/my/{id}/rhr.h5', f'output/my/{id}/rhrf.h5', f'output/my/{id}/step.h5', f'output/my/{id}/data.json']
    exi = [os.path.isfile(f) for f in files]
    if not (force or (False in exi)):
        return
        
    try:
        covid_test_date = pd.to_datetime(info['covid_test_date'])
    except:
        covid_test_date=None    
    try:
        symptom_date = pd.to_datetime(info['symptom_date']) 
    except:
        symptom_date=None    
    device=info['device']
    device,hr, step = read(device, hr_file, step_file)
    rhr = createRestingHR(device, hr=hr, step=step)
    rhrf = createRestingHR_new(device, hr=hr, step=step)
    hr.to_hdf(f'output/my/{id}/hr.h5', 'hr', complevel=9, complib='bzip2')
    step.to_hdf(f'output/my/{id}/step.h5', 'step', complevel=9, complib='bzip2')
    rhr.to_hdf(f'output/my/{id}/rhr.h5', 'rhr', complevel=9, complib='bzip2')
    rhrf.to_hdf(f'output/my/{id}/rhrf.h5', 'rhr', complevel=9, complib='bzip2')
    info = {'id': id,'device': device, 'covid_test_date': str(covid_test_date), 'symptom_date': str(symptom_date)}
    with open(f'output/my/{id}/data.json', 'w') as f:
        json.dump(info, f)
    # return load(id)

def run(id, args={}):
    hr,step,rhr,rhrf,info=load(id)
    device=info['device']
    if args['debug']:
        print(info)
    os.makedirs(f'output/my/{id}', exist_ok=True)
    
    try:
            allAlarms = pd.read_csv(f'output/my/{id}/alarm.csv',parse_dates=['datetime'],index_col='datetime')
            allAlarms=allAlarms[[c for c in allAlarms.columns if 'alisignal' not in c]]
    except:
        allAlarms = rhrf.resample('1d').count()[[]]
    run_methods=args['methods']
    for method in run_methods:
        if not args['rerun'] and method in allAlarms.columns:
            continue
        args['current_method']=method
        try:
            if method == 'nightsignal':
                out = nightsignal(device=device, rhr=rhr)
            elif  'ns-orig' in method:
                out = nightsignal_orig(device=device, hr=hr, step=step,info=info,params={
                    'red_threshold': int(method.split('_')[1])
                })
            elif 'alisignal' in method:
                seg = int(method.split('_')[1])
                out = alisignal(device=device, rhr=rhr, seg=seg)
            elif 'random' in method:
                out = randomSignal(device=device, rhr=rhrf,rate=float(method.split('_')[1]))
            elif method == 'CuSum':
                out = CuSum(device=device, hr=hr, step=step, info=info)
            elif method == 'if':
                out = isolationforest(device=device, rhr=rhr)
            elif method == 'if2':
                out = isolationforest2(device=device, rhr=rhr)
            elif method == 'laad':
                out = laad(info=info,args=args, hr=hr,step=step)
            elif 'rhrad' in method:

                out = rhrad(info=info, args=args, hr=hr, step=step, rhr=rhr,params= {
                    'mode': 'v7',
                    'outliers_fraction': method.split('_')[1]})
            elif 'ad_' in method:
                #flags> a: avg feature
                #flags> b: avg of previous avg feature
                #flags> d: day feature
                #flags> f: time segment
                #flags> l: transfer learning
                #flags> m: median feature
                #flags> n: avg of previous median feature
                #flags> o: only over night
                #flags> t: time feature
                
                params={'flags': method.split('_')[1],
                        'seg': pd.to_timedelta(method.split('_')[2]),
                        'overlap':pd.to_timedelta(method.split('_')[3]),
                        'resolution':method.split('_')[4],
                        'model':'auto-encoder',
                        # 'model':'lstm',
                        # 'use_time_feature':0,
                        #'use_time_feature':1,
                        'test_days':2,
                        'min_train_days':7,
                        'max_train_days':50,
                        # 'future_data_if_not_enough_data':14,
                        'future_data_if_not_enough_data':0,
                        'only_new_points':1,
                        # 'use_median':1,
                        **args
                        }
                # print(params)
                out = anomaly_detection(rhr, rhrf, info, params)
            else:
                continue
            allAlarms[method] = out['alarm']
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f'{id} {method} ignored because of exception {e}')
            if args.get('show_exception',0):
                print(f'id={id} {e}'[0:200])
                traceback.print_exc()

            continue
        
        
    if 'use-all' in run_methods:
        out = usingAll(allAlarms)
        if out is not None:
            allAlarms['use-all'] = out['alarm']
    allAlarms=allAlarms.fillna(-3)
    allAlarms.to_csv(f'output/my/{id}/alarm.csv')
    
    if args.get('save_plots_for_user', 0):
        ui.plotAll(rhr=rhrf, alerts=allAlarms, info=info, 
               file=f'output/my/{id}/all.png', args=args)
    
    ev = {}
    # display(rhrf)
    for method in allAlarms.columns:
        alarm = allAlarms[[method]].rename(columns={method: 'alarm'})
        res = eval.eval_both(rhr=rhrf, alerts=alarm, info=info)
        ev[method] = pd.DataFrame(res).T.stack()

    df = pd.concat(ev, axis=1).T
    # df = df.round(2)#.sort_index(axis=1)
    # print(df)
    df.round(2).to_csv(f'output/my/{id}/eval.csv')
    
        
    
    if args.get('draw_eval',0):
        # display(df.round(2))
        ui.plot_evals(df)
        
            
    
    return df


if __name__ == '__main__':
    import glob, os, getopt, sys
    full_cmd_arguments = sys.argv
    argument_list = full_cmd_arguments[1:]

    short_options = "h"
    long_options = ["heartrate=", "step=", "device=", "restinghr=","method="]

    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        print(str(err))
        sys.exit(2)
    covid_test_date=None
    symthom_date=None
    import random

    id = -1
    for current_argument, current_value in arguments:
        if current_argument in ("-h", "--help"):
            print("Please use: python3 ali.py --method=nightsignal --device=Fitbit --restinghr=<RHR_FILE> || python3 --method=isolationforest ali.py --device=AppleWatch  --heartrate=<HR_FILE> --step=<STEP_FILE> ")
        elif current_argument in ("--heartrate"):
            hr_file = current_value
        elif current_argument in ("--step"):
            step_file = current_value
        elif current_argument in ("--device"):
            device = current_value
        # elif current_argument in ("--restinghr"):
        #     restinghr_file = current_value
        elif current_argument in ("--method"):
            method = current_value
        elif current_argument in ("--covid-test-date"):
            covid_test_date = current_value
        elif current_argument in ("--symthom-date"):
            symthom_date = current_value
        elif current_argument in ("--id"):
            id = current_value
        
    if id==-1:
        id=hash(step_file+hr_file)

    convert(hr_file=hr_file, step_file=step_file,info={
        'id':id,
        'device':device,
        'covid_test_date':covid_test_date, 'symptom_date':symthom_date
    },force=True)
    out = run(runmethods=[method], id=id,args={} )
    # display(out)
