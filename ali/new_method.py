
import pandas as pd
from IPython.display import display
import os
import traceback
from . import ui
from . import eval
import json
# from . import methods
def usingAll():
    for id in os.listdir('output/my'):
        try:
            print(id)
            alarms=pd.read_csv(f'output/my/{id}/alarm.csv').fillna(0)
            alarms['datetime']=pd.to_datetime(alarms['datetime'])
            alarms=alarms.set_index('datetime')
            # display(alarms)
            alarms=alarms.clip(lower=0)
            su = alarms[['if', 'nightsignal', 'CuSum']].sum(axis=1)
            alarms['use-all'] = (su >= 2)*1+(su >= 3)*1
            # display(alarms)
            
            rhr=pd.read_hdf(f'output/my/{id}/rhr.h5','rhr')
            import json
            with open(f'output/my/{id}/data.json', 'r') as f:
                info=json.load(f)
                info['covid_test_date'] = pd.to_datetime(info['covid_test_date']) if info['covid_test_date']!='None' else None
                info['symptom_date'] = pd.to_datetime(info['symptom_date'])if info['symptom_date'] != 'None' else None
            
            alarms.to_csv(f'output/my/{id}/alarm.csv')

            df=pd.read_csv(f'output/my/{id}/eval.csv',index_col=0)
            alarm = alarms[['use-all']].rename(columns={'use-all': 'alarm'})
            ev = eval.eval_both(rhr=rhr, alerts=alarm, covid_test_date=info['covid_test_date'], symptom_date=info['symptom_date'])
            for item in ev:
                df.loc['use-all',item]=ev[item]
            df['tp_rate'] = df['tp_nature']/(df['tp_nature']+df['fn_nature'])
            df['tn_rate'] = df['tn_nature']/(df['tn_nature']+df['fp_nature'])
            df.round(2).to_csv(f'output/my/{id}/eval.csv')
            # display(df)
            
            ui.plot(rhr=rhr, alerts=alarm, covid_test_date=info['covid_test_date'], symptom_date=info['symptom_date'], title=f'{id} use-all',
                    file=f'output/my/{id}/use-all.png', show=1)
        except Exception as e:
            print(f'id={id} {e}'[0:200])
            traceback.print_exc()


# from . import methods


def evalAll():
    total=None
    for id in os.listdir('output/my'):
        try:
            eval = pd.read_csv(f'output/my/{id}/eval.csv',index_col=0).fillna(0)
            if total is None:
                total=eval
            else:
                total = total.add(eval, fill_value=0)
            # display(alarms)
            
        except Exception as e:
            print(f'id={id} {e}'[0:200])
            traceback.print_exc()

    total = total.drop(['tp_rate', 'tn_rate'], axis=1)
    df = total
    df['tp_rate'] = df['tp_nature']/(df['tp_nature']+df['fn_nature'])
    df['tn_rate'] = df['tn_nature']/(df['tn_nature']+df['fp_nature'])
    df.round(2).to_csv(f'output/my/eval.csv')
    display(df)


def reEvalAll():
    total=None
    for id in os.listdir('output/my'):
        if not "P" in id :continue
        try:
            print(f'\r{id}',end='')
            
            with open(f'output/my/{id}/data.json', 'r') as f:
                info=json.load(f)
                info['covid_test_date'] = pd.to_datetime(info['covid_test_date']) if info['covid_test_date']!='None' else None
                info['symptom_date'] = pd.to_datetime(info['symptom_date'])if info['symptom_date'] != 'None' else None
            if not info['covid_test_date']:
                continue
            alarms = pd.read_csv(f'output/my/{id}/alarm.csv', parse_dates=['datetime'], index_col='datetime').fillna(0)
            rhr = pd.read_hdf(f'output/my/{id}/rhr.h5', 'rhr', mode='r')

            ev = {}
            for method in alarms.columns:
                # if 'anomaly-detection' not in method:
                #     continue
                alarm = alarms[[method]].rename(columns={method: 'alarm'})
                res = eval.eval_both(rhr=rhr, alerts=alarm, info=info)
                ev[method] = pd.DataFrame(res).T.stack()
            if len(ev)==0:continue
            # ui.plotAll(rhr,alarms,info,show=True)
            df = pd.concat(ev, axis=1).T
            # df = df.round(2)#.sort_index(axis=1)
            df.round(2).to_csv(f'output/my/{id}/eval.csv')
            # display(df)
            if total is None:
                total=df
            else:
                total = total.add(df, fill_value=0)
            
        except Exception as e:
            print(f'id={id} {e}'[0:200])
            traceback.print_exc()

    for typ in total.columns.levels[0]:
        tpr = total[(typ, 'tp')]/(total[(typ, 'tp')]+total[(typ, 'fn')])
        prc = total[(typ, 'tp')]/(total[(typ, 'tp')]+total[(typ, 'fp')])

        total[(typ, 'TPR')] = tpr
        total[(typ, 'TNR')] = total[(typ, 'tn')]/(total[(typ, 'tn')]+total[(typ, 'fp')])      
        total[(typ, 'PRC')] = prc
        total[(typ, 'F1')] = 2*prc*tpr/(tpr+prc)
        # total = total.drop([(typ, t) for t in ['tp', 'fp', 'fn', 'tn']],axis=1)
    
    total.round(2).to_csv('output/my/eval.csv')
    return total.round(2)
