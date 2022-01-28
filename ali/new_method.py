import multiprocessing
import traceback
import numpy as np
import functools
from tqdm.notebook import tqdm

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
        except KeyboardInterrupt:
            raise
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

def reEval(id, args):
        if not "P" in id:
            return


        try:
            if not os.path.isfile(f'output/my/{id}/data.json'):
                return
            if not os.path.isfile(f'output/my/{id}/alarm.csv'):
                return
            if not os.path.isfile(f'output/my/{id}/rhrf.h5'):
                return

            # print(f'\r{id}', end='')
            with open(f'output/my/{id}/data.json', 'r') as f:
                info = json.load(f)
                info['covid_test_date'] = pd.to_datetime(info['covid_test_date']) if info['covid_test_date'] != 'None' else None
                info['symptom_date'] = pd.to_datetime(info['symptom_date'])if info['symptom_date'] != 'None' else None
            # if not info['covid_test_date']:
            #     continue
            if args.get('only_positive', 0) and not info['covid_test_date']:
                return

            alarms = pd.read_csv(f'output/my/{id}/alarm.csv', parse_dates=['datetime'], index_col='datetime').fillna(0)

            rhr = pd.read_hdf(f'output/my/{id}/rhrf.h5', 'rhr', mode='r')
            # display(rhr)
            
            ev = {}
            for method in alarms.columns:
                # if 'anomaly-detection' not in method:
                #     continue
                alarm = alarms[[method]].rename(columns={method: 'alarm'})
                res = eval.eval_both(rhr=rhr, alerts=alarm, info=info)
                ev[method] = pd.DataFrame(res).T.stack()
            if len(ev) == 0:
                return
            # ui.plotAll(rhr,alarms,info,show=True)
            df = pd.concat(ev, axis=1).T
            # df=df.astype(int)
            # df = df.round(2)#.sort_index(axis=1)
            df.round(2).to_csv(f'output/my/{id}/eval.csv')
            # display(df)
            # if args.get('show_')
            return df
        except Exception as e:
            print(f'id={id} {e}'[0:200])
            traceback.print_exc()

def reEvalAllParallel(args={}):
    total=None
    ids = os.listdir('output/my')
    pool = multiprocessing.Pool(8)
    runner = functools.partial(reEval, args=args)
    result = pool.imap(runner, ids)
    pbar = tqdm(total=len(ids))
    eval_plot = ui.eval_ploter(add2display=True)
    try:
     for i, id in enumerate(ids):
        res = result.next()
        pbar.update(1)
        if res is not None:
                if args.get('methods',0):
                    if sum(np.in1d(args['methods'], res.index))!=len(args['methods']):
                        continue
                    res = res.loc[args['methods']]
                if(total is None):
                    total = res
                else:
                    total = total.add(res, fill_value=0)
                # eval_plot.plot_evals(total)
                if args.get('show_final_graph', 0):
                    display(Image(f'output/my/{id}/all.png'))
                if(i % 1 == 0):
                    eval_plot.plot_evals(total)
                    # printEvals(total)
        # total = total.drop([(typ, t) for t in ['tp', 'fp', 'fn', 'tn']],axis=1)
    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        pool.close()
    total.round(2).to_csv('output/my/eval.csv')
    eval_plot.plot_evals(total,True)
    return total.round(2)

def reEvalAll():
    total=None
    eval_plot = ui.eval_ploter()
    display(eval_plot.out)
    from tqdm.notebook import tqdm
    for i,id in tqdm(enumerate(os.listdir('output/my')),total=len(os.listdir('output/my'))):
        if not "P" in id :continue
        try:
            print(f'\r{id}',end='')
            
            with open(f'output/my/{id}/data.json', 'r') as f:
                info=json.load(f)
                info['covid_test_date'] = pd.to_datetime(info['covid_test_date']) if info['covid_test_date']!='None' else None
                info['symptom_date'] = pd.to_datetime(info['symptom_date'])if info['symptom_date'] != 'None' else None
            # if not info['covid_test_date']:
            #     continue
            
            alarms = pd.read_csv(f'output/my/{id}/alarm.csv', parse_dates=['datetime'], index_col='datetime').fillna(0)

            rhr = pd.read_hdf(f'output/my/{id}/rhr.h5', 'rhr', mode='r')
            # if info['covid_test_date'] <rhr.index[0]+pd.to_timedelta('21d'):continue
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
            if i%10==2:
                eval_plot.plot_evals(total)
        except KeyboardInterrupt:
            raise
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
    total=total.sort_index()
    eval_plot.plot_evals(total,True)
    # return total.round(2)
