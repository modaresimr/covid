import multiprocessing
import concurrent.futures
import traceback
import os
import functools
import pandas as pd
from . import methods
from zipfile import ZipFile
from IPython.display import display, Image
import matplotlib.pylab as plt
from . import ui
import ipyumbrella as uw
from tqdm.notebook import tqdm

def getFileInfo(args):
    info = pd.read_excel('41591_2021_1593_MOESM3_ESM.xlsx', 'SourceData_COVID19_Positives', index_col='Participant ID')
    error_files = {}

    from zipfile import ZipFile
    import re
    pattern = re.compile("[^_]*/(P\d+)/(.*\.csv)$")
    
    with ZipFile('../ssd2/COVID-19-Phase2-Wearables.zip') as myzip:
        matches = [pattern.match(n) for n in myzip.namelist() if pattern.match(n)]
        ids = {}
        for mm in matches:
            m = mm.groups()

            id = m[0]
            if args.get('only_person',0):
                if id not in args.get('only_person', []):
                    continue
            
            
            if id not in ids:
                ids[id] = {'id': id}
                
            
            file = m[1]
            device = 'AppleWatch' if 'NonFitbit' in file else 'Fitbit'
            typ = "hr" if 'HR' in file else "step"
            ids[id][typ] = mm.string
            ids[id]['device'] = device

            ids[id]['covid_test_date'] = None
            ids[id]['symptom_date'] = None
            if id in info.index:
                user_data = info.loc[id]
                ids[id]['covid_test_date'] = user_data['COVID-19 Test Date']
                ids[id]['symptom_date'] = user_data['COVID-19 Symptom Onset']

            

    if args.get('only_positive', 0):
        ids = {id: ids[id] for id in info.index if id in ids}
    if args.get('only_device', False):
        ids = {id: ids[id] for id in info.index if id in ids and ids[id]['device'] == args.get('only_device')}
    
    # ids = {id: ids[id] for i,id in enumerate(ids) if i>2000}
    return ids;

def run(args):
    print(f'runing... args={args}')
    ids=getFileInfo(args)
    eval_plot = ui.eval_ploter(args=args)
    pbar = tqdm(total=len(ids))
    
    
    total = None
    
    if args.get('parallel'):
        display(eval_plot.out)
        runner = functools.partial(parrun, args=args)
        from contextlib import closing

        pool = multiprocessing.Pool(8,maxtasksperchild=8)
        try:
            result = pool.imap(runner, ids.values())
            
            # pool= concurrent.futures.ProcessPoolExecutor(8)
            # result = pool.map(runner, ids.values())

            for i,(id, out, data) in enumerate(result):
            # for i, id in enumerate(ids):
            #     (id, out, data) = result.next()
                pbar.update(1)
                if 'eval' in out:
                    res = out['eval']
                    if(total is None):
                        total = res
                    else:
                        total = total.add(res, fill_value=0)
                    # eval_plot.plot_evals(total)
                    if args.get('show_final_graph', 0):
                        display(Image(f'output/my/{id}/all.png'))
                    if(i % 19 == 0):
                        print(f'{(i/len(ids)*100):0.0f}%    {"*"*int(i/len(ids)*50)}')
                    if(i % 1 == 0):
                        eval_plot.plot_evals(total)
                    # printEvals(total)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            pool.close()

    else:
        uw.Accordion().D.append(eval_plot.out, title='Evaluation')
        acc = uw.Accordion().D
        runner = functools.partial(parrun, args=args,acc=acc)
        for i, data in enumerate(ids.values()):
            (id, out, data) = runner(data)
            pbar.update(1)
            if 'eval' in out:
                res = out['eval']
                if(total is None):
                    total = res
                else:
                    total = total.add(res, fill_value=0)
                # if(total == None):
                #     total = res
                # else:
                #     total = {k: {cm: total[k][cm]+res[k][cm] for cm in res[k]} for k in res}
            if args.get('show_final_graph', 0):
                display(Image(f'output/my/{id}/all.png'))
            if(i % 10 == 0 and not(total is None)):
                print(f'{(i/len(ids)*100):0.0f}%    {"*"*int(i/len(ids)*50)}')

            if(i % 1 == 0 and not(total is None)):
                eval_plot.plot_evals(total)
                # printEvals(total)
                # ui.plot_evals(total,fig)

    print(f'result============')

    eval_plot.plot_evals(total,True)
    # printEvals(total)
    # eval_plot.close()

    return total




def parrun(data, args,acc=None):
    
    id = data['id']

    
    
    print(' ', end='', flush=True)
    if acc is None:
        return parrun2(data,args)
    with acc.item(f'{id}'):
        return parrun2(data,args)

def parrun2(data, args):
        id = data['id']
        print(' ', end='', flush=True)
        # print (id)
        try:
            # run_methods = args['methods']
            # print(f'{id}, run_methods={run_methods}')
            
            # print(f'{id}, run_methods={run_methods}')        
            # run_methods=[]
            with ZipFile('../ssd2/COVID-19-Phase2-Wearables.zip') as myzip, myzip.open(data['step']) as stepf, myzip.open(data['hr']) as hrf:
                methods.convert(hr_file=hrf, step_file=stepf, info=data, force=False)
            # data=pd.DataFrame()
            # data[id,info['device']]=1
            
            return (id, {'eval': methods.run(id=id, args=args)}, data)

        #             print(f'res={res} total={total_res}')
                # if total_res==None:total_res=res
                # else:
                #     total_res={k:{cm:total_res[k][cm]+res[k][cm] for cm in res[k]} for k in res}

            # if args.get('debug') and ('nightsignal_orig' in methods):
            #     from IPython.display import IFrame
            #     display(Image("./NightSignalResult.png"))

        except Exception as e:
            print(f'{id} ignored because of exception {e}')
            if args.get('show_exception'):
                print(f'id={id} {e}'[0:200])
                traceback.print_exc()

            return (id, {'exception': e}, data)
        return (id, {'finish': 'yes'}, data)
