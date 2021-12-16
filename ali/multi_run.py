import multiprocessing
import traceback

import functools
import pandas as pd
from . import methods
from zipfile import ZipFile
from IPython.display import display

def run(args):
    print(f'runing... args={args}')
    info=pd.read_excel('41591_2021_1593_MOESM3_ESM.xlsx','SourceData_COVID19_Positives',index_col='Participant ID')
    error_files={}

    from zipfile import ZipFile 
    import re
    pattern = re.compile("[^_]*/(P\d+)/(.*\.csv)$")



    with ZipFile('COVID-19-Phase2-Wearables.zip') as myzip:
        matches=[pattern.match(n) for n in myzip.namelist() if pattern.match(n)]
        ids={}
        for mm in matches:
            m=mm.groups()
            
            id=m[0]
            if id != args.get('only_person', id):continue
            if id not in ids:ids[id]={'id':id}
            file=m[1]
            device='AppleWatch' if 'NonFitbit' in file else 'Fitbit'
            typ="hr" if 'HR' in file else "step"
            ids[id][typ]=mm.string
            ids[id]['device']=device

            ids[id]['covid_test_date'] = None
            ids[id]['symptom_date'] = None
            if id in info.index:
                user_data = info.loc[id]
                ids[id]['covid_test_date'] = user_data['COVID-19 Test Date']
                ids[id]['symptom_date'] = user_data['COVID-19 Symptom Onset']
    
    
    if args.get('only_positive',0):
        ids = {id: ids[id] for id in info.index if id in ids}
    if args.get('only_device',False):
        ids = {id: ids[id] for id in info.index if ids[id]['device'] == args.get('only_device')}

    runner = functools.partial(parrun, args=args)
    total = None
    if args.get('parallel'):
        pool = multiprocessing.Pool(8)
        result = pool.imap(runner, ids.values())

        for i,id in enumerate(ids):
            (id,out,data)=result.next()
            if 'eval' in out:
                res=out['eval']
                if(total is None):
                    total=res
                else:
                    total = total.add(res, fill_value=0)
                if(i%10==0):
                    print(f'{(i/len(ids)*100):0.0f}%    {"*"*int(i/len(ids)*50)}')
                if(i % 100 == 0):
                    printEvals(total)
            
    else:

        result = [runner(data) for data in ids.values()]
        for i, (id, out, data) in enumerate(result):
            if 'eval' in out:
                res = out['eval']
                if(total is None):
                    total=res
                else:
                    total = total.add(res, fill_value=0)
                # if(total == None):
                #     total = res
                # else:
                #     total = {k: {cm: total[k][cm]+res[k][cm] for cm in res[k]} for k in res}
            if(i % 10 == 0 and not(total is None)):
                print(f'{(i/len(ids)*100):0.0f}%    {"*"*int(i/len(ids)*50)}')
            if(i % 100 == 0 and not(total is None)):
                printEvals(total)

    print (f'result============')
    printEvals(total)
    return total

def printEvals(total_res):
    total=total_res.copy()
    # df=pd.DataFrame(total_res).T
    # df = total_res.drop(['tp_rate', 'tn_rate'], axis=1)

    # df['tp_rate'] = df['tp_nature']/(df['tp_nature']+df['fn_nature'])
    # df['tn_rate'] = df['tn_nature']/(df['tn_nature']+df['fp_nature'])
    # for method in total_res:
    #     tp_rate = total_res[method]['tp_nature']/(total_res[method]['tp_nature']+total_res[method]['fn_nature']) if total_res[method]['tp_nature']>0 else 0
    #     tn_rate=total_res[method]['tn_nature']/(total_res[method]['tn_nature']+total_res[method]['fp_nature']) if total_res[method]['tn_nature']>0 else 0
    #     print (f'method={method} TP rate={tp_rate:0.2f} TN rate={tn_rate:0.2f}')
    # display(df.round(2))
        
    for typ in total.columns.levels[0]:
        tpr = total[(typ, 'tp')]/(total[(typ, 'tp')]+total[(typ, 'fn')])
        prc = total[(typ, 'tp')]/(total[(typ, 'tp')]+total[(typ, 'fp')])

        total[(typ, 'TPR')] = tpr
        total[(typ, 'TNR')] = total[(typ, 'tn')]/(total[(typ, 'tn')]+total[(typ, 'fp')])
        total[(typ, 'PRC')] = prc
        total[(typ, 'F1')] = 2*prc*tpr/(tpr+prc)
        # total = total.drop([(typ, t) for t in ['tp', 'fp', 'fn', 'tn']],axis=1)

    # total.round(2).to_csv('output/my/eval.csv')
    display(total.round(2))

def parrun(data,args):
    id=data['id']
    try:
        run_methods=args['methods']
        # run_methods=[]
        with ZipFile('COVID-19-Phase2-Wearables.zip') as myzip, myzip.open(data['step']) as stepf, myzip.open(data['hr']) as hrf:
                # print(f'id={id} data={data}')
                
                # if(id == 'P678649'):
                    # myzip.extract(data['hr'],'/tmp/covidtmp/{id}/hr.csv')
                methods.convert(hr_file=hrf, step_file=stepf, info=data,force=False)
                return (id, {'eval': methods.run(run_methods=run_methods, id=id, args=args)},data)
                
    #             print(f'res={res} total={total_res}')
                # if total_res==None:total_res=res
                # else:
                #     total_res={k:{cm:total_res[k][cm]+res[k][cm] for cm in res[k]} for k in res}
    

            # if args.get('debug') and ('nightsignal_orig' in methods):
            #     from IPython.display import IFrame
            #     display(Image("./NightSignalResult.png"))
    
    except Exception as e:
        print(f'{id} ignored because of exception')
        if args.get('show_exception'):
            print(f'id={id} {e}'[0:200])
            traceback.print_exc()

        return (id, {'exception': e},data)
    return (id, {'finish': 'yes'},data)
        
