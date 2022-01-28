import statistics
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pyplot as plt
from IPython.display import display
from IPython.display import clear_output
import ipyumbrella as uw
from exdec.decorator import catch

# plot settings
font = {'family': 'sans-serif',
        'size': 4,
        }
plt.rc('font', **font)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=10)
plt.style.use('seaborn-dark-palette')
plt.rc('axes', titlesize=20)


def round10Base(n):
    a = (n // 10) * 10
    b = a + 10
    return (b if n - a > b - n else a)


def plot(rhr, alerts, info, title='', file=None, show=False):

    window = None
    symptom_date = info['symptom_date']
    covid_test_date = info['covid_test_date']
    if(symptom_date != None):
        window = [symptom_date-pd.to_timedelta('21d'), symptom_date]
    elif covid_test_date != None:
        window = [covid_test_date-pd.to_timedelta('21d'), covid_test_date]
    #################################  Plot  #################################
    # print(f"Plotting...covid_test_date={covid_test_date},symptom_date={symptom_date}, window={window}")
    figure = plt.gcf()
    ax = plt.gca()
    rhr.to_csv('a.csv')

    rhr24 = rhr.resample('24H').mean()
    rhr24median = rhr24.expanding().median()  # .astype(int)

    missings = rhr24.index[rhr24['heartrate'].isnull()]
    missings_estimate = rhr24.interpolate().loc[missings]
    plt.plot(missings, missings_estimate['heartrate'], color='white', marker='o', linestyle='', markersize=3, markerfacecolor='gray', markeredgecolor='gray', label="Imputed Avg RHR \n over night")

    min_ = rhr24['heartrate'].min()
    max_ = rhr24['heartrate'].max()

    plt.plot(rhr24.index, rhr24['heartrate'], color='black',  marker='o',  markersize=2.5, linewidth=1.5, label="Avg RHR over night")
    plt.plot(rhr24median.index, rhr24median['heartrate'], color='green', linewidth=1.5, linestyle='dashed', label="Med RHR over night")
    # plt.plot(rhr24median.index, rhr24median['heartrate']+3, color='yellow', linewidth=1, linestyle='dashed', label="Med RHR over night + 3")
    # plt.plot(rhr24median.index, rhr24median['heartrate']+4, color='red', linewidth=1, linestyle='dashed', label="Med RHR over night + 4")

    # display(alerts)
    if not alerts is None:
        for d, row in alerts.iterrows():
            v = row['alarm']
            if(v > 0):
                plt.axvline(x=d, linestyle='-', color='red' if v == 2 else 'yellow', linewidth=1)

    if(covid_test_date != None):
        plt.axvspan(covid_test_date, min(rhr24.index[-1], covid_test_date+pd.to_timedelta('14d')), facecolor='red', alpha=0.25, label='Covid')

    if(symptom_date != None):
        maxSymthom = min(rhr24.index[-1], symptom_date+pd.to_timedelta('14d'))
        if(covid_test_date != None and symptom_date < covid_test_date):
            maxSymthom = min(covid_test_date, maxSymthom)
        plt.axvspan(symptom_date, maxSymthom, facecolor='yellow', alpha=0.25, label='Sympthom')
    if(window != None):
        ax.axvspan(window[0], window[1], facecolor='green', alpha=0.25, label='Acceptable range')

    # Title & Symptom Onset & Save plot
    plt.xticks(rotation=90)
    plt.ylim(int(min_)-1, int(max_)+1)
    h = plt.ylabel('Resting\n heart rate\n over night', fontsize=12)
    plt.xlabel('    Day', fontsize=12)
    h.set_rotation(90)
    plt.yticks(np.arange(round10Base(min_), round10Base(max_), 10))

    plt.xticks(rhr24.index)
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 3 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)

    # uncomment for legend
    # lgd = ax.legend(prop={'size': 8.5}, bbox_to_anchor= (1.0, 1.0), loc="upper left", frameon=False)
    # plt.grid(False)
    ax.spines["bottom"].set_color('black')
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["top"].set_color('black')
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_color('black')
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["left"].set_color('black')
    ax.spines["left"].set_linewidth(1.5)

    figure = plt.gcf()
    figure.set_size_inches(16, 2.5)
    # plt.legend()
    ax.set_title(title)

    if file:
        plt.gcf().savefig(file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    plt.close()

from ipywidgets import widgets
class debugView:
    def __init__(self,show):
        self.show = show
        if show:
            self.out = widgets.HTML()
            display(self.out)

    def set_html(self,html):
        if self.show:
            self.out.value=html
    def add_html(self,html):
        if self.show:
            self.out.value+=html

    def print(self,text):
        if self.show:
            self.out.value+=f'<code>{text}</code>'
    def clear(self):
        if self.show:
            self.out.value=''
        


class onlinePloter:
    def __init__(self, args={}):
        self.args=args
        self.show=args.get('debug',0)
        if self.show:
            self.out = uw.Output()
            display(self.out)
        self.plot_an_score = False
        # self.fig = plt.figure(figsize=(16, 2.5*2))
        self.fig=None

        
        
        
    def close(self):
        if self.show:
            plt.close(self.fig)
    def plot(self, info, rhr1, test, alerts, anomay_score_avg=None):
     if self.show:
      with self.out:
        
        
        # for ax in self.axs:
        #     ax.clear()
        # if not self.fig:
        self.fig, ax = plt.subplots(1+self.plot_an_score, 1, figsize=(16, 2.5*(1+self.plot_an_score)), sharex=True)
        if not self.plot_an_score:
            self.axs = [ax, ax.twinx()]
        else:
            self.axs = ax        
        if anomay_score_avg is not None and anomay_score_avg.shape[0]>0:
            ax2 = self.axs[1]
            anomay_score_avg.plot.area(ax=ax2, stacked=False, color='b')
            ax2.set_ylim(-.5, +.2)
        ax = self.axs[0]

        window = None
        symptom_date = info['symptom_date']
        covid_test_date = info['covid_test_date']
        if(symptom_date != None):
            window = [symptom_date-pd.to_timedelta('21d'), symptom_date]
        elif covid_test_date != None:
            window = [covid_test_date-pd.to_timedelta('21d'), covid_test_date]
        #################################  Plot  #################################
        # print(f"Plotting...covid_test_date={covid_test_date},symptom_date={symptom_date}, window={window}")

        rhr24 = rhr1.resample('24H').mean()
        
            # after = rhr24.loc[rhr24.index >= test[-1]]

        # testarea=rhr24.loc[(rhr24.index>=before.index[-1]) &(rhr24.index<=after.index[0])]

        rhr24median = rhr24.expanding().median()  # .astype(int)

        missings = rhr24.index[rhr24['heartrate'].isnull()]
        missings_estimate = rhr24.interpolate().loc[missings]
        ax.plot(missings, missings_estimate['heartrate'], color='white', marker='o', linestyle='', markersize=3, markerfacecolor='gray', markeredgecolor='gray', label="Imputed Avg RHR \n over night")

        min_ = rhr24['heartrate'].min()
        max_ = rhr24['heartrate'].max()

        if test is not None:
            ax.plot(rhr24.index, rhr24['heartrate'], color='gray',  marker='o',  markersize=2.5, linewidth=1.5, label="Avg RHR over night")
            before = rhr24.loc[rhr24.index < test[-1]]
            ax.plot(before.index, before['heartrate'], color='black',  marker='o',  markersize=2.5, linewidth=1.5, label="Avg RHR over night")
        # plt.plot(testarea.index, testarea['heartrate'], color='green',  marker='o',  markersize=2.5, linewidth=1.5, label="Avg RHR over night")
            ax.axvspan(test[0]-pd.to_timedelta('1d'), test[-1], facecolor='blue', alpha=0.25, label='Covid')
        else:
            ax.plot(rhr24.index, rhr24['heartrate'], color='black',  marker='o',  markersize=2.5, linewidth=1.5, label="Avg RHR over night")

        ax.plot(rhr24median.index, rhr24median['heartrate'], color='green', linewidth=1.5, linestyle='dashed', label="Med RHR over night")
        # plt.plot(rhr24median.index, rhr24median['heartrate']+3, color='yellow', linewidth=1, linestyle='dashed', label="Med RHR over night + 3")
        # plt.plot(rhr24median.index, rhr24median['heartrate']+4, color='red', linewidth=1, linestyle='dashed', label="Med RHR over night + 4")

        # display(alerts)
        if not alerts is None:
            for d, row in alerts.iterrows():
                v = row['alarm']
                if(v > 0):
                    ax.axvline(x=d, linestyle='-', color='red' if v >= 2 else 'yellow', linewidth=2)

        if(covid_test_date != None):
            ax.axvspan(covid_test_date, min(rhr24.index[-1], covid_test_date+pd.to_timedelta('14d')), facecolor='red', alpha=0.25, label='Covid')

        if(symptom_date != None):
            maxSymthom = min(rhr24.index[-1], symptom_date+pd.to_timedelta('14d'))
            if(covid_test_date != None and symptom_date < covid_test_date):
                maxSymthom = min(covid_test_date, maxSymthom)
            ax.axvspan(symptom_date, maxSymthom, facecolor='yellow', alpha=0.25, label='Sympthom')
        if(window != None):
            ax.axvspan(window[0], window[1], facecolor='green', alpha=0.25, label='Acceptable range')

        ax.set_ylim(int(min_)-1, int(max_)+1)
        h = ax.set_ylabel('Resting\n heart rate\n over night', fontsize=12)
        # ax.xlabel('    Day', fontsize=12)
        h.set_rotation(90)
        ax.set_yticks(np.arange(round10Base(min_), round10Base(max_), 10))

        ax.set_xticks(rhr24.index)
        ax.set_xticklabels(labels=rhr24.index.date, rotation=90)

        # ax2.set_xticklabels(rhr24.index, rotation=90)
        for index, label in enumerate(ax.xaxis.get_ticklabels()):
            if index % 3 == 0:
                label.set_visible(True)
            else:
                label.set_visible(False)

        # uncomment for legend
        # lgd = ax.legend(prop={'size': 8.5}, bbox_to_anchor= (1.0, 1.0), loc="upper left", frameon=False)
        # plt.grid(False)
        ax.spines["bottom"].set_color('black')
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["top"].set_color('black')
        ax.spines["top"].set_linewidth(1.5)
        ax.spines["right"].set_color('black')
        ax.spines["right"].set_linewidth(1.5)
        ax.spines["left"].set_color('black')
        ax.spines["left"].set_linewidth(1.5)

        # fig.set_size_inches(16, 2.5)
        # plt.legend()
        ax.set_title(f"{info['id']}-{self.args.get('current_method','')}")
        clear_output(wait=True)
        # clear_output()
        self.fig.canvas.draw()
        # plt.show()
        # self.out.clear_output(wait=True)

# @catch
def plotAll(rhr, alerts, info, file=None, args={}):
    show=args.get('debug',0)
    window = None
    symptom_date = info['symptom_date']
    covid_test_date = info['covid_test_date']
    indection_detation_window=pd.to_timedelta('21d')
    # if symptom_date != None and covid_test_date != None:
    #     if (covid_test_date-symptom_date).days > 7:
    #         window = [covid_test_date-indection_detation_window, covid_test_date]
    #     else:
    #         window = [symptom_date-indection_detation_window, symptom_date]
    # el
    if(symptom_date != None):
        window = [symptom_date-indection_detation_window, symptom_date]
    elif covid_test_date != None:
        window = [covid_test_date-indection_detation_window, covid_test_date]
    #################################  Plot  #################################
    # print(f"Plotting...covid_test_date={covid_test_date},symptom_date={symptom_date}, window={window}")
    ax = plt.subplot(211)
    # ax2 = plt.subplot(212, sharex=ax)
    ax2 = ax.twinx()
    rhr.to_csv('a.csv')

    rhr24 = rhr.resample('24H').mean()
    rhr24median = rhr24.expanding().median()  # .astype(int)

    missings = rhr24.index[rhr24['heartrate'].isnull()]
    missings_estimate = rhr24.interpolate().loc[missings]
    ax.plot(missings, missings_estimate['heartrate'], color='white', marker='o', linestyle='', markersize=3, markerfacecolor='gray', markeredgecolor='gray', label="Imputed Avg RHR \n over night")

    min_ = rhr24['heartrate'].min()
    max_ = rhr24['heartrate'].max()

    ax.plot(rhr24.index, rhr24['heartrate'], color='black',  marker='o',  markersize=2.5, linewidth=1.5, label="Avg RHR over night")
    ax.plot(rhr24median.index, rhr24median['heartrate'], color='green', linewidth=1.5, linestyle='dashed', label="Med RHR over night")
    # plt.plot(rhr24median.index, rhr24median['heartrate']+3, color='yellow', linewidth=1, linestyle='dashed', label="Med RHR over night + 3")
    # plt.plot(rhr24median.index, rhr24median['heartrate']+4, color='red', linewidth=1, linestyle='dashed', label="Med RHR over night + 4")

    # display(alerts)
    draw_train_test_range = -1
    cols=[]
    for m in alerts.columns:
        
        if 'anomaly-detection' in m:continue
        if 'random' == m:continue
        # if m not in args['methods'] and 'ad_' in m and '_all' not in m:
        if m not in args['methods']:
            continue
        cols.append(m)
    cols = args['methods'][::-1]
    # cols = [m for m in alerts.columns if 'anomaly-detection' not in m]
    for i, method in enumerate(cols):
        # if 'anomaly-detection' in method:continue
        mAlarms = alerts[[method]]
        red = mAlarms[mAlarms >= 2].dropna()
        yellow = mAlarms[mAlarms == 1].dropna()
        # print(red)
        for k in red.index:
            ax2.scatter(x=k, y=i, s=mAlarms.loc[k]/mAlarms.max()*30, color='red')
        for k in yellow.index:
            ax2.scatter(x=k, y=i, s=1*10, color='yellow')
        if 'anomaly' in method:
            draw_train_test_range = i

    if draw_train_test_range >= 0 and 'train' in info:
        draw_range(ax2, info['train'], color='blue', y=draw_train_test_range)
        draw_range(ax2, info['test'], color='brown', y=draw_train_test_range)

    ax2.set_yticks(range(len(cols)), labels=cols)


    if(covid_test_date != None):
        # ax.axvspan(covid_test_date, min(rhr24.index[-1], covid_test_date+pd.to_timedelta('14d')), facecolor='red', alpha=0.25, label='Covid')
        ax.axvline(covid_test_date, color='red', zorder=1, linestyle='--',  lw=3, alpha=1)  # Symptom date
        # ax.axvline(min(rhr24.index[-1], covid_test_date+pd.to_timedelta('14d')), color='purple', zorder=1, linestyle=':',  lw=3, alpha=0.5)  # Symptom date
        # ax.axvline(max(rhr24.index[0], covid_test_date-pd.to_timedelta('14d')), color='purple', zorder=1, linestyle=':',  lw=3, alpha=0.5)  # Symptom date

    if(symptom_date != None):
        ax.axvline(symptom_date, color='yellow', zorder=1, linestyle='--',  lw=6, alpha=1)  # Symptom date
        # maxSymthom = min(rhr24.index[-1], symptom_date+pd.to_timedelta('14d'))
        ax.axvline(max(rhr24.index[0], symptom_date-pd.to_timedelta('7d')), color='green', zorder=1, linestyle='--', lw=1, alpha=0.5)  # Symptom date
        ax.axvline(max(rhr24.index[0], symptom_date-pd.to_timedelta('14d')), color='green', zorder=1, linestyle='--', lw=1, alpha=0.5)  # Symptom date
        # ax.axvline(min(rhr24.index[-1], symptom_date+pd.to_timedelta('14d')), color='green', zorder=1, linestyle='--', lw=6, alpha=0.5)  # Symptom date
        # ax.axvline(max(rhr24.index[0], symptom_date-pd.to_timedelta('14d')), color='green', zorder=1, linestyle='--', lw=6, alpha=0.5)  # Symptom date

        # if(covid_test_date != None and symptom_date<covid_test_date):
        #     maxSymthom = min(covid_test_date, maxSymthom)
        # ax.axvspan(symptom_date, maxSymthom, facecolor='yellow', alpha=0.25,label='Sympthom')

    if(window != None):
        ax.axvspan(window[0], window[1], facecolor='green', alpha=0.25, label='Acceptable range')

    # Title & Symptom Onset & Save plot
    # ax.set_xticks(rotation=90)
    ax.set_ylim(int(min_)-1, int(max_)+1)
    h = ax.set_ylabel('Resting\n heart rate\n over night', fontsize=12)
    # ax.xlabel('    Day', fontsize=12)
    h.set_rotation(90)
    ax.set_yticks(np.arange(round10Base(min_), round10Base(max_), 10))

    ax.set_xticks(rhr24.index,minor=True)
    # ax.set_xticklabels(labels=rhr24.index.date, rotation=90)
    minor_label=list(range(0,len(rhr24)))
    
    ref=0
    if window != None:
        covid_indx = np.where(rhr24.index == covid_test_date)[0][0]
        symp_indx = np.where(rhr24.index == window[-1])[0][0] 
        ref = symp_indx if symp_indx!=0 else covid_indx
        

        minor_label = minor_label-ref
        # print(symp_indx, symp_indx%7)
        # ax.set_xticks(rhr24.iloc[symp_indx % 7::7].index, minor=True)

    ax.set_xticks(rhr24.iloc[ref % 7::7].index,minor=False)
    ax.set_xticklabels(labels=minor_label[ref % 7::7], rotation=90)

    # ax2.set_xticklabels(rhr24.index, rotation=90)
    # for index, label in enumerate(ax.xaxis.get_ticklabels()):
    #     if (index-symp_indx) % 3 == 0 and abs(covid_indx - index)>2 or covid_indx == index:
    #         label.set_visible(True)
    #     else:
    #         label.set_visible(False)

    # uncomment for legend
    # lgd = ax.legend(prop={'size': 8.5}, bbox_to_anchor= (1.0, 1.0), loc="upper left", frameon=False)
    
    
    ax.spines["bottom"].set_color('black')
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["top"].set_color('black')
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_color('black')
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["left"].set_color('black')
    ax.spines["left"].set_linewidth(1.5)
    ax2.grid(True)
    ax.grid(False)
    ax.grid(True,axis='x')
    figure = plt.gcf()
    figure.set_size_inches(16, 5+len(cols)/2)
    # plt.legend()
    ax.set_title(info['id'])

    if file:
        plt.gcf().savefig(file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    plt.close()


def draw_range(ax, data, color='black', y=0):
    from matplotlib.patches import Rectangle
    s = -1
    e = -1

    for k in data.index.floor('1D').unique():
        if s == -1:
            s = k
            e = k

        if (k-e).days > 7:

            ax.add_patch(Rectangle((s, y+.5), e-s, .2, facecolor=color, edgecolor=color, alpha=.3, ec='k', lw=2))
            ax.text(s+(e-s)/2, y+.7, f'{len(data.loc[s:e].index.floor("1D").unique())} days', fontsize=20, color='black')
            s = k
        e = k

    ax.add_patch(Rectangle((s, y+.5), e-s, .2, facecolor=color, edgecolor=color, alpha=.3, ec='k', lw=2))
    ax.text(s+(e-s)/2, y+.7, f'{len(data.loc[s:e].index.floor("1D").unique())} days', fontsize=20, color='black')

eps=0.000000001
class eval_ploter:
    
    def __init__(self,add2display=False,args={}):
    #   self.fig=plt.figure(figsize=(1,1))
        self.lasttime=None
        self.args=args
        self.out=uw.Output()
        if add2display:
            display(self.out)
    #   self.axs=None

    #   plt.ion()
    #   self.changeSize=False

    #   fig.
    #   self.fig.canvas.draw()
    #   def close(self):
    #   plt.close(self.fig)
    def printEvals(self,total_res):
        total = total_res.copy()#.astype(int)
        
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
            if typ=='total':continue
            if typ=='cov_score':continue
            tpr = total[(typ, 'tp')]/(total[(typ, 'tp')]+total[(typ, 'fn')]+eps)
            prc = total[(typ, 'tp')]/(total[(typ, 'tp')]+total[(typ, 'fp')]+eps)

            total[(typ, 'TPR')] = tpr
            total[(typ, 'TNR')] = total[(typ, 'tn')]/(total[(typ, 'tn')]+total[(typ, 'fp')]+eps)
            total[(typ, 'PRC')] = prc
            total[(typ, 'F1')] = 2*prc*tpr/(tpr+prc+eps)
            # total = total.drop([(typ, t) for t in ['tp', 'fp', 'fn', 'tn']],axis=1)
        # total = total.sort_values(('nature', 'TPR'), ascending=False)
        total[('cov_score','cov')] = (total[('cov_score','score')]-total[('cov_score','s_null')])/(total[('cov_score','s_best')]-total[('cov_score','s_null')])
        # total = total.sort_values([('nature', 'tp'),('nature', 'fp')], ascending=[False,True])
        total = total.sort_index(axis=0)
        total = total.sort_index(axis=1, level=[0, 1], ascending=[True, False])

        # total.round(2).to_csv('output/my/eval.csv')
        pd.set_option("display.max_columns", None)
        display(total.round(2))
        pd.reset_option("display.max_columns")

    @catch
    def plot_evals(self, evals,force=False):
      if not force and self.lasttime != None and (pd.Timestamp.now()-self.lasttime).seconds<30:
          return
      if self.args.get('methods',None) is not None:
        evals=evals.loc[np.intersect1d(evals.index,self.args['methods'])]
      self.lasttime=pd.Timestamp.now()
        
      with self.out:
        clear_output(wait=True)
        #         team_m=team_m.sort_index()
        evals=evals[evals[('nature','tn')]>=(evals[('nature','tn')].max()//3)]
        # evals = evals.sort_values([('nature', 'tp'),('nature', 'fp')], ascending=[True,False])
        evals = evals.sort_index(axis=0)
        ind = [i for i, _ in enumerate(evals.index)]
        # cols = evals.columns.get_level_values(0).drop('total').unique()
  #     team_m=(evals.T/evals.sum(axis=1)).T
        width = .6
        cols=['cov_score','nature','new']
        self.fig, self.axs = plt.subplots(1, 3*len(cols), sharey=True, figsize=(15, len(ind)*.3))
        # if(self.axs is None):
        # f, axs=plt.subplots(1,3*len(cols),sharey=True,figsize=(15,len(ind)*.3))
        # self.axs=self.fig.subplots(1,3*len(cols),sharey=True)
        # if not self.changeSize:
        #     self.fig.set_size_inches(15, len(ind)*.3)
        #     self.changeSize=True
        # ax=None
        # for i in range(3*len(cols)):
        #     ax=f.add_subplot(111,sharey=ax)

        # for ax in self.axs:
        #         print(ax)
        #         ax.clear()

        axs = self.axs
        
        for ci, c in enumerate(cols):
            i = ci*3
            evs = evals[(c,)].copy()
            evs['tpr'] = (evs['tp']/(evs['tp']+evs['fn']+eps)).fillna(0)
            evs['fnr'] = (evs['fn']/(evs['tp']+evs['fn']+eps)).fillna(0)
            evs['tnr'] = (evs['tn']/(evs['tn']+evs['fp']+eps)).fillna(0)
            evs['fpr'] = (evs['fp']/(evs['tn']+evs['fp']+eps)).fillna(0)
            evs['prc'] = (evs['tp']/(evs['tp']+evs['fp']+eps)).fillna(0)
            evs['f1'] = (2*evs['prc']*evs['tpr']/(evs['tpr']+evs['prc']+eps)).fillna(0)
            axs[i].set_xlabel(c)
            if c=='cov_score':
                evs['cov'] = (evs['score']-evs['s_null'])/(evs['s_best']-evs['s_null'])
                axs[i].barh(ind, evs['cov'], width, label='COV_score', color='g')  # , color='red'
                # evs[['tp', 'fn']].plot.barh(ax=axs[i+1], legend=False, width=.9, color={'tp': 'g', 'fn': 'r'})
                axs[i+1].barh(ind, evs['tp'], width, label='TP', color='g')  # , color='red'
                axs[i+1].barh(ind, evs['fn'], width, left=evs['tp'], label='FN', color='#fdce4e', alpha=1)  # , color='yellow'
                for v in ind:
                    row=evs.iloc[v]
                    axs[i+1].text( row['tp']/2,v, round(row['tp'],1), color='w',fontdict=dict(fontsize=12), fontweight='bold', ha='center', va='center')
                    # if row['fnr']>.2:
                    axs[i+1].text( row['tp']+row['fn']/2,v, round(row['fn'],1), color='b',fontdict=dict(fontsize=7), fontweight='bold', ha='center', va='center')

                axs[i+2].barh(ind, evs['fp'], width, label='FP', color='r')  # , color='red'
            else:
                axs[i].barh(ind, evs['tpr'], width, label='TPR', color='g')  # , color='red'
                axs[i].barh(ind, evs['fnr'], width, left=evs['tpr'], label='FNR', color='#fdce4e', alpha=1)  # , color='yellow'
                
                for v in ind:
                    row=evs.iloc[v]
                    axs[i].text( row['tpr']/2,v, round(row['tp']), color='w',fontdict=dict(fontsize=12), fontweight='bold', ha='center', va='center')
                    # if row['fnr']>.2:
                    axs[i].text( row['tpr']+row['fnr']/2,v, round(row['fn']), color='b',fontdict=dict(fontsize=7), fontweight='bold', ha='center', va='center')

                axs[i+1].barh(ind, evs['tnr'], width, left=0, label='TNR', color='#00c2b8', alpha=1)  # , color='red'
                axs[i+1].barh(ind, evs['fpr'], width, left=evs['tnr'], label='FPR', color='r', alpha=1)  # , color='red'

                evs[['tpr', 'prc']].plot.barh(ax=axs[i+2], legend=False, width=.9, color={'tpr': 'g', 'prc': 'b'})
                # evs[['f1']].plot.barh(ax=axs[i+2],legend=False,width=width/3,color='r',edgecolor='r')
                axs[i+2].scatter(x=evs['f1'], y=ind, color='red', label='F1', marker='x')
                # axs[i+2].barh(ind, evs['tpr'], width/2, left=0, label='TNR', color='#00c2b8', alpha=1)  # , color='red'
                # axs[i+2].barh(ind, evs['prc'], width/2, left=0, label='FPR', color='r', alpha=1)  # , color='red'
                # axs[i+2].barh(ind, evs['prc'], width/4, left=0, label='FPR', color='r', alpha=1)  # , color='red'


                axs[i].set_xticks([0,.2,.4,.6,.8,1])
#         axs[i].set_xticklabels([f'{x-math.floor(x)}' for x in range(0,2,.25)])
#         axs[i].set(xticks=range(0,2,.25), xticklabels=[f'{x-math.floor(x)}' for i,x in enumerate(range(0,2,.25))]
            if ci == 0:
                axs[i].legend(loc='upper center', bbox_to_anchor=(0, -0.15), fancybox=True, shadow=True)
                axs[i+1].legend(loc='upper center', bbox_to_anchor=(0, -0.15), fancybox=True, shadow=True)
                axs[i+2].legend(loc='upper center', bbox_to_anchor=(0, -0.15), fancybox=True, shadow=True)
#         axs[i].xaxis.set_ticks_position('none')

            axs[i].grid(True, axis='x', which='both')


        # if ci==0:return
        # self.fig.canvas.draw()
        # self.fig.show()
        plt.show()
        
        self.printEvals(evals)
        
        # plt.close(self.fig)
        


