import statistics
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pyplot as plt
from IPython.display import display

#plot settings
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


def plot(rhr,alerts,info,title='',file=None,show=False):

    
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

    rhr24=rhr.resample('24H').mean()
    rhr24median = rhr24.expanding().median()#.astype(int)
    
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
        for d,row in alerts.iterrows():
            v=row['alarm']
            if(v>0):
                plt.axvline(x=d, linestyle='-', color='red' if v==2 else 'yellow', linewidth=1)
    
    
    if(covid_test_date!=None):
        plt.axvspan(covid_test_date, min(rhr24.index[-1], covid_test_date+pd.to_timedelta('14d')), facecolor='red', alpha=0.25, label='Covid')

    if(symptom_date!=None):
        maxSymthom = min(rhr24.index[-1], symptom_date+pd.to_timedelta('14d'))
        if(covid_test_date != None and symptom_date<covid_test_date):
            maxSymthom = min(covid_test_date, maxSymthom)
        plt.axvspan(symptom_date, maxSymthom, facecolor='yellow', alpha=0.25,label='Sympthom')
    if(window != None):
        ax.axvspan(window[0], window[1], facecolor='green', alpha=0.25, label='Acceptable range')

    #Title & Symptom Onset & Save plot
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

    #uncomment for legend
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



def plotAll(rhr,alerts,info,file=None,show=False):
    
    
    window = None
    symptom_date = info['symptom_date']
    covid_test_date = info['covid_test_date']
    if(symptom_date != None):
        window = [symptom_date-pd.to_timedelta('14d'), symptom_date]
    elif covid_test_date != None:
        window = [covid_test_date-pd.to_timedelta('14d'), covid_test_date]
    #################################  Plot  #################################
    # print(f"Plotting...covid_test_date={covid_test_date},symptom_date={symptom_date}, window={window}")
    ax = plt.subplot(211)
    # ax2 = plt.subplot(212, sharex=ax)
    ax2 = ax.twinx()
    rhr.to_csv('a.csv')

    rhr24=rhr.resample('24H').mean()
    rhr24median = rhr24.expanding().median()#.astype(int)
    
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
    for i,method in enumerate(alerts.columns):
        red = alerts[[method]][alerts[method]==2]
        red['i']=i
        yellow = alerts[[method]][alerts[method] == 1]
        yellow['i']=i
        for k,row in red.iterrows():
            ax2.scatter(x=k, y=row['i'], color='red')    
        for k, row in yellow.iterrows():
            ax2.scatter(x=k, y=row['i'], color='yellow')
        # ax2.scatter(x=list(red.index), y=red['i'], color='red')
        # ax2.scatter(x=list(yellow.index), y=yellow['i'], color='yellow')
    
    ax2.set_yticks(range(len(alerts.columns)), labels=alerts.columns)
    ax2.grid(True)
    
    if(covid_test_date!=None):
        ax.axvspan(covid_test_date, min(rhr24.index[-1], covid_test_date+pd.to_timedelta('14d')), facecolor='red', alpha=0.25, label='Covid')
        

    if(symptom_date!=None):
        maxSymthom = min(rhr24.index[-1], symptom_date+pd.to_timedelta('14d'))
        if(covid_test_date != None and symptom_date<covid_test_date):
            maxSymthom = min(covid_test_date, maxSymthom)
        ax.axvspan(symptom_date, maxSymthom, facecolor='yellow', alpha=0.25,label='Sympthom')
        

    if(window != None):
        ax.axvspan(window[0], window[1], facecolor='green', alpha=0.25, label='Acceptable range')
        

    #Title & Symptom Onset & Save plot
    # ax.set_xticks(rotation=90)
    ax.set_ylim(int(min_)-1, int(max_)+1)
    h = ax.set_ylabel('Resting\n heart rate\n over night', fontsize=12)
    # ax.xlabel('    Day', fontsize=12)
    h.set_rotation(90)
    ax.set_yticks(np.arange(round10Base(min_), round10Base(max_), 10))

    ax.set_xticks(rhr24.index)
    ax.set_xticklabels(labels=rhr24.index, rotation=90)
    
    # ax2.set_xticklabels(rhr24.index, rotation=90)
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 3 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)

    #uncomment for legend
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
    figure.set_size_inches(16, 10)
    # plt.legend()
    # ax.set_title(title)
    
    if file:
        plt.gcf().savefig(file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    
    plt.close()
