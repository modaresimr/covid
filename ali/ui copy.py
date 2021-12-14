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


def round10Base(n):
    a = (n // 10) * 10
    b = a + 10
    return (b if n - a > b - n else a)


def plot(rhr,alerts,covid_test_date,symptom_date):

    
    window = None
    if(symptom_date != None):
        window = [symptom_date-pd.to_timedelta('21d'), symptom_date]
    elif covid_test_date != None:
        window = [covid_test_date-pd.to_timedelta('21d'), covid_test_date]
    #################################  Plot  #################################
    print(f"Plotting...covid_test_date={covid_test_date},symptom_date={symptom_date}, window={window}")
    figure = plt.gcf()
    ax = plt.gca()
    rhr.to_csv('a.csv')

    rhr24=rhr.resample('24H').mean()
    rhr24median = rhr24.expanding().median().astype(int)
    
    missings = rhr24.index[rhr24['heartrate'].isnull()]
    missings_estimate=rhr24.interpolate().loc[missings]
    haveData = rhr24.index[~rhr24['heartrate'].isnull()]
    allX=rhr24.index
    allX_value = rhr24.loc[haveData]
    min = allX_value['heartrate'].min()
    max = allX_value['heartrate'].max()

    date_set = rhr24.index

    sorted_allX = sorted(allX)
    # plt.plot(sorted_allX, range(len(sorted_allX)), color='white')

    #plot average rhr per night
    
    plt.plot(rhr24.index, rhr24['heartrate'], color='black',  marker='o',  markersize=2.5, linewidth=1.5, label="Avg RHR over night")

    #plot average rhr per night for missing nights
    # plt.plot(missings, missings_estimate['heartrate'], color='white', marker='o', linestyle='', markersize=3, markerfacecolor='gray', markeredgecolor='gray', label="Imputed Avg RHR \n over night")

    #plot median rhr per night
    plt.plot(rhr24median.index, rhr24median['heartrate'], color='green', linewidth=1.5, linestyle='dashed', label="Med RHR over night")

    plt.plot(rhr24median.index, rhr24median['heartrate']+3, color='yellow', linewidth=1, linestyle='dashed', label="Med RHR over night + 3")
    plt.plot(rhr24median.index, rhr24median['heartrate']+4, color='red', linewidth=1, linestyle='dashed', label="Med RHR over night + 4")

    #find consecutive red and yellow days
    
    newJoin = rhr24.join(rhr24median.rename(columns={'heartrate': 'hr_median'})).sort_index()
    red_and_yellow_alert_dates = newJoin[newJoin['heartrate'] > newJoin['hr_median']+3].index
    red_alert_dates = newJoin[newJoin['heartrate'] > newJoin['hr_median']+4].index
    yellow_alert_dates = newJoin[(newJoin['heartrate'] <= newJoin['hr_median']+4) & (newJoin['heartrate'] > newJoin['hr_median']+3)].index
    
    sorted_red_and_yellow_alert_dates = sorted(red_and_yellow_alert_dates)

    allalarms=newJoin.copy()
    # display(red_and_yellow_alert_dates)
    allalarms['red_alarm'] = 0
    allalarms.loc[red_and_yellow_alert_dates, 'red_alarm'] = 1
    # print(allalarms)
    # newJoin['yellow_alarm'].loc[yellow_alert_dates] = 1
    allalarms=allalarms[['red_alarm']].rolling(3).sum()
    red_alarms = allalarms[allalarms['red_alarm']>=3]
    
    for d in red_alarms.index:
        plt.axvline(x=d, linestyle='-', color='red', linewidth=1)
    
    if(symptom_date!=None):
        plt.axvspan(symptom_date, symptom_date+pd.to_timedelta('7d'), ymin=0, ymax=500, facecolor='yellow', alpha=0.25,label='Sympthom')
    if(covid_test_date!=None):
        plt.axvspan(covid_test_date, covid_test_date+pd.to_timedelta('7d'), ymin=0, ymax=500, facecolor='red', alpha=0.25, label='Covid')
    if(window != None):
        ax.axvspan(window[0], window[1], facecolor='green', alpha=0.25, label='Acceptable range')

    # plt.axhspan(xmin=window[0], xmax=window[1], facecolor='red')
    # ax.axvspan(window[0],window[1])
    # plt.axvspan(window[0],window[1])
    # plt.axhspan(window[0], window[1])

        # x, y = zip(*sorted_res)
        # plt.plot(x, y, color='white', marker='o' , linestyle='' , markersize=3.5 , markerfacecolor='red' , markeredgecolor='red' , label="Red alert")

    #plot yellow alerts
    # plt.axvline(x=key, linestyle='-', color='orange' , linewidth=1)
      
    #Title & Symptom Onset & Save plot
    plt.xticks(rotation=90)
    plt.ylim(int(min)-1, int(max)+1)
    h = plt.ylabel('Resting\n heart rate\n over night', fontsize=12)
    plt.xlabel('    Day', fontsize=12)
    h.set_rotation(90)
    plt.yticks(np.arange(round10Base(min), round10Base(max), 10))

    plt.xticks(rhr24.index)
    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 3 == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)

    #uncomment for legend
    lgd = ax.legend(prop={'size': 8.5}, bbox_to_anchor= (1.0, 1.0), loc="upper left", frameon=False)
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
    plt.show()
    # plt.savefig("NightSignalResult2" + '.png', dpi=300, bbox_inches= "tight")
    # plt.close()
