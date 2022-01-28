import pandas as pd
def IsolationForest(args):
    rhr=args['rhr']
    from sklearn.ensemble import IsolationForest
    df = rhr.resample("1d").mean().reset_index()
    df['heartrate'] = df['heartrate'].interpolate()
    model = IsolationForest(contamination="auto")
    model.fit(df[['heartrate']])
    df['alarm'] = pd.Series(model.predict(df[['heartrate']])).apply(lambda x: 1 if (x == -1) else 0)
    
    
    allalarms = df.set_index('datetime')[['alarm']]
    allalarms['cum'] = allalarms.rolling(2).sum()
    allalarms['alarm']=allalarms['alarm']*allalarms['cum']
    
    return allalarms[['alarm']].dropna()


def IsolationForest2(args):

    rhr=args['rhr']
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
    return allalarms[['alarm']].dropna()


