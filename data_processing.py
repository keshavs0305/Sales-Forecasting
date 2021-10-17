import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

folder_path = "..\Sales-Forecasting\csv_files"
df = pd.read_csv(folder_path + "\df.csv")


def time_series_bin(right, col):
    ret = pd.DataFrame(pd.date_range(df['Order date'].min(), df['Order date'].max(), freq='d'), columns=['Date'])
    try:
        new_data = pd.merge(ret, right, left_on='Date', right_on=col, how='outer')
        print(len(new_data),'new data')
    except:
        new_data = pd.concat([ret, right], axis=1, join="outer")
        print(len(new_data), 'new data except')
    new_data['ret'] = new_data[col].isna() == 0
    new_data['ret'].replace({True: 1, False: 0}, inplace=True)
    return new_data['ret']


def recency(dfr, cusid):
    t_ = time_series_bin(dfr.loc[dfr['Custumer Id'] == cusid][['Order date']].drop_duplicates(), 'Order date')
    t = np.where(t_.to_numpy() == 1)[0]
    t1 = [t[i] - t[i + 1] for i in range(len(t) - 1)]
    tt = [0 - t[0]]
    [tt.append(i) for i in t1]
    tt.append(t[-1] - len(t_))
    data = []
    for rec in tt:
        [data.append(_) for _ in range(abs(rec))]
    return data


def buy_freq(dfr, cusid):
    t = time_series_bin(dfr.loc[dfr['Custumer Id'] == cusid][['Order date']].drop_duplicates(), 'Order date')
    ret = [0]
    [ret.append(t[:i + 1].mean()) for i in range(len(t) - 1)]
    return ret


def price_inc_30(df1, df2):
    df2 = pd.concat(
        [df2[['Date of price increase']],
         pd.DataFrame.sparse.from_spmatrix(
             OneHotEncoder().fit_transform(df2.drop(['Date of price increase'], axis=1)))],
        axis=1)
    df2 = df2.groupby(['Date of price increase']).sum()

    x = pd.DataFrame(index=pd.date_range(df1['Order date'].min(), df1['Order date'].max(), freq='d'))
    x = pd.concat([x, df2], axis=0)

    x.fillna(0, inplace=True)
    x.index = pd.to_datetime(x.index,)
    x = x[~x.index.duplicated(keep='last')]
    x.sort_index(inplace=True)

    return x


def check_next_n(dfp, n):
    for col in dfp.columns:
        x = np.where(dfp[col].to_numpy() == 1)[0]
        ar = [0 for _ in range(x[0] - n)]
        for i in range(1, n + 1):
            ar.append(i)
        for i in range(x[1] - x[0] - n):
            ar.append(0)
        for i in range(1, n + 1):
            ar.append(i)
        for i in range(x[2] - x[1] - n):
            ar.append(0)
        for i in range(1, n + 1):
            ar.append(i)
        for i in range(len(dfp[col]) - len(ar)):
            ar.append(0)
        dfp[col] = ar
    return dfp


def next_buy(dfr, cusid):
    t_ = time_series_bin(dfr.loc[dfr['Custumer Id'] == cusid][['Order date']].drop_duplicates(), 'Order date')
    t = np.where(t_.to_numpy() == 1)[0]
    t1 = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    t2 = []
    tt = [t[0]]
    [tt.append(i) for i in t1]
    tt.append(len(t_) - t[-1])

    for i in range(len(tt) - 1):
        for j in range(tt[i]):
            t2.append(tt[i] - j - 1)

    x = len(t_) - len(t2)
    for i in range(len(t_) - len(t2)):
        t2.append(90)
    return t2, x
