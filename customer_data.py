import pandas as pd
from data_processing import recency, buy_freq, price_inc_n, check_next_n, next_buy

folder_path = "..\Sales-Forecasting\csv_files"
df = pd.read_csv(folder_path + "\df.csv")
df2 = pd.read_csv(folder_path + "\df2.csv")


def data_cus(df1, df2, cusid, n):
    ret = pd.DataFrame()
    ret['recency'] = recency(df1, cusid)
    ret['buy_freq'] = buy_freq(df1, cusid)
    dft = price_inc_n(df1, df2, n)
    for col in dft.columns:
        ret['price_inc_' + str(col)] = list(dft[col])

    ret['next_buy'], n = next_buy(df1, cusid)

    return ret, n


def new_data(cusid, f):
    t, n = data_cus(df, df2, cusid, f)
    recency = t.recency.values[-1] + 1
    buy_freq = t.buy_freq.values[-1] * len(t) / (len(t) + 1)
    return [recency, buy_freq] + [0 for _ in range(10)] + [0]
