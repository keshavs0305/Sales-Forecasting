import pandas as pd
from sklearn.preprocessing import StandardScaler
from customer_data import data_cus, new_data
import numpy as np
import xgboost as xgb
import datetime
import warnings
warnings.filterwarnings("ignore")

folder_path = "..\Sales-Forecasting\csv_files"
df = pd.read_csv(folder_path + "\df.csv")
df2 = pd.read_csv(folder_path + "\df2.csv")


def preds(cusid, f):
    scale = StandardScaler()
    data, n = data_cus(df, df2, cusid, f)
    data = scale.fit_transform(data.head(len(data) - n))

    model = xgb.XGBRegressor().fit(data[:, :-1], data[:, -1])

    x = scale.transform(np.array(new_data(cusid, f)).reshape(1, -1))
    pred = model.predict(x[:,:-1])

    r3 = scale.inverse_transform(np.array([0 for _ in range(12)] + [pred[0]]).reshape(1,-1))[-1]

    return int(r3[-1])


pred = {'cusid': [], 'pred': []}
c=0
for id_ in df['Custumer Id'].unique():
    if id_ in [903]:
        pred['cusid'].append(id_)
        pred['pred'].append(np.nan)
        continue
    pred['cusid'].append(id_)
    pred['pred'].append(preds(id_, 15))
    print(preds(id_, 15), id_)
    c += 1
    if c == 10:
        break

#pd.DataFrame(pred).to_csv('csv_files/preds_' + str(datetime.date.today()) + '.csv', index=False)
