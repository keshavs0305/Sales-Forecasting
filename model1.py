import pandas as pd
from sklearn.preprocessing import StandardScaler
from customer_data import data_cus, new_data
import numpy as np
import xgboost as xgb

folder_path = "..\Sales-Forecasting\csv_files"
df = pd.read_csv(folder_path + "\df.csv")
df2 = pd.read_csv(folder_path + "\df2.csv")


def preds(cusid):
    scale = StandardScaler()
    data, n = data_cus(df, df2, cusid)
    data = scale.fit_transform(data.head(len(data) - n))

    # train, test = data[:round(len(data)*0.8)], data[round(len(data)*0.8):]\n",
    # x_train, x_test, y_train, y_test = train[:,:-1], test[:,:-1], train[:,-1], test[:,-1]\n",

    # pred = [np.append(x_test[i,:],model.predict(x_test[i,:].reshape(1,-1))) for i in range(len(x_test))]\n",
    # true = [np.append(x_test[i,:],y_test[i].reshape(1,-1)) for i in range(len(x_test))]\n",
    # plt.plot(scale.inverse_transform(pred)[:,-1],label='preds')\n",
    # plt.plot(scale.inverse_transform(true)[:,-1])\n",
    # plt.legend()\n",
    # plt.show()\n",

    model = xgb.XGBRegressor().fit(data[:, :-1], data[:, -1])
    data, n = data_cus(df, df2, cusid)
    pred = model.predict(scale.transform(data)[-1][:-1].reshape(1, -1))

    r3 = scale.inverse_transform([0 for _ in range(12)] + [pred[0]])[-1]

    return r3


pred = {'cusid': [], 'pred': []}
for id_ in df['Custumer Id'].unique():
    if id_ in [903]:
        pred['cusid'].append(id_)
        pred['pred'].append(np.nan)
        continue
    pred['cusid'].append(id_)
    pred['pred'].append(preds(id_))

pd.DataFrame(pred).to_csv('csv_files/sample.csv', index=False)
