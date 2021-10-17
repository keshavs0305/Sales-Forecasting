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
    #data, n = data_cus(df, df2, cusid)
    x = scale.transform(np.array(new_data(cusid)).reshape(1, -1))
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
    pred['pred'].append(preds(id_))
    print(preds(id_), id_)
    c += 1
    if c == 10:
        break

#pd.DataFrame(pred).to_csv('csv_files/preds_' + str(datetime.date.today()) + '.csv', index=False)
