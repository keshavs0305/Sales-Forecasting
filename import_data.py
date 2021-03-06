import pandas as pd
from sklearn import preprocessing


filepath = r'C:\Users\MADHAV\PycharmProjects\Sales-Forecasting\csv_files\2 years data before 29-09-2021.xlsm'
df1 = pd.read_excel(filepath, '2 years data before 29-09-2021')
df2 = pd.read_excel(filepath, 'Price increase dates')
df3 = pd.read_excel(filepath, 'Mandatory vacation dates', names=['start', 'end'])


df3.drop([0], inplace=True)
df1.drop(['Order nº', 'Custumer Name', 'Item Name', 'Subgroup_description'], axis=1, inplace=True)
df2.drop(['Subgroup of products (description)'], axis=1, inplace=True)

le = preprocessing.LabelEncoder()
for col in ['Country', 'State', 'City']:
    df1[col] = le.fit_transform(df1[[col]])
df = df1.groupby(['Order date', 'Custumer Id', 'Item Id', 'Country', 'State', 'City', 'Subgruop_ID']) \
    .mean().reset_index()

folder_path ="..\Sales-Forecasting\csv_files"
df1.to_csv(folder_path + '\df1.csv', index=False)
df2.to_csv(folder_path + '\df2.csv', index=False)
df3.to_csv(folder_path + '\df3.csv', index=False)
df.to_csv(folder_path + '\df.csv', index=False)
