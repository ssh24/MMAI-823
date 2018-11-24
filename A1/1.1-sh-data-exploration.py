# import libraries
import numpy as np
import os
import pandas as pd

# environment settings
cwd = os.getcwd()
data = os.path.join(cwd, 'data', 'sh-cleansed.csv')

# read in data
df = pd.read_csv(data)

unique_ids = np.unique(df['cId'])

out_df = pd.DataFrame(columns = df.columns)

for i in range(0, len(unique_ids)):
    uId = unique_ids[i]
    new_df = df.loc[df['cId'] == uId]
    cols_nan = new_df.columns[new_df.isna().any()].tolist()
    for j in range(0, len(cols_nan)):
        mean = new_df[cols_nan[j]].mean()
        new_df[cols_nan[j]] = new_df[cols_nan[j]].replace(np.nan, mean)
        out_df = out_df.append(new_df)

# drop the date column: not required
out_df = out_df.drop('Data Year - Fiscal', axis = 1)

# drop not needed columns
out_df = out_df.drop('cId', axis = 1)
out_df = out_df.drop('Return on Equity', axis = 1)

# get head and tail
print(out_df.head())
print(out_df.tail())

# get shape
print(out_df.shape)

# get the correlation between each variables
print(out_df.corr())

# write the cleansed data into csv
out_df.to_csv(os.path.join(cwd, 'data', '1.0-sh-data-exploration.csv'), index=False)
