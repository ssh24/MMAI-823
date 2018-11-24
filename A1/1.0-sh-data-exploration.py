# import libraries
import numpy as np
import os
import pandas as pd

# environment settings
cwd = os.getcwd()
data = os.path.join(cwd, 'data', 'raw-data.csv')

# read in data
df = pd.read_csv(data)

rows = len(df.index)

id = 1
df['cId'] = id

for i, row in df.iterrows():
    if i + 1 < rows:
        curr_date = df['Data Year - Fiscal'][i]
        next_date = df['Data Year - Fiscal'][i+1]
        if (next_date <= curr_date):
            df.loc[i, 'cId'] = id
            id = id + 1
        else:
            df.loc[i, 'cId'] = id
    else:
        df.loc[i, 'cId'] = id

# write the cleansed data into csv
df.to_csv(os.path.join(cwd, 'data', 'sh-cleansed.csv'), index=False)
