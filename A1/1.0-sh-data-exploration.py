# import libraries
import numpy as np
import os
import pandas as pd

# environment settings
cwd = os.getcwd()
data = os.path.join(cwd, 'data', 'data.csv')

# read in data
df = pd.read_csv(data)

# fill in nan values with 0
df = df.replace(np.nan, 0)

# drop the date column: not required
df = df.drop('Data Year - Fiscal', axis = 1)

# get the shape of the data
print(df.shape)

# describe the data
print(df.describe())

# get the first 20 rows of the data
print(df.head(20))

# get the correlation between each variables
print(df.corr())

# write the cleansed data into csv
df.to_csv(os.path.join(cwd, 'data', 'sh-cleansed.csv'))
