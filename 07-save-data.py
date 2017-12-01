
# Load Data
import numpy as np
import urllib

url = "http://goo.gl/j0Rvxq"
raw_data = urllib.urlopen(url)
dataset = np.loadtxt(raw_data , delimiter=",")

print(dataset.shape)

# Format
import pandas as pd 
from pandas import DataFrame 

data = pd.DataFrame(dataset)

print data.head(4)

# Replace
data = data.replace(0, np.nan)

#pregnant nan to 0
data[0].fillna(0, inplace=True)

print data[0].head(10)

# target nan to -1
data[8].fillna(-1, inplace=True) 

# drop the attr columns with too many missing values
data = data.drop(3, 1) 
data = data.drop(4, 1) 
# drop all the data with missing values.
data=data.dropna()

data.info()

# store target to y, attrs to x
y=data[8] 
data = data.drop(8, 1) 
x = data

###
###standardize the data
from sklearn import preprocessing 
x_MinMax = preprocessing.MinMaxScaler() 
y_MinMax = preprocessing.MinMaxScaler()
y.as_matrix(y) 
y = np.array(y).reshape((len(y), 1)) 
x = np.array(x).reshape((len(x), 6))
x = x_MinMax.fit_transform(x) 
y = y_MinMax.fit_transform(y)
x.mean(axis=0)


