import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['agg.path.chunksize'] = 10000
import tensorflow as tf
import numpy as np
import copy
import itertools
import seaborn as sns
import os
import time
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
#import tensorflow.keras.backend as K
from keras.backend.tensorflow_backend import set_session
import glob
from keras.callbacks import EarlyStopping, ModelCheckpoint

config = tf.ConfigProto()#tf.compat.v1.ConfigProto() # tensorflow version 2.1.0 

# put in path to folder with files you want to append
# *.xlsx or *.csv will get all files of that type
path = "/home/parvathy/Desktop/RCS_analytical_expression/ReflectionPhase/*.xlsx"

# initialize a empty df
appended_data = pd.DataFrame()

#loop through each file in the path
for file in glob.glob(path):
    print(file)

    # create a df of that file path
    df = pd.read_excel(file, sheet_name = 0)
    #df = pd.read_csv(file, sep=',')

    # appened it
    appended_data = appended_data.append(df)
    df1 = appended_data.reset_index(drop = True) #Reset index helps to avoid same index values as two excel files are appended

openingangle_list = df1["openingangle"].unique().tolist()
length_list = df1["length"].unique().tolist()

# fix random seed for reproducibility
#seed = 7
#np.random.seed(seed)

df1.columns = df1.columns.str.replace(' ','')
input_features = list(df1.columns)
input_features.remove('reflectionphase')

d1 = {} # dictionary is required for unwrapping the phase of every combination of opening angle and length separately
for i in openingangle_list:
    for j in length_list:
        d1['df1_openingangle_length_%d_%%.2f' %i %j] = df1.loc[(df1["openingangle"] == i) & (df1["length"] == j)]    

for key in d1.keys():
    #print(key)
    phase=np.deg2rad(d1[key]["reflectionphase"]) # np.unwrap() accepts as input angle in radians
    phase_unwrapped=np.unwrap(phase)
    d1[key]["reflectionphase_unwrapped_radians"]=phase_unwrapped

reflectionphase_list = []
for key in d1.keys():
    reflectionphase_list.extend(d1[key]["reflectionphase_unwrapped_radians"])

df1["reflectionphase_unwrapped"] = reflectionphase_list
