from itertools import product
import pandas as pd
import numpy as np
import glob 

dataframe = pd.DataFrame(columns = range(100)) # creating a dataframe with 100 columns

def get_combinations(n,m):
    for flat in product([0,1], repeat = n*m):
        yield np.reshape(flat,(n,m))

from itertools import islice
for m in islice(get_combinations(10,10),100):
    print(m)
    data_to_append = {}
     
    for i in range(len(dataframe.columns)):
        data_to_append[dataframe.columns[i]] = m[0:].ravel()[i]
    dataframe = dataframe.append(data_to_append, ignore_index = True)


dataframe.to_excel("Database_with_combinations_of_1_and_0.xlsx", sheet_name='Sheet1')
dataframe.to_excel("Database_with_combinations_of_1_and_0_length.xlsx", sheet_name='Sheet1')
dataframe.to_excel("Database_with_combinations_of_1_and_0_openingangle.xlsx", sheet_name='Sheet1')

# put in path to folder with files you want to append
# *.xlsx or *.csv will get all files of that type
path = "/home/parvathy/Desktop/RCS_analytical_expression/Database/*.xlsx"

# initialize a empty df
appended_data = pd.DataFrame()

#loop through each file in the path
for file in glob.glob(path):
    print(file)

    # create a df of that file path
    df = pd.read_excel(file, sheet_name = 0)
    #df = pd.read_csv(file, sep=',')

    # appened it
    appended_data = pd.concat([df, appended_data], axis = 1)#appended_data.append(df)
    df_appended = appended_data
    #df_appended.to_excel('Trial.xlsx')#('Length_Openingangle_V_Elements_Pattern_Design.xlsx')
