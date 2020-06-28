from Finding_dataframe_rows_with_reflectionphasedifference_180 import *
value_list = [10] # frequency = 10 GHz

index_list = df1.index[df1['frequency'] == 10].tolist()

df2 = df1.loc[index_list]
df2.reset_index(drop = True, inplace = True) # if inplace is defined no need to assign



for i in range(1,len(df2),1):
    #df2['reflectionphase_difference_%d' %i] = df2['reflection phase'].diff(periods = i) # difference between consecutive rows according to the period
    df2['reflectionphase_difference_%d' %i] = df2.loc[i:,'reflectionphase_unwrapped'] - df2.at[i-1,'reflectionphase_unwrapped']   
    
df2 = round(df2, 4) #need to assign for round() to work

df2 = df2.replace(np.nan,0) # need to assign to df2 inorder to save the changes

df2.to_excel('Reflection_phase_difference_radians_between_rows.xlsx')  


#print(df2[df2['reflectionphase_difference_322'] == -0.5806].index.values)

def getIndexes(dfobj, value):
    listOfPosition = list()
    result = dfobj.isin([value])
    #print('result' + result)
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    print(columnNames)
    return result

listOfPositions = getIndexes(df2,value = -0.5806)
print('Index positions of the value in dataframe :',listOfPositions)

#for i in range(len(listOfPositions)):
    #print('Position', i , '(Row index, Column Name) : ', listOfPositions[i])
