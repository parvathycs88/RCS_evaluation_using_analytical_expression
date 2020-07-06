import scipy.optimize as opt
import numpy as np
import math
import keras
from keras.models import load_model
import pandas as pd
import time
import matplotlib.pyplot as plt

lambda0 = 0.03
pi=np.pi
k = 2*pi/lambda0
D = 0.03 #d=0.015 used for simulation by haoyang #lambda0/2
theta_= np.linspace(0,pi/2,90)
phi_= np.linspace(0,2*pi,360)
theta,phi = np.meshgrid(theta_,phi_) # Makind 2D grid

#omsriramajayam

N = 10 # 10 by 10 array of 2 different v-shaped element
df = pd.DataFrame([])
df1 = pd.DataFrame([])

result = []

def fun(x):
    L_v = x[:N**2]
    theta_v = x[N**2:]
    phase_pred = []
    for t,l in zip(theta_v,L_v):
        if((t == 0) & (l == 0)):
            phase_pred.append(-68.5282)
        if((t == 1) & (l == 1)):
            phase_pred.append(117.1988)   
    print(phase_pred)
        #omsriramajayam
    reflection_phase = np.reshape(phase_pred, (N,N))
    #omsriramajayam
    S1 = []
    S = 0
    for m in range(N):
        for n in range(N):    
            S =  S + np.exp(-1j * (reflection_phase[m,n] + k*D*np.sin(theta)*((m-1/2)*np.cos(phi)+((n-1/2)*np.sin(phi)))))
            #S = 0.5*((np.cos((k*L_v[m,n]/2)*np.sin(theta)*np.sin(phi)) - np.cos(k*L_v[m,n]/2))/(np.sqrt(1-(np.sin(theta)*np.sin(theta)*np.sin(phi)*np.sin(phi))))) * S
    #S = ((np.cos((3*pi/4)*np.sin(theta)*np.sin(phi)) - np.cos(3*pi/4))/(np.sqrt(1-(np.sin(theta)*np.sin(theta)*np.sin(phi)*np.sin(phi))))) * S
    S = np.cos(theta) * S
    #S1.append(S)
    H = np.trapz(np.trapz(np.abs(S)**2*np.sin(theta),theta_),phi_) # integration using trapezoid function
    directivity = 4 * pi * np.abs(S)**2 / H
    rcs = (1/(4*pi*N**2)) * np.max(directivity)  
    return rcs
#omsriramajayam

#dataframe = pd.read_excel('Length_Openingangle_V_Elements_Pattern_Design.xlsx')
state_list = []  
x = np.zeros((5,200))  #Initialise numpy array x
for times in range(5):#(dataframe.shape[0]):# number of instances
    #t = ((np.random.random((N**2)))*pi/2).tolist()
    #l = ((np.random.random((N**2)))*(11-6) + 6).tolist()
    #for i in range(N**2):
        #df1['OpeningAngle_%s'%str(i)] = [round(t[i],3)]
        #df1['Length_%s'%str(i)] = [round(l[i],3)]    
    #x = dataframe.iloc[[times]].to_numpy()    
    state = pd.DataFrame(np.random.randint(2, size=(1,100)))
    
    
    x[times][:N**2] = state #np.array((t,l)).ravel() 
    x[times][N**2:] = state
    result.append(fun(x[times]))
    state_list.append(state)
    print(result)
pd.DataFrame(state_list).to_excel('random_combination_of_one_and_zero.xlsx', header = None, index = False)
df1 = pd.DataFrame(state_list)
#df[0:100] = state
df = pd.DataFrame(result, columns = ['RCS'])  #df.append(df1)
#df.reset_index(drop=True)
#print(t,l,fun(x))
df.to_excel("Database_with_RCS_explicit_for_loop.xlsx", sheet_name='Sheet_name_1', index = False)
pd.concat([df1,df], axis = 1).to_excel('Database_with_random_combination_of_ones_and_zeros_with_overall_RCS.xlsx', index = False)
