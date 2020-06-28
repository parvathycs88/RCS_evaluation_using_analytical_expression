import scipy.optimize as opt
import numpy as np
import math
import keras
from keras.models import load_model
import pandas as pd
import time
import matplotlib.pyplot as plt

#nn = load_model('my_model.h5') # Loading the saved model
lambda0 = 0.03
pi=np.pi
k = 2*pi/lambda0
D = 0.03#d=0.015 used for simulation by haoyang #lambda0/2
theta_= np.linspace(0,pi/2,90)
phi_= np.linspace(0,2*pi,360)
theta,phi = np.meshgrid(theta_,phi_) # Makind 2D grid

#omsriramajayam

N = 10 # 10 by 10 array of 2 different v-shaped element
df = pd.DataFrame([])
df1 = pd.DataFrame([])


#def ml(theta, L):
    #x00 = np.array([[[theta], [L], [10]]]) # 10 corresponds to third argument frequency. In this case restricting to 10 GHz
    #out = nn.predict(x00, batch_size=1000)[0,0]
    #return out

result = []

def fun(x):
    L_v = x[:N**2]
    theta_v = x[N**2:]
    phase_pred = []
    for t,l in zip(theta_v,L_v):
        if((t == 10) & (l == 6)):
            phase_pred.append(2.0207)#117.1988)
        if((t == 85) & (l == 10)):
            phase_pred.append(0.1734)#-68.5282)   
    print(phase_pred)
        #omsriramajayam
    reflection_phase = np.reshape(phase_pred, (N,N))
    S = 0
    for m in range(N):
        for n in range(N):    
            S =  (S + np.exp(-1j * (reflection_phase[m,n] + k*D*np.sin(theta)*((m-1/2)*np.cos(phi)+((n-1/2)*np.sin(phi))))))
            #S = 0.5*((np.cos((k*L_v[m,n]/2)*np.sin(theta)*np.sin(phi)) - np.cos(k*L_v[m,n]/2))/(np.sqrt(1-(np.sin(theta)*np.sin(theta)*np.sin(phi)*np.sin(phi))))) * S
    #S = ((np.cos((3*pi/4)*np.sin(theta)*np.sin(phi)) - np.cos(3*pi/4))/(np.sqrt(1-(np.sin(theta)*np.sin(theta)*np.sin(phi)*np.sin(phi))))) * S
    S = np.cos(theta) *S
    H = np.trapz(np.trapz(np.abs(S)**2*np.sin(theta),theta_),phi_) # integration using trapezoid function
    directivity = 4 * pi * np.abs(S)**2 / H
    rcs = (1/(4*pi*N**2)) * np.max(directivity)
    #result.append(rcs) 
    return rcs 

dataframe = pd.read_excel('Length_Openingangle_V_Elements_Pattern_Design.xlsx', header = None)
x = np.zeros((100,200))  #Initialise numpy array x
for times in range(dataframe.shape[0]):# number of instances
    #t = ((np.random.random((N**2)))*pi/2).tolist()
    #l = ((np.random.random((N**2)))*(11-6) + 6).tolist()
    #for i in range(N**2):
        #df1['OpeningAngle_%s'%str(i)] = [round(t[i],3)]
        #df1['Length_%s'%str(i)] = [round(l[i],3)]    
    #x = dataframe.iloc[[times]].to_numpy()    
    x[times][:N**2] = dataframe.iloc[times,:N**2].to_numpy()#np.array((t,l)).ravel() 
    x[times][N**2:] = dataframe.iloc[times,N**2:].to_numpy()   
    result.append(fun(x[times]))
    print(result)

df["RCS"] = result #df.append(df1)
df["RCS in dB"] = 10 * np.log10(result)
df.reset_index(drop=True)
    #print(t,l,fun(x))
df.to_excel("Database_with_rcs.xlsx", sheet_name='Sheet_name_1')
