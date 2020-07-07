import scipy.optimize as opt
import numpy as np
import math
import keras
from keras.models import load_model
import pandas as pd
import time
import matplotlib.pyplot as plt

pi = np.pi

theta_= np.linspace(0,pi/2,90)
phi_= np.linspace(0,2*pi,360)
theta,phi = np.meshgrid(theta_,phi_) # Makind 2D grid

df_v1 = pd.read_excel('ReflectionPhase_1_openingangle_10_length_6.xlsx') # dimensions of '0'th element V1
df_v2 = pd.read_excel('ReflectionPhase_1_openingangle_85_length_10.xlsx') # dimensions of '0'th element V2
lambda0 = pd.DataFrame([])
lambda0 = 1/(df_v1["frequency"].div(3*10^8))
k = 2*pi/lambda0
D = lambda0

#omsriramajayam

N = 10 # 10 by 10 array of 2 different v-shaped element
df = pd.DataFrame([])
df1 = pd.DataFrame([])

result = []

df_v1["reflectionphase_unwrapped"] = np.unwrap((np.deg2rad(df_v1["reflectionphase"])) % 2*np.pi) # modulo 2*pi helps to change range from -pi to +pi to 0 to 2*pi
df_v2["reflectionphase_unwrapped"] = np.unwrap((np.deg2rad(df_v2["reflectionphase"])) % 2*np.pi)

def fun(x,i):
    L_v = x[:N**2]
    theta_v = x[N**2:]
    phase_pred = []
    for t,l in zip(theta_v,L_v):
        if((t == 0) & (l == 0)):
            phase_pred.append(df_v1["reflectionphase_unwrapped"][i])
        if((t == 1) & (l == 1)):
            phase_pred.append(df_v2["reflectionphase_unwrapped"][i])   
    print(phase_pred)
        #omsriramajayam
    reflection_phase = np.reshape(phase_pred, (N,N))
    #omsriramajayam
    S1 = []
    S = 0
    for m in range(N):
        for n in range(N):    
            S =  S + np.exp(-1j * (reflection_phase[m,n] + k[i]*D[i]*np.sin(theta)*((m-1/2)*np.cos(phi)+((n-1/2)*np.sin(phi)))))
            #S = 0.5*((np.cos((k*L_v[m,n]/2)*np.sin(theta)*np.sin(phi)) - np.cos(k*L_v[m,n]/2))/(np.sqrt(1-(np.sin(theta)*np.sin(theta)*np.sin(phi)*np.sin(phi))))) * S
    #S = ((np.cos((3*pi/4)*np.sin(theta)*np.sin(phi)) - np.cos(3*pi/4))/(np.sqrt(1-(np.sin(theta)*np.sin(theta)*np.sin(phi)*np.sin(phi))))) * S
    S = np.cos(theta) * S
    #S1.append(S)
    H = np.trapz(np.trapz(np.abs(S)**2*np.sin(theta),theta_),phi_) # integration using trapezoid function
    directivity = 4 * pi * np.abs(S)**2 / H
    rcs = 10 * np.log10((1/(4*pi*N**2)) * np.max(directivity)) 
    return rcs
#omsriramajayam

#dataframe = pd.read_excel('Length_Openingangle_V_Elements_Pattern_Design.xlsx')
state_list = []  
number_of_frequency_points = len(df_v1)
number_of_combinations = 5
x = np.zeros((number_of_combinations,200))  #Initialise numpy array x
np.random.seed(30)

rcs_over_frequency = {} #pd.DataFrame([])
list_of_rcs_over_frequency = pd.DataFrame([])


for times in range(number_of_combinations):#(dataframe.shape[0]):# number of instances
    state = pd.DataFrame(np.random.randint(2, size=(1,100)))
    for i in range(number_of_frequency_points):
        x[times][:N**2] = state #np.array((t,l)).ravel() 
        x[times][N**2:] = state
        rcs_over_frequency['Combination_number_%d_%%d' %times %i] = fun(x[times],i)
        list_of_rcs_over_frequency.loc['%d' %times ,'%d' %i] = fun(x[times],i)
    state_list.append(state)
        #print(result)
pd.DataFrame(state_list).to_excel('random_combination_of_one_and_zero.xlsx', header = None, index = False)
df1 = pd.DataFrame(state_list)

list_of_rcs_over_frequency.to_excel("RCS_over_all_frequencies_for_random_combinations.xlsx") 


