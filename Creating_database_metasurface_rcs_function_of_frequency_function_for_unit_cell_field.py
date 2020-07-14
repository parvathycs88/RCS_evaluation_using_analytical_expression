import scipy.optimize as opt
import numpy as np
import math
import keras
from keras.models import load_model
import pandas as pd
import time
import matplotlib.pyplot as plt

N = 10
pi=np.pi

df_10_6 = pd.read_excel('Efield_10degree_6mm.xlsx')
df_85_10 = pd.read_excel('Efield_85degree_10mm.xlsx')

theta_=  np.arange(0,pi,0.087266).round(4) #np.linspace(0,pi,10).round(4) #theta_= np.linspace(0,pi/2,90)
phi_=  np.arange(0,6.1959,0.087266).round(4) #np.linspace(0,6.1959,10).round(4)  #phi_= np.linspace(0,2*pi,360)

theta_value = df_10_6["Theta_radians"]
phi_value = df_10_6["Phi_radians"]

theta,phi = np.meshgrid(theta_,phi_) #np.meshgrid(theta_,phi_) # Makind 2D grid


df_v1 = pd.read_excel('ReflectionPhase_1_openingangle_10_length_6.xlsx') # dimensions of '0'th element V1
df_v2 = pd.read_excel('ReflectionPhase_1_openingangle_85_length_10.xlsx') # dimensions of '0'th element V2
lambda0 = pd.DataFrame([])
lambda0 = 1/(df_v1["frequency"].div(3*10^8))
k = 2*pi/lambda0
D = lambda0


df_v1["reflectionphase_unwrapped"] = np.unwrap((np.deg2rad(df_v1["reflectionphase"])) % 2*np.pi) # modulo 2*pi helps to change range from -pi to +pi to 0 to 2*pi
df_v2["reflectionphase_unwrapped"] = np.unwrap((np.deg2rad(df_v2["reflectionphase"])) % 2*np.pi)

#omsriramajayam

def fun(x,i, angle1,angle2):
    L_v = x[:N**2]
    theta_v = x[N**2:]
    phase_pred = []
    for t,l in zip(theta_v,L_v):
        if((t == 10) & (l == 6)):
            phase_pred.append(df_v1["reflectionphase_unwrapped"][i])
        if((t == 85) & (l == 10)):
            phase_pred.append(df_v2["reflectionphase_unwrapped"][i])   
    #print(phase_pred)
        #omsriramajayam
    reflection_phase = np.reshape(phase_pred, (N,N))
    #phase_pred = np.repeat(df["reflectionphase_in_radians"][i], N**2)
    #reflection_phase = np.reshape(phase_pred, (N,N))
    result = []
    S = 0 #np.zeros((len(theta_),len(phi_)))
    #H = np.zeros((len(theta_),len(phi_)))
    #directivity = np.zeros((len(theta_),len(phi_)))
    #rcs = np.zeros((len(theta_),len(phi_)))
    #for angle1 in theta_value: 
        #for angle2 in phi_value: 
    #for angle1 in range(len(theta_)):
        #for angle2 in range(len(phi_)):    
    for m in range(N):
        for n in range(N):
            S =  S  + np.exp(-1j * (reflection_phase[m,n] + k[i]*D[i]*np.sin(theta_[angle1])*((m-1/2)*np.cos(phi_[angle2])+((n-1/2)*np.sin(phi_[angle2])))))
                    
            S = np.multiply(S,(df_10_6.loc[((df_10_6['Phi_radians'] == phi_[angle2]) & (df_10_6['Theta_radians'] == theta_[angle1])), 'Abs(E)']))   
            #print(S[int(theta_[angle1]),int(phi_[angle2])].shape)
            #omsriramajayam 
    H = np.abs(S)**2*np.sin(theta_[angle1]) #np.trapz(np.trapz(np.abs(S)**2*np.sin(theta_[angle1]),theta_),phi_) # integration using trapezoid function
    directivity = 4 * pi * np.abs(S)**2 / H
    rcs = (1/(4*pi*N**2)) * np.max(directivity)  
    result.append(10 * np.log10(rcs)) 
    print(len(result))
    return result
    
    
rcs_over_frequency = {} 
list_of_rcs_over_frequency = []
list_for_many_combinations = pd.DataFrame([])
 

dataframe = pd.read_excel('Length_Openingangle_V_Elements_Pattern_Design.xlsx', header = None)
#x = np.zeros((99,200))  #Initialise numpy array x

    #k[i] = 2*pi/lambda0[i]
    #D[i] = lambda0[i]#0.03#d=0.015 used for simulation by haoyang #lambda0/2
x = np.zeros((100,200))  #Initialise numpy array x
number_of_frequency_points = 10#len(df_v1)
Combination_number = 3
#rcs_over_frequency = np.zeros((Combination_number,len(phi_)))#pd.DataFrame([]) #{}

for times in range(Combination_number):#dataframe.shape[0]):  
    for i in range(number_of_frequency_points):#len(df_v1)): 
        for angle1 in range(len(theta_)):
            for angle2 in range(len(phi_)):     
                x[times][:N**2] = dataframe.iloc[times,:N**2].to_numpy()#np.array((t,l)).ravel() 
                x[times][N**2:] = dataframe.iloc[times,N**2:].to_numpy()   
        #print(type(fun(x[times],i)))                
                rcs_over_frequency['Combination_number_%d_%d_%d' %(times,angle1,angle2)] = fun(x[times],i,int(theta_[angle1]),int(phi_[angle2]))
                list_of_rcs_over_frequency.extend(rcs_over_frequency['Combination_number_%d_%d_%d' %(times,angle1,angle2)])
        #S4 = fun(x[times],i,angle1,angle2)[1]
                
for j in range(Combination_number):
    list_for_many_combinations['Combination_number_%d' %j] = list_of_rcs_over_frequency[(number_of_frequency_points*len(theta_)*len(phi_))*j:(number_of_frequency_points*len(theta_)*len(phi_))*j+(number_of_frequency_points*len(theta_)*len(phi_))]#[len(df_v1)*j:len(df_v1)*j+len(df_v1)]

#list_for_many_combinations.to_excel("RCS_over_all_frequencies_for_all_combinations.xlsx") 

for k in range(list_for_many_combinations.shape[1]):
    plt.figure()
    plt.xlabel("Frequency GHz")
    plt.ylabel("RCS in dB")
    plt.title("RCS for %d combination of V1 and V2 from 6GHz to 14GHz \n" %k, loc = 'right')
    plt.plot(np.repeat([df_v1["frequency"][0:number_of_frequency_points]],len(theta_)*len(phi_)*number_of_frequency_points,0) ,list_for_many_combinations['Combination_number_%d' %k])
    #plt.savefig("RCS over frequency considering unit field function for Combination_number_%d_%%d.png" %k %number_of_frequency_points)
plt.show()
plt.ion() # helps to come to next line in command window without cosing figures
    



   
