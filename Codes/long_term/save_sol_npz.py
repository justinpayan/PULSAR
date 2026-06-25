# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:34:40 2023

@author: user
"""

#%% Save data from iterations
# open file
saveName = 'solutions'
#get position of last iteration saved
Bcurr = len(datafile)

# B = [] #execute if want to create an empty/new dictionary to save information
#initial benchmark
B.append({'cycle':[], 'sb_uids':[], 'sbs':[], 'prj':[], 'days':[], 'cant_var':[], 'percentage':[], 'bins':[], 'sol_s':[], 'z':[], 'assigned_vars':[]})
# Bcurr = 0
#save data
B[Bcurr]['cycle'] = year
B[Bcurr]['sb_uids'] = df_main['SB_UID'] #list of sb_uids
B[Bcurr]['sbs'] = df_main.index #list with eb numbers
B[Bcurr]['prj'] = df_main['PRJ_CODE'].unique() #list of projects codes
B[Bcurr]['days'] = d #days of flexibility
B[Bcurr]['cant_var'] = m.numVars #variables created in model
B[Bcurr]['percentage'] = porcentaje #lambda value
B[Bcurr]['bins'] = total_bins #available bins
B[Bcurr]['sol_s'] = m.Runtime #runtime to get the solution
B[Bcurr]['z'] = m.objVal #optimal solution value
B[Bcurr]['assigned_vars'] = binary_vars #dictionary with binary variables assgined
Bcurr+=1 #increase count to save next iteration

#%% save file with B dictionary 
np.savez('C:/Users/user/solutions' + '.npz', B=B)
#%% import B dictionary
#load data
npzfile = np.load('C:/Users/user/solutions.npz', allow_pickle=True)
#choose benchmark instance
C = npzfile['B'][()]
B = [] #create empty dictionary to save iterations
for i in range(len(C)): #for each iteration in B dictionary
    #save iterations
    B.append({'cycle':[], 'sb_uids':[], 'sbs':[], 'prj':[], 'days':[], 'cant_var':[], 'percentage':[], 'bins':[], 'sol_s':[], 'z':[], 'assigned_vars':[]})
    B[i] = C[i]
Bcurr = len(C)