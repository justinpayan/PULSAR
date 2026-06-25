# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 11:21:34 2023

@author: user
"""

import pandas as pd
import numpy as np
from configuration import configurable_dict

#%% set actual date and get list of SB_UIDs completed
set_date = configurable_dict['set_date'] #'YYYY-MM-DD'
set_date = pd.to_datetime(set_date)

#delete completed SB_UID
#completed_sbs should be a list of SB_UID codes that are already observed
completed_sbs = configurable_dict['completed_sbs']#random.choices(sb_uid_list_main, k=120)

#set lambda for constraint (optional)
lambda_p = configurable_dict['lambda_p']

completed_executions_df = df_match_sb_sbuid[df_match_sb_sbuid['SB_UID'].isin(completed_sbs)] #get EBs of SB_UIDs
completed_executions_list = completed_executions_df['sb'].tolist() #get completed eb list
#drop eb completed from eb_temp
for eb in completed_executions_list: #for each eb that is completed
    if eb in eb_temp: #if the eb exists in eb_temp
        del eb_temp[eb] #delete eb information
#drop rows from df_availability before set date
df_availability = df_availability[df_availability['timestamp']>set_date] #if there is a row with informaation before the set date, drop row
df_availability = df_availability.reset_index() #reset index
df_availability = df_availability.drop('level_0',axis=1) #drop row created
total_bins = round(df_availability.available_time.sum())*2 #set new total_bins number to execute model with available time

#%% update necessary information of the parameters and re run the model
calendar = pd.DataFrame(columns=['Configuration', 'Start', 'End']) #create df to save calendar
begin_calendar = pd.DataFrame({'Configuration': df_availability['conf'][0] , 'Start': df_availability['timestamp'][0]}, index=[0]) #create first row of calendar
calendar = pd.concat([begin_calendar,calendar.loc[:]]) #append first row to calendar
count = 0 #create count variable
actual_c = calendar['Configuration'][0] #set actual configuration to the first one
for row in range(len(df_availability)): #for each row of df_availability
    if df_availability['conf'][row] != actual_c: #if the configuration is different than the actual one
        count += 1 #add one to the count variable
        add_row_calendar = pd.DataFrame({'Configuration': df_availability['conf'][row] , 'Start': df_availability['timestamp'][row]}, index=[0]) #create following row of calendar
        calendar = pd.concat([add_row_calendar,calendar.loc[:]]).reset_index(drop=True) #append row to calendar
        actual_c = df_availability['conf'][row] #update the actual configuration
calendar['Start'] = calendar['Start'].apply(lambda a: pd.to_datetime(a).date()) #set start date in datetime format
calendar = calendar.sort_values(by=['Start']).reset_index() #sort calendar according to start date
calendar = calendar.drop(['index'],axis=1) #drop index column
for row in range(1,len(calendar['Start'])): #for the second row till the end of the calendar
    calendar['End'][row-1] = calendar['Start'][row] - timedelta(days=1) #set end date by subtracting a day to start date of the following configuration
calendar['End'].iloc[-1] = pd.to_datetime(df_availability['timestamp'].iloc[-1], format='%Y-%m-%d %H:%M:%S').date() #set end date for the last configuration using last row of df_availability
# print('Calendar created')
# add column with just day to df_availability
df_availability['day'] = pd.to_datetime(df_availability['timestamp']) #create a column with just the date of the timestamp in df_availability
df_availability['day'] = df_availability['day'].dt.date #set date format
# create config array
config = []
lista_config = calendar['Configuration'].str.split('-',n=1,expand=True)[1]
for i in lista_config:
    config.append(i)
# print(config)
duplicados = {x for x in config if config.count(x) > 1}
# Use a defaultdict to keep track of element counts
element_counts = defaultdict(int)
# Create a new list to store the modified elements
new_config = []
# Iterate through the original list
for element in config:
    # Increment the count for the current element
    element_counts[element] += 1
    # Determine the letter to add based on the count
    letter = chr(ord('A') + element_counts[element] - 1)
    # Append the element with the letter to the new list
    new_config.append(element + letter)
# print(new_config)
config = new_config
# dict conf_duration
durations = [] #create empty array
for c in range(len(calendar['Start'])): #for each row in calendar
    durations.append(0) #append a 0 to each place in durations list
    for row in range(len(df_availability)): #for each row in df_availability
        if bool((calendar['Start'][c] - timedelta(days = 1)) < df_availability['day'][row] and (df_availability['day'][row] < (calendar['End'][c]+timedelta(days=1)))):
            #if the day of df_availability is between the start and end date of the configuration in calendar
            durations[c] += df_availability['available_time'][row] #add available time to the duration
conf_duration = {config[i]: durations[i] for i in range(len(config))} #create a dictionary with configurations as keys and durations as values (in hours)
# print('dict conf_duration')
# dict conf_duration_bin
conf_duration_bin = {key: value * 2 for key, value in conf_duration.items()} #create dictionary with configurations and durations with bins
# print('dict conf_duration_bin')
# add columns to calendar
calendar['Config'] = config #add config column to calendar
calendar['Days'] = (calendar['End'] - calendar['Start']).dt.days +1 #get amount of days for each configuration into a column
calendar['Durations'] = list(conf_duration.values()) #get durations in hours into a column
calendar['Durations_bins'] = list(conf_duration_bin.values()) #get durations in bins into a column
calendar['Days'][5] -= 27 #restamos días de febrero #subtract days of feb of configuration
# print(calendar)
# dict Tti
df_availability['array']=0
for s in range(0,len(df_availability['day'])):
    for i in range(0,len(calendar['Start'])):
      if bool((calendar['Start'][i] - timedelta(days = 1)) < df_availability['day'][s] and (df_availability['day'][s] < (calendar['End'][i]+timedelta(days=1)))):
          df_availability['array'][s] = config[i]
# Calculo Tti_hr
Tti_hr = {} #create empty dictionary
for c in config: #for each configuration
    temp_dict = {} #create a temporary dictionary
    for t in range(48): #for each half of hour
        temp_dict[t]=0 #create as many keys as bins with all values = 0
    Tti_hr[c] = temp_dict #fill the dictionary with the temporary dict as values and configuration as keys
for row in range(len(df_availability['array'])): #for each row in df_availability
    c = df_availability['array'][row] #get configuration that have the available time
    t = df_availability['LST_bin'][row]*2 #get bins that are available
    Tti_hr[c][t] += df_availability['available_time'][row] #add to the value of the configuration and bin the amount of available time
# print('dict Tti_hr')
# dejar en bins Tti
Tti = Tti_hr.copy()
multiply_dict_values(Tti, (2))
# multiply_dict_values(Tti, (3000/3900*2))
approximate_dict_values(Tti)
# print('dict Tti en bins')
epsilon = 1e-10 #set tiny value to avoid symmetry problems in each bin

sb_value = {} #create empty dictionary to have the value of each eb (profit)
for s in eb_temp.keys(): #for each eb in the dictionary eb_temp (that tells us if can be scheduled)
    if s not in sb_value: #if the eb is not in the dictionary
        sb_value[s] = {} #create a key with the eb with empty dictionary inside
    for c in eb_temp[s].keys(): #for each configuration where the eb can be executed
        key_c = "".join([ele for ele in c if ele.isdigit()]) #get the number of the configuration
        if element_counts[key_c] == 3: #if the configuration is scheduled 3 times throughout the cycle
            extra_val = 0.002 #the extra value is 0.002
        elif element_counts[key_c] == 2: #if is scheduled twice throughout the cycle
            extra_val = 0.001 #the extra value is 0.001
        elif element_counts[key_c] == 1: #if is scheduled just once
            extra_val = 0 #there is no extra value
        sb_value[s][c] = {} #create a empty dictionary for the EB in the configuration
        for t in eb_temp[s][c].keys(): #for each bin where can be the EB s in the configutaion c excecuted
            if np.isnan(pref_conf[c][s]) == False: #if there is a value of preference in the dataframe
                sb_value[s][c][t] = (df_main['SB_Value'][s] * pref_conf[c][s]) + extra_val + (epsilon*(float(t)+1)) #assign value of EB s in configuration c and bin t
# print('dict sb_value')
#%% re run model
print('------------------------------------- Model -------------------------------------')

# upper y lower limit with flexibility days
d = 0 #ammount of days

conf_upperlimit = {} #create dictionry with upper limit days of flexibility for each configuration
for i in config: #for each configuration
    if config[len(config)-1] == i or i == borderconf_1: #if the configuration is the last one or the one before february starts
        conf_upperlimit[i] = 0 #the upper limit is 0
    else: #for any other configuration
        conf_upperlimit[i] = d #the limit is the set days

conf_lowerlimit = {} #create dictionry with lower limit days of flexibility for each configuration
for i in config: #for each configuration
    if config[0] == i or i == borderconf_2: #if the configuration is the first one or the one after february starts
        conf_lowerlimit[i] = 0 #the lower limit is 0
    else: #for any other configuration
        conf_lowerlimit[i] = -d #the limit is the set days
# modficiacion de bins
#dictionary to add the extra available time per configurations if neccesary
Tti_mod = {} #create empty dictionary to save extra time for each configuration
for c, b in Tti.items(): #for each configuration and bin
    duration = calendar.loc[calendar['Config'] == c, 'Durations'].values[0] #get the duration from the calendar
    new_bins = {bin_number: bin_value / duration for bin_number, bin_value in b.items()}  #get the time for each bin
    Tti_mod[c] = new_bins #set the times of bins for each configuration
# model
#import libraries
import gurobipy as gp
from gurobipy import GRB

#Model creation
m = gp.Model('GAPM') #set model name

### Variables ###
#Xij, binary, 1: if SBj assing to config i. 0: o.w.
x = {} #create empty dictionary for x_ijt variable
for j in eb_temp.keys(): #for each EB
    x[j] = {} #create empty dictionary to add subindexes
    for i in eb_temp[j].keys(): #for each configuration where can be the EB executed
        x[j][i] = {} #create empty dictionary
        for bin_t in eb_temp[j][i].keys(): #for each bin where can be executed
            if eb_temp[j][i][bin_t] == 1: #if the EB j in configuration i on bin bin_t can be scheduled
                x[j][i][bin_t] = m.addVar(obj = sb_value[j][i][bin_t]/max_value, vtype=GRB.BINARY, name = 'x[%s,%s,%s]'%(j,i,bin_t))
                #create binary variable

#di, units of time that is added or decreased to config i
extra_duration = {} #create empty dictionary for d_i variable
for i in config: #for each configuration
    extra_duration[i] = m.addVar(lb=conf_lowerlimit[i], ub=conf_upperlimit[i] , vtype=GRB.INTEGER, name='extra_duration_of[%s]'%(i))
    #create integer variable with lower and upper limit from the dictionaries

#yk, binary, 1: if all EBs from project k are assigned
y = {} #create empty dictionary for y_k variables
for k in projects_eb.keys(): #for each project
    y[k] = m.addVar(vtype=GRB.BINARY, name='y[%s]'%k) #create binary variable

m.modelSense = gp.GRB.MAXIMIZE #set the type of model (to maximize to obtain the highest profit on the asignation)
m.update() #update the information of the model (save the variables)
#
print('------------------------ Creating constraints --------------------------')
# assign constraint
for j in sb_value.keys(): #for each EB j
    lexp = gp.LinExpr() #create linear expression
    for i in x[j].keys(): #for each configuration i of EB j variable
        for t in x[j][i].keys(): #and for each bin where can be executed
            lexp.addTerms(1.0, x[j][i][t]) #add to the linear expression the variable for x_ijt
    #for each EB, there will be a linear expression with all possible slots where can be scheduled but only one variable can be one
    m.addConstr(lexp, gp.GRB.LESS_EQUAL, 1, name='Assign[%s]'%(j))
m.update() #save constraint

if problem != 'plan': #if the problem is not the planning one
    # constraint of projects that must be assigned
    # for p in proj_accepted:
    #     if p in project_sb.keys():
    #         #si el proyecto existe y pertenece a los que deben si o si estar asignados, la variable y será 1
    #         m.addConstr(y[p], gp.GRB.EQUAL, 1, name='Forced_project_[%s]'%(p))
    # m.update()

    ###
    # constraint of do not exceed 45 hrs
    lexp = gp.LinExpr() #create linear expression
    for j in eb_temp.keys(): #for each EB
        if df_main['PRJ_CODE'][j] in proj_max_45: #if the project code of the EB is in the list
            for i in x[j].keys(): #for each configuration of the EB
                for t in x[j][i].keys(): #and for each bin where can be executed
                    lexp.addTerms(bins_eb[j], x[j][i][t]) #add the variable to the linear expression with the time that uses
    m.addConstr(lexp <= 90, name = 'Max_45_hrs') #create the constraint with all EBs options so they do not exceed the time
    m.update() #save constraint

    ###
    # constraint of do not exceed 50 hrs in .P projects
    lexp = gp.LinExpr() #create linear expression
    for j in eb_temp.keys(): #for each EB
        if df_main['PRJ_CODE'][j] in proj_P_max_50: #if the project code of the EB is in the list
            for i in x[j].keys(): #for each configuration of the EB
                for t in x[j][i].keys(): #and for each bin where can be executed
                    lexp.addTerms(bins_eb[j], x[j][i][t]) #add the variable to the linear expression with the time that uses
    m.addConstr(lexp <= 100, name = 'Max_50_hrs_P') #create the constraint with all EBs options so they do not exceed the time
    m.update() #save constraint

# time constraint
for c in config: #for each configuration
    for t in Tti[c].keys(): #for each bin of each configuration
        lexp = gp.LinExpr() #create a linear expression
        for s in eb_temp.keys(): #for each EB
            if c in eb_temp[s].keys(): #and for each configuration where can be executed
                if t in eb_temp[s][c].keys(): #and for each bin where can be executed
                    for u in range(max(0,t-bins_eb[s]+1), t+1): #
                        if u in eb_temp[s][c].keys(): #
                            lexp.addTerms(1.0, x[s][c][u]) #add to the linear expression the variables
        #create constraint for each bin so that all assigned EB do not exceed the available time
        m.addConstr(lexp, gp.GRB.LESS_EQUAL, (Tti[c][t] + (Tti_mod[c][t]*extra_duration[c])), name='time_bin[%s,%s]'%(t,c))
m.update() #save constraint

# constraint same available time
lexp = gp.LinExpr() #create linear expression
for i in config: #for each configuration
    lexp.addTerms(1.0, extra_duration[i]) #add variable of how many days are added or decreased
m.addConstr(lexp == 0, name='durations_sum_0') #create constraint that all days added or decreased must add up 0
m.update() #save constraint

# all EBs per project constraint
for p in projects_eb.keys(): #for each project
    for j in x.keys(): #and for each EB
        if bool(j in projects_eb[p]): #if the EB belongs to project p
            lexp = gp.LinExpr() #create linear expression
            for c in x[j].keys(): #for each configuration where can be the EB executed
                for t in x[j][c].keys(): #and for each bin where can be scheduled
                    lexp.addTerms(1.0, x[j][c][t]) #add variable to the linear expression
            #create constraint that forces all EB to be assigned if the project is
            m.addConstr(y[p], gp.GRB.LESS_EQUAL, lexp, name='All_sb_per_proj[%s]_and_eb[%s]'%(p,j) )
m.update() #save constraint

# % of projects forced to be assigned constraint
porcentaje = lambda_p #use set lambda by user
lexp = gp.LinExpr() #create linear expression
for i in y.keys(): #for each project
    # if i in grade_prj.keys() and grade_prj[i] in ['A','B']:
    lexp.addTerms(1.0, y[i]) #add the project variable to the linear expression
m.addConstr((lexp) >= porcentaje*(df_main.PRJ_CODE.nunique()), name='cant_proj_assigned')
#create constraint that forces a % of the total available projects to be completly assigned
m.update() #save constraint

# constraint of executive balance
for organization in org_p.keys(): #for each organization
    lexp = gp.LinExpr() #create linear expression
    for j in x.keys(): #for each EB
        for i in x[j].keys(): #for each configuration where can be executed
            for t in x[j][i].keys(): #and for each bin where can be scheduled
                for org in ex_bal[j].keys(): #for each organization of the EB
                    if org == organization: #if the organization of the constaint is in the dict of the EB
                        lexp.addTerms(ex_bal[j][org]*bins_eb[j], x[j][i][t]) #add variable with time usage to the linear expression
    #create constraint that limits the time that is assignt to the project for each organization
    m.addConstr((lexp) <= (org_p[organization]*total_bins), name= 'balance_max[%s]'%(organization))
m.update() #save constraint

# min % of executive balance constraint
for organization in org_p.keys(): #for each organization
    lexp = gp.LinExpr() #create linear expression
    for j in x.keys(): #for each EB
        for i in x[j].keys(): #for each configuration where can be executed
            for t in x[j][i].keys(): #and for each bin where can be scheduled
                for org in ex_bal[j].keys(): #for each organization of the EB
                    if org == organization: #if the organization fo the constraint is in the dict of the EB
                        lexp.addTerms(ex_bal[j][org]*bins_eb[j], x[j][i][t]) #add variable with time usage to the linear expression
    #create constraint that forces at minimum of time per organization
    m.addConstr((lexp) >= (org_p_min[organization]*total_bins), name= 'balance_min[%s]'%(organization))
m.update() #save constraint

# set gap for limit execution time
m.Params.MIPGap = model_gap #set gap for model

print('------------------------ Optimizing --------------------------')
# optimize model to obtain results
m.optimize() #get results of model

# Generate output with codes on 'opt_model.py'
