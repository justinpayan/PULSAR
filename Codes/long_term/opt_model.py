# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:48:09 2023

@author: user
"""
#import libraries
import gurobipy as gp
from gurobipy import GRB

from Codes.long_term.pre_process import project_to_ranks
from Codes.short_term.src.fixed_params import PROJECT_WEIGHT

# from Codes.configuration import configurable_dict

#%% Values definition
# Configurations that end before february and start after, the dates should not be modified.
borderconf_1 = configurable_dict['borderconf_1'] #cant last longer
borderconf_2 = configurable_dict['borderconf_2'] #cant begin earlier
#The names of this configurations must have "letters", should be extract from the calendar df in pre_process.py

#Amount of days that can be modified the duration of the configurations.
d = configurable_dict['d']

#Percentage of projects that will be completely assigned.
lambda_p = configurable_dict['lambda_p']

#Total amount of bins that are available to observe
total_bins = configurable_dict['total_bins']

#gap of model
model_gap = configurable_dict['model_gap']

#type of problem
problem = configurable_dict['problem']

#name of outputfile
out_file = configurable_dict['out_file']


#%%
print('------------------------ Optimization model --------------------------')

#%% upper y lower limit with flexibility days
d = d #amount of days
#create dictionary of limit of days per configuration (upper and lower limit)
#the configurations that are between february are limited

conf_upperlimit = {} #create dictionary with upper limit days of flexibility for each configuration
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
#%% modficiacion de bins
Tti_mod = {} #create empty dictionary to save extra time for each configuration
for c, b in Tti.items(): #for each configuration and bin
    duration = calendar.loc[calendar['Config'] == c, 'Durations'].values[0] #get the duration from the calendar
    new_bins = {bin_number: bin_value / duration for bin_number, bin_value in b.items()}  #get the time for each bin
    Tti_mod[c] = new_bins #set the time in bins for each configuration
#%% model
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
                #create binary variable
                x[j][i][bin_t] = m.addVar(obj = sb_value[j][i][bin_t]/max_value, vtype=GRB.BINARY, name = 'x[%s,%s,%s]'%(j,i,bin_t))

#di, units of time that is added or decreased to config i
extra_duration = {} #create empty dictionary for d_i variable
for i in config: #for each configuration
    #create integer variable with lower and upper limit from the dictionaries
    extra_duration[i] = m.addVar(lb=conf_lowerlimit[i], ub=conf_upperlimit[i] , vtype=GRB.INTEGER, name='extra_duration_of[%s]'%(i))

PROJECT_WEIGHT = configurable_dict['project_weight']
#yk, binary, 1: if all EBs from project k are assigned
y = {} #create empty dictionary for y_k variables
for k in projects_eb.keys(): #for each project
    y[k] = m.addVar(obj=PROJECT_WEIGHT/(project_to_ranks[k]*max_value), vtype=GRB.BINARY, name='y[%s]'%k) #create binary variable

m.modelSense = gp.GRB.MAXIMIZE #set the type of model (to maximize to obtain the highest profit on the asignation)
m.update() #update the information of the model (save the variables)
#%%
print('------------------------ Creating constraints --------------------------')

#%% assign constraint (cada EB puede asignarse a lo más una vez)
for j in sb_value.keys(): #for each EB j
    lexp = gp.LinExpr() #create linear expression
    for i in x[j].keys(): #for each configuration i of EB j variable
        for t in x[j][i].keys(): #and for each bin where can be executed
            lexp.addTerms(1.0, x[j][i][t]) #add to the linear expression the variable for x_ijt
    #for each EB, there will be a linear expression with all possible slots where can be scheduled but only one variable can be one
    m.addConstr(lexp <= 1, name='Assign[%s]'%j)
m.update() #save constraint
print("assign constraint done")
#%%
if problem != 'plan': #if the problem is not the planning one
    # constraint of projects that must be assigned
    # for p in proj_accepted:
    #     if p in project_sb.keys():
    #         #if the project exists and belong to the subset 'must be assigned, variable y will be 1
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

#%% time constraint
# for c in config: #for each configuration
#     for t in Tti[c].keys(): #for each bin of each configuration
#         lexp = gp.LinExpr() #create a linear expression
#         for s in eb_temp.keys(): #for each EB
#             if c in eb_temp[s].keys(): #and for each configuration where can be executed
#                 if t in eb_temp[s][c].keys(): #and for each bin where can be executed
#                     for u in range(max(0,t-bins_eb[s]+1), t+1):
#                         if u in eb_temp[s][c].keys():
#                             #add to the linear expression all the variables that may add time
#                             lexp.addTerms(1.0, x[s][c][u])
#         #create constraint for each bin so that all assigned EB do not exceed the available time
#         m.addConstr(lexp <= (Tti[c][t] + (Tti_mod[c][t]*extra_duration[c])), name='time_bin[%s,%s]'%(t,c))
# m.update() #save constraint

for c in config:
    for t in Tti[c]:
        lexp = gp.LinExpr() #create a linear expression
        for s in eb_temp: # each EB
            if c in eb_temp[s]: #if it's the same configuration
                for t_prime in eb_temp[s][c]: #and for each bin where can be executed
                    if t_prime <= t < t_prime + bins_eb[s]:
                        lexp.addTerms(1.0, x[s][c][t_prime])
#create constraint for each bin so that all assigned EB do not exceed the available time
        m.addConstr(lexp <= (Tti[c][t] + (Tti_mod[c][t]*extra_duration[c])), name='time_bin[%s,%s]'%(t,c))
m.update() #save constraint

#%% constraint same available time
lexp = gp.LinExpr() #create linear expression
for i in config: #for each configuration
    lexp.addTerms(1.0, extra_duration[i]) #add variable of how many days are added or decreased
m.addConstr(lexp == 0, name='durations_sum_0') #create constraint that all days added or decreased must add up 0
m.update() #save constraint

#%% all EBs per project constraint
for p in projects_eb.keys(): #for each project
    for j in x.keys(): #and for each EB
        if bool(j in projects_eb[p]): #if the EB belongs to project p
            lexp = gp.LinExpr() #create linear expression
            for c in x[j].keys(): #for each configuration where can be the EB executed
                for t in x[j][c].keys(): #and for each bin where can be scheduled
                    lexp.addTerms(1.0, x[j][c][t]) #add variable to the linear expression
            #create constraint that forces all EB to be assigned if the project is
            m.addConstr(y[p] <= lexp, name='All_sb_per_proj[%s]_and_eb[%s]'%(p,j) )
m.update() #save constraint

#%% % of projects forced to be assigned constraint
porcentaje = lambda_p #use set lambda by user
lexp = gp.LinExpr() #create linear expression
for i in y.keys(): #for each project
    # if i in grade_prj.keys() and grade_prj[i] in ['A','B']:
    lexp.addTerms(1.0, y[i]) #add the project variable to the linear expression
#create constraint that forces a % of the total available projects to be completly assigned
m.addConstr((lexp) >= porcentaje*(df_main.PRJ_CODE.nunique()), name='cant_proj_assigned')
m.update() #save constraint

#%% constraint of executive balance
for organization in org_p.keys(): #for each organization
    lexp = gp.LinExpr() #create linear expression
    for j in x.keys(): #for each EB
        for i in x[j].keys(): #for each configuration where can be executed
            for t in x[j][i].keys(): #and for each bin where can be scheduled
                for org in ex_bal[j].keys(): #for each organization of the EB
                    if org == organization: #if the organization of the constaint is in the dict of the EB
                        lexp.addTerms(ex_bal[j][org]*bins_eb[j], x[j][i][t]) #add variable with time usage to the linear expression
    #create constraint that limits the time that is assigned to each organization
    m.addConstr((lexp) <= (org_p[organization]*total_bins), name= 'balance_max[%s]'%(organization))
m.update() #save constraint

#%% min % of executive balance constraint
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

#%% set gap for limit execution time
m.Params.MIPGap = model_gap #set gap for model

#%% optimize model to obtain results
m.optimize() #get results of model
#%%
print('------------------------ Output generation --------------------------')

#%% get variables
binary_vars = [v.varName for v in m.getVars() if v.x > 0.5] #get all variables that are 1
eb_one_dict = {} #crate empty dictionary to save assigned EBs
for item in binary_vars: #for each variable that is one
    if item.startswith('x'): #if the variable is an x one
        s = item.split(',')[0].split('[')[1] #get number of EB
        c = item.split(',')[1] #get configuration
        b = item.split(',')[2].split(']')[0] #and get bin where is assigned
        if s not in eb_one_dict: #if the EB doesnt exists as key in the dictionary
            eb_one_dict[s] = {} #create key in the dict
            eb_one_dict[s]['conf'] = c #save the configuration in the dictionary
            eb_one_dict[s]['bin'] = b #save the bin
print('Los eb asignados son:')
print(len(eb_one_dict.keys()))

#%% output problema planificacion
if problem == 'plan': #if we are in the planning problem
    sbs_conf = {} #create empty dictionary
    for idx,row in df_match_eb_sbuid.iterrows(): #for each row in the df with matching sb_uid and eb numbers
        if str(row['eb']) in eb_one_dict.keys(): #if the EB is in the dict with ebs that are assigned
            if row['SB_UID'] not in sbs_conf.keys(): #and the SB_UID is not in the new dict
                sbs_conf[row['SB_UID']] = [] #create key with empty list
            if eb_one_dict[str(row['eb'])]['conf'] not in sbs_conf[row['SB_UID']]: #if the configuration is not in the list
                sbs_conf[row['SB_UID']].append(eb_one_dict[str(row['eb'])]['conf']) #add configuration to list

    df_output = pd.DataFrame(list(sbs_conf.items()))
    df_output.columns = ['SB_UID', 'Configuration']
    df_output.to_csv(out_file, index=False) #export df_output as a file

#%% output asignment problem
if problem == 'assign':
    df_results = df_main.copy() #create copy of df main to create results
    df_results['eb'] = df_results.index #add column with eb numbers
    # df_results = df_results.drop(columns = ['level_0']) #drop column
    # print(len(df_results.columns))
    df_results = df_results.sort_values(by=['PRJ_SCIENTIFIC_RANK']) #sort df_results by scientific ranking
    df_results['GRADE'] = 0 #create column for suggested grade
    time_grade = 0 #create time count
    for idx, row in df_results.iterrows(): #for each row in df_results
        time_grade += df_results['SB_TIME_BY_EXECUTION'][idx] #add the time of the EB
        if time_grade < (1434): #if the time is lower than 4300/3
            df_results['GRADE'][idx] = 'A' #assign A grade
        elif time_grade >= 1434 and time_grade < 4300: #if is more than 4300/3 and less than 4300
            df_results['GRADE'][idx] = 'B' #assign B grade
        else: #if is more than 4300
            df_results['GRADE'][idx] = 'C' #assign filler grade
    df_results.to_csv(out_file, index=False) #export df_results as a file