# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 08:37:28 2023

@author: user
"""
from calendar import error
import sys
sys.path.append("Codes/long_term")
# %% Import libraries
import pandas as pd
import numpy as np
from datetime import timedelta
from random import randint
import random as random
from itertools import cycle
import matplotlib.pyplot as plt
from configuration import configurable_dict
from collections import defaultdict
import re
import tqdm
import math

from fontTools.misc.plistlib import end_date

# %%
# name of file with information of sbs
main = configurable_dict['main']
# name of file with available time per bin
availability = configurable_dict['availability']
# name of file of where can be the sb executed
simulation = configurable_dict['simulation']
# name of file with extra information for projects
modes = configurable_dict['modes']

# year of the cycle (for code of projects)
year = configurable_dict['year']

# type of problem
problem = configurable_dict['problem']  # planning problem 'plan' (P2), assign problem 'assign' (P1)

# name of file with information of already accepted projects per cycle
cycle_accepted = configurable_dict['cycle_accepted']  # only needed when problem = 'plan'

# %% load dataframes
df_main = pd.read_csv(main, index_col=False)
# df_availability = pd.read_excel(availability, index_col=False)
df_availability = pd.read_csv(availability, index_col=False)
import os
print(os.listdir("Files\\Cycle 10"))
df_simulation = pd.read_csv(simulation, index_col=False)
df_modes = pd.read_csv(modes)
# set 'timestamp' column in timestamp
df_availability['timestamp'] = df_availability.timestamp.apply(lambda x: pd.Timestamp(x))

# load dataframe of accepted projects if exists
if problem == 'plan':
    df_cycle_accepted = pd.read_csv(cycle_accepted, index_col=False)

# %% keep only projects from the year in df_main and df_simulation
# if the project starts with 'year' keep in df_main
print("Loading is done. Moving to filtering by year")
df_main = df_main[df_main['PRJ_CODE'].str.startswith(year)].copy()
df_main = df_main.reset_index()  # reset index
df_main = df_main.drop('index', axis=1)  # drop index column
# if the project starts with 'year' keep in df_simulation
df_simulation = df_simulation[df_simulation['CODE'].str.startswith(year)].copy()
df_simulation = df_simulation.reset_index()  # reset index
df_simulation = df_simulation.drop('index', axis=1)  # drop index column

print("Checking which projects need to be less than 45 hours")
# %% list of projects that can not exceed 45 hrs
proj_max_45 = []  # create empty list for projects
for row in range(len(df_modes['CODE'])):  # for each row in df_modes
    if df_modes['MODE_NAME'][row] in ['BandToBand Interferometry',
                                      'BandwidthSwitching Interferometry']:  # if the mode of the projects match with the right options
        if df_modes['CODE'][row] not in proj_max_45:  # if the project code is not already on the list
            proj_max_45.append(df_modes['CODE'][row])  # append to the list the project code

print("Checking which projects need to be less than 50 hours")
# %% list of project '.P' that can not exceed 50 hrs
proj_P_max_50 = []  # create empty list for projects
for row in range(len(df_main['PRJ_CODE'])):  # for each row in df_main
    name = df_main['PRJ_CODE'][row]  # save the code of the project
    if name[
       -2:] == '.P' and name not in proj_P_max_50:  # if the last character of the code match with .P, and is not already on the list
        proj_P_max_50.append(df_main['PRJ_CODE'][row])  # append to the list the project code

# print("Dropping large program projects")
# #%% drop rows of Large program projects
# for row in range(len(df_main['PRJ_CODE'])): #for each row in df_main
#     name = df_main['PRJ_CODE'][row] #save the code of the project
#     if name[-2:] == '.L':# and name not in proj_accepted: #if the type of project is .L
#         df_main.drop(index=row, inplace= True) #drop row of large program project from df_main

# %% ignore rows with no information in fraction_obs_cx
df_main = df_main.dropna()  # drop rows with na values
df_main = df_main.reset_index()  # reset index of df_main

# MODIFY THE PROJECT RANKS FOR PROJECTS WHICH ARE FOUND IN THE DSA CORRECTED
# df_extra = pd.read_csv("Codigos/master_dsa_c9_prepared_camila_corrected.csv")
# for prj_code in df_main['PRJ_CODE'].values:
#     print(prj_code)
#     if prj_code in df_extra['CODE'].values:
#         print("MATCH")
#         df_main.loc[df_main['PRJ_CODE'] == prj_code, 'PRJ_SCIENTIFIC_RANK'] = \
#         df_extra[df_extra['CODE'] == prj_code]['PRJ_SCIENTIFIC_RANK'].unique()[0]
#         print(df_main.loc[df_main['PRJ_CODE'] == prj_code, 'PRJ_SCIENTIFIC_RANK'])
#         print('\n\n')


# %% function for dictionaries
# function that multiplies a dictionary value for a number
def multiply_dict_values(dictionary, mult):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            multiply_dict_values(value, mult)
        else:
            dictionary[key] = value * mult


# function that sums up all values of a key in a dictionary
def sum_dict_by_key(new_dict):
    res = dict()
    for sub in new_dict.values():
        for key, ele in sub.items():
            res[key] = ele + res.get(key, 0)
    return res


# function that approximates values of a dictionary
def approximate_dict_values(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            approximate_dict_values(value)
        else:
            dictionary[key] = math.floor(value)


# %% keep top 50% per organization
df_CL = pd.DataFrame()  # create empty dataframe for each organization
entry = df_main.loc[df_main['CL'] > 0]  # if the SB belongs to a organizacion, loc row
df_CL = pd.concat([df_CL, entry])  # append to the empty dataframe the loc rows
df_CL = df_CL.sort_values(by=['PRJ_SCIENTIFIC_RANK'])  # sort values by scientific rank to keep the "best" ones
df_CL = df_CL.drop_duplicates(subset=['PRJ_CODE'], keep='last')  # keep only one row per project code
df_CL = df_CL.reset_index()  # reset index of dataframe

df_EA = pd.DataFrame()
entry = df_main.loc[df_main['EA'] > 0]
df_EA = pd.concat([df_EA, entry])
df_EA = df_EA.sort_values(by=['PRJ_SCIENTIFIC_RANK'])
df_EA = df_EA.drop_duplicates(subset=['PRJ_CODE'], keep='last')
df_EA = df_EA.reset_index()

df_EU = pd.DataFrame()
entry = df_main.loc[df_main['EU'] > 0]
df_EU = pd.concat([df_EU, entry])
df_EU = df_EU.sort_values(by=['PRJ_SCIENTIFIC_RANK'])
df_EU = df_EU.drop_duplicates(subset=['PRJ_CODE'], keep='last')
df_EU = df_EU.reset_index()

df_NA = pd.DataFrame()
entry = df_main.loc[df_main['NA'] > 0]
df_NA = pd.concat([df_NA, entry])
df_NA = df_NA.sort_values(by=['PRJ_SCIENTIFIC_RANK'])
df_NA = df_NA.drop_duplicates(subset=['PRJ_CODE'], keep='last')
df_NA = df_NA.reset_index()

# create list top50 project
top50 = []  # create empty list for projects
for row in range(len(df_CL['PRJ_CODE']) // 2):  # for the first half of the project
    if df_CL['PRJ_CODE'][row] not in top50:  # if the project code is not in the list
        top50.append(df_CL['PRJ_CODE'][row])  # append the project code

for row in range(len(df_EA['PRJ_CODE']) // 2):
    if df_EA['PRJ_CODE'][row] not in top50:
        top50.append(df_EA['PRJ_CODE'][row])

for row in range(len(df_EU['PRJ_CODE']) // 2):
    if df_EU['PRJ_CODE'][row] not in top50:
        top50.append(df_EU['PRJ_CODE'][row])

for row in range(len(df_NA['PRJ_CODE']) // 2):
    if df_NA['PRJ_CODE'][row] not in top50:
        top50.append(df_NA['PRJ_CODE'][row])

# %% keep in df_main only rows from the top 50 projects or accepted projects
if problem == 'plan':  # if the problem is planning problem
    grade_prj = df_cycle_accepted.groupby('PRJ_CODE')[
        'GRADE'].first().to_dict()  # create dict with projects as keys and the grade as values
    cycle_accepted_prj_list = df_cycle_accepted[
        'PRJ_CODE'].tolist()  # create list of project codes that are accepted for the cycle
    df_main = df_main.loc[df_main['PRJ_CODE'].isin(
        cycle_accepted_prj_list)]  # keep rows that belong to code of projects that are accepted
    df_main['PRJ_GRADE'] = ''  # create empty column to save grades in df_main
    for idx, row in df_main.iterrows():  # for each row of df_main
        p = row['PRJ_CODE']  # save the project code
        df_main['PRJ_GRADE'][idx] = grade_prj[p]  # save in the right row the grade of the project
else:  # if the problem is not planning
    df_main = df_main.loc[df_main['PRJ_CODE'].isin(top50)]  # keep rows of only top 50 projects according to ranking

# %% add index for each SB
indice = list(range(1, len(df_main) + 1))  # create a list of numbers according to the lenght of the df_main
df_main.insert(0, 'eb', indice)  # add eb column to df_main to identify each eb with just a number
# print('columns eb added')

# %% duplicate rows to create as many variable as execution may be needed
# duplicate rows based on the 'NUMBER_OF_EXECUTIONS' column
duplicated_rows = df_main.loc[df_main.index.repeat(df_main['NUMBER_OF_EXECUTIONS'])]
# modify the 'eb' column in the duplicated rows
duplicated_rows['eb'] = duplicated_rows['eb'].astype(str) + '.' + (
            duplicated_rows.groupby(level=0).cumcount() + 1).astype(str)
# reset the index of the duplicated rows
duplicated_rows.reset_index(drop=True, inplace=True)
# concatenate the original DataFrame with the duplicated rows
df_main = pd.concat([df_main, duplicated_rows], ignore_index=True)

# %% drop rows of eb that are duplicated
df_main = df_main.loc[~df_main['eb'].apply(lambda x: isinstance(x, int))]  # ignore rows that are duplicated
df_main.set_index('eb', inplace=True)  # set eb column as index

# %% create a dataframe with matching eb and SB_UID and save it in a csv file
df_match_eb_sbuid = df_main.reset_index()  # duplicate df_main and reset index
df_match_eb_sbuid = df_match_eb_sbuid[
    ['eb', 'SB_UID', 'PRJ_CODE']]  # keep only needed columns with information about project, eb and sb_uid
df_match_eb_sbuid.to_csv('match_eb_sbuid.csv', index=False)  # export file with matching information

# %% create calendar with start and end dates
calendar = pd.DataFrame(columns=['Configuration', 'Start', 'End'])  # create df to save calendar
begin_calendar = pd.DataFrame({'Configuration': df_availability['conf'][0], 'Start': df_availability['timestamp'][0]},
                              index=[0])  # create first row of calendar
calendar = pd.concat([begin_calendar, calendar.loc[:]])  # append first row to calendar

count = 0  # create count variable
actual_c = calendar['Configuration'][0]  # set first configuration to the actual one
for row in range(len(df_availability)):  # for each row of df_availability
    if df_availability['conf'][row] != actual_c:  # if the configuration is different than the actual one
        count += 1  # add one to the count variable
        add_row_calendar = pd.DataFrame(
            {'Configuration': df_availability['conf'][row], 'Start': df_availability['timestamp'][row]},
            index=[0])  # create following row of calendar
        calendar = pd.concat([add_row_calendar, calendar.loc[:]]).reset_index(drop=True)  # append row to calendar
        actual_c = df_availability['conf'][row]  # update the actual configuration
# calendar['Start'] = calendar['Start'].apply(lambda a: pd.to_datetime(a).date()) #set start date in datetime format
calendar['Start'] = pd.to_datetime(calendar['Start']).apply(lambda a: pd.to_datetime(a).date())
calendar['End'] = pd.to_datetime(calendar['End']).apply(lambda a: pd.to_datetime(a).date())
calendar['Start'] = pd.to_datetime(calendar['Start'])
calendar['End'] = pd.to_datetime(calendar['End'])

calendar = calendar.sort_values(by=['Start']).reset_index()  # sort calendar according to start date
calendar = calendar.drop(['index'], axis=1)  # drop index column
for row in range(1, len(calendar['Start'])):  # for the second row till the end of the calendar
    calendar['End'][row - 1] = calendar['Start'][row] - timedelta(
        days=1)  # set end date by subtracting a day to start date of the following configuration
calendar['End'].iloc[-1] = pd.to_datetime(pd.to_datetime(df_availability['timestamp'].iloc[-1],
                                                         format='%Y-%m-%d %H:%M:%S').date())  # set end date for the last configuration using last row of df_availability
print('Calendar created')
# %% add column with just day to df_availability
df_availability['day'] = pd.to_datetime(
    df_availability['timestamp'])  # create a column with just the date of the timestamp in df_availability
df_availability['day'] = pd.to_datetime(df_availability['day'].dt.date)  # set date format

# %% create config array
lista_config = calendar['Configuration'].str.split('-', n=1, expand=True)[
    1]  # get from calendar just the number of the configuration
config = lista_config.to_list()  # create config list
# print(config)
duplicados = {x for x in config if config.count(x) > 1}  # get set of repeated configurations

# use a defaultdict to keep track of element counts
element_counts = defaultdict(int)
new_config = []  # create new list to save configurations
for element in config:  # for each configuration of the original list
    element_counts[element] += 1  # increment the count of the current configuration
    letter = chr(ord('A') + element_counts[element] - 1)  # determine the letter to add based on the count
    new_config.append(element + letter)  # append configuration with letter to the new list
# print(new_config)
config = new_config  # replace with new list
# %% dict conf_duration
durations = []  # create empty array
for c in range(len(calendar['Start'])):  # for each row in calendar
    durations.append(0)  # append a 0 to each place in durations list
    for row in range(len(df_availability)):  # for each row in df_availability
        if bool((calendar['Start'][c] - timedelta(days=1)) < df_availability['day'][row] and (
                df_availability['day'][row] < (calendar['End'][c] + timedelta(days=1)))):
            # if the day of df_availability is between the start and end date of the configuration in calendar
            durations[c] += df_availability['available_time'][row]  # add available time to the duration
conf_duration = {config[i]: durations[i] for i in range(
    len(config))}  # create a dictionary with configurations as keys and durations as values (in hours)
print('dict conf_duration')

# %% dict conf_duration_bin
conf_duration_bin = {key: value * 2 for key, value in
                     conf_duration.items()}  # create dictionary with configurations and durations with bins
print('dict conf_duration_bin')

# %% add columns to calendar
calendar['Config'] = config  # add config column to calendar
calendar['Days'] = (calendar['End'] - calendar[
    'Start']).dt.days + 1  # get amount of days for each configuration into a column
calendar['Durations'] = list(conf_duration.values())  # get durations in hours into a column
calendar['Durations_bins'] = list(conf_duration_bin.values())  # get durations in bins into a column
calendar['Days'][5] -= 28  # subtract days of feb of configuration
# TODO: make it depend on the year, since sometimes there is a leap day.
print(calendar)
# %% assign value to category in a new column (inversa de ranking)
df_main['SB_Value'] = 1 / df_main['PRJ_SCIENTIFIC_RANK']  # create sb_value column
# if you want to add extra points to the profit equation, you can add it in the 'SB_Value' column
# for idx, row in df_main.iterrows(): #for each row
#     if row['PRJ_GRADE'] == 'A': #if the project is A graded
#         df_main['SB_Value'][idx] += 1000 #add 1000 points
#     elif row['PRJ_GRADE'] == 'B': if the project is B graded
#         df_main['SB_Value'][idx] += 10 #add 1000 points

# %% dict eb_duration
eb_duration = {}  # create empty dictionary to save the hours that each EB needs
for idx, row in df_main.iterrows():  # for each row in df_main
    eb_duration[idx] = (row[
        'SB_TIME_BY_EXECUTION'])  # save in the dictionary with EB as keys and duration in hours of the execution as values
print('dict eb_duration')

# %% dict bins_eb
bins_eb = {}  # create empty dictionary to save how many bins each EB needs
for idx, row in df_main.iterrows():  # for each row in df_main
    cant_bins = (row['SB_TIME_BY_EXECUTION']) * 2  # get amount of bins that each EB lasts
    bins_eb[idx] = round(cant_bins)  # add to de dictionary EB as keys and duration in bins of the execution as values
# bins_eb = {key: 1 if value == 0 else value for key, value in bins_eb.items()} #set in 1 all values that are 0
print('dict bins_eb')

# %% dict projects_eb
code_count = df_main['PRJ_CODE'].nunique()  # get amount of projects
unique_code = df_main['PRJ_CODE'].unique()  # get list of project codes
projects_eb = {}  # create empty dictionary to save which EBs belong to each project
grp_proj = df_main.groupby('PRJ_CODE')['PRJ_CODE']

for i in tqdm.tqdm(list(range(code_count))):  # for each project
    projects_eb[unique_code[i]] = grp_proj.get_group(unique_code[i]).index.to_list()

# Projects to ranks
project_to_ranks = {}
for _, row in df_main.iterrows():
    project_to_ranks[row['PRJ_CODE']] = row['PRJ_SCIENTIFIC_RANK']

# LESS EFFICIENT CODE USED TO BE THIS:
# for i in tqdm.tqdm(list(range(code_count))): #for each project
#     eb_per_project = [] #create empty array
#     for idx, row in df_main.iterrows(): #for each row in df_main
#         if row['PRJ_CODE'] == unique_code[i]: #if the information of the row of df_main matches the code of project
#             eb_per_project.append(idx) #append to list of EBs the sb value
#     projects_eb[unique_code[i]] = eb_per_project #save in dictionary the information per project
print('dict projects_eb')

# %% get a list of the names of FRACTION_OBS_columns
# extract columns with 'FRACTION_OBS_'
cols = [col for col in df_main.columns if 'FRACTION_OBS_' in col and len(col) > len('FRACTION_OBS_')]
# print(cols)
# %% find max profit (pmax)

max_value = 0  # set max_value as 0
preferencias = df_main[cols]  # create df with just fraction obs columns
preferencias = preferencias.dropna()  # drop na values
if 3 in element_counts.values():  # if any configurations is scheduled 3 times in a cycle
    valor_extra_por_config = 0.002  # the extra value is 0.002
else:  # if a configuration is just scheduled at most twice
    valor_extra_por_config = 0.001  # the extra value is 0.001
for idx, row in preferencias.iterrows():  # for each row in 'preferencias'
    indice = pd.to_numeric(preferencias.loc[idx]).idxmax().split('_')[2].split('C')[
        1]  # get the configuration of the greatest value of row
    valor = preferencias.loc[idx].max()  # get the value of the configuration selected previously
    nota = df_main['SB_Value'][idx]  # get value of EB from df_main
    if indice in duplicados:  # if the configuration is scheduled more than once in the cycle
        valor += valor_extra_por_config  # the value gets an "extra"
    max_value += nota * valor  # add to max_value the points of executing the EB in the best option available (the biggest preference * value of eb)
print(max_value)
print('max_value')

# %% create bins
bins_as_string = np.arange(0, 24., 24. / (24 * 60. / 30.)).astype(
    str)  # create array of bins representing each time slot of the day as strings
bins = np.arange(0, 24.5, 24. / (24 * 60. / 30.))  # create array of bins as numbers

# %% merge dataframes to have number of sb in df_simulation
df_sb_index = df_main[['SB_UID']]  # create df with just sb_uid column
df_sb_index['eb'] = df_main.index  # add eb column
df_disponibilidad = df_simulation.copy()  # copy df_simulation
df_sb_index.set_index('SB_UID', inplace=True)  # SB_UID as index in df_sb_index


# %% add 'array' to df_disponibilidad
# print('adding array to df_disponibilidad')
# df_disponibilidad['array']=0 #add array columns to df_disponibilidad
# df_disponibilidad['Day'] = df_disponibilidad['Date'].str.split('T',n=1,expand=True)[0] #get just date in a new column
# df_disponibilidad['Day'] = df_disponibilidad['Day'].apply(lambda a: pd.to_datetime(a).date()) #set datetimedate format to day column
# for i in range(0,len(calendar['Start'])): #for each row in calendar
#     for s in range(0,len(df_disponibilidad['Day'])): #for each row in df_disponibilidad
#         if bool((calendar['Start'][i] - timedelta(days = 1)) < df_disponibilidad['Day'][s] and (df_disponibilidad['Day'][s] < (calendar['End'][i]+timedelta(days=1)))):
#             #if the date from df_disponibilidad is between the start and end date of the calendar
#             df_disponibilidad['array'][s] = str(config[i]) #set value of column array in row as the corresponding configuration


def extract_date(df):
    df['Day'] = df['Date'].str.split('T', n=1, expand=True)[0]  # get just date in a new column
    df['Day'] = pd.to_datetime(df['Day']).dt.date  # set datetimedate format to day column
    return df


def date_in_range(start_date, end_date, df_row):
    return start_date < df_row['Day'] < end_date


def update_df_disponibilidad(df_disponibilidad, calendar, config):
    df_disponibilidad = extract_date(df_disponibilidad)

    calendar['Start'] = pd.to_datetime(calendar['Start']).dt.date
    calendar['End'] = pd.to_datetime(calendar['End']).dt.date

    config_avail_list = []
    for index, calendar_row in calendar.iterrows():  # for each row in calendar
        start_date = calendar_row['Start']
        end_date = calendar_row['End'] + pd.Timedelta(days=1)
        day_count = (end_date - start_date).days
        for single_date in (start_date + timedelta(n) for n in range(day_count)):
            config_avail_list.append([single_date, str(config[index])])
    config_avail_df = pd.DataFrame(config_avail_list)
    config_avail_df.columns = ['Day', 'array']
    df_disponibilidad = df_disponibilidad.merge(config_avail_df, on="Day", how="left")

    return df_disponibilidad


df_disponibilidad = update_df_disponibilidad(df_disponibilidad, calendar, config)
#
print("Finished with update_df_disponibilidad")
#
# #%% sort df_disponibilidad
df_disponibilidad = df_disponibilidad.sort_values(by=['SB_UID'])  # sort values of df_disponibilidad by SB_UID code
df_disponibilidad = df_disponibilidad.dropna()  # drop na rows
df_disponibilidad = df_disponibilidad.reset_index()  # reset index

# %%
sb_uid_list_main = df_main['SB_UID'].unique().tolist()  # get list of SB_UIDs of df_main
sb_uid_list_disp = df_disponibilidad['SB_UID'].unique().tolist()  # get list of SB_UIDs of df_disponibilidad
# %%
df_disponibilidad_filtered = df_disponibilidad[
    df_disponibilidad['SB_UID'].isin(sb_uid_list_main)]  # filter rows of df_disponibilidad if the SB_UID is in df_main
df_disponibilidad_filtered.reset_index(inplace=True)  # reset index

print("Creating eb_temp")

# %% create eb_temp
df_disponibilidad_filtered['lst_bin'] = pd.cut(df_disponibilidad_filtered['lst'], bins, labels=bins_as_string).astype(
    float) * 2
df_eb_array_lstbin = df_disponibilidad_filtered[['SB_UID', 'array', 'lst_bin']].merge(df_sb_index, on='SB_UID',
                                                                                      how='left').drop_duplicates()

eb_groups = df_eb_array_lstbin.groupby('eb')

eb_temp = {}
for eb_group in tqdm.tqdm(list(eb_groups)):
    eb, eb_group = eb_group
    for _, row in eb_group.iterrows():
        c, lstbin = row['array'], row['lst_bin']
        if eb not in eb_temp:
            eb_temp[eb] = {}
        if c not in eb_temp[eb]:
            eb_temp[eb][c] = {}
        eb_temp[eb][c][lstbin] = 1

# Old code took > 40 minutes, now 10 seconds.
# eb_temp={} #create an empty dict to save the availability of the EB
# for row in tqdm.tqdm(list(range(len(df_disponibilidad_filtered)))): # for each row in df_disponibilidad
#     sb_uid = df_disponibilidad_filtered['SB_UID'][row] #save the sb_uid of row
#     if type(df_sb_index['eb'][sb_uid]) is str: #if the SB only have one execution
#         eb_list=[] #create empty list
#         eb_list.append(df_sb_index['eb'][sb_uid]) #append to this list the eb number
#     else: #the sb need to be executed more than once
#         eb_list = df_sb_index['eb'][sb_uid].tolist() #create a list with all eb numbers representing the executions per sb
#     c = str(df_disponibilidad_filtered['array'][row]) #get the configuration where can be the eb can be executed
#     idx_lst = float(pd.cut([df_disponibilidad_filtered['lst'][row]], bins, labels=bins_as_string)[0])*2 #get the bin where can be executed according to df_disponibilidad
#     for eb in eb_list: #for each value of the executions of the sb
#         if eb not in eb_temp: #if the eb number is not in the dictionary
#             eb_temp[eb] = {} #add empty dictionary
#         if c not in eb_temp[eb]: #if the configuration where can be executed is not in the values of the eb number
#             eb_temp[eb][c]={} #create an empty dictionary inside the eb number
#         eb_temp[eb][c][idx_lst] = 1 #insert a 1, that means that for EB 's' in configuration 'c' at bin 'idx_lst' can be scheduled
# print('dict disponiblidad (eb_temp)')

# %% dict Tti
calendar['Start'] = pd.to_datetime(calendar['Start'])
calendar['End'] = pd.to_datetime(calendar['End'])
df_availability['array'] = 0  # create array column in df_availability
for s in range(0, len(df_availability['day'])):  # for each row in df_availability
    for i in range(0, len(calendar['Start'])):  # for each row in calendar
        if bool((calendar['Start'][i] - timedelta(days=1)) < df_availability['day'][s] and (
                df_availability['day'][s] < (calendar['End'][i] + timedelta(days=1)))):
            # if the date of df_availability is between the start and end date of the row in the calendar
            df_availability['array'][s] = config[i]  # add to row the configuration where can be executed

# %% Calculo Tti_hr
Tti_hr = {}  # create empty dictionary
for c in config:  # for each configuration
    temp_dict = {}  # create a temporary dictionary
    for t in range(48):  # for each half of hour
        temp_dict[t] = 0  # create as many keys as bins with all values = 0
    Tti_hr[c] = temp_dict  # fill the dictionary with the temporary dict as values and configuration as keys

for row in range(len(df_availability['array'])):  # for each row in df_availability
    c = df_availability['array'][row]  # get configuration that have the available time
    t = df_availability['LST_bin'][row] * 2  # get bins that are available
    Tti_hr[c][t] += df_availability['available_time'][
        row]  # add to the value of the configuration and bin the amount of available time
print('dict Tti_hr')
# %% respaldo Tti_hr
# Tti={}
Tti = Tti_hr.copy()
# %% dejar en bins Tti
multiply_dict_values(Tti, (2))  # multiply values to get the available time in bins
approximate_dict_values(Tti)  # aproximate amount of bins available to intergers
print('dict Tti en bins')

# %%
pref_conf = pd.DataFrame()  # create empty dataframe to save the preference of the configurations
for c in config:  # for each configuration
    i = [int(s) for s in re.findall(r'-?\d+\.?\d*', c)]  # get the name of the configuration
    col_name = cols[
        i[0] - 1]  # set the name of the column from list of fraction_obs_ names and the number of the configuration
    pref_conf[c] = df_main[col_name]  # create a column in the new df with the information for each configuration

# %% dict sb_value (Pijt)
# 0.015625 min value of fraction_obs...
epsilon = 1e-10  # set tiny value to avoid symmetry problems in each bin

sb_value = {}  # create empty dictionary to save the value of each eb (profit)
for s in tqdm.tqdm(list(eb_temp.keys())):  # for each eb in the dictionary eb_temp (that tells us if can be scheduled)
    if s not in sb_value:  # if the eb is not in the dictionary
        sb_value[s] = {}  # create a key with the eb with empty dictionary inside
    for c in eb_temp[s].keys():  # for each configuration where the eb can be executed
        key_c = "".join([ele for ele in c if ele.isdigit()])  # get the number of the configuration
        if element_counts[key_c] == 3:  # if the configuration is scheduled 3 times throughout the cycle
            extra_val = 0.002  # the extra value is 0.002
        elif element_counts[key_c] == 2:  # if is scheduled twice throughout the cycle
            extra_val = 0.001  # the extra value is 0.001
        elif element_counts[key_c] == 1:  # if is scheduled just once
            extra_val = 0  # there is no extra value
        sb_value[s][c] = {}  # create a empty dictionary for the EB in the configuration
        for t in eb_temp[s][c].keys():  # for each bin where can be the EB s in the configutaion c excecuted
            if np.isnan(pref_conf[c][s]) == False:  # if there is a value of preference in the dataframe
                sb_value[s][c][t] = (df_main['SB_Value'][s] * pref_conf[c][s]) + extra_val + (
                            epsilon * (float(t) + 1))  # assign value of EB s in configuration c and bin t
print('dict sb_value')

# %% dict org_o (porcentaje correspondiente a cada organizacion)
org_p = {}  # create empty dictionary to save % of executive balances per organizations
org_p['CL'] = 0.1  # ExecBal of Chile
org_p['EA'] = 0.225  # ExecBal of East Asia
org_p['EU'] = 0.3375  # ExecBal of Europe
org_p['NA'] = 0.3375  # ExecBal of North America
# %% dict org_p_min
org_p_min = {}  # create dictionary to set lowest limit of % of executive balance for each organization
org_p_min['CL'] = 0.09322
org_p_min['EA'] = 0.20950
org_p_min['EU'] = 0.31761
org_p_min['NA'] = 0.31939
# %% dict ex_bal (% de pertenencia del sb a cada organizacion)
ex_bal = {}  # create empty dictionary to save % of executive balance per organization as values and EB as keys
for s in df_sb_index['eb']:  # for each EB in df_main
    marks = {}  # create empty dictionary
    for subject in list(df_main[['CL', 'EA', 'EU', 'NA']].columns):  # for each organization
        marks[subject] = df_main[subject][s]  # get the % of executive balances
    ex_bal[s] = marks  # save the values for each EB
print('dict ex_bal')

# %% time per organization per sb
tiempo_a = {}  # create empty dictionary to save information of how much time each organization could use
for s in ex_bal.keys():  # for each EB
    tiempo_a[s] = {}  # create en empty dictionary
    for o in ex_bal[s].keys():  # for each organization
        tiempo_a[s][o] = ex_bal[s][o] * df_main['SB_TIME_BY_EXECUTION'][s]  # save the estimated time