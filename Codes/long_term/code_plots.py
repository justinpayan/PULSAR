# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 15:58:40 2023

@author: user
"""

import matplotlib.pyplot as plt
import math

#%% Pareto frontier plot, using dicionaries from 'pareto array' from loop of probabilities

#To generate this pareto arrays, once the model is created (until the definition of model gap in opt_model.py)
#the following lines should be executed for each line (scenario):
pareto_array = [] #create empty array
probabilities = np.arange(start = 0, stop= 1.01, step = 0.05).round(2) #create range of probabilities to try in lambda
for prob in probabilities: #for each probability
    lexp = gp.LinExpr() #create constraint
    for i in y.keys(): #for each project
        lexp.addTerms(1.0, y[i]) #add project to the constraint
    m.addConstr((lexp) >= prob*len(projects_sb), name='cant_proj_assigned') #create constraint in model 
    m.update() #update model
    m.optimize() #optimize model
    profit = m.objVal #get profit obtained
    lambda_p = prob #save probability used
    tiempo = m.Runtime #get runtime of model
    pareto_array.append([lambda_p, profit, tiempo]) #save values in array
    m.remove(m.getConstrByName('cant_proj_assigned')) #remove constraint of model
    m.update() #update model
    # print('----------------', prob) #print end of iteration with probability value

#some examples of pareto arrays for cycle 10
# pareto c10 0 days recieved proposals
pareto_0d_c10 = [[0.0, 0.7985707227169712, 13.575999975204468],
 [0.05, 0.7992171085793188, 19.566999912261963],
 [0.1, 0.7992163338184726, 22.085000038146973],
 [0.15, 0.799174550901392, 19.569999933242798],
 [0.2, 0.7986240154259352, 23.06500005722046],
 [0.25, 0.7987949856466365, 24.21499991416931],
 [0.3, 0.7964381095908045, 22.313000202178955],
 [0.35, 0.7963808648264636, 37.9229998588562],
 [0.4, 0.789677316742558, 22.799999952316284],
 [0.45, 0.775345599506217, 49.68400001525879],
 [0.5, 0.7484219045083205, 96.47900009155273],
 [0.55, 0.6647705662885771, 3696.734999895096]]
for sublist in pareto_0d_c10:
    sublist[0] *= 100
    sublist[1] *= 100

# pareto c10 5 days recieved proposals
pareto_5d_c10 = [[0.0, 0.7985640774148842, 16.317999839782715],
 [0.05, 0.7987925063837827, 21.187000036239624],
 [0.1, 0.7990513883988167, 20.283999919891357],
 [0.15, 0.7990698297642695, 21.585999965667725],
 [0.2, 0.7990723544831057, 20.513000011444092],
 [0.25, 0.7991861390755869, 19.777999877929688],
 [0.3, 0.7982296329004205, 25.97000002861023],
 [0.35, 0.7954470648856675, 29.247999906539917],
 [0.4, 0.7841706180507443, 35.38599991798401],
 [0.45, 0.7754441854009625, 69.09500002861023],
 [0.5, 0.7486712922266503, 619.558000087738],
 [0.55, 0.6608116426247325, 731.7599999904633]]
for sublist in pareto_5d_c10:
    sublist[0] *= 100
    sublist[1] *= 100

# pareto c10 0 dias accepted projects
pareto_0d_c10_accepted = [[0.0, 0.7775217104092026, 5.438999891281128],
 [0.05, 0.77876403386757, 7.83299994468689],
 [0.1, 0.7787678326487011, 8.069999933242798],
 [0.15, 0.7787848393717558, 7.230000019073486],
 [0.2, 0.7788050890694581, 7.9040000438690186],
 [0.25, 0.7783210107380318, 8.20799994468689],
 [0.3, 0.7767957961389265, 8.062999963760376],
 [0.35, 0.7745787877435424, 9.361999988555908]]
for sublist in pareto_0d_c10_accepted:
    sublist[0] *= 100
    sublist[1] *= 100

##extracting x and y values from each scenario
x1, y1, z1 = zip(*pareto_5d_c10)
x3, y3, z3 = zip(*pareto_0d_c10_accepted)
x2, y2, z2 = zip(*pareto_0d_c10)

#creating a scatter plot for each scenario
plt.plot(x1, y1, 'o-', label='791 proj', markersize =5, color = 'b')#, linewidth = 4)
plt.plot(x3, y3, 'o-', label='455 proj', markersize = 5, color = 'r')
plt.plot(x2, y2, 'o-', label='791 proj - 0 days', markersize = 5, color = 'g')

#setting plot title and labels
plt.title('Profit vs Percentage')
plt.xlabel('Percentage of lambda')
plt.ylabel('Percentage of max profit obtained')
#adding a legend
plt.legend()
plt.figure(figsize=(8, 6), dpi=300)
#displaying the plot
plt.show()

#%% grid of bar plots with available time per configuration (1ro en paper)
num_plots = len(Tti) #get number of configurations
#get number of rows and columns for the grid
num_cols = int(math.ceil(math.sqrt(num_plots)))
num_rows = int(math.ceil(num_plots / num_cols))
fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 10))
#flatten the axes array for easy indexing
axs = axs.flatten()
#iterate over configurations and plot each bar plot
for i, c in enumerate(Tti):
    b = list(Tti[c].keys())
    values = list(Tti[c].values())
    #plot the bar plot on the corresponding axis
    axs[i].bar(b, values)
    axs[i].set_title(c)
#hide any empty subplots
for i in range(num_plots, num_rows * num_cols):
    axs[i].axis('off')
#adjust the spacing between subplots
plt.tight_layout()
#show the grid of bar plots
plt.show()

#%% bar plot with avaiable hours per configuration
plt.bar(list(conf_duration.keys()), list(conf_duration.values())) #create plot with information of configurations name and durations
#set labels
plt.xlabel('Configurations')
plt.ylabel('Hours')
plt.title('Total time per configuration')
plt.show() #show plots

#%% grid of bar plot of where can be the EBs assigned
#get number of rows and columns for the grid layout
num_rows = (len(config) + 2) // 3
num_cols = 3
#create the grid of subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
fig.tight_layout(pad=3.0)
#iterate over configurations and plot bar charts
for i, c in enumerate(config):
    row = i // num_cols
    col = i % num_cols
    #count the number of ebs with value 1 for each bin in the configuration
    bins_counts = {}
    for sb, sb_data in eb_temp.items():
        if c in sb_data:
            bin_data = sb_data[c]
            for bin, count in bin_data.items():
                if bin not in bins_counts:
                    bins_counts[bin] = 0
                bins_counts[bin] += count
    bins = list(bins_counts.keys())
    counts = list(bins_counts.values())
    #plot the bar chart on the corresponding subplot
    ax = axes[row, col] if num_rows > 1 else axes[col]
    ax.bar(bins, counts)
    ax.set_xlabel('Bins')
    ax.set_ylabel('Count')
    ax.set_title(f'Configuration {c}')
#hide empty subplots if any
if len(config) < num_rows * num_cols:
    for i in range(len(config), num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].axis('off')
#show the grid of plots
plt.show()

#%% create csv with information of % completion per project to solve with 'project_completion_plots.ipynb'
#to run this cell 'eb_one_dict'(opt_model), 'grade_prj'(pre_process) dictionaries must exists
#eb_one_dict is created in the solution generation at the end of 'opt_model.py' or can be executed in the 'get eb_one_dict....'cell
percentage_prj = {} #create empty dictionary
for p,l_eb in projects_sb.items(): #for each project
    assigned_ebs = 0 #set ammount of assigned EBs to 0
    for e in l_eb: #for each EB of the project
        if e in eb_one_dict.keys(): #if the EB is in the dictionary of assigned EBs
            assigned_ebs +=1 #add a number to the ammount of assigned EBs per project p
    ebs_prj = len(projects_sb[p]) #get the number of all EBs in project p
    percentage_prj[p] = assigned_ebs/ebs_prj*100 #calculate the percentage of project p 
df_gra = pd.DataFrame.from_dict(grade_prj, orient='index', columns=['grade']) #create df with grade of projects
df_per = pd.DataFrame.from_dict(percentage_prj, orient='index', columns=['compl']) #create df with % of projects
df = pd.concat([df_per, df_gra], axis=1) #concat both dfs into new df
# df.to_csv('2023_11_03_3950_AB_compl.csv') #export df as csv file

#%% create csv with information of % completion per sb_uid
# eb_match_sbuid = B[39]['sb_uids'] #export sb_uids from a simulation of B dictionary in the npz file
eb_match_sbuid = df_main['SB_UID'] #get list of SB_UIDs
ebs_per_sbuid = {} #create empty dictionary to save EB per sb_uid
for execution_id, sb_uid in eb_match_sbuid.items(): #for each EB and SB_UID
    if sb_uid in ebs_per_sbuid: #if the SB_UID is in the dictionary
        ebs_per_sbuid[sb_uid].append(execution_id) #add the EB id to a list
    else:
        ebs_per_sbuid[sb_uid] = [execution_id] #create key of SB_UID in the dicionary
sbu_prj = {} #create empty dictionary to save percentage of completion per SB_UID
for sb_uid,l_eb in ebs_per_sbuid.items(): #for each sb_uid and list of EB that belong to it
    assigned_ebs = 0 #set count in 0
    for e in l_eb: #for each EB of the SB_UID
        if e in eb_one_dict.keys(): #if the EB is assigned (the variable is one)
            assigned_ebs +=1 #increase the count
    ebs_sbuid = df_main['NUMBER_OF_EXECUTIONS'][e] #get Execution number of the SB_UID
    sbu_prj[sb_uid] = assigned_ebs/ebs_sbuid*100 #calculate the percentage of SB_UID
sbu_grad = {} #create empty dictionary to save grades per SB_UID
for key in ebs_per_sbuid.keys(): #for each SB_UID
    sbu_grad[key] = df_main['PRJ_GRADE'][ebs_per_sbuid[key][0]]  #save the grade    
df_gra_s = pd.DataFrame.from_dict(sbu_grad, orient='index', columns=['grade']) #create df with grades of SB_UIDs
df_sbu = pd.DataFrame.from_dict(sbu_prj, orient='index', columns=['compl']) #create df with % of SB_UIDs
df = pd.concat([df_sbu, df_gra_s], axis=1) #concat both dfs into new df 
# df.to_csv('2023_11_14_3950_AB_compl.csv') #export df to csv file

#%% Plot with original and proposed durations of configurations
modified_durations = {} #create empty dictionary
for c in config: #for each configuration
    modified_durations[c] = extra_duration[c].x #get added or decreased ammount of days
time_table = pd.DataFrame.from_dict(modified_durations, orient='index') #create df to save information of the calendar
first_config = config[0] #get first configuration

#add original and total durations per configurations
def total_config_duration(c, conf_duration, extra_duration):
    suma=0
    suma = conf_duration[c] + (extra_duration[c].x * sum(Tti_mod[c].values()))
    return suma #function that adds time to the original calendar.

time_table.rename(columns = {0:'Extra_time'}, inplace = True) #rename column in dataframe
time_table['Total_duration'] = 0 #create new columns with total duration
time_table['Original_duration'] = 0 #create new column to get original duration
for c in config: #for each configuration
    time_table['Total_duration'][c] = total_config_duration(c, conf_duration, extra_duration) #get the total duration
    time_table['Original_duration'][c] = conf_duration[c] #get duration of the original calendar

# add original start_time column
time_table['start_time_original'] = 0
duration=0
for c in time_table.index:
    if c == first_config:
        time_table['start_time_original'][c] = 0
        duration += conf_duration[c]
    else:
        time_table['start_time_original'][c] = duration
        duration += conf_duration[c]
# add new start_time
time_table['start_time_results'] = 0
duration = 0
for c in time_table.index:
    if c == first_config:
        time_table['start_time_results'][c] = 0
        duration += time_table['Total_duration'][c]
    else:
        time_table['start_time_results'][c] = duration
        duration += time_table['Total_duration'][c]
        
#declaring a figure "gnt"
fig, gnt = plt.subplots()
#setting Y-axis limits
gnt.set_ylim(0, 45)
#setting X-axis limits
gnt.set_xlim(0, 4350)
#setting labels for x-axis and y-axis
gnt.set_xlabel('Hours since the beginning of the cycle')
gnt.set_ylabel('Configuration')
#setting position of tickets
ticket_position = np.arange(start = 3, stop = 45, step  = 3)
#setting ticks on y-axis
gnt.set_yticks(ticket_position)
#labelling tickes of y-axis
config_invert = config[::-1]
gnt.set_yticklabels(config_invert)
#set grid
gnt.grid(True)
#set dpi of figure
plt.figure(dpi=500)
#adding a legend
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, color='tab:orange', label='Original'),
    plt.Rectangle((0, 0), 1, 1, color='tab:blue', label='Results')]
gnt.legend(handles=legend_elements)
#setting starting positions                          
h0 = 42
h1 = 40
for c in config: #for each configuration
    start_t_original = time_table['start_time_original'][c]
    start_t_resultados = time_table['start_time_results'][c]
    original_duration = time_table['Original_duration'][c]
    results_duration = time_table['Total_duration'][c]
    gnt.broken_barh([(start_t_original, original_duration)], (h0, 2), facecolors =('tab:orange'))
    gnt.broken_barh([(start_t_resultados, results_duration)], (h1, 2), facecolors =('tab:blue'))
    h0-=3
    h1-=3
    # print(c)

#%% Histograms of estimated time per projects
#two plots, with estimated time per projects to compare lenght of accepted projects by ALMA vs Model

#to create the list of the accepted projects, a list with the PRJ_CODE is needed (c9_proj_list)
prj_hist = {} #create empty dictionary
for key in projects_sb.keys(): #for each project
    total_time = 0 #set count for time in 0
    if key in c9_proj_list: #if the project is accepted
    # if key in c10_prj_list:        
        for eb in projects_sb[key]: #for each EB of the project
            # total_time += bins_sb[eb] #add the time of the eb on bins
            total_time += df_main['SB_TIME_BY_EXECUTION'][eb] #ass the time of the eb in hours
        prj_hist[key] = total_time #save total time per project
values_1 = list(prj_hist.values()) #convert the dictionary values to a list for plotting

#to create the list of the accepted projects by the model the 'eb_one_dict' is needed
prj1_hist = {} #create empty ditionary
for key in projects_sb.keys(): #for each project
    total_time = 0 #set count for time in 0
    for eb in projects_sb[key]: #for each EB of the project
        if key not in prj1_hist.keys() and eb in eb_one_dict.keys(): #if the project is not in the dicitonary and the eb is assigned
            prj1_hist[key] = 0 #add project to the dictionary
    if key in prj1_hist.keys(): #if the project exist in the dictionary
        for eb in projects_sb[key]: #for each EB of the project
              # total_time += bins_sb[eb] #add time of the EB in bins
               total_time += df_main['SB_TIME_BY_EXECUTION'][eb] #add the time of the eb in hours
        prj1_hist[key] = total_time #save total time per project
values_2 = list(prj1_hist.values()) #convert the dictionary values to a list for plotting

#create a 1x2 grid for subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6)) #set size of the plot
#plot the first histogram in the first subplot
axs[0].hist(values_1, bins=50, range=[0, 50], edgecolor='black')
axs[0].set_xlabel('Total time')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Estimated time per accepted projects in hours C9')
axs[0].set_ylim(0, 42) #set the y-axis range

#plot the second histogram in the second subplot
axs[1].hist(values_2, bins=50, range=[0, 50], edgecolor='black')
axs[1].set_xlabel('Total time')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Estimated time per projects in hours Model')
axs[1].set_ylim(0, 42)
#adjust layout to prevent overlapping
plt.tight_layout()
#show the plot
plt.show()
