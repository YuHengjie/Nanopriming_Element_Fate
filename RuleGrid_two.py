# %%
import pandas as pd
import os
import re
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %%
file_list = os.listdir('./Model_RuleFit')
xlsx_files = [file for file in file_list if file.endswith('.xlsx')]

df_list = []

for i in range(1,11,1):
    exec("temp_pd = pd.read_excel(\"./Model_RuleFit/rules_{}.xlsx\",index_col = 0,)".format(i))
    temp_pd['Model'] = i
    df_list.append(temp_pd)

rules_all = pd.concat(df_list)
rules_all = rules_all.reset_index()
rules_all.to_excel('RuleFit_all_rules.xlsx')
rules_all = rules_all.drop(columns=['index'])
rules_all

# %%
print("Total number of generated rules: ",str(rules_all.shape[0]))

# %%
feature_dict = {'C': 'Concentration (mg/L)',
           'Z': 'Zeta potential (mV)',
           'H': 'Hydrodynamic diameter (nm)',
           'R': 'Relative weight',
           'B': 'BET surface area (m2/g)',
           'S': 'Seedling part',}
feature_dict

# %% simplify the rules
rules_all['rule'] = rules_all['rule'].str.replace('Concentration (mg/L)','C',regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('BET surface area (m2/g)','B',regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('Relative weight','R',regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('Hydrodynamic diameter (nm)','H',regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('Zeta potential (mV)','Z',regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('Seedling part','S',regex=False)
rules_all

# %%
def count_feature_letters(s):
    letters = set(['C', 'B', 'R', 'H', 'Z', 'S'])
    return len(set(filter(lambda c: c in letters, s)))

# %%
rules_all['Feature number'] = rules_all['rule'].apply(count_feature_letters)
rules_all = rules_all[rules_all['importance']>0.1]
rules_all

# %%
rules_two = rules_all[rules_all['Feature number']==2]
rules_two

# %%
B_C_index = []
H_R_index = []
C_R_index = []

for i,rule in enumerate(rules_two['rule']):
    if ('C' in rule) and ('B' in rule):
        B_C_index.append(i)
    if ('H' in rule) and ('R' in rule):
        H_R_index.append(i)
    if ('C' in rule) and ('R' in rule):
        C_R_index.append(i)

rule_B_C = rules_two.iloc[B_C_index,:]
rule_B_C = rule_B_C.reset_index(drop=True)

rule_H_R = rules_two.iloc[H_R_index,:]
rule_H_R = rule_H_R.reset_index(drop=True)

rule_C_R = rules_two.iloc[C_R_index,:]
rule_C_R = rule_C_R.reset_index(drop=True)

print(rule_B_C.shape[0],rule_H_R.shape[0],rule_C_R.shape[0])

# %%
rule_B_C.to_excel('rule_B_C.xlsx')

# %% the min and max value of each feature
C_range = [25, 200]
B_range = [4.07,200.84]
H_range = [197, 933.73]
R_range = [1, 1.87]

# can use to set same scale for plot
# give these two variables maxmimum values
larger_coeff_limit = 0.8314820655244579
larger_import_limit = 0.3412127346229156

# %% rule_to_interval_two
def rule_to_interval_two(matches: list, feature_1: str, feature_2: str, feature_1_range: list, feature_2_range: list): # order: 'B' > 'C' > 'R']

    interval_1 = 'None'
    interval_2 = 'None'

    if len(matches) == 1: # if only one statement, so only one feature
        # judge the feature, determin interval based on the logical operator and feature range
        if matches[0][0] == feature_1: 
            if matches[0][1] == '>':
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '<':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ']'
        if matches[0][0] == feature_2:
            if matches[0][1] == '>':
                interval_2 = '(' + str(matches[0][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_2 = '[' + str(matches[0][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[0][1] == '<':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[0][2]) + ']'

    if len(matches) == 2: # if the rule have two statements
        # judge the number of feature(s)
        if matches[0][0] == matches[1][0]:
            # if only one feature
            if matches[0][0] == feature_1:
                if (matches[0][1] == '>') & (matches[1][1] == '<'):
                        interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
                if (matches[0][1] == '>') & (matches[1][1] == '<='):
                        interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
                if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                        interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
                if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                        interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'         
            if matches[0][0] == feature_2:
                if (matches[0][1] == '>') & (matches[1][1] == '<'):
                        interval_2 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
                if (matches[0][1] == '>') & (matches[1][1] == '<='):
                        interval_2 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
                if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                        interval_2 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
                if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                        interval_2 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
        # if two feature
        else:
            if matches[0][1] == '>':
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '<':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ']'

            if matches[1][1] == '>':
                interval_2 = '(' + str(matches[1][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[1][1] == '>=':
                interval_2 = '[' + str(matches[1][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[1][1] == '<':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[1][2]) + ')'
            if matches[1][1] == '<=':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[1][2]) + ']'

    if len(matches) == 3:
        # judge which feature appear twice
        if matches[0][0] == matches[1][0]:
            # the first feature appear twice
            if (matches[0][1] == '>') & (matches[1][1] == '<'):
                    interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
            if (matches[0][1] == '>') & (matches[1][1] == '<='):
                    interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
            if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                    interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
            if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                    interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'  
            # so the second feature appear once       
            if matches[2][1] == '>':
                interval_2 = '(' + str(matches[2][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[2][1] == '>=':
                interval_2 = '[' + str(matches[2][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[2][1] == '<':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[2][2]) + ')'
            if matches[2][1] == '<=':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[2][2]) + ']'

        else:
            # the second feature appear twice
            if (matches[1][1] == '>') & (matches[2][1] == '<'):
                    interval_2 = '(' + str(matches[1][2]) + ', ' + str(matches[2][2]) + ')'
            if (matches[1][1] == '>') & (matches[2][1] == '<='):
                    interval_2 = '(' + str(matches[1][2]) + ', ' + str(matches[2][2]) + ']' 
            if (matches[1][1] == '>=') & (matches[2][1] == '<'):
                    interval_2 = '[' + str(matches[1][2]) + ', ' + str(matches[2][2]) + ')'
            if (matches[1][1] == '>=') & (matches[2][1] == '<='):
                    interval_2 = '[' + str(matches[1][2]) + ', ' + str(matches[2][2]) + ']'  
            # so the first feature appear once  
            if matches[0][1] == '>':
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '<':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ']'

    if len(matches) == 4:
        # the first feature appear twice
        if (matches[0][1] == '>') & (matches[1][1] == '<'):
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
        if (matches[0][1] == '>') & (matches[1][1] == '<='):
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
        if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
        if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'    
        
        if (matches[2][1] == '>') & (matches[3][1] == '<'):
                interval_2 = '(' + str(matches[2][2]) + ', ' + str(matches[3][2]) + ')'
        if (matches[2][1] == '>') & (matches[3][1] == '<='):
                interval_2 = '(' + str(matches[2][2]) + ', ' + str(matches[3][2]) + ']' 
        if (matches[2][1] == '>=') & (matches[3][1] == '<'):
                interval_2 = '[' + str(matches[2][2]) + ', ' + str(matches[3][2]) + ')'
        if (matches[2][1] == '>=') & (matches[3][1] == '<='):
                interval_2 = '[' + str(matches[2][2]) + ', ' + str(matches[3][2]) + ']'  

    return interval_1, interval_2


# %% 
feature_1 = 'B'
feature_2 = 'C'
feature_1_range = B_range
feature_2_range = C_range

rule = 'C <= 75.0 and B <= 60 and C > 37.5'
matches_raw = re.findall(r'(B|C)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number

rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)


# %% 1、 for rule map of B and C
feature_1 = 'B'
feature_2 = 'C'
feature_1_range = B_range
feature_2_range = C_range

rule_B_C['BET surface area (m2/g)'] = ''
rule_B_C['Concentration (mg/L)'] = ''

for i,rule in enumerate(rule_B_C['rule']):
    matches_raw = re.findall(r'(B|C)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_B_C.loc[i,['BET surface area (m2/g)','Concentration (mg/L)',]] = rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)

rule_B_C['BET surface area (m2/g)'].astype(str)
rule_B_C['Concentration (mg/L)'].astype(str)
rule_B_C

# %%
rule_B_C.to_excel('rule_B_C.xlsx')

# %%
print(rule_B_C.sort_values('Concentration (mg/L)')['Concentration (mg/L)'].unique())

# %%
rule_B_C_sort_C = {'[25, 75.0]':0,'[25, 150.0]':1,
                  '(75.0, 200]':2,
       }

# %%
print(rule_B_C.sort_values('BET surface area (m2/g)')['BET surface area (m2/g)'].unique())



# %%
rule_B_C_sort_B = {'[4.07, 5.01]':0,'[4.07, 15.31]':1, '[4.07, 33.53]':2, 
                   '[4.07, 45.705]':3,'[4.07, 73.47]':4, '[4.07, 92.805]':5,
                   '(45.705, 200.84]':6,
       }

# %%
rule_map_B_C_importance = pd.DataFrame(0,index=list(rule_B_C_sort_B.keys()),columns=list(rule_B_C_sort_C),)
rule_map_B_C_frequency = pd.DataFrame(0,index=list(rule_B_C_sort_B.keys()),columns=list(rule_B_C_sort_C),)
rule_map_B_C_coefficient = pd.DataFrame(0,index=list(rule_B_C_sort_B.keys()),columns=list(rule_B_C_sort_C),)
rule_map_B_C_importance

# %%
for i in range(rule_B_C.shape[0]):
    rule_map_B_C_importance.loc[rule_B_C.loc[i,'BET surface area (m2/g)'],rule_B_C.loc[i,'Concentration (mg/L)']] += rule_B_C.loc[i,'importance']/10
    rule_map_B_C_coefficient.loc[rule_B_C.loc[i,'BET surface area (m2/g)'],rule_B_C.loc[i,'Concentration (mg/L)']] += rule_B_C.loc[i,'coef']/10
    rule_map_B_C_frequency.loc[rule_B_C.loc[i,'BET surface area (m2/g)'],rule_B_C.loc[i,'Concentration (mg/L)']] += 1/10

rule_map_B_C_importance

# %%
rule_map_B_C_plot = pd.DataFrame(columns=['BET surface area (m2/g)','Concentration (mg/L)','Importance','Coefficient','Frequency'])
for B_item in  list(rule_B_C_sort_B.keys()):
    for C_item in list(rule_B_C_sort_C.keys()):
        temp_rule_map = pd.DataFrame({'BET surface area (m2/g)':B_item,
                                        'Concentration (mg/L)':C_item,
                                        'Importance': rule_map_B_C_importance.loc[B_item,C_item],
                                        'Coefficient':rule_map_B_C_coefficient.loc[B_item,C_item],
                                        'Frequency':rule_map_B_C_frequency.loc[B_item,C_item],
                                        }, index=[0])
        rule_map_B_C_plot = pd.concat([rule_map_B_C_plot,temp_rule_map],ignore_index=True)
rule_map_B_C_plot


# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (1.5, 2.5))
palette = sns.diverging_palette(220, 20, n=10,  as_cmap=True)

if max([abs(rule_map_B_C_coefficient.min().min()), abs(rule_map_B_C_coefficient.max().max())])>larger_coeff_limit:
    larger_coeff_limit = max([abs(rule_map_B_C_coefficient.min().min()), abs(rule_map_B_C_coefficient.max().max())])
    print('Warning: get larger coef limit for plot!')

if rule_map_B_C_importance.max().max() > larger_import_limit:
    larger_import_limit = rule_map_B_C_importance.max().max()
    print('Warning: get larger importance limit for plot!')


g = sns.scatterplot(
    data=rule_map_B_C_plot,x="Concentration (mg/L)", y="BET surface area (m2/g)", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(20,250*rule_map_B_C_importance.max().max()/larger_import_limit),
    ax=ax, palette = palette,hue_norm=(-larger_coeff_limit,larger_coeff_limit),
)

plt.xticks(rotation=45,ha='right')
plt.xlim(-0.5,2.5)
plt.ylim(-0.5,6.5)

plt.legend(bbox_to_anchor=(3.5, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )

norm = plt.Normalize(-larger_coeff_limit, larger_coeff_limit)
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm,)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Coefficient',shrink=0.7, anchor=(0, 1.0))

ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/RuleGrid_B_C.jpg",dpi=600,bbox_inches='tight')


# %%






# %% simple the intervals for better visualization, if several intervals are close
def simple_interval_func(intervals,threshold=10):
    interval_data = []
    for interval in intervals:
        left_paren, nums, right_paren = interval[0], interval[1:-1], interval[-1]
        num1, num2 = [float(num) for num in nums.split(', ')]
        interval_data.append([left_paren, num1, num2, right_paren])
    interval_df = pd.DataFrame(interval_data, columns=['left_paren', 'num1', 'num2', 'right_paren'])

    interval_df = interval_df.sort_values(by=['num1','num2'], ascending=True)
    interval_df['group'] = ''
    interval_df.iloc[0,4] = 0

    group_start = 0
    for index in range(1,interval_df.shape[0]):
        interval_df_temp = interval_df[interval_df['group']==group_start]
        if (abs(interval_df.iloc[index,1]-interval_df_temp['num1'].mean())<=threshold) and (abs(interval_df.iloc[index,2]-interval_df_temp['num2'].mean())<=threshold):
            interval_df.iloc[index,4] = group_start
        else:
            group_start += 1
            interval_df.iloc[index,4] = group_start

    interval_df['new num1'] = ''
    interval_df['new num2'] = ''

    for group_idnex in range(0,interval_df['group'].max()+1):
        interval_df_temp = interval_df[interval_df['group']==group_idnex]
        interval_df.loc[interval_df_temp.index,'new num1'] = round(interval_df_temp['num1'].mean(),3)
        interval_df.loc[interval_df_temp.index,'new num2'] = round(interval_df_temp['num2'].mean(),3)

    interval_df['simple'] = ''
    for index in interval_df.index:
        interval_df.loc[index,'simple'] = interval_df.loc[index,'left_paren']+str(interval_df.loc[index,'new num1'])+','+str(interval_df.loc[index,'new num2'])+interval_df.loc[index,'right_paren']
    
    interval_df=interval_df.sort_index(axis=0)

    return interval_df['simple'].values

# %% 2、 for rule map of H and R
feature_1 = 'H'
feature_2 = 'R'
feature_1_range = H_range
feature_2_range = R_range

rule_H_R['Hydrodynamic diameter (nm)'] = ''
rule_H_R['Relative weight'] = ''

for i,rule in enumerate(rule_H_R['rule']):
    matches_raw = re.findall(r'(H|R)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_H_R.loc[i,['Hydrodynamic diameter (nm)','Relative weight',]] = rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)

rule_H_R['Hydrodynamic diameter (nm)'].astype(str)
rule_H_R['Relative weight'].astype(str)
rule_H_R

# %%
print(rule_H_R.sort_values('Hydrodynamic diameter (nm)')['Hydrodynamic diameter (nm)'].unique())

# %%
rule_H_R_simply_H = pd.DataFrame({'Raw interval':[
                                '[197, 276.06999]','(220.665, 933.73]' ,
                                '(276.06999, 840.285]', '(344.45, 933.73]',
                                '(363.47, 900.76498]', '(363.47, 933.73]' ,
                                '(900.76498, 933.73]',
                                
                                     ],})
rule_H_R_simply_H.loc[0:,'Simply interval'] = simple_interval_func(rule_H_R_simply_H['Raw interval'].values[0:],
                                                                       threshold=(H_range[1]-H_range[0])*0.05)
len(rule_H_R_simply_H['Simply interval'].unique())

# %%
rule_H_R_simply_H.to_excel('H_R_simply_H.xlsx')
rule_H_R_simply_H

# %% replace the interval for better visualization
rule_H_R_simply = rule_H_R.copy()
for i,item in enumerate(rule_H_R_simply['Hydrodynamic diameter (nm)']):
     index = list(rule_H_R_simply_H['Raw interval']).index(item)
     rule_H_R_simply.loc[i,'Hydrodynamic diameter (nm)'] = rule_H_R_simply_H.loc[index,'Simply interval']
rule_H_R_simply

# %%
print(rule_H_R.sort_values('Relative weight')['Relative weight'].unique())

# %%
rule_H_R_simply_R = pd.DataFrame({'Raw interval':[
                                '[1, 1.11]', '[1, 1.185]', '[1, 1.465]', '[1, 1.795]',
                                '(1.11, 1.87]', '(1.125, 1.87]', '(1.135, 1.455]' ,
                                '(1.175, 1.355]','(1.175, 1.87]', '(1.185, 1.355]',
                                 '(1.185, 1.5]' ,'(1.225, 1.87]',
                                     ],})
rule_H_R_simply_R.loc[0:,'Simply interval'] = simple_interval_func(rule_H_R_simply_R['Raw interval'].values[0:],
                                                                       threshold=(R_range[1]-R_range[0])*0.05)
len(rule_H_R_simply_R['Simply interval'].unique())

# %%
rule_H_R_simply_R.to_excel('H_R_simply_R.xlsx')
rule_H_R_simply_R

# %% replace the interval for better visualization
for i,item in enumerate(rule_H_R_simply['Relative weight']):
     index = list(rule_H_R_simply_R['Raw interval']).index(item)
     rule_H_R_simply.loc[i,'Relative weight'] = rule_H_R_simply_R.loc[index,'Simply interval']
rule_H_R_simply

# %%
rule_map_H_R_importance = pd.DataFrame(0,index=list(rule_H_R_simply_H['Simply interval'].unique()),columns=list(rule_H_R_simply_R['Simply interval'].unique()),)
rule_map_H_R_frequency = pd.DataFrame(0,index=list(rule_H_R_simply_H['Simply interval'].unique()),columns=list(rule_H_R_simply_R['Simply interval'].unique()),)
rule_map_H_R_coefficient = pd.DataFrame(0,index=list(rule_H_R_simply_H['Simply interval'].unique()),columns=list(rule_H_R_simply_R['Simply interval'].unique()),)
rule_map_H_R_importance

# %%
for i in range(rule_H_R_simply.shape[0]):
    rule_map_H_R_importance.loc[rule_H_R_simply.loc[i,'Hydrodynamic diameter (nm)'],rule_H_R_simply.loc[i,'Relative weight']] += rule_H_R_simply.loc[i,'importance']/10
    rule_map_H_R_coefficient.loc[rule_H_R_simply.loc[i,'Hydrodynamic diameter (nm)'],rule_H_R_simply.loc[i,'Relative weight']] += rule_H_R_simply.loc[i,'coef']/10
    rule_map_H_R_frequency.loc[rule_H_R_simply.loc[i,'Hydrodynamic diameter (nm)'],rule_H_R_simply.loc[i,'Relative weight']] += 1/10

rule_map_H_R_importance

# %%
rule_map_H_R_plot = pd.DataFrame(columns=['Hydrodynamic diameter (nm)','Relative weight','Importance','Coefficient','Frequency'])
for H_item in  list(rule_H_R_simply_H['Simply interval'].unique()):
    for R_item in list(rule_H_R_simply_R['Simply interval'].unique()):
        temp_rule_map = pd.DataFrame({'Hydrodynamic diameter (nm)':H_item,
                                        'Relative weight':R_item,
                                        'Importance': rule_map_H_R_importance.loc[H_item,R_item],
                                        'Coefficient':rule_map_H_R_coefficient.loc[H_item,R_item],
                                        'Frequency':rule_map_H_R_frequency.loc[H_item,R_item],
                                        }, index=[0])
        rule_map_H_R_plot = pd.concat([rule_map_H_R_plot,temp_rule_map],ignore_index=True)
rule_map_H_R_plot


# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (2.5, 3.5))
palette = sns.diverging_palette(220, 20, n=10,  as_cmap=True)

if max([abs(rule_map_H_R_coefficient.min().min()), abs(rule_map_H_R_coefficient.max().max())])>larger_coeff_limit:
    larger_coeff_limit = max([abs(rule_map_H_R_coefficient.min().min()), abs(rule_map_H_R_coefficient.max().max())])
    print('Warning: get larger coef limit for plot!')

if rule_map_H_R_importance.max().max() > larger_import_limit:
    larger_import_limit = rule_map_H_R_importance.max().max()
    print('Warning: get larger importance limit for plot!')


g = sns.scatterplot(
    data=rule_map_H_R_plot,x="Hydrodynamic diameter (nm)", y="Relative weight", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(20,250*rule_map_H_R_importance.max().max()/larger_import_limit),
    ax=ax, palette = palette,hue_norm=(-larger_coeff_limit,larger_coeff_limit),
)

plt.xticks(rotation=45,ha='right')

plt.xlim(-0.5,4.5)
plt.ylim(-0.5,10.5)

plt.legend(bbox_to_anchor=(2.5, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )

norm = plt.Normalize(-larger_coeff_limit, larger_coeff_limit)
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm,)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Coefficient',shrink=0.7, anchor=(0, 1.0))

ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/RuleGrid_H_R.jpg",dpi=600,bbox_inches='tight')








# %% 3、 for rule map of C and R
feature_1 = 'C'
feature_2 = 'R'
feature_1_range = C_range
feature_2_range = R_range

rule_C_R['Concentration (mg/L)'] = ''
rule_C_R['Relative weight'] = ''

for i,rule in enumerate(rule_C_R['rule']):
    matches_raw = re.findall(r'(C|R)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_C_R.loc[i,['Concentration (mg/L)','Relative weight',]] = rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)

rule_C_R['Concentration (mg/L)'].astype(str)
rule_C_R['Relative weight'].astype(str)
rule_C_R

# %%
print(rule_C_R.sort_values('Concentration (mg/L)')['Concentration (mg/L)'].unique())

# %%
rule_C_R_sort_C = {'[25, 37.5]':0, '[25, 150.0]':1,
                  '(37.5, 200]':2, '(150.0, 200]':3,
       }

# %%
print(rule_C_R.sort_values('Relative weight')['Relative weight'].unique())

# %%
rule_C_R_simply_R = pd.DataFrame({'Raw interval':[
                                '(1.1, 1.87]', '(1.105, 1.87]', '(1.235, 1.305]',
                                  '(1.245, 1.87]', '(1.265, 1.87]', '(1.305, 1.87]', 
                                  '(1.355, 1.87]' ,'(1.435, 1.87]',
                                     ],})
rule_C_R_simply_R.loc[0:,'Simply interval'] = simple_interval_func(rule_C_R_simply_R['Raw interval'].values[0:],
                                                                       threshold=(R_range[1]-R_range[0])*0.05)
len(rule_C_R_simply_R['Simply interval'].unique())

# %%
rule_C_R_simply_R.to_excel('C_R_simply_R.xlsx')
rule_C_R_simply_R

# %% replace the interval for better visualization
rule_C_R_simply = rule_C_R.copy()
for i,item in enumerate(rule_C_R_simply['Relative weight']):
     index = list(rule_C_R_simply_R['Raw interval']).index(item)
     rule_C_R_simply.loc[i,'Relative weight'] = rule_C_R_simply_R.loc[index,'Simply interval']
rule_C_R_simply

# %%
rule_map_C_R_importance = pd.DataFrame(0,index=list(rule_C_R_sort_C.keys()),columns=list(rule_C_R_simply_R['Simply interval'].unique()),)
rule_map_C_R_frequency = pd.DataFrame(0,index=list(rule_C_R_sort_C.keys()),columns=list(rule_C_R_simply_R['Simply interval'].unique()),)
rule_map_C_R_coefficient = pd.DataFrame(0,index=list(rule_C_R_sort_C.keys()),columns=list(rule_C_R_simply_R['Simply interval'].unique()),)
rule_map_C_R_importance

# %%
for i in range(rule_C_R_simply.shape[0]):
    rule_map_C_R_importance.loc[rule_C_R_simply.loc[i,'Concentration (mg/L)'],rule_C_R_simply.loc[i,'Relative weight']] += rule_C_R_simply.loc[i,'importance']/10
    rule_map_C_R_coefficient.loc[rule_C_R_simply.loc[i,'Concentration (mg/L)'],rule_C_R_simply.loc[i,'Relative weight']] += rule_C_R_simply.loc[i,'coef']/10
    rule_map_C_R_frequency.loc[rule_C_R_simply.loc[i,'Concentration (mg/L)'],rule_C_R_simply.loc[i,'Relative weight']] += 1/10

rule_map_C_R_importance

# %%
rule_map_C_R_plot = pd.DataFrame(columns=['Concentration (mg/L)','Relative weight','Importance','Coefficient','Frequency'])
for C_item in  list(rule_C_R_sort_C.keys()):
    for R_item in list(rule_C_R_simply_R['Simply interval'].unique()):
        temp_rule_map = pd.DataFrame({'Concentration (mg/L)':C_item,
                                        'Relative weight':R_item,
                                        'Importance': rule_map_C_R_importance.loc[C_item,R_item],
                                        'Coefficient':rule_map_C_R_coefficient.loc[C_item,R_item],
                                        'Frequency':rule_map_C_R_frequency.loc[C_item,R_item],
                                        }, index=[0])
        rule_map_C_R_plot = pd.concat([rule_map_C_R_plot,temp_rule_map],ignore_index=True)
rule_map_C_R_plot


# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (1.8, 2.2))
palette = sns.diverging_palette(220, 20, n=10,  as_cmap=True)

if max([abs(rule_map_C_R_coefficient.min().min()), abs(rule_map_C_R_coefficient.max().max())])>larger_coeff_limit:
    larger_coeff_limit = max([abs(rule_map_C_R_coefficient.min().min()), abs(rule_map_C_R_coefficient.max().max())])
    print('Warning: get larger coef limit for plot!')

if rule_map_C_R_importance.max().max() > larger_import_limit:
    larger_import_limit = rule_map_C_R_importance.max().max()
    print('Warning: get larger importance limit for plot!')


g = sns.scatterplot(
    data=rule_map_C_R_plot,x="Concentration (mg/L)", y="Relative weight", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(20,250*rule_map_C_R_importance.max().max()/larger_import_limit),
    ax=ax, palette = palette,hue_norm=(-larger_coeff_limit,larger_coeff_limit),
)

plt.xticks(rotation=45,ha='right')

plt.xlim(-0.5,3.5)
plt.ylim(-0.5,5.5)

plt.legend(bbox_to_anchor=(3, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )

norm = plt.Normalize(-larger_coeff_limit, larger_coeff_limit)
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm,)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Coefficient',shrink=0.7, anchor=(0, 1.0))

ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/RuleGrid_C_R.jpg",dpi=600,bbox_inches='tight')

# %%
