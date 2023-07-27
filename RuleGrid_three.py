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
rules_three = rules_all[rules_all['Feature number']==3]
rules_three

# %%
B_C_H_index = []
B_C_R_index = []

for i,rule in enumerate(rules_three['rule']):
    if ('B' in rule) and ('C' in rule) and ('H' in rule):
        B_C_H_index.append(i)
    if ('B' in rule) and ('C' in rule) and ('R' in rule):
        B_C_R_index.append(i)

rule_B_C_H = rules_three.iloc[B_C_H_index,:]
rule_B_C_H = rule_B_C_H.reset_index(drop=True)

rule_B_C_R = rules_three.iloc[B_C_R_index,:]
rule_B_C_R = rule_B_C_R.reset_index(drop=True)

print(rule_B_C_H.shape[0],rule_B_C_R.shape[0])

# %% the min and max value of each feature
C_range = [25, 200]
H_range = [197, 933.73]
R_range = [1, 1.87]
B_range = [4.07,200.84]

# set to 0 when first run, then change to the max value by manual
larger_coeff_limit = 0.47060894589716795
larger_import_limit = 0.18859064080101479

# %% rule_to_interval_three
def rule_to_interval_three(matches: list, feature_1: str, feature_2: str, feature_3: str, feature_1_range: list, feature_2_range: list, feature_3_range: list): # order: 'B' > 'C' > 'R']

    interval_1 = 'None'
    interval_2 = 'None'
    interval_3 = 'None'

    feature_in = [matches[i][0] for i in range(len(matches))]

    feature_1_count = feature_in.count(feature_1)
    feature_2_count = feature_in.count(feature_2)
    feature_3_count = feature_in.count(feature_3)

    start = 0

    if feature_1_count > 0:
        if feature_1_count == 1:
            if matches[0][1] == '>':
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '>=':
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
            if matches[0][1] == '<':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ')'
            if matches[0][1] == '<=':
                interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ']'
        else:
            if (matches[0][1] == '>') & (matches[1][1] == '<'):
                    interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
            if (matches[0][1] == '>') & (matches[1][1] == '<='):
                    interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
            if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                    interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
            if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                    interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
        start += feature_1_count

    if feature_2_count > 0:
        if feature_2_count == 1:
            if matches[start+0][1] == '>':
                interval_2 = '(' + str(matches[start+0][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[start+0][1] == '>=':
                interval_2 = '[' + str(matches[start+0][2]) + ', ' + str(feature_2_range[1]) + ']'
            if matches[start+0][1] == '<':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[start+0][2]) + ')'
            if matches[start+0][1] == '<=':
                interval_2 = '[' + str(feature_2_range[0]) + ', ' + str(matches[start+0][2]) + ']'
        else:
            if (matches[start+0][1] == '>') & (matches[start+1][1] == '<'):
                interval_2 = '(' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ')'
            if (matches[start+0][1] == '>') & (matches[start+1][1] == '<='):
                interval_2 = '(' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ']' 
            if (matches[start+0][1] == '>=') & (matches[start+1][1] == '<'):
                interval_2 = '[' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ')'
            if (matches[start+0][1] == '>=') & (matches[start+1][1] == '<='):
                interval_2 = '[' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ']' 
        start += feature_2_count

    if feature_3_count > 0:
        if feature_3_count == 1:
            if matches[start+0][1] == '>':
                interval_3 = '(' + str(matches[start+0][2]) + ', ' + str(feature_3_range[1]) + ']'
            if matches[start+0][1] == '>=':
                interval_3 = '[' + str(matches[start+0][2]) + ', ' + str(feature_3_range[1]) + ']'
            if matches[start+0][1] == '<':
                interval_3 = '[' + str(feature_3_range[0]) + ', ' + str(matches[start+0][2]) + ')'
            if matches[start+0][1] == '<=':
                interval_3 = '[' + str(feature_3_range[0]) + ', ' + str(matches[start+0][2]) + ']'
        else:
            if (matches[start+0][1] == '>') & (matches[start+1][1] == '<'):
                interval_3 = '(' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ')'
            if (matches[start+0][1] == '>') & (matches[start+1][1] == '<='):
                interval_3 = '(' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ']' 
            if (matches[start+0][1] == '>=') & (matches[start+1][1] == '<'):
                interval_3 = '[' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ')'
            if (matches[start+0][1] == '>=') & (matches[start+1][1] == '<='):
                interval_3 = '[' + str(matches[start+0][2]) + ', ' + str(matches[start+1][2]) + ']' 

    return interval_1, interval_2, interval_3

# %% test rule_to_interval_three function
feature_1 = 'B'
feature_2 = 'C'
feature_3 = 'H'
feature_1_range = B_range
feature_2_range = C_range
feature_3_range = H_range

rule = 'C <= 150.0 and H <= 363.47 and B <= 60.325'
matches_raw = re.findall(r'(B|C|H)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number

rule_to_interval_three(matches, feature_1, feature_2, feature_3, feature_1_range, feature_2_range, feature_3_range)

# %% 1. rule map for B C H

feature_1 = 'B'
feature_2 = 'C'
feature_3 = 'H'
feature_1_range = B_range
feature_2_range = C_range
feature_3_range = H_range

rule_B_C_H['BET surface area (m2/g)'] = ''
rule_B_C_H['Concentration (mg/L)'] = ''
rule_B_C_H['Hydrodynamic diameter (nm)'] = ''


for i,rule in enumerate(rule_B_C_H['rule']):
    matches_raw = re.findall(r'(B|C|H)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_B_C_H.loc[i,['BET surface area (m2/g)','Concentration (mg/L)','Hydrodynamic diameter (nm)',]] = rule_to_interval_three(matches, feature_1, feature_2, feature_3, feature_1_range, feature_2_range, feature_3_range)

rule_B_C_H

# %%
print(rule_B_C_H.sort_values('BET surface area (m2/g)')['BET surface area (m2/g)'].unique())

# %%
rule_B_C_H_sort_B = {'[4.07, 45.705]':0, '(45.705, 200.84]':1, 
                    }

# %%
print(rule_B_C_H.sort_values('Concentration (mg/L)')['Concentration (mg/L)'].unique())

# %%
rule_B_C_H_sort_C = {'[25, 75.0]':0, '(75.0, 200]':1, 
                    }

# %%
print(rule_B_C_H.sort_values('Hydrodynamic diameter (nm)')['Hydrodynamic diameter (nm)'].unique())

# %%
rule_B_C_H_sort_H = {'[197, 357.45]':0, '[197, 763.89999]':1, 
                     '[197, 840.285]':2, '[197, 900.76498]':3,
                     '(220.665, 933.73]':4, '(253.07999, 933.73]':5,
                    }

# %%
# %%
rule_B_C_H_combnination = rule_B_C_H.copy()

for i in range(rule_B_C_H_combnination.shape[0]):
    rule_B_C_H_combnination.loc[i,'Combination'] = rule_B_C_H_combnination.loc[i,'BET surface area (m2/g)'] + ' and ' + rule_B_C_H_combnination.loc[i,'Concentration (mg/L)']

rule_B_C_H_combnination


# %%
rule_map_B_C_H_importance = pd.DataFrame(0,index=list(rule_B_C_H_combnination['Combination'].unique()),columns=list(rule_B_C_H_sort_H),)
rule_map_B_C_H_frequency = pd.DataFrame(0,index=list(rule_B_C_H_combnination['Combination'].unique()),columns=list(rule_B_C_H_sort_H),)
rule_map_B_C_H_coefficient = pd.DataFrame(0,index=list(rule_B_C_H_combnination['Combination'].unique()),columns=list(rule_B_C_H_sort_H),)
rule_map_B_C_H_importance

# %%
for i in range(rule_B_C_H_combnination.shape[0]):
    rule_map_B_C_H_importance.loc[rule_B_C_H_combnination.loc[i,'Combination'],rule_B_C_H_combnination.loc[i,'Hydrodynamic diameter (nm)']] += rule_B_C_H_combnination.loc[i,'importance']/10
    rule_map_B_C_H_coefficient.loc[rule_B_C_H_combnination.loc[i,'Combination'],rule_B_C_H_combnination.loc[i,'Hydrodynamic diameter (nm)']] += rule_B_C_H_combnination.loc[i,'coef']/10
    rule_map_B_C_H_frequency.loc[rule_B_C_H_combnination.loc[i,'Combination'],rule_B_C_H_combnination.loc[i,'Hydrodynamic diameter (nm)']] += 1/10

rule_map_B_C_H_importance

# %%
rule_map_B_C_H_plot = pd.DataFrame(columns=['Hydrodynamic diameter (nm)','Combination','Importance','Coefficient','Frequency'])
for Com_item in list(rule_B_C_H_combnination['Combination'].unique()):
    for H_item in  list(rule_B_C_H_sort_H):
        temp_rule_map = pd.DataFrame({'Hydrodynamic diameter (nm)':H_item,
                                        'Combination':Com_item,
                                        'Importance': rule_map_B_C_H_importance.loc[Com_item,H_item],
                                        'Coefficient':rule_map_B_C_H_coefficient.loc[Com_item,H_item],
                                        'Frequency':rule_map_B_C_H_frequency.loc[Com_item,H_item],
                                        }, index=[0])
        rule_map_B_C_H_plot = pd.concat([rule_map_B_C_H_plot,temp_rule_map],ignore_index=True)
rule_map_B_C_H_plot

# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (1.8, 2.6))
palette = sns.diverging_palette(220, 20, n=10,  as_cmap=True)

if max([abs(rule_map_B_C_H_coefficient.min().min()), abs(rule_map_B_C_H_coefficient.max().max())])>larger_coeff_limit:
    larger_coeff_limit = max([abs(rule_map_B_C_H_coefficient.min().min()), abs(rule_map_B_C_H_coefficient.max().max())])
    print('Warning: get larger coef limit for plot!')

if rule_map_B_C_H_importance.max().max() > larger_import_limit:
    larger_import_limit = rule_map_B_C_H_importance.max().max()
    print('Warning: get larger importance limit for plot!')

g = sns.scatterplot(
    data=rule_map_B_C_H_plot,x="Combination", y="Hydrodynamic diameter (nm)", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(20,250*rule_map_B_C_H_importance.max().max()/larger_import_limit),
    ax=ax, palette = palette,hue_norm=(-larger_coeff_limit,larger_coeff_limit),
)

plt.xticks(rotation=45,ha='right')

plt.xlim(-0.5,2.5)
plt.ylim(-0.5,5.5)

plt.legend(bbox_to_anchor=(3.5, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )

plt.xlabel('BET surface area (m2/g) and Concentration (mg/L)')

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

fig.savefig("./Image/RuleGrid_B_C_H.jpg",dpi=600,bbox_inches='tight')

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


# %% 2. rule map for B C R
feature_1 = 'B'
feature_2 = 'C'
feature_3 = 'R'
feature_1_range = B_range
feature_2_range = C_range
feature_3_range = R_range

rule_B_C_R['BET surface area (m2/g)'] = ''
rule_B_C_R['Concentration (mg/L)'] = ''
rule_B_C_R['Relative weight'] = ''

for i,rule in enumerate(rule_B_C_R['rule']):
    matches_raw = re.findall(r'(B|C|R)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_B_C_R.loc[i,['BET surface area (m2/g)','Concentration (mg/L)','Relative weight',]] = rule_to_interval_three(matches, feature_1, feature_2, feature_3, feature_1_range, feature_2_range, feature_3_range)

rule_B_C_R

# %%
print(rule_B_C_R.sort_values('BET surface area (m2/g)')['BET surface area (m2/g)'].unique())

# %%
B_C_R_interval_simply_B = pd.DataFrame({'Raw interval':[
                                    '[4.07, 5.01]','[4.07, 27.46]', '[4.07, 45.705]' ,
                                    '[4.07, 127.18999]', '(5.01, 200.84]','(45.705, 149.7]' ,
                                    '(45.705, 200.84]','(48.595, 200.84]',  '(56.85, 200.84]',
                                    '(127.18999, 200.84]',
                                     ],})
B_C_R_interval_simply_B.loc[0:,'Simply interval'] = simple_interval_func(B_C_R_interval_simply_B['Raw interval'].values[0:],
                                                                       threshold=(B_range[1]-B_range[0])*0.05)

len(B_C_R_interval_simply_B['Simply interval'].unique())

# %%
B_C_R_interval_simply_B.to_excel('B_C_R_interval_simply_B.xlsx')
B_C_R_interval_simply_B

# %% replace the interval for better visualization
rule_B_C_R_simply = rule_B_C_R.copy()
for i,item in enumerate(rule_B_C_R_simply['BET surface area (m2/g)']):
     index = list(B_C_R_interval_simply_B['Raw interval']).index(item)
     rule_B_C_R_simply.loc[i,'BET surface area (m2/g)'] = B_C_R_interval_simply_B.loc[index,'Simply interval']
rule_B_C_R_simply

# %%
print(rule_B_C_R.sort_values('Concentration (mg/L)')['Concentration (mg/L)'].unique())

# %%
rule_B_C_R_sort_C = {'[25, 37.5]':0, '[25, 75.0]':1, 
                     '[25, 150.0]':2, '(75.0, 200]':3, 
                     '(150.0, 200]':4,
                    }
    
# %%
print(rule_B_C_R.sort_values('Relative weight')['Relative weight'].unique())

# %%
B_C_R_interval_simply_R = pd.DataFrame({'Raw interval':[
                                    '[1, 1.225]', '[1, 1.305]' ,'[1, 1.335]', '[1, 1.385]',
                                    '[1, 1.405]', '[1, 1.415]', '[1, 1.43]', '[1, 1.865]',
                                    '(1.035, 1.305]', '(1.075, 1.87]', '(1.105, 1.505]', '(1.105, 1.87]',
                                    '(1.125, 1.395]', '(1.16, 1.245]', '(1.175, 1.305]', '(1.175, 1.87]',
                                    '(1.385, 1.87]' ,

                                     ],})
B_C_R_interval_simply_R.loc[0:,'Simply interval'] = simple_interval_func(B_C_R_interval_simply_R['Raw interval'].values[0:],
                                                                       threshold=(R_range[1]-R_range[0])*0.05)

len(B_C_R_interval_simply_R['Simply interval'].unique())

# %%
B_C_R_interval_simply_R.to_excel('B_C_R_interval_simply_R.xlsx')
B_C_R_interval_simply_R

# %% replace the interval for better visualization
for i,item in enumerate(rule_B_C_R_simply['Relative weight']):
     index = list(B_C_R_interval_simply_R['Raw interval']).index(item)
     rule_B_C_R_simply.loc[i,'Relative weight'] = B_C_R_interval_simply_R.loc[index,'Simply interval']
rule_B_C_R_simply

# %%
rule_B_C_R_combnination = rule_B_C_R_simply.copy()

for i in range(rule_B_C_R_combnination.shape[0]):
    rule_B_C_R_combnination.loc[i,'Combination'] = rule_B_C_R_combnination.loc[i,'BET surface area (m2/g)'] + ' and ' + rule_B_C_R_combnination.loc[i,'Concentration (mg/L)']

rule_B_C_R_combnination


# %%
rule_map_B_C_R_importance = pd.DataFrame(0,index=list(rule_B_C_R_combnination['Combination'].unique()),columns=list(B_C_R_interval_simply_R['Simply interval'].unique()),)
rule_map_B_C_R_frequency = pd.DataFrame(0,index=list(rule_B_C_R_combnination['Combination'].unique()),columns=list(B_C_R_interval_simply_R['Simply interval'].unique()),)
rule_map_B_C_R_coefficient = pd.DataFrame(0,index=list(rule_B_C_R_combnination['Combination'].unique()),columns=list(B_C_R_interval_simply_R['Simply interval'].unique()),)
rule_map_B_C_R_importance

# %%
for i in range(rule_B_C_R_combnination.shape[0]):
    rule_map_B_C_R_importance.loc[rule_B_C_R_combnination.loc[i,'Combination'],rule_B_C_R_combnination.loc[i,'Relative weight']] += rule_B_C_R_combnination.loc[i,'importance']/10
    rule_map_B_C_R_coefficient.loc[rule_B_C_R_combnination.loc[i,'Combination'],rule_B_C_R_combnination.loc[i,'Relative weight']] += rule_B_C_R_combnination.loc[i,'coef']/10
    rule_map_B_C_R_frequency.loc[rule_B_C_R_combnination.loc[i,'Combination'],rule_B_C_R_combnination.loc[i,'Relative weight']] += 1/10

rule_map_B_C_R_importance

# %%
rule_map_B_C_R_plot = pd.DataFrame(columns=['Relative weight','Combination','Importance','Coefficient','Frequency'])
for Com_item in list(rule_B_C_R_combnination['Combination'].unique()):
    for R_item in  list(B_C_R_interval_simply_R['Simply interval'].unique()):
        temp_rule_map = pd.DataFrame({'Relative weight':R_item,
                                        'Combination':Com_item,
                                        'Importance': rule_map_B_C_R_importance.loc[Com_item,R_item],
                                        'Coefficient':rule_map_B_C_R_coefficient.loc[Com_item,R_item],
                                        'Frequency':rule_map_B_C_R_frequency.loc[Com_item,R_item],
                                        }, index=[0])
        rule_map_B_C_R_plot = pd.concat([rule_map_B_C_R_plot,temp_rule_map],ignore_index=True)
rule_map_B_C_R_plot

# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (5.7, 5))
palette = sns.diverging_palette(220, 20, n=10,  as_cmap=True)

if max([abs(rule_map_B_C_R_coefficient.min().min()), abs(rule_map_B_C_R_coefficient.max().max())])>larger_coeff_limit:
    larger_coeff_limit = max([abs(rule_map_B_C_R_coefficient.min().min()), abs(rule_map_B_C_R_coefficient.max().max())])
    print('Warning: get larger coef limit for plot!')

if rule_map_B_C_R_importance.max().max() > larger_import_limit:
    larger_import_limit = rule_map_B_C_R_importance.max().max()
    print('Warning: get larger importance limit for plot!')

g = sns.scatterplot(
    data=rule_map_B_C_R_plot,x="Combination", y="Relative weight", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(20,250*rule_map_B_C_R_importance.max().max()/larger_import_limit),
    ax=ax, palette = palette,hue_norm=(-larger_coeff_limit,larger_coeff_limit),
)

plt.xticks(rotation=45,ha='right')

#plt.xlim(-0.5,2.5)
#plt.ylim(-0.5,6.5)

plt.legend(bbox_to_anchor=(1.8, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )

plt.xlabel('BET surface area (m2/g) and Concentration (mg/L)')

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

fig.savefig("./Image/RuleGrid_B_C_R.jpg",dpi=600,bbox_inches='tight')

# %%
