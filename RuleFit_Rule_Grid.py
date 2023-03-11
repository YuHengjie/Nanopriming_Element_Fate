# %%
import pandas as pd
import os
import re

import numpy as np
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,GradientBoostingClassifier

from sklearn import metrics
from imodels import RuleFitClassifier

import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
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
def find_mk(input_vars:list, rule:str):

    var_count = 0
    for var in input_vars:
        if var in rule:
            var_count += 1
    return(var_count)

def get_feature_importance(feature_set: list, rule_set: pd.DataFrame, scaled = False):

    feature_imp = list()
    
    rule_feature_count = rule_set.rule.apply(lambda x: find_mk(feature_set, x))

    for feature in feature_set:
        
        # find subset of rules that apply to a feature
        feature_rk = rule_set.rule.apply(lambda x: feature in x)
        
        # find importance of linear features
        linear_imp = rule_set[(rule_set.type=='linear')&(rule_set.rule==feature)].importance.values
        
        # find the importance of rules that contain feature
        rule_imp = rule_set.importance[(rule_set.type=='rule')&feature_rk]
        
        # find the number of features in each rule that contain feature
        m_k = rule_feature_count[(rule_set.type=='rule')&feature_rk]
        
        # sum the linear and rule importances, divided by m_k
        if len(linear_imp)==0:
            linear_imp = 0
        # sum the linear and rule importances, divided by m_k
        if len(rule_imp) == 0:
            feature_imp.append(float(linear_imp))
        else:
            feature_imp.append(float(linear_imp + (rule_imp/m_k).sum()))
        
    if scaled:
        feature_imp = 100*(feature_imp/np.array(feature_imp).max())
    
    return(feature_imp)

# %%
feature_names = ['Concentration (mg/L)','BET surface area (m2/g)','Relative weight']
feature_importances = get_feature_importance(feature_names, rules_all, scaled=False)
importance_df = pd.DataFrame(feature_importances, index = feature_names, columns = ['importance']).sort_values(by='importance',ascending=False)
importance_df['importance'] = importance_df['importance']/10
importance_df

# %%
color_list = ['#FCFEA4','#FB9B06','#CF4446',]
fig, ax1 = plt.subplots(figsize=(2,2)) 

plt.bar(importance_df.index,importance_df['importance'],width=0.4,color=color_list,edgecolor='k',alpha=0.6)

plt.xticks(rotation=45,ha='right')
plt.ylim(0,1.75)
plt.ylabel('Average feature importance')
fig.savefig("./Image/RuleFit_feature_importanca.jpg",dpi=600,bbox_inches='tight')

# %% simplify the rules

rules_all['rule'] = rules_all['rule'].str.replace('Concentration (mg/L)','C',regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('BET surface area (m2/g)','B',regex=False)
rules_all['rule'] = rules_all['rule'].str.replace('Relative weight','R',regex=False)
rules_all

# %% important rules
three_index = []
two_index = []

for i,rule in enumerate(rules_all['rule']):
    if ('C' in rule) & ('B' in rule) & ('R' in rule):
        three_index.append(i)

for i,rule in enumerate(rules_all['rule']):
    if ('C' in rule) & ('B' in rule) & ('R' not in rule):
        two_index.append(i)
    if ('C' in rule) & ('B' not in rule) & ('R' in rule):
        two_index.append(i)
    if ('C' not in rule) & ('B' in rule) & ('R' in rule):
        two_index.append(i)

rules_three = rules_all.iloc[three_index,:]
rules_three = rules_three.reset_index(drop=True)

rules_two = rules_all.iloc[two_index,:]
rules_two = rules_two.reset_index(drop=True)

rules_one_two = rules_all.drop(index=three_index)
rules_one = rules_one_two.drop(index=two_index)
rules_one = rules_one.reset_index(drop=True)

print(rules_one.shape[0],rules_two.shape[0],rules_three.shape[0],)

# %%
rule_number_importance = pd.DataFrame(columns=['Rule number','Importance'],index=range(3))
rule_number_importance.loc[0,:] = [1,rules_one['importance'].sum()/10]
rule_number_importance.loc[1,:] = [2,rules_two['importance'].sum()/10]
rule_number_importance.loc[2,:] = [3,rules_three['importance'].sum()/10]
rule_number_importance

# %%
color_list = ['#FCFEA4','#FB9B06','#CF4446',]

fig, ax1 = plt.subplots(figsize=(2,2)) 

plt.bar(rule_number_importance['Rule number'],rule_number_importance['Importance'],
        width=0.4,color=color_list,edgecolor='k',alpha=0.6)

plt.xlabel('Rule number')
plt.ylim(0,2.65)
plt.ylabel('Average rule importance')
fig.savefig("./Image/RuleFit_number_importanca.jpg",dpi=600,bbox_inches='tight')

# %%
C_B_index = []
C_R_index = []
B_R_index = []

for i,rule in enumerate(rules_two['rule']):
    if 'R' not in rule:
        C_B_index.append(i)
    if 'B' not in rule:
        C_R_index.append(i)
    if 'C' not in rule:
        B_R_index.append(i)

rule_B_C = rules_two.iloc[C_B_index,:]
rule_B_C = rule_B_C.reset_index(drop=True)

rule_C_R = rules_two.iloc[C_R_index,:]
rule_C_R = rule_C_R.reset_index(drop=True)

rule_B_R = rules_two.iloc[B_R_index,:]
rule_B_R = rule_B_R.reset_index(drop=True)


print(rule_B_C.shape[0],rule_C_R.shape[0],rule_B_R.shape[0])


# %%
rule_combined_importance = pd.DataFrame(columns=['Features','Importance'],index=range(3))
rule_combined_importance.loc[0,:] = ['B and C',rule_B_C['importance'].sum()/10]
rule_combined_importance.loc[1,:] = ['B and R',rule_B_R['importance'].sum()/10]
rule_combined_importance.loc[2,:] = ['C and R',rule_C_R['importance'].sum()/10]
rule_combined_importance

# %%
color_list = ['#FCFEA4','#FB9B06','#CF4446',]

fig, ax1 = plt.subplots(figsize=(2,2)) 

plt.bar(rule_combined_importance['Features'],rule_combined_importance['Importance'],
        width=0.4,color=color_list,edgecolor='k',alpha=0.6)

plt.xticks([0,1,2],['BET surface area (m2/g)\nand Concentration (mg/L)','BET surface area (m2/g)\nand Relative weight','Concentration (mg/L)\nand Relative weight'],
           rotation=45,ha='right')
plt.ylim(0,2.1)

plt.ylabel('Average rule importance')
fig.savefig("./Image/RuleFit_combined_importanca.jpg",dpi=600,bbox_inches='tight')

# %%
threshold_importance = 0.1
rules_important = rules_all[rules_all['importance']>threshold_importance]
rules_important = rules_important.reset_index(drop=True)

rules_important.to_excel('RuleFit_important_rules.xlsx')

print("Important rules: ",str(rules_important.shape[0]))
print("Important rules in linear terms: ",str(rules_important[rules_important['type']=='linear'].shape[0]),)
print("Important rules in rule terms: ", str(rules_important[rules_important['type']=='rule'].shape[0]),)

# %% important rules
three_index = []
for i,rule in enumerate(rules_important['rule']):
    if ('C' in rule) & ('B' in rule) & ('R' in rule):
        three_index.append(i)

rules_important_three = rules_important.iloc[three_index,:]
rules_important_three = rules_important_three.reset_index(drop=True)

rules_important_one_two = rules_important.drop(index=three_index)
rules_important_one_two = rules_important_one_two.reset_index(drop=True)

rules_important_one_two.to_excel('RuleFit_important_one_two_rules.xlsx')

print(rules_important_one_two.shape[0],rules_important_three.shape[0])

# %% 
C_B_index = []
C_R_index = []
B_R_index = []

for i,rule in enumerate(rules_important_one_two['rule']):
    if 'R' not in rule:
        C_B_index.append(i)
    if 'B' not in rule:
        C_R_index.append(i)
    if 'C' not in rule:
        B_R_index.append(i)

rule_B_C = rules_important_one_two.iloc[C_B_index,:]
rule_B_C = rule_B_C.reset_index(drop=True)

rule_C_R = rules_important_one_two.iloc[C_R_index,:]
rule_C_R = rule_C_R.reset_index(drop=True)

rule_B_R = rules_important_one_two.iloc[B_R_index,:]
rule_B_R = rule_B_R.reset_index(drop=True)


print(rule_B_C.shape[0],rule_C_R.shape[0],rule_B_R.shape[0])


# %% the min and max value of each feature
B_range = [4.07, 200.84]
C_range = [25, 200]
R_range = [1, 1.87]

# %% 
feature_1 = 'B'
feature_2 = 'C'
feature_1_range = [4.07, 200.84]
feature_2_range = [25, 200]

rule = 'C <= 75.0 and B <= 127.18999 and C > 37.5'
matches_raw = re.findall(r'(C|B)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number

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
    print('please add the code for 4 feature statements') # no this situation in the study

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
        print('please add the code for 4 feature statements') # no this situation in the study

    return interval_1, interval_2


# %% 1、 for rule map of B and C

# %% 
feature_1 = 'B'
feature_2 = 'C'
feature_1_range = [4.07, 200.84]
feature_2_range = [25, 200]

rule = 'C <= 75.0 and B <= 127.18999 and C > 37.5'
matches_raw = re.findall(r'(C|B)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number

rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)


# %%
feature_1 = 'B'
feature_2 = 'C'
feature_1_range = [4.07, 200.84]
feature_2_range = [25, 200]

rule_B_C['BET surface area (m2/g)'] = ''
rule_B_C['Concentration (mg/L)'] = ''

for i,rule in enumerate(rule_B_C['rule']):
    matches_raw = re.findall(r'(B|C)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule) # 
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_B_C.loc[i,['BET surface area (m2/g)','Concentration (mg/L)']] = rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)

rule_B_C['BET surface area (m2/g)'].astype(str)
rule_B_C['Concentration (mg/L)'].astype(str)
rule_B_C

# %%
print(rule_B_C.sort_values('BET surface area (m2/g)')['BET surface area (m2/g)'].unique())

rule_B_C_sort_B = {'None':0,'[4.07, 15.31]':1,'[4.07, 27.46]':2,'[4.07, 33.53]':3,
                   '[4.07, 41.785]':4, '[4.07, 45.705]':5, '[4.07, 48.595]':6, '[4.07, 73.47]':7,
                    '[4.07, 107.75]':8,'(45.705, 92.805]':9,'(45.705, 107.75]':10,
                   '(73.47, 92.805]':11, '(73.47, 107.75]':12, '(73.47, 200.84]':13, 
                    '(107.75, 200.84]':14, '(149.7, 200.84]':15,
       }

# %%
print(rule_B_C.sort_values('Concentration (mg/L)')['Concentration (mg/L)'].unique())

rule_B_C_sort_C = {'None':0,'[25, 37.5]':1, '[25, 75.0]':2,'(37.5, 75.0]':3, 
                   '(37.5, 150.0]':4,
                    '(37.5, 200]':5, '(75.0, 200]':6,

                   }

# %%
rule_map_B_C_importance = pd.DataFrame(0,columns=list(rule_B_C_sort_B.keys()),index=list(rule_B_C_sort_C.keys()),)
rule_map_B_C_frequency = pd.DataFrame(0,columns=list(rule_B_C_sort_B.keys()),index=list(rule_B_C_sort_C.keys()),)
rule_map_B_C_coefficient = pd.DataFrame(0,columns=list(rule_B_C_sort_B.keys()),index=list(rule_B_C_sort_C.keys()),)

for i in range(rule_B_C.shape[0]):
    rule_map_B_C_importance.loc[rule_B_C.loc[i,'Concentration (mg/L)'],rule_B_C.loc[i,'BET surface area (m2/g)']] += rule_B_C.loc[i,'importance']/10
    rule_map_B_C_coefficient.loc[rule_B_C.loc[i,'Concentration (mg/L)'],rule_B_C.loc[i,'BET surface area (m2/g)']] += rule_B_C.loc[i,'coef']/10
    rule_map_B_C_frequency.loc[rule_B_C.loc[i,'Concentration (mg/L)'],rule_B_C.loc[i,'BET surface area (m2/g)']] += 1/10

rule_map_B_C_importance

# %%
rule_map_B_C_plot = pd.DataFrame(columns=['BET surface area (m2/g)','Concentration (mg/L)','Importance','Coefficient','Frequency'])
for B_item in list(rule_B_C_sort_B.keys()):
    for C_item in  list(rule_B_C_sort_C.keys()):
        temp_rule_map = pd.DataFrame({'BET surface area (m2/g)':B_item,
                                        'Concentration (mg/L)':C_item,
                                        'Importance': rule_map_B_C_importance.loc[C_item,B_item],
                                        'Coefficient':rule_map_B_C_coefficient.loc[C_item,B_item],
                                        'Frequency':rule_map_B_C_frequency.loc[C_item,B_item],
                                        }, index=[0])
        rule_map_B_C_plot = pd.concat([rule_map_B_C_plot,temp_rule_map],ignore_index=True)
rule_map_B_C_plot


# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (3, 4.5))

palette = sns.diverging_palette(220, 20, n=10,  as_cmap=True)
larger_absolute_limit = max([abs(rule_map_B_C_coefficient.min().min()), abs(rule_map_B_C_coefficient.max().max())])

g = sns.scatterplot(
    data=rule_map_B_C_plot,x="Concentration (mg/L)", y="BET surface area (m2/g)", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(50,250),ax=ax, palette = palette,hue_norm=(-larger_absolute_limit,larger_absolute_limit),
)

plt.xticks(rotation=45,ha='right')
plt.xlim(-0.5,6.5)

plt.legend(bbox_to_anchor=(2.8, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )
norm = plt.Normalize(-larger_absolute_limit, larger_absolute_limit)
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm,)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Coefficient',shrink=0.5, anchor=(1.05, 1.0))

ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/Rule_Map_B_C.jpg",dpi=600,bbox_inches='tight')


# %%
# %% 2、 for rule map of B and R

# %%  test
feature_1 = 'B'
feature_2 = 'R'
feature_1_range = [4.07, 200.84]
feature_2_range = [1, 1.87]

rule = 'R <= 1.5 and B <= 127.18999 and R > 1.2'
matches_raw = re.findall(r'(B|R)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number

rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)


# %%
feature_1 = 'B'
feature_2 = 'R'
feature_1_range = [4.07, 200.84]
feature_2_range = [1, 1.87]

rule_B_R['BET surface area (m2/g)'] = ''
rule_B_R['Relative weight'] = ''

for i,rule in enumerate(rule_B_R['rule']):
    matches_raw = re.findall(r'(B|R)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule) # 
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_B_R.loc[i,['BET surface area (m2/g)','Relative weight']] = rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)

rule_B_R['BET surface area (m2/g)'].astype(str)
rule_B_R['Relative weight'].astype(str)
rule_B_R

# %%
print(rule_B_R.sort_values('BET surface area (m2/g)')['BET surface area (m2/g)'].unique())

rule_B_R_sort_B = {'None':0,'[4.07, 73.47]':1, '[4.07, 92.805]':2, '[4.07, 117.85]':3,
                    '[4.07, 181.475]':4,'(33.53, 92.805]':5,'(33.53, 200.84]':6, 
                    '(45.705, 92.805]':7, '(56.85, 73.47]':8,'(73.47, 107.75]':9, '(73.47, 200.84]':10,
                    '(107.75, 200.84]':11, '(127.18999, 200.84]':12, '(149.7, 200.84]':13,
       }
# %% simple the intervals for better visualization, if several intervals are close
def simple_interval_func(intervals,threshold=0.05):
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

# %% test the function
original_list = ['[1, 1.065]', '[1, 1.105]','[1, 1.175]','(1.235, 1.465]', '(1.235, 1.87]', '(1.255, 1.87]',
                '(1.265, 1.395]', '(1.285, 1.395]', '(1.305, 1.395]', '(1.385, 1.645]',
                '(1.395, 1.87]', '(1.465, 1.49]',
                '[1, 1.235]', '[1, 1.255]', '[1, 1.3]','[1, 1.395]', 
                '[1, 1.505]', '[1, 1.635]','(1.055, 1.465]', '(1.065, 1.075]',
                '(1.065, 1.425]', '(1.065, 1.645]','(1.105, 1.645]', '(1.16, 1.87]',
                '(1.165, 1.87]', '(1.175, 1.515]',
                '(1.175, 1.645]', '(1.185, 1.295]', '(1.185, 1.635]', '(1.195, 1.295]',
                '(1.195, 1.645]', ]
interval_df = simple_interval_func(original_list,)
interval_df

# %%

# %% simplify the interval for better visualization
print(rule_B_R.sort_values('Relative weight')['Relative weight'].unique())
B_R_interval_simply_R = pd.DataFrame({'Raw interval':['None','[1, 1.065]', '[1, 1.105]','[1, 1.175]',
                                                    '[1, 1.235]', '[1, 1.255]', '[1, 1.3]','[1, 1.395]', 
                                                    '[1, 1.505]', '[1, 1.635]','(1.055, 1.465]', '(1.065, 1.075]',
                                                    '(1.065, 1.425]', '(1.065, 1.645]','(1.105, 1.645]', '(1.16, 1.87]',
                                                    '(1.165, 1.87]', '(1.175, 1.515]',
                                                    '(1.175, 1.645]', '(1.185, 1.295]', '(1.185, 1.635]', '(1.195, 1.295]',
                                                    '(1.195, 1.645]', '(1.235, 1.465]', '(1.235, 1.87]', '(1.255, 1.87]',
                                                    '(1.265, 1.395]', '(1.285, 1.395]', '(1.305, 1.395]', '(1.385, 1.645]',
                                                    '(1.395, 1.87]', '(1.465, 1.49]',
                                         ],})
B_R_interval_simply_R.loc[0,'Simply interval'] = 'None'
B_R_interval_simply_R.loc[1:,'Simply interval'] = simple_interval_func(B_R_interval_simply_R['Raw interval'].values[1:])

len(B_R_interval_simply_R['Simply interval'].unique())

# %%
B_R_interval_simply_R.to_excel('simply_R_in_B_R.xlsx')
B_R_interval_simply_R

# %% replace the interval for better visualization
rule_B_R_simply = rule_B_R.copy()
for i,item in enumerate(rule_B_R_simply['Relative weight']):
     index = list(B_R_interval_simply_R['Raw interval']).index(item)
     rule_B_R_simply.loc[i,'Relative weight'] = B_R_interval_simply_R.loc[index,'Simply interval']
rule_B_R_simply

# %%
print(len(rule_B_R_simply['Relative weight'].unique()))
rule_B_R_sort_R = B_R_interval_simply_R['Simply interval'].unique()

# %%
rule_map_B_R_importance = pd.DataFrame(0,columns=rule_B_R_sort_R, index=list(rule_B_R_sort_B.keys()),)
rule_map_B_R_frequency = pd.DataFrame(0,columns=rule_B_R_sort_R, index=list(rule_B_R_sort_B.keys()),)
rule_map_B_R_coefficient = pd.DataFrame(0,columns=rule_B_R_sort_R, index=list(rule_B_R_sort_B.keys()),)

for i in range(rule_B_R.shape[0]):
    print(i)
    rule_map_B_R_importance.loc[rule_B_R_simply.loc[i,'BET surface area (m2/g)'],rule_B_R_simply.loc[i,'Relative weight']] += rule_B_R_simply.loc[i,'importance']/10
    rule_map_B_R_coefficient.loc[rule_B_R_simply.loc[i,'BET surface area (m2/g)'],rule_B_R_simply.loc[i,'Relative weight']] += rule_B_R_simply.loc[i,'coef']/10
    rule_map_B_R_frequency.loc[rule_B_R_simply.loc[i,'BET surface area (m2/g)'],rule_B_R_simply.loc[i,'Relative weight']] += 1/10

rule_map_B_R_importance

# %%
rule_map_B_R_plot = pd.DataFrame(columns=['BET surface area (m2/g)','Relative weight','Importance','Coefficient','Frequency'])
for B_item in list(rule_B_R_sort_B.keys()):
    for R_item in  rule_B_R_sort_R:
        temp_rule_map = pd.DataFrame({'BET surface area (m2/g)':B_item,
                                        'Relative weight':R_item,
                                        'Importance': rule_map_B_R_importance.loc[B_item,R_item],
                                        'Coefficient':rule_map_B_R_coefficient.loc[B_item,R_item],
                                        'Frequency':rule_map_B_R_frequency.loc[B_item,R_item],
                                        }, index=[0])
        rule_map_B_R_plot = pd.concat([rule_map_B_R_plot,temp_rule_map],ignore_index=True)
rule_map_B_R_plot


# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (6, 6.8))

palette = sns.diverging_palette(220, 20, n=10,  as_cmap=True)

larger_absolute_limit = max([abs(rule_map_B_R_coefficient.min().min()), abs(rule_map_B_R_coefficient.max().max())])

g = sns.scatterplot(
    data=rule_map_B_R_plot,x="BET surface area (m2/g)", y="Relative weight", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(50,250),ax=ax, palette = palette,hue_norm=(-larger_absolute_limit,larger_absolute_limit),
)

plt.xticks(rotation=45,ha='right')
plt.xlim(-0.5,13.5)

plt.legend(bbox_to_anchor=(2, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )

norm = plt.Normalize(-larger_absolute_limit, larger_absolute_limit)
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm,)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Coefficient',shrink=0.5, anchor=(0.3, 1.0))

ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/Rule_Map_B_R.jpg",dpi=600,bbox_inches='tight')

# %%
# %%
# %% 3、 for rule map of C and R

# %%  test
# %%  test
feature_1 = 'C'
feature_2 = 'R'
feature_1_range = [25, 200]
feature_2_range = [1, 1.87]

rule = 'R <= 1.7 and C <= 127.18999 and R > 1.5'
matches_raw = re.findall(r'(C|R)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number

rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)


# %%
feature_1 = 'C'
feature_2 = 'R'
feature_1_range = [25, 200]
feature_2_range = [1,1.87]

rule_C_R['Concentration (mg/L)'] = ''
rule_C_R['Relative weight'] = ''

for i,rule in enumerate(rule_C_R['rule']):
    matches_raw = re.findall(r'(C|R)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule) # 
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_C_R.loc[i,['Concentration (mg/L)','Relative weight']] = rule_to_interval_two(matches, feature_1, feature_2, feature_1_range, feature_2_range)

rule_C_R['Concentration (mg/L)'].astype(str)
rule_C_R['Relative weight'].astype(str)
rule_C_R

# %%
print(rule_C_R.sort_values('Concentration (mg/L)')['Concentration (mg/L)'].unique())

rule_C_R_sort_C = {'None':0, '[25, 37.5]':1, '[25, 75.0]':2,'(37.5, 150.0]':3, '(37.5, 200]':4,
                    '(75.0, 200]':5,'(150.0, 200]':6,
}

# %% simplify the interval for better visualization
print(rule_C_R.sort_values('Relative weight')['Relative weight'].unique())
C_R_interval_simply_R = pd.DataFrame({'Raw interval':['None','[1, 1.065]', '[1, 1.105]', '[1, 1.175]', '[1, 1.195]', 
                                                    '[1, 1.235]', '[1, 1.505]', '[1, 1.645]','(1.055, 1.465]',
                                                    '(1.065, 1.075]', '(1.065, 1.425]', '(1.065, 1.645]',
                                                    '(1.105, 1.645]', '(1.105, 1.87]', '(1.165, 1.87]', '(1.175, 1.515]',
                                                    '(1.175, 1.645]', '(1.185, 1.295]', '(1.185, 1.635]', '(1.195, 1.295]',
                                                    '(1.195, 1.645]', '(1.21, 1.285]', '(1.235, 1.465]', '(1.235, 1.87]',
                                                    '(1.285, 1.87]', '(1.295, 1.87]', '(1.385, 1.645]', '(1.465, 1.49]',
                                         ],
})

C_R_interval_simply_R.loc[0,'Simply interval'] = 'None'
C_R_interval_simply_R.loc[1:,'Simply interval'] = simple_interval_func(C_R_interval_simply_R['Raw interval'].values[1:])

len(C_R_interval_simply_R['Simply interval'].unique())

#%%
C_R_interval_simply_R.to_excel('simply_R_in_C_R.xlsx')
C_R_interval_simply_R

# %% replace the interval for better visualization
rule_C_R_simply = rule_C_R.copy()
for i,item in enumerate(rule_C_R_simply['Relative weight']):
     index = list(C_R_interval_simply_R['Raw interval']).index(item)
     rule_C_R_simply.loc[i,'Relative weight'] = C_R_interval_simply_R.loc[index,'Simply interval']
rule_C_R_simply

# %%
print(len(rule_C_R_simply['Relative weight'].unique()))
rule_C_R_sort_R = C_R_interval_simply_R['Simply interval'].unique()

# %%
rule_map_C_R_importance = pd.DataFrame(0,columns=rule_C_R_sort_R, index=list(rule_C_R_sort_C.keys()),)
rule_map_C_R_coefficient = pd.DataFrame(0,columns=rule_C_R_sort_R, index=list(rule_C_R_sort_C.keys()),)
rule_map_C_R_frequency = pd.DataFrame(0,columns=rule_C_R_sort_R, index=list(rule_C_R_sort_C.keys()),)

for i in range(rule_C_R.shape[0]):
    rule_map_C_R_importance.loc[rule_C_R_simply.loc[i,'Concentration (mg/L)'],rule_C_R_simply.loc[i,'Relative weight']] += rule_C_R_simply.loc[i,'importance']/10
    rule_map_C_R_coefficient.loc[rule_C_R_simply.loc[i,'Concentration (mg/L)'],rule_C_R_simply.loc[i,'Relative weight']] += rule_C_R_simply.loc[i,'coef']/10
    rule_map_C_R_frequency.loc[rule_C_R_simply.loc[i,'Concentration (mg/L)'],rule_C_R_simply.loc[i,'Relative weight']] += 1/10

rule_map_C_R_importance

# %%
rule_map_C_R_plot = pd.DataFrame(columns=['Concentration (mg/L)','Relative weight','Importance','Coefficient','Frequency'])
for C_item in list(rule_C_R_sort_C.keys()):
    for R_item in  rule_C_R_sort_R:
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

fig, ax= plt.subplots(figsize = (2.7, 5.8))

palette = sns.diverging_palette(220, 20, n=10,  as_cmap=True)

larger_absolute_limit = max([abs(rule_map_B_R_coefficient.min().min()), abs(rule_map_B_R_coefficient.max().max())])

g = sns.scatterplot(
    data=rule_map_C_R_plot,x="Concentration (mg/L)", y="Relative weight", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(50,250),ax=ax, palette = palette,hue_norm=(-larger_absolute_limit,larger_absolute_limit),
)

plt.xticks(rotation=45,ha='right')
plt.xlim(-0.5,6.5)

plt.legend(bbox_to_anchor=(3.8, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )

norm = plt.Normalize(-larger_absolute_limit, larger_absolute_limit)
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm,)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Coefficient',shrink=0.5, anchor=(1.2, 1.0))

ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/Rule_Map_C_R.jpg",dpi=600,bbox_inches='tight')








# %% rule map for three features

# %% 
feature_1 = 'B'
feature_2 = 'C'
feature_3 = 'R'
feature_1_range = [4.07, 200.84]
feature_2_range = [25, 200]
feature_3_range = [1,1.87]

rule = 'R > 1.5 and C > 56.85 and B <= 65.7 and B > 38.15'
matches_raw = re.findall(r'(B|C|R)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number

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

# %% test
feature_1 = 'B'
feature_2 = 'C'
feature_3 = 'R'
feature_1_range = [4.07, 200.84]
feature_2_range = [25, 200]
feature_3_range = [1, 1.87]

rule = 'R > 1.5 and C > 56.85 and B <= 65.7 and B > 38.15'
matches_raw = re.findall(r'(B|C|R)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number

rule_to_interval_three(matches, feature_1, feature_2, feature_3, feature_1_range, feature_2_range, feature_3_range)

# %%
rule_B_C_R = rules_important_three.copy()
rule_B_C_R['BET surface area (m2/g)'] = ''
rule_B_C_R['Concentration (mg/L)'] = ''
rule_B_C_R['Relative weight'] = ''

for i,rule in enumerate(rule_B_C_R['rule']):
    matches_raw = re.findall(r'(B|C|R)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_B_C_R.loc[i,['BET surface area (m2/g)','Concentration (mg/L)','Relative weight']] = rule_to_interval_three(matches, feature_1, feature_2, feature_3, feature_1_range, feature_2_range, feature_3_range)

rule_B_C_R

# %%
print(rule_B_C_R.sort_values('BET surface area (m2/g)')['BET surface area (m2/g)'].unique())

rule_B_C_R_sort_B = { '[4.07, 45.705]':0, '[4.07, 56.85]':1, '[4.07, 73.47]':2,
                    '[4.07, 92.805]':3,'[4.07, 127.18999]':4, '(33.53, 73.47]':5,'(33.53, 200.84]':6,
                     '(45.705, 92.805]':7,'(45.705, 200.84]':8, '(92.805, 200.84]':9,'(127.18999, 200.84]':10,
                    }

# %%
print(rule_B_C_R.sort_values('Concentration (mg/L)')['Concentration (mg/L)'].unique())

rule_B_C_R_sort_C = {  '[25, 75.0]':0, '(37.5, 200]':1,  '(75.0, 200]':2,
                    }

# %%
print(rule_B_C_R.sort_values('Relative weight')['Relative weight'].unique())

B_C_R_interval_simply_R = pd.DataFrame({'Raw interval':['[1, 1.295]', '[1, 1.29]', '[1, 1.345]', '[1, 1.35]',
                                        '[1, 1.355]','[1, 1.365]', '[1, 1.4]', '[1, 1.405]', '[1, 1.465]', '[1, 1.475]',
                                        '[1, 1.495]', '(1.065, 1.87]', '(1.09, 1.525]', '(1.095, 1.87]',
                                        '(1.16, 1.87]', '(1.175, 1.46]', '(1.195, 1.35]', '(1.195, 1.87]' ,
                                        '(1.235, 1.355]', '(1.265, 1.87]',
                                        ],
})

B_C_R_interval_simply_R.loc[0:,'Simply interval'] = simple_interval_func(B_C_R_interval_simply_R['Raw interval'].values[0:])

len(B_C_R_interval_simply_R['Simply interval'].unique())

# %%
B_C_R_interval_simply_R.to_excel('simply_R_in_B_C_R.xlsx')

B_C_R_interval_simply_R

# %% replace the interval for better visualization
rule_B_C_R_simply_R = rule_B_C_R.copy()
for i,item in enumerate(rule_B_C_R_simply_R['Relative weight']):
     index = list(B_C_R_interval_simply_R['Raw interval']).index(item)
     rule_B_C_R_simply_R.loc[i,'Relative weight'] = B_C_R_interval_simply_R.loc[index,'Simply interval']
rule_B_C_R_simply_R

# %%
print(len(rule_B_C_R_simply_R['Relative weight'].unique()))
rule_B_C_R_sort_R = B_C_R_interval_simply_R['Simply interval'].unique()

# %% 
rule_map_B_C_R_plot = pd.DataFrame(columns=['BET surface area (m2/g)','Concentration (mg/L)','Relative weight','Importance','Coefficient','Frequency'])
for B_item in list(rule_B_C_R_sort_B.keys()):
    for C_item in list(rule_B_C_R_sort_C.keys()):
        for R_item in  rule_B_C_R_sort_R:
             temp_rule_map = pd.DataFrame({'BET surface area (m2/g)':B_item,
                                        'Concentration (mg/L)':C_item,
                                        'Relative weight':R_item,
                                        'Importance': rule_B_C_R_simply_R[(rule_B_C_R_simply_R['BET surface area (m2/g)']==B_item) & (rule_B_C_R_simply_R['Concentration (mg/L)']==C_item) & (rule_B_C_R_simply_R['Relative weight']==R_item)]['importance'].sum()/10,
                                        'Coefficient':rule_B_C_R_simply_R[(rule_B_C_R_simply_R['BET surface area (m2/g)']==B_item) & (rule_B_C_R_simply_R['Concentration (mg/L)']==C_item) & (rule_B_C_R_simply_R['Relative weight']==R_item)]['coef'].sum()/10,
                                        'Frequency':rule_B_C_R_simply_R[(rule_B_C_R_simply_R['BET surface area (m2/g)']==B_item) & (rule_B_C_R_simply_R['Concentration (mg/L)']==C_item) & (rule_B_C_R_simply_R['Relative weight']==R_item)].shape[0]/10,
                                        }, index=[0])
             rule_map_B_C_R_plot = pd.concat([rule_map_B_C_R_plot,temp_rule_map],ignore_index=True)
rule_map_B_C_R_plot

# %%
rule_map_B_C_R_plot[rule_map_B_C_R_plot['Importance']>0]['Concentration (mg/L)'].value_counts()

# %% c1 :'Concentration (mg/L)' - (75.0, 200]
rule_map_B_C_R_plot_c1 = rule_map_B_C_R_plot[rule_map_B_C_R_plot['Concentration (mg/L)']=='(75.0, 200]']
rule_map_B_C_R_plot_c1

# %%
rule_map_B_C_R_plot_c1[rule_map_B_C_R_plot_c1['Importance']>0]

# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (6, 6))

palette = sns.diverging_palette(220, 20, n=10, as_cmap=True)

larger_absolute_limit = max([abs(rule_map_B_C_R_plot_c1['Coefficient'].min()), abs(rule_map_B_C_R_plot_c1['Coefficient'].max())])

g = sns.scatterplot(
    data=rule_map_B_C_R_plot_c1,x="BET surface area (m2/g)", y="Relative weight", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(50,250),ax=ax, palette = palette, hue_norm=(-larger_absolute_limit,larger_absolute_limit),
)

plt.xticks(rotation=45,ha='right')
plt.title('Concentration (mg/L): (75.0, 200]')
plt.legend(bbox_to_anchor=(1.8, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )

norm = plt.Normalize(-larger_absolute_limit, larger_absolute_limit)
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm,)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Coefficient',shrink=0.5, anchor=(0, 1.0))

ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/Rule_Map_B_C_R_c1.jpg",dpi=600,bbox_inches='tight')



# %%
# %% c2 :'Concentration (mg/L)' - [25, 75.0]
rule_map_B_C_R_plot_c2 = rule_map_B_C_R_plot[rule_map_B_C_R_plot['Concentration (mg/L)']=='[25, 75.0]']
rule_map_B_C_R_plot_c2

# %%
rule_map_B_C_R_plot_c2[rule_map_B_C_R_plot_c2['Importance']>0]

# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (6, 6))

palette = sns.diverging_palette(220, 20, n=10, as_cmap=True)

larger_absolute_limit = max([abs(rule_map_B_C_R_plot_c2['Coefficient'].min()), abs(rule_map_B_C_R_plot_c2['Coefficient'].max())])

g = sns.scatterplot(
    data=rule_map_B_C_R_plot_c2,x="BET surface area (m2/g)", y="Relative weight", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(50,250),ax=ax, palette = palette, hue_norm=(-larger_absolute_limit,larger_absolute_limit),
)

plt.xticks(rotation=45,ha='right')
plt.title('Concentration (mg/L): [25, 75.0]')
plt.legend(bbox_to_anchor=(1.8, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1
          )

norm = plt.Normalize(-larger_absolute_limit, larger_absolute_limit)
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm,)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Coefficient',shrink=0.5, anchor=(0, 1.0))

ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/Rule_Map_B_C_R_c2.jpg",dpi=600,bbox_inches='tight')
# %%



# %% c3 :'Concentration (mg/L)' - (37.5, 200]
rule_map_B_C_R_plot_c3 = rule_map_B_C_R_plot[rule_map_B_C_R_plot['Concentration (mg/L)']=='(37.5, 200]']
rule_map_B_C_R_plot_c3

# %%
rule_map_B_C_R_plot_c3[rule_map_B_C_R_plot_c3['Importance']>0]

# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (6, 6))

palette = sns.diverging_palette(220, 20, n=10, as_cmap=True)

larger_absolute_limit = max([abs(rule_map_B_C_R_plot_c3['Coefficient'].min()), abs(rule_map_B_C_R_plot_c3['Coefficient'].max())])

g = sns.scatterplot(
    data=rule_map_B_C_R_plot_c3,x="BET surface area (m2/g)", y="Relative weight", hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(50,250),ax=ax, palette = palette, hue_norm=(-larger_absolute_limit,larger_absolute_limit),
)


plt.xticks(rotation=45,ha='right')
plt.title('Concentration (mg/L): (37.5, 200]')


plt.legend(bbox_to_anchor=(2.2, 1), loc='upper right', 
            borderaxespad=0,labelspacing=0.8, scatterpoints=1,
            frameon = False, ncol=1,
          )

norm = plt.Normalize(-larger_absolute_limit, larger_absolute_limit)
sm = plt.cm.ScalarMappable(cmap=palette, norm=norm,)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Coefficient',shrink=0.5, anchor=(0, 1.0))


ax.spines['top'].set_linewidth(1.2)
ax.spines['top'].set_color('dimgray')
ax.spines['bottom'].set_linewidth(1.2)
ax.spines['bottom'].set_color('dimgray')
ax.spines['left'].set_linewidth(1.2)
ax.spines['left'].set_color('dimgray')
ax.spines['right'].set_linewidth(1.2)
ax.spines['right'].set_color('dimgray')

fig.savefig("./Image/Rule_Map_B_C_R_c3.jpg",dpi=600,bbox_inches='tight')
# %%
