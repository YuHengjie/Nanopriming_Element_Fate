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
rules_all.to_excel('rules_all_0.1.xlsx')

# %%
rules_one = rules_all[rules_all['Feature number']==1]
rules_one

# %%

R_index = []

for i,rule in enumerate(rules_one['rule']):
    if 'R' in rule:
        R_index.append(i)


rule_R = rules_one.iloc[R_index,:]
rule_R = rule_R.reset_index(drop=True)

print(rule_R.shape[0])

# %% the min and max value of each feature
C_range = [25, 200]
B_range = [4.07,200.84]
H_range = [197, 933.73]
R_range = [1, 1.87]

# can use to set same scale for plot
# give these two variables maxmimum values
larger_coeff_limit = 0
larger_import_limit = 0

# %% rule_to_interval_one
def rule_to_interval_one(matches: list, feature_1: str,feature_1_range: list,):

    interval_1 = 'None'

    if len(matches) == 1: # if only one statement
        if matches[0][1] == '>':
            interval_1 = '(' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
        if matches[0][1] == '>=':
            interval_1 = '[' + str(matches[0][2]) + ', ' + str(feature_1_range[1]) + ']'
        if matches[0][1] == '<':
            interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ')'
        if matches[0][1] == '<=':
            interval_1 = '[' + str(feature_1_range[0]) + ', ' + str(matches[0][2]) + ']'

    if len(matches) == 2: # if the rule has two statements
        if (matches[0][1] == '>') & (matches[1][1] == '<'):
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
        if (matches[0][1] == '>') & (matches[1][1] == '<='):
                interval_1 = '(' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']' 
        if (matches[0][1] == '>=') & (matches[1][1] == '<'):
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ')'
        if (matches[0][1] == '>=') & (matches[1][1] == '<='):
                interval_1 = '[' + str(matches[0][2]) + ', ' + str(matches[1][2]) + ']'         

    return interval_1


# %% 
feature_1 = 'R'
feature_1_range = R_range

rule = 'R <= 1.5 and R > 1.1'
matches_raw = re.findall(r'(R)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number

rule_to_interval_one(matches, feature_1, feature_1_range,)


# %% 1、 for rule map of R
feature_1 = 'R'
feature_1_range = R_range

rule_R['Relative weight'] = ''

for i,rule in enumerate(rule_R['rule']):
    matches_raw = re.findall(r'(R)\s*([><]=?|==)\s*(-?\d+(?:\.\d+)?)', rule)
    matches =  sorted(matches_raw, key=lambda x: (x[0], float(x[2]))) # sort according the letter and float number
    rule_R.loc[i,['Relative weight']] = rule_to_interval_one(matches, feature_1, feature_1_range)

rule_R['Relative weight'].astype(str)
rule_R

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

# %%
print(rule_R.sort_values('Relative weight')['Relative weight'].unique())

# %%
rule_R_simply_R = pd.DataFrame({'Raw interval':[
                                '[1, 1.1]','[1, 1.105]' ,
                                '(1.105, 1.155]' ,'(1.125, 1.645]',
                                  '(1.175, 1.465]', '(1.245, 1.635]' ,
                                
                                     ],})
rule_R_simply_R.loc[0:,'Simply interval'] = simple_interval_func(rule_R_simply_R['Raw interval'].values[0:],
                                                                       threshold=(R_range[1]-R_range[0])*0.05)
len(rule_R_simply_R['Simply interval'].unique())

# %%
rule_R_simply_R.to_excel('R_simply_R.xlsx')
rule_R_simply_R

# %% replace the interval for better visualization
rule_R_simply = rule_R.copy()
for i,item in enumerate(rule_R_simply['Relative weight']):
     index = list(rule_R_simply_R['Raw interval']).index(item)
     rule_R_simply.loc[i,'Relative weight'] = rule_R_simply_R.loc[index,'Simply interval']
rule_R_simply

# %%
rule_map_R_importance = pd.DataFrame(0,index=list(rule_R_simply_R['Simply interval'].unique()),columns=[''])
rule_map_R_frequency = pd.DataFrame(0,index=list(rule_R_simply_R['Simply interval'].unique()),columns=[''])
rule_map_R_coefficient = pd.DataFrame(0,index=list(rule_R_simply_R['Simply interval'].unique()),columns=[''])
rule_map_R_importance

# %%
for i in range(rule_R_simply.shape[0]):
    rule_map_R_importance.loc[rule_R_simply.loc[i,'Relative weight'],:] += rule_R_simply.loc[i,'importance']/10
    rule_map_R_coefficient.loc[rule_R_simply.loc[i,'Relative weight'],:] += rule_R_simply.loc[i,'coef']/10
    rule_map_R_frequency.loc[rule_R_simply.loc[i,'Relative weight'],:] += 1/10

rule_map_R_importance

# %%
rule_map_R_plot = pd.DataFrame(columns=['Relative weight','y','Importance','Coefficient','Frequency'])
for R_item in  list(rule_R_simply_R['Simply interval'].unique()):
    temp_rule_map = pd.DataFrame({'Relative weight':R_item,
                                  'y':0,
                                    'Importance': rule_map_R_importance.loc[R_item,:].values,
                                    'Coefficient':rule_map_R_coefficient.loc[R_item,:].values,
                                    'Frequency':rule_map_R_frequency.loc[R_item,:].values,
                                    }, index=[0])
    rule_map_R_plot = pd.concat([rule_map_R_plot,temp_rule_map],ignore_index=True)
rule_map_R_plot


# %%
sns.set_theme(style="whitegrid")

fig, ax= plt.subplots(figsize = (3, 1))
palette = sns.diverging_palette(220, 20, n=10,  as_cmap=True)

if max([abs(rule_map_R_coefficient.min().min()), abs(rule_map_R_coefficient.max().max())])>larger_coeff_limit:
    larger_coeff_limit = max([abs(rule_map_R_coefficient.min().min()), abs(rule_map_R_coefficient.max().max())])
    print('Warning: get larger coef limit for plot!')

if rule_map_R_importance.max().max() > larger_import_limit:
    larger_import_limit = rule_map_R_importance.max().max()
    print('Warning: get larger importance limit for plot!')


g = sns.scatterplot(
    data=rule_map_R_plot,x="Relative weight", y='y', hue="Coefficient",linewidth=0.2,
    size="Importance",sizes=(20,250*rule_map_R_importance.max().max()/larger_import_limit),
    ax=ax, palette = palette,hue_norm=(-larger_coeff_limit,larger_coeff_limit),
)

plt.xticks(rotation=45,ha='right')
ax.set_yticks([0])
ax.set_yticklabels('')
ax.set_ylabel('')
plt.xlim(-0.5,4.5)
plt.ylim(-0.5,0.5)

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

fig.savefig("./Image/RuleGrid_R.jpg",dpi=600,bbox_inches='tight')


# %%
