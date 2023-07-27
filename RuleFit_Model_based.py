# %%
import pandas as pd
import os
import re

import numpy as np
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,GradientBoostingClassifier
import itertools
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
data = pd.read_excel("./Dataset_RMC.xlsx",index_col = 0,)
X = data.drop(columns=['RMC','Composition','Morphology','Solubility']) 
X.columns

# %%
feature_names = X.columns
feature_importances = get_feature_importance(feature_names, rules_all, scaled=False)
importance_df = pd.DataFrame(feature_importances, index = feature_names, columns = ['importance']).sort_values(by='importance',ascending=False)
importance_df['importance'] = importance_df['importance']/10
importance_df

# %%
palette = sns.diverging_palette(220, 20, n=7,  )
hex_values = sns.color_palette(palette, as_cmap=True).as_hex()
# Print the HEX values
for hex_code in hex_values:
    print(hex_code)

# %%
color_list = ['#c3553a','#d28976','#e3beb4','#b5ccd3','#79a5b3','#3f7f93',]
fig, ax1 = plt.subplots(figsize=(2.2,1.5)) 

plt.bar(importance_df.index,importance_df['importance'],width=0.4,color=color_list,edgecolor='k',alpha=0.6)
plt.xticks(rotation=45,ha='right')
plt.ylim(0,importance_df['importance'].max()*1.15)
plt.ylabel('Feature importance')
fig.savefig("./Image/RuleFit_feature_importance.jpg",dpi=600,bbox_inches='tight')

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
rules_all

# %%
rules_all['Feature number'].unique()

# %%
rule_number_importance = pd.DataFrame(columns=['Rule number','Importance'],index=range(6))
rule_number_importance.loc[0,:] = [1,rules_all[rules_all['Feature number']==1]['importance'].sum()/10]
rule_number_importance.loc[1,:] = [2,rules_all[rules_all['Feature number']==2]['importance'].sum()/10]
rule_number_importance.loc[2,:] = [3,rules_all[rules_all['Feature number']==3]['importance'].sum()/10]
rule_number_importance.loc[3,:] = [4,rules_all[rules_all['Feature number']==4]['importance'].sum()/10]
rule_number_importance.loc[4,:] = [5,rules_all[rules_all['Feature number']==5]['importance'].sum()/10]
rule_number_importance.loc[5,:] = [6,rules_all[rules_all['Feature number']==6]['importance'].sum()/10]
rule_number_importance

# %%
fig, ax1 = plt.subplots(figsize=(2,1.2)) 
plt.bar(rule_number_importance['Rule number'],rule_number_importance['Importance'],
        width=0.4,color=color_list,edgecolor='k',alpha=0.6)

plt.xlabel('Feature number')
plt.xticks([1,2,3,4,5,6])
plt.ylim(0,rule_number_importance['Importance'].max()*1.1)
plt.ylabel('Rule importance')
#ax1.spines['top'].set_visible(False)
#ax1.spines['top'].set_visible(False)
fig.savefig("./Image/RuleFit_number_importance_RMC.jpg",dpi=600,bbox_inches='tight')


# %%
rules_all_one = rules_all[rules_all['Feature number']==1]
rules_all_one

# %%
rules_all_two = rules_all[rules_all['Feature number']==2]
rules_all_two

# %%
rules_all_three = rules_all[rules_all['Feature number']==3]
rules_all_three

# %%
feature_dict = {'C': 'Concentration (mg/L)',
           'Z': 'Zeta potential (mV)',
           'H': 'Hydrodynamic diameter (nm)',
           'R': 'Relative weight',
           'B': 'BET surface area (m2/g)',
           'S': 'Seedling part',}
feature_dict

# %% rules containing one feature
col1 = feature_dict.keys()
rules_one_importance = pd.DataFrame({'Feature 1': col1, })
rules_one_importance

# %%
for i in range(0,rules_one_importance.shape[0]):
    index = []
    for j in range(0,rules_all_one.shape[0]):
        if (rules_one_importance.loc[i,'Feature 1'] in rules_all_one.iloc[j,0]) :
            index.append(j)
    rules_one_importance.loc[i,'Importance'] = rules_all_one.iloc[index,:]['importance'].sum()/10
rules_one_importance

# %%
rules_one_importance['Importance'].sum()

# %% rules containing two features
rules_one_importance = rules_one_importance.sort_values(by='Importance', ascending=False)
rules_one_importance

# %%
letters = feature_dict.keys()
combinations = list(itertools.combinations(letters, 2))
col1, col2 = zip(*combinations)
rules_two_importance = pd.DataFrame({'Feature 1': col1, 'Feature 2': col2})
rules_two_importance

# %%
for i in range(0,rules_two_importance.shape[0]):
    index = []
    for j in range(0,rules_all_two.shape[0]):
        if (rules_two_importance.loc[i,'Feature 1'] in rules_all_two.iloc[j,0]) and (rules_two_importance.loc[i,'Feature 2'] in rules_all_two.iloc[j,0]):
            index.append(j)
    rules_two_importance.loc[i,'Importance'] = rules_all_two.iloc[index,:]['importance'].sum()/10
rules_two_importance

# %%
rules_two_importance['Importance'].sum()

# %% 
rules_two_importance = rules_two_importance.sort_values(by='Importance', ascending=False)
rules_two_importance

# %% rules containing three features
letters = feature_dict.keys()
combinations = list(itertools.combinations(letters, 3))
col1, col2, col3 = zip(*combinations)
rules_three_importance = pd.DataFrame({'Feature 1': col1, 'Feature 2': col2, 'Feature 3': col3})
rules_three_importance

# %%
for i in range(0,rules_three_importance.shape[0]):
    index = []
    for j in range(0,rules_all_three.shape[0]):
        if (rules_three_importance.loc[i,'Feature 1'] in rules_all_three.iloc[j,0]) and (rules_three_importance.loc[i,'Feature 2'] in rules_all_three.iloc[j,0]) and (rules_three_importance.loc[i,'Feature 3'] in rules_all_three.iloc[j,0]):
            index.append(j)
    rules_three_importance.loc[i,'Importance'] = rules_all_three.iloc[index,:]['importance'].sum()/10
rules_three_importance

# %%
rules_three_importance = rules_three_importance.sort_values(by='Importance', ascending=False)
rules_three_importance

# %%
feaure_combinations = pd.DataFrame(columns=['Feature (combinations)','Importance'],index=range(len(rules_one_importance)+len(rules_two_importance)+len(rules_three_importance)))
feaure_combinations

# %%
feaure_combinations.iloc[0:len(rules_one_importance),0] = rules_one_importance['Feature 1']
feaure_combinations.iloc[0:len(rules_one_importance),1] = rules_one_importance['Importance']

feaure_combinations.iloc[len(rules_one_importance):len(rules_one_importance)+len(rules_two_importance),0] = rules_two_importance.apply(lambda row: row['Feature 1'] + ' and ' + row['Feature 2'], axis=1)
feaure_combinations.iloc[len(rules_one_importance):len(rules_one_importance)+len(rules_two_importance),1] = rules_two_importance['Importance']

feaure_combinations.iloc[len(rules_one_importance)+len(rules_two_importance):,0] = rules_three_importance.apply(lambda row: row['Feature 1'] + ', ' + row['Feature 2'] + ' and ' + row['Feature 3'], axis=1)
feaure_combinations.iloc[len(rules_one_importance)+len(rules_two_importance):,1] = rules_three_importance['Importance']

feaure_combinations['Importance'] = feaure_combinations['Importance'].astype('float')
feaure_combinations

# %%
feaure_combinations_plot = feaure_combinations[feaure_combinations['Importance']>0.05]

def colors_from_values(values, palette_name):
    normalized = (values - min(values)) / (max(values) - min(values))
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

fig, ax = plt.subplots(figsize=(1.1,4.7)) 

s = sns.barplot(x = feaure_combinations_plot['Importance'], 
            y = feaure_combinations_plot['Feature (combinations)'],
            palette=colors_from_values(feaure_combinations_plot['Importance'], "vlag"), 
            orient='h',alpha=0.6, edgecolor=".1",width=0.6)
s.set_yticklabels(s.get_yticklabels(), size = 9)
plt.xlim(0,feaure_combinations_plot['Importance'].max()*1.1)

#plt.xlabel('Rule importance (>0.05)')
plt.xlabel('')
ax.text(-0.45,20.5, "Rule importance (>0.05)")

plt.margins(0.02)
fig.savefig("./Image/RuleFit_detailed_importance.jpg",dpi=600,bbox_inches='tight')

# %%
