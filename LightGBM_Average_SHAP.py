# %% 导入包
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import optimize
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 
import joblib
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

from sklearn.inspection import permutation_importance
import shap
from pdpbox import pdp
import lime
import lime.lime_tabular
from scipy.interpolate import interp1d
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %% 
data = pd.read_excel("./Dataset_RMC.xlsx",index_col = 0,)
data

# %% 
X = data.drop(columns=['RMC']) # 'TEM size (nm)'
X_raw = X.copy() # 用于shap可视化
X_raw

# %%
le_composition = LabelEncoder()
le_composition.fit(X['Composition'])
X['Composition'] = le_composition.transform(X['Composition'])
print(list(le_composition.inverse_transform([0,1,2,3,4,5,6,7])))


le_morphology = LabelEncoder()
le_morphology.fit(X['Morphology'])
X['Morphology'] = le_morphology.transform(X['Morphology'])
print(list(le_morphology.inverse_transform([0,1])))
X

#X['Seedling part'] = X['Seedling part'].map({'Root':0,'Stem':1,'Leaf':2})
X['Seedling part'] = X['Seedling part'].map({'Root':0,'Stem':1,'Leaf':2,'Shoot':3,'Whole':4})
seedling_inverse_dict = {0:'Root',1:'Stem',2:'Leaf',3:'Shoot',4:'Whole'}
X

# %%
target_name = 'RMC'
y = data.loc[:,target_name]
y

# %%
feature_name_list = list(X.columns)
feature_name_list

# %%
# %%
para_pd = pd.read_excel("./LightGBM_parameters.xlsx",index_col = 0,)
para_pd

# %% check the model performance again
test_ratio = 0.25

for random_seed in range(1,11,1):
    print('Processing: ', random_seed)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed) 

    for train,test in sss.split(X, y):
        X_cv = X.iloc[train]
        y_cv = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]
    X_cv_raw = X_raw.iloc[X_cv.index,:]

    model = lgb.LGBMClassifier(n_jobs=-1,max_cat_to_onehot=9, random_state=42,objective='binary',
                                bagging_fraction = para_pd.loc[random_seed,'bagging_fraction'],
                                bagging_freq = para_pd.loc[random_seed,'bagging_freq'],
                                boosting_type = para_pd.loc[random_seed,'boosting_type'],
                                feature_fraction = para_pd.loc[random_seed,'feature_fraction'],
                                learning_rate = para_pd.loc[random_seed,'learning_rate'],
                                max_bin = para_pd.loc[random_seed,'max_bin'],
                                max_depth = para_pd.loc[random_seed,'max_depth'],
                                num_iterations = para_pd.loc[random_seed,'num_iterations'],
                                num_leaves = para_pd.loc[random_seed,'num_leaves'],
                            )

    model.fit(X_cv, y_cv,categorical_feature=['Composition','Morphology','Seedling part'])

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_cv_proba = model.predict_proba(X_cv)[:, 1]

    print('Test AUC: %.2f'%metrics.roc_auc_score(y_test,y_proba))
    print('Test F1: %.2f'%metrics.f1_score(y_test,y_pred,average='weighted'))
    print('Test Accuracy: %.2f'%metrics.accuracy_score(y_test,y_pred))

    
# %%
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test) # each feature contribution is given as a contribution to the probability of the positive class.

# %%
shap.summary_plot(shap_values, X_test)

# %%
shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_test)

# %% global average effects/interactions

for random_seed in range(1,11,1):
    print('Processing: ', random_seed)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed) 

    for train,test in sss.split(X, y):
        X_cv = X.iloc[train]
        y_cv = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]
    X_test_raw = X_raw.iloc[X_test.index,:]

    model = lgb.LGBMClassifier(n_jobs=-1,max_cat_to_onehot=9, random_state=42,objective='binary',
                                bagging_fraction = para_pd.loc[random_seed,'bagging_fraction'],
                                bagging_freq = para_pd.loc[random_seed,'bagging_freq'],
                                boosting_type = para_pd.loc[random_seed,'boosting_type'],
                                feature_fraction = para_pd.loc[random_seed,'feature_fraction'],
                                learning_rate = para_pd.loc[random_seed,'learning_rate'],
                                max_bin = para_pd.loc[random_seed,'max_bin'],
                                max_depth = para_pd.loc[random_seed,'max_depth'],
                                num_iterations = para_pd.loc[random_seed,'num_iterations'],
                                num_leaves = para_pd.loc[random_seed,'num_leaves'],
                            )

    model.fit(X_cv, y_cv,categorical_feature=['Composition','Morphology','Seedling part'])

    if random_seed == 1:
        shap_interaction_values_all = shap.TreeExplainer(model).shap_interaction_values(X_test)
        shap_values_all = shap.TreeExplainer(model).shap_values(X_test)[1]
        X_test_all = X_test
    else:     
        shap_interaction_values = shap.TreeExplainer(model).shap_interaction_values(X_test)
        shap_interaction_values_all = np.concatenate((shap_interaction_values_all,shap_interaction_values),axis=0)
        shap_values = shap.TreeExplainer(model).shap_values(X_test)[1]
        shap_values_all = np.append(shap_values_all,shap_values,axis=0)
        X_test_all = pd.concat([X_test_all, X_test])

print(shap_interaction_values_all.shape, shap_values_all.shape, X_test_all.shape,)

# %%
index = feature_name_list.index('Concentration (mg/L)')
feature_shap_values = shap_values_all[:,index]
print('Average absoulte SHAP value for Concentration (mg/L): ', abs(feature_shap_values).mean())

index = feature_name_list.index('BET surface area (m2/g)')
feature_shap_values = shap_values_all[:,index]
print('Average absoulte SHAP value for BET surface area (m2/g)s: ', abs(feature_shap_values).mean())

index = feature_name_list.index('Concentration (mg/L)')
feature_shap_values = shap_interaction_values_all[:,index,index]
print('Average absoulte SHAP interaction value for Concentration (mg/L): ', abs(feature_shap_values).mean())

index = feature_name_list.index('BET surface area (m2/g)')
feature_shap_values = shap_interaction_values_all[:,index,index]
print('Average absoulte SHAP interaction value for BET surface area (m2/g)s: ', abs(feature_shap_values).mean())

# %% global SHAP values for numerical features
for feature in ['Concentration (mg/L)', 'Solubility','Hydrodynamic diameter (nm)', 'Zeta potential (mV)',
       'BET surface area (m2/g)','Relative weight']:

    index = feature_name_list.index(feature)
    global_shap_values = shap_values_all[:,index]
    fig, ax = plt.subplots(figsize=(3, 1.8))
    plt.xlabel(feature)
    plt.ylabel('SHAP value')
    plt.scatter(X_test_all[feature], global_shap_values, marker='o',s=45,c='#EA8D6C',linewidth=0.4,edgecolors='#FFFFFF')
    fig.savefig("./Image/SHAP_values_%s_global.jpg"%feature[0:6],dpi=600,bbox_inches='tight')

# %% global  SHAP values for categorical features
for feature in ['Composition','Morphology','Seedling part']:

    index = feature_name_list.index(feature)
    global_shap_values = shap_values_all[:,index]

    fig, ax = plt.subplots(figsize=(3, 1.8))
    plt.scatter(X_test_all[feature], global_shap_values, marker='o',s=45,c='#EA8D6C',linewidth=0.4,edgecolors='#FFFFFF')

    plt.xlabel(feature)
    plt.ylabel('SHAP value')
    if feature == 'Composition':
        plt.xticks(range(8),list(le_composition.inverse_transform([0,1,2,3,4,5,6,7])),rotation = 45,horizontalalignment='right')
    elif feature == 'Morphology':
        plt.xticks(range(2),list(le_morphology.inverse_transform([0,1])),)
    else:
        plt.xticks(range(5),['Root','Stem','Leaf','Shoot','Whole'])
        
    fig.savefig("./Image/SHAP_values_%s_global.jpg"%feature[0:6],dpi=600,bbox_inches='tight')


# %% global average main effects for numerical features
for feature in ['Concentration (mg/L)', 'Solubility','Hydrodynamic diameter (nm)', 'Zeta potential (mV)',
       'BET surface area (m2/g)','Relative weight']:
    index = feature_name_list.index(feature)
    global_main_effects = shap_interaction_values_all[:,index,index]
    fig, ax = plt.subplots(figsize=(3, 1.8))
    plt.xlabel(feature)
    plt.ylabel('SHAP main effect')
    plt.scatter(X_test_all[feature], global_main_effects, marker='o',s=45,c='#EA8D6C',linewidth=0.4,edgecolors='#FFFFFF')
    fig.savefig("./Image/SHAP_main_%s_global.jpg"%feature[0:6],dpi=600,bbox_inches='tight')
# small figures for plot in manuscript

# %%
feature = 'Solubility'
index = feature_name_list.index(feature)
global_main_effects = shap_interaction_values_all[:,index,index]
fig, ax = plt.subplots(figsize=(2, 1.2))
plt.xlabel(feature)
plt.ylabel('SHAP main effect')
plt.scatter(X_test_all[feature], global_main_effects, marker='o',s=30,c='#EA8D6C',linewidth=0.4,edgecolors='#FFFFFF')
fig.savefig("./Image/SHAP_main_%s_global.jpg"%feature[0:6],dpi=600,bbox_inches='tight')


# %% global average main effects for categorical 
for feature in ['Composition','Morphology','Seedling part']:

    index = feature_name_list.index(feature)

    index = feature_name_list.index(feature)
    global_main_effects = shap_interaction_values_all[:,index,index]

    fig, ax = plt.subplots(figsize=(3,1.8))
    plt.scatter(X_test_all[feature], global_main_effects, marker='o',s=45,c='#EA8D6C',linewidth=0.4,edgecolors='#FFFFFF')

    plt.xlabel(feature)
    plt.ylabel('SHAP main effect')
    if feature == 'Composition':
        plt.xticks(range(8),list(le_composition.inverse_transform([0,1,2,3,4,5,6,7])),rotation = 45,horizontalalignment='right')
    elif feature == 'Morphology':
        plt.xticks(range(2),list(le_morphology.inverse_transform([0,1])),)
    else:
        plt.xticks(range(5),['Root','Stem','Leaf','Shoot','Whole'])
        
    fig.savefig("./Image/SHAP_main_%s_global.jpg"%feature[0:6],dpi=600,bbox_inches='tight')

# small figures for plot in manuscript
for feature in ['Composition','Morphology']:

    index = feature_name_list.index(feature)

    index = feature_name_list.index(feature)
    global_main_effects = shap_interaction_values_all[:,index,index]

    if feature == 'Composition':
        fig, ax = plt.subplots(figsize=(2,1.2))
    else:
        fig, ax = plt.subplots(figsize=(2,1.1))

    plt.scatter(X_test_all[feature], global_main_effects, marker='o',s=30,c='#EA8D6C',linewidth=0.4,edgecolors='#FFFFFF')

    plt.xlabel(feature)
    plt.ylabel('SHAP main effect')
    if feature == 'Composition':
        plt.xticks(range(8),list(le_composition.inverse_transform([0,1,2,3,4,5,6,7])),rotation = 45,horizontalalignment='right')
    elif feature == 'Morphology':
        plt.xticks(range(2),list(le_morphology.inverse_transform([0,1])),)
    else:
        plt.xticks(range(5),['Root','Stem','Leaf','Shoot','Whole'])
        
    fig.savefig("./Image/SHAP_main_%s_global.jpg"%feature[0:6],dpi=600,bbox_inches='tight')






# %% SHAP interaction values
fig, ax= plt.subplots(figsize = (8, 8))
plt.style.use('default')
tmp = np.abs(shap_interaction_values_all).sum(0)/10
for i in range(tmp.shape[0]):
    tmp[i,i] = 0
inds = np.argsort(-tmp.sum(0))[:50]
tmp2 = tmp[inds,:][:,inds]
df_temp2 = pd.DataFrame(tmp2)
df_temp2.columns = X.columns[inds]
df_temp2.index = X.columns[inds]

h=sns.heatmap(df_temp2, cmap='flare', square=True, center=12,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':12})
bottom, top = ax.get_ylim()

cb = h.figure.colorbar(h.collections[0],shrink=0.85) #显示colorbar
cb.ax.tick_params(labelsize=15)  # 设置colorbar刻度字体大小。
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='left',fontsize=15)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45,horizontalalignment='right',fontsize=15, rotation_mode='anchor')

fig.savefig("./Image/Global_interactions_all.jpg",dpi=600,bbox_inches='tight')


fig, ax= plt.subplots(figsize = (4, 4))
plt.style.use('default')
tmp = np.abs(shap_interaction_values_all).sum(0)/10
for i in range(tmp.shape[0]):
    tmp[i,i] = 0
inds = np.argsort(-tmp.sum(0))[:50]
tmp2 = tmp[inds,:][:,inds]
df_temp2 = pd.DataFrame(tmp2)
df_temp2.columns = X.columns[inds]
df_temp2.index = X.columns[inds]

df_temp2 = df_temp2.iloc[0:3,0:3]

h=sns.heatmap(df_temp2, cmap='flare', square=True, center=12,
            fmt=".2f", annot=True, linewidths=0.4, ax=ax, cbar=False,annot_kws={'size':12})
bottom, top = ax.get_ylim()
#ax.set_ylim(bottom + 0.5, top - 0.5)            
cb = h.figure.colorbar(h.collections[0],shrink=0.85) #显示colorbar
cb.ax.tick_params(labelsize=12)  # 设置colorbar刻度字体大小。
ax.xaxis.tick_top()
ax.set_xticklabels(ax.get_xticklabels(), rotation=45,horizontalalignment='left',fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45,horizontalalignment='right',fontsize=12, rotation_mode='anchor')

fig.savefig("./Image/Global_interactions_important.jpg",dpi=600,bbox_inches='tight')




# %% global average interactions
feature_1 = 'BET surface area (m2/g)'
feature_2 = 'Concentration (mg/L)'

index_1 = feature_name_list.index(feature_1)
index_2 = feature_name_list.index(feature_2)

global_main_effects = shap_interaction_values_all[:,index_1,index_2]

data_plot = pd.DataFrame(columns=[feature_1,feature_2,'SHAP interactions'])
data_plot[feature_1] = X_test_all[feature_1].values
data_plot[feature_2] = X_test_all[feature_2].values
data_plot['SHAP interactions'] = global_main_effects

fig, ax = plt.subplots(figsize=(4, 2.5))
sns.scatterplot(x=feature_1, y='SHAP interactions',hue=feature_2,data=data_plot,
                palette='flare',s=45)

#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

norm = plt.Normalize(25, 200)
sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm,label=feature_2)

fig.savefig("./Image/SHAP_inter_BET_Conc_global.jpg",dpi=600,bbox_inches='tight')


# %% global average interactions
feature_1 = 'BET surface area (m2/g)'
feature_2 = 'Relative weight'

index_1 = feature_name_list.index(feature_1)
index_2 = feature_name_list.index(feature_2)

global_main_effects = shap_interaction_values_all[:,index_1,index_2]

data_plot = pd.DataFrame(columns=[feature_1,feature_2,'SHAP interactions'])
data_plot[feature_1] = X_test_all[feature_1].values
data_plot[feature_2] = X_test_all[feature_2].values
data_plot['SHAP interactions'] = global_main_effects

fig, ax = plt.subplots(figsize=(4, 2.5))
sns.scatterplot(x=feature_1, y='SHAP interactions',hue=feature_2,data=data_plot,
                palette='flare',s=45)

#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

norm = plt.Normalize(25, 200)
sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm,label=feature_2)

fig.savefig("./Image/SHAP_inter_BET_relat_weight_global.jpg",dpi=600,bbox_inches='tight')


# %% global average interactions
feature_1 = 'Relative weight'
feature_2 = 'Concentration (mg/L)'

index_1 = feature_name_list.index(feature_1)
index_2 = feature_name_list.index(feature_2)

global_main_effects = shap_interaction_values_all[:,index_1,index_2]

data_plot = pd.DataFrame(columns=[feature_1,feature_2,'SHAP interactions'])
data_plot[feature_1] = X_test_all[feature_1].values
data_plot[feature_2] = X_test_all[feature_2].values
data_plot['SHAP interactions'] = global_main_effects

fig, ax = plt.subplots(figsize=(4, 2.5))
sns.scatterplot(x=feature_1, y='SHAP interactions',hue=feature_2,data=data_plot,
                palette='flare',s=45)

#plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

norm = plt.Normalize(25, 200)
sm = plt.cm.ScalarMappable(cmap="flare", norm=norm)
sm.set_array([])

# Remove the legend and add a colorbar
ax.get_legend().remove()
ax.figure.colorbar(sm,label=feature_2)

fig.savefig("./Image/SHAP_inter_Relatweight_Conc_global.jpg",dpi=600,bbox_inches='tight')

# %%
