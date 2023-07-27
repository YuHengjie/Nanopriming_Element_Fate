# %% 导入包
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns
from matplotlib import rcParams
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
X = data.drop(columns=['RMC']) 
X_raw = X.copy() # for SHAP
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
para_pd = pd.read_excel("./LightGBM_parameters.xlsx",index_col = 0,)
para_pd

# %% test the models are same with grid-search and obtain ten models
for random_seed in range(1,11,1):

    print('Processing: ', random_seed)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_seed) 

    for train,test in sss.split(X, y):
        X_cv = X.iloc[train]
        y_cv = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]

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
    LightGBM_importance_split = model.feature_importances_

    exec("model_{} = model".format(random_seed))

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print('Test AUC: %.2f'%metrics.roc_auc_score(y_test,y_proba))
    print('Test F1: %.2f'%metrics.f1_score(y_test,y_pred,average='weighted'))
    print('Test Accuracy: %.2f'%metrics.accuracy_score(y_test,y_pred))


# %%  numerical features for all models
ratio_show_lines = 1
for feature in ['Concentration (mg/L)', 'Solubility','Hydrodynamic diameter (nm)', 'Zeta potential (mV)',
       'BET surface area (m2/g)','Relative weight']:
    for i in range(1,11,1):
        exec("LGBM_GS_best = model_{}".format(i))

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=i) 
        for train,test in sss.split(X, y):
            X_cv = X.iloc[train]
            y_cv = y.iloc[train]
            X_test = X.iloc[test]
            y_test = y.iloc[test]

        pdp_NP_none_M = pdp.pdp_isolate(model=LGBM_GS_best,
                            dataset=X_test,
                            model_features=X_test.columns,
                            feature=feature,
                            percentile_range=(0, 100),
                            n_jobs=-1, num_grid_points = min(10, len(np.unique(X_test[feature]))))
        if i == 1:
            ICE_lines = pdp_NP_none_M.ice_lines
            PDP_lines = pd.DataFrame(columns=ICE_lines.columns)
            PDP_lines.loc[len(PDP_lines)] = pdp_NP_none_M.pdp
        else:
            ICE_lines = pd.concat([ICE_lines, pdp_NP_none_M.ice_lines])
            ICE_lines = ICE_lines.reset_index(drop=True)
            PDP_lines_temp = pd.DataFrame(columns=pdp_NP_none_M.ice_lines.columns)
            PDP_lines_temp.loc[len(PDP_lines_temp)] = pdp_NP_none_M.pdp
            PDP_lines = pd.concat([PDP_lines, PDP_lines_temp])
            PDP_lines = PDP_lines.reset_index(drop=True)

    # delete the columns with less than 3 non-null values
    non_null_counts = PDP_lines.count()
    PDP_lines = PDP_lines[non_null_counts[non_null_counts >= 3].index] 
    PDP_lines = PDP_lines.reindex(sorted(PDP_lines.columns), axis=1) # sort columns for plot
    ICE_lines = ICE_lines.reindex(sorted(ICE_lines.columns), axis=1) # sort columns for plot

    # show part of the ice lines
    num_lines = int(ICE_lines.shape[0] * ratio_show_lines)
    select_lines = np.random.choice(ICE_lines.index, size=num_lines, replace=False)
    ICE_lines_show = ICE_lines.loc[select_lines]

    fig, ax = plt.subplots(figsize=(3, 1.85))

    pdp_mean = PDP_lines.mean()
    pdp_std = PDP_lines.std()

    #plt.plot(PDP_lines.columns,ICE_lines.values.T, label='ICE lines', linewidth=0.2,color='#696969',zorder=-1)
    for i in range(len(ICE_lines_show)):
        pd_temp = ICE_lines_show.iloc[i,:].dropna()
        plt.plot(pd_temp.index,pd_temp.values,linewidth=0.1,color='#696969',zorder=-1)

    plt.plot(pdp_mean, marker='o',markersize=4,label='Average PDP',zorder=2,linewidth=1.5,color='#FF8C00')
    plt.fill_between(PDP_lines.columns, pdp_mean-pdp_std, pdp_mean+pdp_std, alpha=0.5,zorder=1,color='#FFDAB9')

    plt.xlabel(feature)
    plt.ylabel('Predicted RMC')
    fig.savefig("./Image/PDP_ICE_%s_global.jpg"%feature[0:6],dpi=600,bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(1, 0.625))
    pdp_mean = PDP_lines.mean()
    pdp_std = PDP_lines.std()
    #plt.yticks([0.47,0.50])
    plt.plot(pdp_mean, marker='o',markersize=2,label='Average PDP',zorder=2,linewidth=1.5,color='#FF8C00')
    plt.fill_between(PDP_lines.columns, pdp_mean-pdp_std, pdp_mean+pdp_std, alpha=0.5,zorder=1,color='#FFDAB9')
    fig.savefig("./Image/PDP_%s_global.jpg"%feature[0:6],dpi=600,bbox_inches='tight')
    
    if  feature == 'Solubility':
        fig, ax = plt.subplots(figsize=(2, 1))
        pdp_mean = PDP_lines.mean()
        pdp_std = PDP_lines.std()
        #plt.yticks([0.47,0.50])
        plt.xlabel(feature)
        plt.ylabel('Predicted RMC')
        plt.plot(pdp_mean, marker='o',markersize=2,label='Average PDP',zorder=2,linewidth=1.5,color='#FF8C00')
        plt.fill_between(PDP_lines.columns, pdp_mean-pdp_std, pdp_mean+pdp_std, alpha=0.5,zorder=1,color='#FFDAB9')
        fig.savefig("./Image/PDP_%s_global.jpg"%feature[0:6],dpi=600,bbox_inches='tight')

# %% category features for all models
for feature in ['Composition','Morphology','Seedling part']:
    for i in range(1,11,1):
        exec("LGBM_GS_best = model_{}".format(i))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=i) 
        for train,test in sss.split(X, y):
            X_cv = X.iloc[train]
            y_cv = y.iloc[train]
            X_test = X.iloc[test]
            y_test = y.iloc[test]

        pdp_NP_none_M = pdp.pdp_isolate(model=LGBM_GS_best,
                            dataset=X_test,
                            model_features=X_test.columns,
                            feature=feature,
                            percentile_range=(0, 100),
                            n_jobs=-1, num_grid_points = min(10, len(np.unique(X_test[feature]))))
        if i == 1:
            ICE_lines = pdp_NP_none_M.ice_lines
            PDP_lines = pd.DataFrame(columns=ICE_lines.columns)
            PDP_lines.loc[len(PDP_lines)] = pdp_NP_none_M.pdp
        else:
            ICE_lines = pd.concat([ICE_lines, pdp_NP_none_M.ice_lines])
            ICE_lines = ICE_lines.reset_index(drop=True)
            PDP_lines_temp = pd.DataFrame(columns=pdp_NP_none_M.ice_lines.columns)
            PDP_lines_temp.loc[len(PDP_lines_temp)] = pdp_NP_none_M.pdp
            PDP_lines = pd.concat([PDP_lines, PDP_lines_temp])
            PDP_lines = PDP_lines.reset_index(drop=True)

    # delete the columns with less than 3 non-null values
    non_null_counts = PDP_lines.count()
    PDP_lines = PDP_lines[non_null_counts[non_null_counts >= 3].index] 
    PDP_lines = PDP_lines.reindex(sorted(PDP_lines.columns), axis=1) # sort columns for plot
    ICE_lines = ICE_lines.reindex(sorted(ICE_lines.columns), axis=1) # sort columns for plot

    # show part of the ice lines
    num_lines = int(ICE_lines.shape[0] * ratio_show_lines)
    select_lines = np.random.choice(ICE_lines.index, size=num_lines, replace=False)
    ICE_lines_show = ICE_lines.loc[select_lines]

    fig, ax = plt.subplots(figsize=(3, 1.85))

    pdp_mean = PDP_lines.mean()
    pdp_std = PDP_lines.std()

    #plt.plot(PDP_lines.columns,ICE_lines.values.T, label='ICE lines', linewidth=0.2,color='#696969',zorder=-1)
    for i in range(len(ICE_lines_show)):
        pd_temp = ICE_lines_show.iloc[i,:].dropna()
        plt.plot(pd_temp.index,pd_temp.values,linewidth=0.2,color='#696969',zorder=-1)

    plt.plot(pdp_mean, marker='o',markersize=4,label='Average PDP',zorder=2,linewidth=1.5,color='#FF8C00')
    plt.fill_between(PDP_lines.columns, pdp_mean-pdp_std, pdp_mean+pdp_std, alpha=0.5,zorder=1,color='#FFDAB9')
    
    if feature == 'Composition':
        plt.xticks(range(8),list(le_composition.inverse_transform([0,1,2,3,4,5,6,7])),rotation = 45,horizontalalignment='right')
    elif feature == 'Morphology':
        plt.xticks(range(2),list(le_morphology.inverse_transform([0,1])),)
    else:
        plt.xticks(range(5),['Root','Stem','Leaf','Shoot','Whole'])


    plt.xlabel(feature)
    plt.ylabel('Predicted RMC')
    fig.savefig("./Image/PDP_ICE_%s_global.jpg"%feature[0:6],dpi=600,bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(1, 0.625))
    pdp_mean = PDP_lines.mean()
    pdp_std = PDP_lines.std()
    #plt.yticks([0.47,0.50])
    if feature == 'Composition':
        plt.xticks(range(8),['','','','','','','',''])
    elif feature == 'Morphology':
        plt.xticks(range(2),['',''])
    else:
        plt.xticks(range(5),['','','','',''])

    plt.plot(pdp_mean, marker='o',markersize=2,label='Average PDP',zorder=2,linewidth=1.5,color='#FF8C00')
    plt.fill_between(PDP_lines.columns, pdp_mean-pdp_std, pdp_mean+pdp_std, alpha=0.5,zorder=1,color='#FFDAB9')

    fig.savefig("./Image/PDP_%s_global.jpg"%feature[0:6],dpi=600,bbox_inches='tight')
    
    if feature in ['Composition','Morphology']:
        fig, ax = plt.subplots(figsize=(2, 1))
        pdp_mean = PDP_lines.mean()
        pdp_std = PDP_lines.std()
        #plt.yticks([0.47,0.50])
        if feature == 'Composition':
            plt.xticks(range(8),list(le_composition.inverse_transform([0,1,2,3,4,5,6,7])),rotation = 45,horizontalalignment='right')
        elif feature == 'Morphology':
            plt.xticks(range(2),list(le_morphology.inverse_transform([0,1])),)
        else:
            plt.xticks(range(5),['Root','Stem','Leaf','Shoot','Whole'])

        plt.plot(pdp_mean, marker='o',markersize=2,label='Average PDP',zorder=2,linewidth=1.5,color='#FF8C00')
        plt.fill_between(PDP_lines.columns, pdp_mean-pdp_std, pdp_mean+pdp_std, alpha=0.5,zorder=1,color='#FFDAB9')
        plt.xlabel(feature)
        plt.ylabel('Predicted RMC')
        fig.savefig("./Image/PDP_%s_global.jpg"%feature[0:6],dpi=600,bbox_inches='tight')


# %%
