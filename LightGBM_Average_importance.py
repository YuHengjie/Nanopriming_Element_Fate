# %% 导入包
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 

from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.inspection import permutation_importance
import shap
import seaborn as sns

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %% 
data = pd.read_excel("./Dataset_RMC.xlsx",index_col = 0,)
data

# %% 
X = data.drop(columns=['RMC']) 
print(len(X.columns))
print(X.columns)

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
X

# %%
target_name = 'RMC'
y = data.loc[:,target_name]
y

# %%
para_pd = pd.read_excel("./LightGBM_parameters.xlsx",index_col = 0,)
para_pd

# %% test the models are same with grid-search
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

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print('Test AUC: %.2f'%metrics.roc_auc_score(y_test,y_proba))
    print('Test F1: %.2f'%metrics.f1_score(y_test,y_pred,average='weighted'))
    print('Test Accuracy: %.2f'%metrics.accuracy_score(y_test,y_pred))

# %%
importance_rank = pd.DataFrame(index=range(1,11,1),columns=X.columns)

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

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    result = permutation_importance(model, X_cv, y_cv, scoring='accuracy', 
                                n_repeats=10, random_state=0, n_jobs=-1)
    Permutation_importance = result.importances_mean

    explainer = shap.TreeExplainer(model=model, data=None, model_output='raw', feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(X_cv)[1]
    global_shap_values = np.abs(shap_values).mean(0)

    # Calculate the relative importance, the maximum is 1
    LightGBM_importance_split_relative = LightGBM_importance_split/max(LightGBM_importance_split)
    Permutation_importance_relative = Permutation_importance/max(Permutation_importance)
    shap_values__relative = global_shap_values/max(global_shap_values)

    # Sort by the sum of relative importance
    importance_sum = LightGBM_importance_split_relative+Permutation_importance_relative+shap_values__relative
    sorted_idx_sum = importance_sum.argsort()
    sorted_features = X_cv.columns[sorted_idx_sum][::-1]

    importance_relatiove_value = pd.DataFrame(columns=['Feature','Relative importance'])
    importance_relatiove_value['Feature'] = X.columns
    importance_relatiove_value['Relative importance'] = importance_sum/3
    importance_relatiove_value.to_excel('./Table/Relative importance_'+str(random_seed)+'.xlsx')

    importance_df = pd.DataFrame({'Feature':X_cv.columns[sorted_idx_sum],
                        'LightGBM (split)':LightGBM_importance_split_relative[sorted_idx_sum],
                        'Permutatio':Permutation_importance_relative[sorted_idx_sum],
                        'SHAP':shap_values__relative[sorted_idx_sum]},
                        )
    
    if importance_df.iloc[0,1:].sum() == 0:
        print('Warning: At least one feature is not used by the {}th model.'.format(random_seed))

    for feature in importance_rank.columns:
        index = importance_df.index[importance_df['Feature'] == feature][0]
        importance_rank.loc[random_seed,feature] = len(importance_rank.columns)-index
importance_rank.to_excel('LightGBM_Importance_Rank.xlsx')
importance_rank

# %%


# %%
if random_seed == 10:
    file_list = os.listdir('./Table')
    xlsx_files = [file for file in file_list if file.endswith('.xlsx')]

    for i in range(1,11,1):
        exec("temp_pd = pd.read_excel(\"./Table/Relative importance_{}.xlsx\",index_col = 0,)".format(i,i))
        if i == 1:
            Relative_importance_all = pd.DataFrame(columns=temp_pd['Feature'],index=range(1,11,1))
            Relative_importance_all.loc[i,:] = temp_pd['Relative importance'].values
        else:
            Relative_importance_all.loc[i,:] = temp_pd['Relative importance'].values

    mean_Relative_importance_all = Relative_importance_all.mean(axis=0)
    mean_Relative_importance_all.sort_values(ascending=False, inplace=True)
    Relative_importance_all = Relative_importance_all[mean_Relative_importance_all.index]

    Relative_importance_all.to_excel('LightGBM_Importance_Relative.xlsx')
Relative_importance_all

# %%
