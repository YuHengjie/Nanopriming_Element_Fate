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
data = pd.read_excel("./Dataset_RMC_qcut.xlsx",index_col = 0,)
data

# %% 
X = data.drop(columns=['RMC']) 

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
random_seed = 10
# run the following code from 1 to 10 (random_state) and record model performance and hyperparameters
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_seed) 

for train,test in sss.split(X, y):
    X_cv = X.iloc[train]
    y_cv = y.iloc[train]
    X_test = X.iloc[test]
    y_test = y.iloc[test]

X_cv

# %%
parameter = {
        #'boosting_type': ['gbdt','rf','dart'],
        #"num_iterations":[100,200,400,800],
        #"max_bin":  [8,16,24,32],
        #"min_child_samples":[4,8,12,20,], # same with min_data_in_leaf, use the aliase for avoiding too much warning messages
        #"min_child_weight": [0.001,0.01,0.1,1], # same with min_sum_hessian_in_leaf
        #"max_depth":[3,4,5,6,7],
        #"num_leaves": [4,6,8,12,16,20],
        #'learning_rate':[0.01,0.05,0.1,0.2],
}

model_gs = lgb.LGBMClassifier(n_jobs=-1,max_cat_to_onehot=9, random_state=42,objective='binary',
                            boosting_type = 'dart',
                            num_iterations = 800,
                            max_bin = 32,
                            min_data_in_leaf=12,
                            min_sum_hessian_in_leaf=0.001,
                            max_depth=6,
                            num_leaves=8,
                            learning_rate=0.01,
                            )

grid_search = GridSearchCV(model_gs, param_grid = parameter, scoring='accuracy', cv=5, n_jobs=-1)
grid_search.fit(X_cv, y_cv,categorical_feature=['Composition','Seedling part','Morphology'])

print('best score: %.3f '%grid_search.best_score_)
print('best_params:', grid_search.best_params_)

LGBM_Gs_best = grid_search.best_estimator_

# %%
AUC_CV = cross_val_score(LGBM_Gs_best, X_cv, y_cv, cv=5, scoring='roc_auc').mean()
F1_CV = cross_val_score(LGBM_Gs_best, X_cv, y_cv, cv=5, scoring='f1_weighted').mean()
Accuracy_CV = cross_val_score(LGBM_Gs_best, X_cv, y_cv, cv=5, scoring='accuracy').mean()

# %%
print('AUC_CV: %.2f ' %AUC_CV)
print('F1_CV: %.2f ' %F1_CV)
print('Accuracy_CV: %.2f ' %Accuracy_CV)

y_pred_cv = LGBM_Gs_best.predict(X_cv)
y_proba_cv = LGBM_Gs_best.predict_proba(X_cv)[:, 1]

print('Train AUC: %.2f'%metrics.roc_auc_score(y_cv,y_proba_cv))
print('Train F1: %.2f'%metrics.f1_score(y_cv,y_pred_cv,average='weighted'))
print('Train Accuracy: %.2f'%metrics.accuracy_score(y_cv,y_pred_cv))
#print('Train MCC: %.2f'%metrics.matthews_corrcoef(y_cv,y_pred))

y_pred = LGBM_Gs_best.predict(X_test)
y_proba = LGBM_Gs_best.predict_proba(X_test)[:, 1]

print('Test AUC: %.2f'%metrics.roc_auc_score(y_test,y_proba))
print('Test F1: %.2f'%metrics.f1_score(y_test,y_pred,average='weighted'))
print('Test Accuracy: %.2f'%metrics.accuracy_score(y_test,y_pred))
#print('Test MCC: %.2f'%metrics.matthews_corrcoef(y_test,y_pred))

# %%

# %%
# run the following code from 1 to 10 (random_state) and record model performance and hyperparameters
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=random_seed) 

for train,test in sss.split(X, y):
    X_cv = X.iloc[train]
    y_cv = y.iloc[train]
    X_test = X.iloc[test]
    y_test = y.iloc[test]

model = lgb.LGBMClassifier(n_jobs=-1,max_cat_to_onehot=9, random_state=42,objective='binary',
                            boosting_type = 'dart',
                            num_iterations = 800,
                            max_bin = 32,
                            min_data_in_leaf=12,
                            min_sum_hessian_in_leaf=0.001,
                            max_depth=6,
                            num_leaves=8,
                            learning_rate=0.01,
                         )

model.fit(X_cv, y_cv,categorical_feature=['Composition','Morphology','Seedling part'])
LightGBM_importance_split = model.feature_importances_

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
y_cv_proba = model.predict_proba(X_cv)[:, 1]

print('Test AUC: %.2f'%metrics.roc_auc_score(y_test,y_proba))
print('Test F1: %.2f'%metrics.f1_score(y_test,y_pred,average='weighted'))
print('Test Accuracy: %.2f'%metrics.accuracy_score(y_test,y_pred))

# %%
fig, ax= plt.subplots(figsize = (3,3))
plt.style.use('classic')
plt.rcParams['font.size'] ='8'
plt.margins(0.02)

fpr, tpr, threshold = metrics.roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, 'dodgerblue',label = 'Test set')

fpr, tpr, threshold = metrics.roc_curve(y_cv, y_cv_proba)
plt.plot(fpr, tpr, 'hotpink',label = 'Train set')

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],c='dimgray',linestyle='--')
margin = 0.03
plt.xlim([0-margin, 1+margin])
plt.ylim([0-margin, 1+margin])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

fig.savefig("./Image/Test_ROC.jpg",dpi=600,bbox_inches='tight')



# %%
figure = plt.figure(figsize=(8,6))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
plt.margins(0.02)
sorted_idx = LightGBM_importance_split.argsort()
sorted_features = X_cv.columns[sorted_idx]
fature_name = X_cv.columns
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

plt.barh(feature_this_plot,
         LightGBM_importance_split[sorted_idx], align='center', color="#1E90FF")

plt.title('LightGBM feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=18)

figure.savefig("./Image/LightGBM_importance.jpg",dpi=600,bbox_inches='tight')

# %%
result = permutation_importance(model, X_cv, y_cv, scoring='accuracy', 
                                n_repeats=10, random_state=0, n_jobs=-1)
Permutation_importance = result.importances_mean

figure = plt.figure(figsize=(8,6))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
plt.margins(0.02)
sorted_idx = Permutation_importance.argsort()
sorted_features = X_cv.columns[sorted_idx]
fature_name = X_cv.columns
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

plt.barh(feature_this_plot,
         Permutation_importance[sorted_idx], align='center', color="#1E90FF")

plt.title('Permutation feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=16)

figure.savefig("./Image/Permutation_importance.jpg",dpi=600,bbox_inches='tight')

# %%
explainer = shap.TreeExplainer(model=model, data=None, model_output='raw', feature_perturbation='tree_path_dependent')
shap_values = explainer.shap_values(X_cv)[1]
global_shap_values = np.abs(shap_values).mean(0)

figure = plt.figure(figsize=(8,6))
plt.style.use('classic')
plt.rcParams['font.size'] ='16'
plt.margins(0.02)
sorted_idx = global_shap_values.argsort()
sorted_features = X_cv.columns[sorted_idx]
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

plt.barh(feature_this_plot,
         global_shap_values[sorted_idx], align='center', color="#1E90FF")

plt.title('SHAP feature importance',fontsize=18)
plt.xlabel('Importance value',fontsize=16)
figure.savefig("./Image//Shap_importance.jpg",dpi=600,bbox_inches='tight')

# %%
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
importance_df

# %%
sorted_features

# %%
importance_df = pd.DataFrame(columns=('Feature','Method','Relative importance value'))
n_feature = len(X_cv.columns)

for i in range(0,n_feature):
    importance_df.loc[i,'Feature'] = X_cv.columns[sorted_idx_sum][-i-1]
    importance_df.loc[i,'Method'] = 'LightGBM'
    importance_df.loc[i,'Relative importance value'] = LightGBM_importance_split_relative[sorted_idx_sum][-i-1]

for i in range(0,n_feature):
    importance_df.loc[i+n_feature,'Feature'] = X_cv.columns[sorted_idx_sum][-i-1]
    importance_df.loc[i+n_feature,'Method'] = 'Permutation'
    importance_df.loc[i+n_feature,'Relative importance value'] = Permutation_importance_relative[sorted_idx_sum][-i-1]
    
for i in range(0,n_feature):
    importance_df.loc[i+n_feature*2,'Feature'] = X_cv.columns[sorted_idx_sum][-i-1]
    importance_df.loc[i+n_feature*2,'Method'] = 'SHAP'
    importance_df.loc[i+n_feature*2,'Relative importance value'] = shap_values__relative[sorted_idx_sum][-i-1]

LightGBM_split_sorted_value = sorted(importance_df.loc[:,'Relative importance value'][0:n_feature].values,reverse=True)
Permutation_sorted_value = sorted(importance_df.loc[:,'Relative importance value'][n_feature*1:n_feature*2].values,reverse=True)
SHAP_sorted_value = sorted(importance_df.loc[:,'Relative importance value'][n_feature*2:n_feature*3].values,reverse=True)

# %%
annotate_LightGBM_split = []
annotate_LightGBM_gain = []
annotate_Permutation = []
annotate_SHAP = []
n_feature = len(X_cv.columns)
for i in range(0,n_feature,1):
    annotate_LightGBM_split.append(LightGBM_split_sorted_value.index(importance_df.loc[:,'Relative importance value'][0:n_feature].values[i])+1)
    annotate_Permutation.append(Permutation_sorted_value.index(importance_df.loc[:,'Relative importance value'][n_feature:n_feature*2].values[i])+1)
    annotate_SHAP.append(SHAP_sorted_value.index(importance_df.loc[:,'Relative importance value'][n_feature*2:n_feature*3].values[i])+1)
    
annotate_value = np.hstack((annotate_LightGBM_split, annotate_LightGBM_gain, annotate_Permutation,annotate_SHAP))
annotate_value

# %%
feature_this_plot = []
for item in sorted_features:
    itemindex = np.argwhere(fature_name == item)
    feature_this_plot.append(fature_name[int(itemindex)])

figure = plt.figure(figsize=(6,5))
plt.style.use('classic')
bar = sns.barplot(data = importance_df,y='Feature',x='Relative importance value',hue='Method',palette="rocket")
bar.set_ylabel('',fontsize=16)
bar.set_xlabel('Relative importance value',fontsize=16)
bar.set_yticklabels(feature_this_plot,fontsize=16)
plt.legend(loc='lower right')
i=0
plt.margins(0.02)
for p in bar.patches:
    if p.get_width()>=0:
        bar.annotate("%d" %annotate_value[i], xy=(p.get_width(), p.get_y()+p.get_height()/2),
                xytext=(1, -0.5), textcoords='offset points', ha="left", va="center",fontsize=7)
    else:
        bar.annotate("%d" %annotate_value[i], xy=(0, p.get_y()+p.get_height()/2),
        xytext=(1, -0.5), textcoords='offset points', ha="left", va="center",fontsize=7)
    i=i+1
figure.savefig("./Image/Importance_summary.jpg",dpi=600,bbox_inches='tight')

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

    Relative_importance_all.to_excel('LightGBM_Relative_importance_all.xlsx')


# %%



# %%
