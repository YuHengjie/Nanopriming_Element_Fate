# %% 导入包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit 
import joblib
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

# %% 
data = pd.read_excel("./Dataset_RMC.xlsx",index_col = 0,)
data

# %%
data.loc[:,'Seedling part'].value_counts()

# %%
data.describe()

# %%
plt.figure(figsize=(6,4))
plt.style.use('default')
seedling_part_list = data.loc[:,'Seedling part'].unique()
plt.hist([data.loc[data['Seedling part'] == seedling_part_list[0], 'Relative weight'],
          data.loc[data['Seedling part'] == seedling_part_list[1], 'Relative weight'],
          data.loc[data['Seedling part'] == seedling_part_list[2], 'Relative weight'],
          data.loc[data['Seedling part'] == seedling_part_list[3], 'Relative weight'],
          data.loc[data['Seedling part'] == seedling_part_list[4], 'Relative weight']],
         bins=10, stacked=True, label=seedling_part_list)

plt.xlabel('Relative weight')
plt.ylabel('Frequency')
plt.legend()
plt.savefig("./Image/dataset_seedling_part_weight.jpg",dpi=600,bbox_inches='tight')


# %% 
plt.figure(figsize=(8,10))
plt.style.use('default')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

for i in range(1,11,1):
    plt.subplot(4,3,i)
    if i in [1,7,10]:
        if i == 1:
            data[data.columns[i-1]].value_counts().plot(kind='bar',color="#FF8C00",
                edgecolor="black", alpha=0.7, width=1)
            plt.xticks(rotation = 45,horizontalalignment='right')
        elif i == 7:
            data[data.columns[i-1]].value_counts().plot(kind='bar',color="#FF8C00",
                edgecolor="black", alpha=0.7, width=0.45)
            plt.xticks(rotation = 0,horizontalalignment='center')
        else:
            data[data.columns[i-1]].value_counts().plot(kind='bar',color="#FF8C00",
                edgecolor="black", alpha=0.7, width=0.15)
            plt.xticks(rotation = 0,horizontalalignment='center') 
    else:
        plt.hist(data.iloc[:,i-1], facecolor="#FF8C00", edgecolor="black", alpha=0.7)
    plt.xlabel(data.columns[i-1])
    plt.ylabel("Freqency")
    
plt.tight_layout()
plt.savefig("./Image/dataset_visual.jpg",dpi=600,bbox_inches='tight')

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
best_parameters = []
test_ratio = 0.25
df_result = pd.DataFrame(columns=('AUC_CV','F1_CV','Accuracy_CV','Train AUC','Train F1','Train Accuracy','Test AUC','Test F1','Test Accuracy'))
for random_seed in range(1,11,1):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed) 

    for train,test in sss.split(X, y):
        X_cv = X.iloc[train]
        y_cv = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]

    parameter = {
        "boosting_type": ['gbdt','dart'],
        "num_iterations":[100,200,400,600],
        "max_bin":  [16,24,32,64,],
        "min_child_weight": [1e-3,1e-2,0.1,1],
        "max_depth":[3,4,5,6,],
        "num_leaves": [4,7,10,14],
        'bagging_fraction': [0.6,0.7,0.8,0.9],
        'bagging_freq': [1,4,8,12],
        'feature_fraction': [0.4,0.6,0.8,1],
        'learning_rate': [0.1,0.2,0.3,0.4],

    }

    model_gs = lgb.LGBMClassifier(n_jobs=-1,max_cat_to_onehot=9, 
                                  random_state=42,objective='binary')

    grid_search = GridSearchCV(model_gs, param_grid = parameter, scoring='accuracy', cv=5, n_jobs=-1, )
    grid_search.fit(X_cv, y_cv,categorical_feature=['Composition','Seedling part','Morphology'])
    LGBM_Gs_best = grid_search.best_estimator_
    
    best_parameters.append(["Best parameters for "+str(random_seed)+": ",grid_search.best_params_,])

    y_pred_cv = LGBM_Gs_best.predict(X_cv)
    y_proba_cv = LGBM_Gs_best.predict_proba(X_cv)[:, 1]

    y_pred = LGBM_Gs_best.predict(X_test)
    y_proba = LGBM_Gs_best.predict_proba(X_test)[:, 1]

    df_result = df_result.append(pd.Series({'AUC_CV':sum(cross_val_score(LGBM_Gs_best, X_cv, y_cv, cv=5, scoring='roc_auc'))/5,
                                                'F1_CV':sum(cross_val_score(LGBM_Gs_best, X_cv, y_cv, cv=5, scoring='f1_weighted'))/5,
                                                'Accuracy_CV':sum(cross_val_score(LGBM_Gs_best, X_cv, y_cv, cv=5, scoring='accuracy'))/5,
                                                'Train AUC':metrics.roc_auc_score(y_cv,y_proba_cv),
                                                'Train F1':metrics.f1_score(y_cv,y_pred_cv,average='weighted'),
                                                'Train Accuracy':metrics.accuracy_score(y_cv,y_pred_cv),
                                                'Test AUC':metrics.roc_auc_score(y_test,y_proba),
                                                'Test F1':metrics.f1_score(y_test,y_pred,average='weighted'),
                                                'Test Accuracy':metrics.accuracy_score(y_test,y_pred)}),ignore_index=True)
   
    df_result.to_excel('./Model_LightGBM/GridSearch'+'_performance_'+str(random_seed)+'.xlsx')
    X_test['Observed RMC'] = y_test
    X_test['Predicted RMC'] = y_pred
    X_test.to_excel('./Model_LightGBM/Predict_observe'+'_'+str(random_seed)+'.xlsx')

    list_str = '\n'.join(str(item) for item in best_parameters)
    # Write string to file
    with open('./Model_LightGBM/GridSearch_parameters_{}.txt'.format(random_seed), 'w') as file:
        file.write(list_str)

df_result.describe()



# %%
