# %%
import pandas as pd
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
X_cv = X_cv.loc[:,['Concentration (mg/L)','BET surface area (m2/g)','Relative weight']]
X_test = X_test.loc[:,['Concentration (mg/L)','BET surface area (m2/g)','Relative weight']]
X_cv

# %%
feature_names = X_cv.columns
feature_names



# %%
test_ratio = 0.25
df_result = pd.DataFrame(columns=('AUC_CV','F1_CV','Accuracy_CV','Train AUC','Train F1','Train Accuracy','Test AUC','Test F1','Test Accuracy'))
for random_seed in range(1,11,1):

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed) 

    for train,test in sss.split(X, y):
        X_cv = X.iloc[train]
        y_cv = y.iloc[train]
        X_test = X.iloc[test]
        y_test = y.iloc[test]

    X_cv = X_cv.loc[:,['Concentration (mg/L)','BET surface area (m2/g)','Relative weight']]
    X_test = X_test.loc[:,['Concentration (mg/L)','BET surface area (m2/g)','Relative weight']]

    parameter = {
        "n_estimators":[100,200,400,800],
        "tree_generator":[None,RandomForestRegressor(),GradientBoostingRegressor(),GradientBoostingClassifier()],
        "tree_size":[3,4,5,6,7],
        "max_rules":[10,15,20,25,30],
    }

    model_gs = RuleFitClassifier(random_state=42,include_linear=False)

    grid_search = GridSearchCV(model_gs, param_grid = parameter, scoring='accuracy', cv=5, n_jobs=-1)
    grid_search.fit(X_cv, y_cv)

    print("Best parameters for "+str(random_seed)+": ",grid_search.best_params_)

    RuleFit_GS_best = grid_search.best_estimator_
    
    y_pred_cv = RuleFit_GS_best.predict(X_cv)
    y_proba_cv = RuleFit_GS_best.predict_proba(X_cv)[:, 1]

    y_pred = RuleFit_GS_best.predict(X_test)
    y_proba = RuleFit_GS_best.predict_proba(X_test)[:, 1]
    
    df_result = df_result.append(pd.Series({'AUC_CV':sum(cross_val_score(RuleFit_GS_best, X_cv, y_cv, cv=5, scoring='roc_auc'))/5,
                                                'F1_CV':sum(cross_val_score(RuleFit_GS_best, X_cv, y_cv, cv=5, scoring='f1_weighted'))/5,
                                                'Accuracy_CV':sum(cross_val_score(RuleFit_GS_best, X_cv, y_cv, cv=5, scoring='accuracy'))/5,
                                                'Train AUC':metrics.roc_auc_score(y_cv,y_proba_cv),
                                                'Train F1':metrics.f1_score(y_cv,y_pred_cv,average='weighted'),
                                                'Train Accuracy':metrics.accuracy_score(y_cv,y_pred_cv),
                                                'Test AUC':metrics.roc_auc_score(y_test,y_proba),
                                                'Test F1':metrics.f1_score(y_test,y_pred,average='weighted'),
                                                'Test Accuracy':metrics.accuracy_score(y_test,y_pred)}),ignore_index=True)
    df_result.to_excel('./Model_RuleFit/GridSearch'+'_performance_'+str(random_seed)+'.xlsx')
    X_test['Observed RMC'] = y_test
    X_test['Predicted RMC'] = y_pred
    X_test.to_excel('./Model_RuleFit/Predict_observe'+'_'+str(random_seed)+'.xlsx')

    #model = RuleFitClassifier(random_state=42,tree_size=grid_search.best_params_['tree_size'])
    #model.fit(X_cv, y_cv)
    rules = RuleFit_GS_best.get_rules()
    rules = rules[rules.coef != 0].sort_values("importance", ascending=False)
    rules.to_excel("./Model_RuleFit/rules_"+str(random_seed)+".xlsx")
df_result.describe()
