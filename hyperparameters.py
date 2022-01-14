#svc
C_svc =[0.001,0.01,0.1, 0.5, 1.0]
gamma_svc=['scale','auto']
kernel_svc = ['linear', 'rbf']
param_names_svc = ['C','gamma','kernel']
#rf:
bootstrap_rf = [True, False]
max_depth_rf = [3, 4, 5, 10,15,20, None]
max_features_rf = ['auto', 'sqrt']
min_samples_leaf_rf =  [1, 2, 4]
min_samples_split_rf = [2, 5, 10]
n_estimators_rf = [50, 100,200, 400]
criterion_rf = ['gini', 'entropy']
param_names_rf = ['bootstrap', 'max_depth', 'max_features'\
     ,'min_samples_leaf', 'min_samples_split','n_estimators']
#logreg
penalties = ['none','l2']
C_logreg = [0.01,0.1,1.0,100]
param_names_logreg = ['penalty', 'C']
#ada
n_estimators_ada = [10,50,100,500]
learning_rate_ada = [0.0001,0.001,0.01,0.1,0.2,1.0]
param_names_ada = ['n_estimators', 'learning_rate']
#gradboost
n_estimators_gradboost = [50,100,200,500]
learning_rate_gradboost = [0.0001,0.001,0.01,0.05,0.1,1.0]
max_depth_gradboost = [3,4,5]
param_names_ada = ['n_estimators', 'learning_rate','max_depth']
