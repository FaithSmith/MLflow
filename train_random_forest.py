from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,\
    f1_score, roc_auc_score,recall_score, roc_curve, classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from utils import *
from load_data import *
from hyperparameters import *
#mlflow imports for tracking experiments
from mlflow import  set_tracking_uri ,log_metrics, log_params, log_artifacts
import mlflow.sklearn
##choose local path where experiments are saved
set_tracking_uri(LOCAL_PATH)
#create experiment
experiment_id = mlflow.create_experiment("training random_forest")

# Scaling data for the Upsampled
upsampled_features_scaled_train , features_up_scaled_test = scale(upsampled_features, features_test)
# Scaling data for the downsampled
downsampled_features_scaled_train , features_ds_scaled_test = scale(downsampled_features, features_test)
SMOTE_features_scaled_train , features_sm_scaled_test = scale(SMOTE_features, features_test)
# Scaling data for the imbalanced data
unbalanced_features_scaled_train , features_imb_scaled_test = scale(imbalanced_features_train, features_test)

#data
X_trains = [upsampled_features_scaled_train, downsampled_features_scaled_train,\
    SMOTE_features_scaled_train,imbalanced_features_train]
y_trains = [upsampled_target, downsampled_target,\
    SMOTE_target,imbalanced_target]
X_tests = [features_up_scaled_test, features_ds_scaled_test,\
    features_sm_scaled_test,features_imb_scaled_test]
y_tests = [y_test]
types_data = ['upsampled_data', 'downsampled_data', 'smote_data', 'imbalanced_data']

for type_data, X_train,y_train,X_test,y_test in zip(types_data, X_trains,y_trains,X_tests,y_tests):
    for bootstrap in bootstrap_rf:
        for max_depth in max_depth_rf:
            for max_features in max_features_rf: 
                for min_samples_leaf in min_samples_leaf_rf:
                    for min_samples_split in  min_samples_split_rf:
                        for n_estimators in n_estimators_rf:
                            for criterion in criterion_rf:
    
                
                                with mlflow.start_run(experiment_id=experiment_id):
                                    model = RandomForestClassifier(bootstrap=bootstrap \
                                        , max_depth=max_depth,max_features=max_features \
                                            ,min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split \
                                                ,n_estimators=n_estimators, criterion=criterion)
                                    model.fit(X_train,y_train)
                                    y_pred = model.predict(X_test)
                                    #log scores
                                    log_metrics({"acc": accuracy_score(y_test, y_pred),
                                    "prec" : precision_score(y_test, y_pred),
                                    "auc" : roc_auc_score(y_test, y_pred),
                                    "recall" : recall_score(y_test, y_pred),
                                    "f1" : f1_score(y_test, y_pred)})
                                    #log hyperparams
                                    for param_name in param_names_rf:
                                        mlflow.log_param(param_name, eval(param_name))
                                        
                                
                                    # mlflow.log_param("max_features", f)
                                    #log model to serialize it
                                    mlflow.sklearn.log_model(model, "model_rf_" + type_data)
                                    
