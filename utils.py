import pandas as pd
# Importing required packages
from sklearn.utils import resample, shuffle
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, plot_confusion_matrix, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve, classification_report


def upsample(features_train, target):
    
    # Balancing with upsampling for churn
    no_churn = features_train[target==0]
    yes_churn = features_train[target ==1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]
    # resample yes_churn
    upsampled_yes_churn = resample(yes_churn, replace=True, n_samples=len(no_churn), random_state=12)

    # Combining no_churn with upsampled yes_churn
    upsampled_features = pd.concat([no_churn,upsampled_yes_churn], axis=0)
    upsampled_target = pd.concat([target_zeros] + [pd.Series([1]*len(upsampled_yes_churn))], axis=0)
    #shuffle
    # upsampled_features, upsampled_target = shuffle(upsampled_features, upsampled_target,random_state=12)
    print('upsampled_features:',upsampled_features.shape)
    print('upsampled_target:',upsampled_target.shape)
    print()

    return upsampled_features, upsampled_target

def downsample(features_train, target):
    
    # Balancing with upsampling for churn
    no_churn = features_train[target==0]
    yes_churn = features_train[target ==1]

    # resample yes_churn
    downsampled_no_churn = resample(no_churn, replace=False, n_samples=len(yes_churn), random_state=12)

    # Combining no_churn with upsampled yes_churn
    downsampled_features = pd.concat([downsampled_no_churn, yes_churn], axis=0)
    downsampled_target = pd.concat([pd.Series([0]*len(downsampled_no_churn))] + [pd.Series([1]*len(yes_churn))], axis=0 )
    #shuffle
    # downsampled_features, downsampled_target = shuffle(downsampled_features, downsampled_target,random_state=12)
    print('downsampled_features:',downsampled_features.shape)
    print('downsampled_target:',downsampled_target.shape)
    print()

    return downsampled_features, downsampled_target

def smote(features_train, target):
    # Creating SMOTE model
    sm = SMOTE(random_state=12)
    X_sm_train, y_sm_train = sm.fit_resample(features_train, target)

    print('smote_sampled_features:',X_sm_train.shape)
    print('smote_sampled_target:',y_sm_train.shape)
    return X_sm_train, y_sm_train

def scale(features_train, features_test):
    scaler = StandardScaler()
    features_train_scaled_train = scaler.fit_transform(features_train)
    features_test_scaled_test = scaler.transform(features_test)
    return features_train_scaled_train, features_test_scaled_test
# Creating model results evaluation function
def model_results(model, X_train,y_train,X_test, y_test, title=""):
    
    # fitting the model
    model.fit(X_train,y_train)
    # predicting
    preds = model.predict(X_test)
    # results evaluation
    print(classification_report(y_test,preds))
    plot_confusion_matrix(model,X_test,y_test)
    # ConfusionMatrixDisplay.from_predictions(y_test,preds)
    plt.title(title)
    plt.show()
def feature_importances(X, model):
    
    '''
    Takes:
    - features_train
    - a specific model that was run through GridSearchCV
    Tries:
    - to display the 3 most important features of the model
    - If it's unable to do so, it prints a message telling that the model does not have the method of features importance
    '''
    
    try:
        feature_importance = pd.DataFrame({'features':X.columns,'importance':model.best_estimator_._final_estimator.feature_importances_})
        print(feature_importance.sort_values('importance',ascending=False).head(3))
        print()
        
    except:
        try:
            feature_importance=pd.DataFrame({'feature':list(X.columns),'importance':[abs(i) for i in model.best_estimator_._final_estimator.coef_[0]]})
            print(feature_importance.sort_values('importance',ascending=False).head(3))
            print()

        except:
            print('This model does not have a feature importance attribute')


def model_scores(model, X_test, models, Y_test, names, model_name):
    
    '''
    Takes:
    - a specific model
    - features_test
    - a set of models
    - target_test
    - a set of names of models
    - the name of specific model
    Creates:
    - predictions usig predict_proba, for the roc auc score
    - predicition using predict method for the other scores
    Appends:
    - the set of models with three scores for the model
    - the set of models names with the current model's name
    Sends:
    - parameters to the function that displays the confusion matrix of the model and it's roc curve (cm_roc_auc_curve)
    - if the model does not have a cunfusion matrix of a roc curve, the function skips this action    
    '''
    
    # for the roc auc score we need to use predict proba
    predictions = model.predict_proba(X_test)[:,1]
    # for accuracy and f1 we will use predict, as usual
    preds = model.predict(X_test)
    
    # collects scores to the dataset
    models['roc auc'].append(roc_auc_score(Y_test, predictions))
    models['accuracy'].append(accuracy_score(Y_test, preds))
    models['f1'].append(f1_score(Y_test, preds))
    
    # add the model's name to the above table
    names.append(model_name)
    
    # send to the function that displays confusion matrix and the roc curve
    try:
        cm_roc_auc_curve(model, X_test, Y_test, preds, models, names)
    except:
        print()

# print confusion matix + roc auc curve
def cm_roc_auc_curve(model, X_test, y_test, predictions, models, names):
    
    '''
    Takes:
    - the specific model run by the GrdSearchCV
    - features_test
    - target_test
    - the roc auc predictions
    - the set of models, updated
    - the set of models' names, updated
    Adds to the models set:
    - the values of the confusion matrix (TP, TN, FP, FN)
    Displays:
    - the models set in its current stage (including all the model so far) with their scores and confusion matrix values
    - a heatmap of the confusion matrix
    - the roc curve of the current model
    
    '''
    probabilities_one_valid = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probabilities_one_valid)
    cm = confusion_matrix(y_test, predictions, labels=[1,0])
    
    # add information about tp,tn,fp,fn
    models['TP'].append(cm[0,0])
    models['TN'].append(cm[1,1])
    models['FP'].append(cm[0,1])
    models['FN'].append(cm[1,0])
    
    # sort the table
    models = pd.DataFrame(models, index=names).sort_values('roc auc', ascending=False)
    
    # display the scores of all the models we have so far
    # display(models)
    
    # plot confusion matrix
    plot_confusion_matrix(model,X_test,y_test)
    
    # roc auc curve
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for the model')
    plt.show()

# a function that gets parameters and train the model via gridsearch to find the best parameters and best score
def train_model(model, params, X, Y, X_test, Y_test, models, names, model_name):
    
    '''
    Takes: 
    - a specific model
    - a specific dictionary of hyper-parameters for this model
    - features_train
    - target_train
    - features_test
    - target test
    - a set of models
    - a set of names
    - the specific model's name
    Creates: 
    - a pipeline to run the model through
    - a cross-validation model that runs the pipeline
    Sends parameters to the following functions:
    - a function that saves the desired scores for the model (model_scores)
    - a function that displays the most important features for this model (feature_importance)
    Prints:
    - best parameters for the models
    - best roc auc score for the train set
   '''
    
    # defining the pipeline
    pipeline = Pipeline([('scale', StandardScaler()), ('model', model) ])
    
    # use gridsearchCV for cross validation
    grid = GridSearchCV(pipeline, param_grid=params, cv=5, scoring=['roc_auc','accuracy', 'f1'], refit='roc_auc')
    grid.fit(X,Y)
    
    # display best score on train
    print('Best roc auc score (on train set):',grid.best_score_)
    
    # display scores
    model_scores(grid, X_test,  models, Y_test, names, model_name)
    
    # display features importances
    feature_importances(X, grid)
       
    # display best parameters
    print('Best parameters:\n',grid.best_params_)
    print()