import pandas as pd
#read data
data = pd.read_csv('TelcoCustomerChurn.csv')
#load features and targets
downsampled_features = pd.read_pickle('downsampled_features.pkl') 
upsampled_features = pd.read_pickle('upsampled_features.pkl')
SMOTE_features = pd.read_pickle('SMOTE_features.pkl')

downsampled_target = pd.read_pickle('downsampled_target.pkl')
upsampled_target = pd.read_pickle('upsampled_target.pkl')
SMOTE_target = pd.read_pickle('SMOTE_target.pkl')

imbalanced_features_train = pd.read_pickle('unbalanced_features_train.pkl')
features_test = pd.read_pickle('features_test.pkl')
imbalanced_target = pd.read_pickle('unbalanced_target.pkl')
y_test = pd.read_pickle('target_test.pkl')
#local path where experiments are saved
LOCAL_PATH = "file:///Users/33645/Desktop/projects/kaggle/CustomerChurn/mlruns"