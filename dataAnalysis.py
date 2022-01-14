import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from utils import *

df = pd.read_csv('TelcoCustomerChurn.csv')
print(df.head(6))
print(df.columns)

#20 Features
print(len(df.columns) - 1)

#Check for nulls and duplicates
print('Nulls',df.isna().sum().sum())
print('Duplicates: ',df.duplicated().sum())
print('Duplicated CustomerID: ', df.customerID.duplicated().sum())
print('Number of Customers: {}'.format(len(df)))
#78.33% of the total customers get InternetService
print('InternetService:', df.InternetService.unique())
#PhoneService is categorical (Yes/No).
#90.31% get PhoneService
print('PhoneService:', df.PhoneService.unique())
print('Customers who have subscribed to internet service: {}%', 100*len(df[df['InternetService'] != 'No'])/len(df))
print('Customers who have phone service: {}%', 100*len(df[df['PhoneService'] != 'No'])/len(df))

#Classification Problem
print(df.Churn.unique())

# sn.countplot(x=df.Churn);
# plt.show();
counts = df.Churn.value_counts()
print(counts)
#Yes only counts for 26.53% of the total samples ==> Data Imbalance
print("Customer Churn rate:", 100*counts['Yes']/len(df))

#TODO
#Models With & Without data balancing(over-sampling, under-sampling & SMOTE)

#Checking for data types: 
# TotalCharges should be changed to float
#All categorical features with Yes/No values should be changed to 0 and 1
print(df.info())

#ValueError: could not convert string to float: '' 
# ==> Must replace ' ' with a numeric value 
#df['TotalCharges'] = df['TotalCharges'].astype(float, errors = 'raise')

df['TotalCharges'] = df['TotalCharges'].apply(lambda x: 0 if x==' ' else x)
df['TotalCharges'] = df['TotalCharges'].astype(float, errors = 'raise')
#Stats
print(df.describe())
#==> It seems like 'SeniorCitizen' is a categorical value (0, 1)
#Tenure (Number of months the customer has stayed with the company).

# change to 1, 0 values instead Of yes/no, then convert types
for col in ['Partner', 'Dependents', 'PaperlessBilling', 'PhoneService', 'OnlineSecurity','OnlineBackup', \
            'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies']:
    df[col]= df[col].apply(lambda x: 1 if x=='Yes' else 0).astype('int32')
    
print(df.info())
print(df.MultipleLines.unique())
#Comparing Customers who left and those who didn't in regards to (charges, tenure, internet service, phone service, contract, multiple lines)
#MonthlyCharges distribution isn't normal
# df['MonthlyCharges'].hist()
# plt.title('MonthlyCharges')
# plt.show()

print(df.pivot_table(index='Churn', values='MonthlyCharges', aggfunc='median'))

#High prices might be the reason why some customers are leaving
# sn.histplot(df, x="MonthlyCharges", hue="Churn", element="step")
# plt.show()
# sn.boxplot(x='Churn', y='MonthlyCharges', data=df)
# plt.show()

# sn.boxplot(x='Churn', y='TotalCharges', data=df)
# plt.show()

#Those who churned have got higher total charges
# sn.lineplot(x='tenure', y='TotalCharges',hue ='Churn',  data=df)
# plt.show()

#Those who churned have mostly subscribed to fiber, or DSL
sn.histplot(x=df.InternetService, hue=df.Churn)
# plt.show()

churned = df[df['Churn'] == 'Yes']
not_churned = df[df['Churn'] == 'No']
plotted = pd.Series(churned['InternetService'].value_counts())[:]
not_churned_plotted = pd.Series(not_churned['InternetService'].value_counts())[:]

#customers who left:
# 69% used fiber optic
#25% of them had DSL
#Only 6% of them did not have internet services.
# plt.pie(labels = plotted[:]/plotted.sum(), x=plotted[:])
# plt.show()

#Those who stayed:
#35% had fiber optic, and 38% had DSL
plt.pie(labels = (not_churned_plotted[:]/not_churned_plotted.sum()).round(2), x=plotted)
# plt.show()

#Seeing as the monthly charges of those who churned was higher than those who stayed,
# most churners having a fiber service, maybe the cost of fiber was cause for churning

# sn.histplot(x=df.MultipleLines, hue=df.Churn)
# plt.show()

# print('churned.MultipleLines: \n', churned.MultipleLines.value_counts()*100/len(churned))
# print('not_churned.MultipleLines: \n',not_churned.MultipleLines.value_counts()*100/len(not_churned))
#Month-to-Month subscriptions have the highest rate of churn  88.55%
sn.histplot(x=df.Contract, hue=df.Churn)
plt.show()
print((churned.Contract.value_counts()*100/len(churned)).round(2))

print((not_churned.Contract.value_counts()*100/len(not_churned)).round(2))

#Those with less tenure are more likely to churn
sn.boxplot(x='Churn', y='tenure', data=df)
# plt.show()

#Preprocessing
#dropping ID column
df.drop(['customerID'], axis=1, inplace=True)
#Replace values
df['Churn'] = df['Churn'].apply(lambda x: 0 if x=='No' else 1)
target = df['Churn']
print((target.value_counts(normalize=True)*100).round(3))

#One Hot Encoding 
# df.hist()
# plt.show()
# OHE on the categorical columns
ohe_cols = ['InternetService','gender','PaymentMethod', 'MultipleLines', 'Contract']
def OHE_encoding():
    global ohe_cols
    global df
    
    # OHE encoding
    data_ohe = pd.get_dummies(df[ohe_cols], drop_first=True)
    
    # join the columns and drop the relevant columns
    new_df = df.join(data_ohe).drop(ohe_cols, axis=1)
    
    return new_df

new_df = OHE_encoding()
print(new_df.head())

ax = plt.axes()
plt.gcf().set_size_inches(3,10)
sn.heatmap(new_df.corr()[['Churn']].sort_values('Churn', ascending=False), cmap='Blues', annot=True)
ax.set_title('Features Correlations')
# plt.show();
# ==>The features that have some correlation to the target are fiber optic(expensive product).
# And payment method of electronic check(complicated paying method?).
#tenure(neg, the less, the more likely to churn), two-year-contract (if yes, then the less likely to churn) .

#Splitting the data
features = new_df.drop(['Churn'], axis=1)
X_train, X_test, y_train, y_test = \
train_test_split(features, target,stratify=target, test_size=0.15, random_state=12)

# Balancing with Downsampling or Undersampling or SMOTE
downsampled_features, downsampled_target = downsample(X_train, y_train)
upsampled_features, upsampled_target = upsample(X_train, y_train)
SMOTE_features, SMOTE_target = smote(X_train, y_train)

#Save the pickled files
# downsampled_features.to_pickle('downsampled_features.pkl')
# upsampled_features.to_pickle('upsampled_features.pkl')
# SMOTE_features.to_pickle('SMOTE_features.pkl')

# downsampled_target.to_pickle()
# upsampled_target.to_pickle()
# SMOTE_target.to_pickle()
map_dic = {'unbalanced_features_train':X_train, 'features_test':X_test, 'unbalanced_target':y_train, 'target_test':y_test,'downsampled_target':downsampled_target,'downsampled_features':downsampled_features,'upsampled_features':upsampled_features, 'upsampled_target':upsampled_target,'SMOTE_features':SMOTE_features, 'SMOTE_target':SMOTE_target}
for name,value in map_dic.items():
    value.to_pickle(name + '.pkl')
