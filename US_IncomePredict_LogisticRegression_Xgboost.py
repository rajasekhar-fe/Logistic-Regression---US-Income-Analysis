# Project to predict the Household income of a person in US based on the US census data provided in the below website
# https://archive.ics.uci.edu/ml/datasets/census+income

# ====================== IMPORT DATA SETS AND LIBRARIES ===================================
import pandas as pd
import numpy as np
import seaborn as sea
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost


# Import the data from csv file into data frame

cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
           "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country","Income"]

inc_df =  pd.read_csv("D:\ML Projects\IncomeData.csv", names=cols)

#print(inc_df[["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation", "relationship",
#          "race", "sex"]].head())

# Convert the Income column to 0 and 1 ie. 0 if Income <=50K or 1 if Income > 50K

inc_df["Income_Upd"] = inc_df['Income'].apply(lambda a: 1 if a == " >50K" else 0)
#print("Number of samples in Income data : {}".format(len(inc_df)))

# ====================== EXPLORATORY DATA ANALYSIS =====================================

# Statistical summary of the data frame
print(inc_df.describe())
print(inc_df.info())

# Number of null values in the data frame
print(inc_df.isnull().sum(axis=0))

# Number of unique values for each column
print(inc_df['workclass'].value_counts())

# Replace ? with mode of work class
#print("Mode of workclass is : {}".format(inc_df['workclass'].mode()))
inc_df['workclass'].replace(" ?",inc_df['workclass'].mode()[0],inplace = True)
print(inc_df['workclass'].value_counts())

# Check for any ? through seaborn countplot method
#plt.figure(figsize=[15,8])
#sea.countplot(x=inc_df['workclass'])
#plt.show()

# Replace ? with mode of native country
inc_df['native-country'].replace(' ?',inc_df['native-country'].mode()[0],inplace=True)
print(inc_df['native-country'].value_counts())

#sea.countplot(x=inc_df['marital-status'])

#==================== PERFORM DATA VISUALIZATION =====================================

# Visualize Income data
#sea.countplot(x=inc_df['Income'])
#plt.show()

# Visualize Education data
#plt.figure(figsize=[25,15])
#sea.countplot(x=inc_df['education'])
#plt.show()

# Visualize age data
plt.figure(figsize=[25,15])
#sea.displot(x=inc_df['age'])
#plt.show()

# Visualize pair plot
#sea.pairplot(inc_df)
#plt.show()

# Correlation Matrix

corr = inc_df.corr()
#sea.heatmap(corr, annot=True)
#plt.show()


# ======================= PREPARE THE DATA TO TRAIN THE MODEL ====================================

#print(type(inc_df.columns))

# Drop the income columns from the data frame for the input features

X = inc_df.drop(columns=['Income','Income_Upd'])
y = inc_df['Income_Upd']

encode = LabelEncoder()

for i in X.columns:
    X[i] = encode.fit_transform(X[i])

features = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']

X = pd.get_dummies(X, columns = features)
#print(X[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']])

# Scale the data before training the model using StandardScaler and MinMaxScaler classes

scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)

# Prepare the train and test data using train test split class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, X_test.shape)

# ================================ Model building ============================================

# Train the logistic regression model with X_train using logistic regression model

lr = LogisticRegression()

lr.fit(X_train, y_train)

# Predict the trained model with test data and check its accuracy through classification report and confusion matrix

predicted_y = lr.predict(X_test)
print("Test Accurancy score: ",accuracy_score(y_test,predicted_y))
print(classification_report(y_test, predicted_y))

# Plot the confusion matrix using heatmap for clarity
plt.figure(figsize=(20,10))

cm = confusion_matrix(y_test, predicted_y)
print(cm)
sea.heatmap(cm, annot=True, fmt='0.5g')
plt.show()

# ============================== XG Boost classifier ======================================

xgb= xgboost.XGBClassifier()

xgb.fit(X_train,y_train)

y_xg_predict = xgb.predict(X_test)
print(y_xg_predict)

print("Test Accurancy score with XGBoost : ",accuracy_score(y_test,y_xg_predict))
print("Classification Report with XGBoost :", classification_report(y_test, y_xg_predict))

cmxg = confusion_matrix(y_test, y_xg_predict)
sea.heatmap(cmxg, fmt='0.5g', annot=True)
plt.show()





