import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

financial_data = pd.read_csv('./dataset/raw.csv')

## EDA: Exploratory Data Analysis
# Information about the dataset
# print(financial_data.info())  # -- uncomment to explore

# Checking the number of missing values in each column
# print(financial_data.isnull().sum()) # -- uncomment to explore

# Since there's only one row with null values, we drop the row
financial_data_cleaned = financial_data.dropna()
# print(financial_data_cleaned.isnull().sum()) # -- uncomment to explore

# Distribution of normal tranasactions and fraudulent transactions
financial_data_cleaned['isFraud'].value_counts()
# print(financial_data_cleaned['isFraud'].value_counts()) # --uncomment to explore

"""
This dataset is highly skewed. Here:
0 -> Normal Transaction 
1 -> Fraudulent Transaction
"""

# Separating the data for analysis
legit = financial_data_cleaned[financial_data_cleaned['isFraud'] == 0]
fraud = financial_data_cleaned[financial_data_cleaned['isFraud'] == 1]

# print(legit.shape) # --uncomment to explore
# print(fraud.shape) # --uncomment to explore

# Statistical measures for each dataset
# print(legit.amount.describe()) # --uncomment to explore
# print(fraud.amount.describe()) # --uncomment to explore

"""
The mean amount that has been transacted for left transactions amount to 1,61,946.7 and that 
of fradulent transactions amount to 12,44,297.
=> The average of the fradulent transactions are significantly higher than the average of 
non-fraudulent transactions
"""

# print(financial_data_cleaned.groupby('isFraud').mean()) # --uncomment to explore
"""
The mean of difference between oldbalanceOrg and newbalanceOrig both cases are noticable: 
In case of fradulent transactions, large sum of money is transacted which lead top lower 
remaining balance in the account
"""

"""
UNDERSAMPLING

Build a sample dataset containing similar distribution of non-fraudulent transactions and 
fraudulent transactions

number of fraudulent transactions = 8213
"""

# Uniform distribution
legit_sample = legit.sample(n=8213)
# print(legit_sample.shape) # --uncomment to explore

# Concatenating both dataframes
uniform_df = pd.concat([legit_sample,fraud],axis=0)
# print(uniform_df) # --uncomment to explore

# print(uniform_df['isFraud'].value_counts()) # --uncomment to explore
# print(uniform_df.groupby('isFraud').mean()) # --uncomment to explore

"""
Splitting the dataset
X: Features
Y: Targets
"""
X = uniform_df.drop(columns=['isFraud','type','nameOrig','nameDest','isFlaggedFraud'],axis=1)
y = uniform_df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)

""" MODEL TRAINING: Logistic Regression """
model = LogisticRegression()
model.fit(X_train,y_train)

""" MODEL EVALUATION: Accuracy score """
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,y_train)
print(training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,y_test)
print(test_data_accuracy)