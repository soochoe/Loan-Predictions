## 1. Recap ##

import pandas as pd
loans = pd.read_csv("cleaned_loans_2007.csv")
print(loans.info())

## 3. Picking an error metric ##

import pandas as pd
# true negative 
tn = (predictions == 0) & (loans["loan_status"]==0)
# true positive
tp = (predictions == 1) & (loans["loan_status"]==1)
# false negative
fn = (predictions == 0) & (loans["loan_status"]==1)
# false positive
fp = (predictions == 1) & (loans["loan_status"]==0)



## 5. Class imbalance ##

import pandas as pd
import numpy

# Predict that all loans will be paid off on time.
predictions = pd.Series(numpy.ones(loans.shape[0]))
# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])
print(predictions[fp_filter])
# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)

## 6. Logistic Regression ##

# use Scikit-learn library 
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
# A good first algorithm to apply to binary classification problems is logistic regression, for the following reasons:
#it's quick to train and we can iterate more quickly,
#it's less prone to overfitting than more complex models like decision trees,
# it's easy to interpret.

# create a new dataframe with all the feature columns (w/o  loan_status)
features = loans
features = features.drop(columns=["loan_status"])

# create a targe column (loan_status)
target = loans["loan_status"]

# Create an instance of Logistic Regression Classifier and fit the data.
lr.fit(features, target)

predictions = lr.predict(features)



## 7. Cross Validation ##

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
# Instantiate model object
lr = LogisticRegression()

# Make predictions using 3-fold cross-validation
predictions = cross_val_predict(lr, features, target, cv=3)
predictions = pd.Series(predictions)

fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)


## 9. Penalizing the classifier ##

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

# Instantiate model object
lr = LogisticRegression(class_weight = "balanced")

# Make predictions using 3-fold cross-validation
predictions = cross_val_predict(lr, features, target, cv=3)
# Converting to Series objects letse us take advantage of boolean filtering and arithmetic operations from pandas.
predictions = pd.Series(predictions)


fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)



## 10. Manual penalties ##

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

penalty = {
    0: 10,
    1: 1
}

lr = LogisticRegression(class_weight = penalty)

# Make predictions using 3-fold cross-validation
predictions = cross_val_predict(lr, features, target, cv=3)
# Converting to Series objects letse us take advantage of boolean filtering and arithmetic operations from pandas.
predictions = pd.Series(predictions)


fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)


## 11. Random forests ##

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_predict


lr = RandomForestClassifier(random_state = 1, class_weight = "balanced")

# Make predictions using 3-fold cross-validation
predictions = cross_val_predict(lr, features, target, cv=3)
# Converting to Series objects letse us take advantage of boolean filtering and arithmetic operations from pandas.
predictions = pd.Series(predictions)


fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])

# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])

# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])

# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])

# Rates
tpr = tp / (tp + fn)
fpr = fp / (fp + tn)

print(tpr)
print(fpr)

