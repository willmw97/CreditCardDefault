import pickle
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('clients6_data.csv')

# dataset['AgeCategory'] = 0
# dataset.loc[((dataset['AGE'] > 20) & (dataset['AGE'] < 30)), 'AgeCategory'] = 1
# dataset.loc[((dataset['AGE'] >= 30) & (dataset['AGE'] < 40)), 'AgeCategory'] = 2
# dataset.loc[((dataset['AGE'] >= 40) & (dataset['AGE'] < 50)), 'AgeCategory'] = 3
# dataset.loc[((dataset['AGE'] >= 50) & (dataset['AGE'] < 60)), 'AgeCategory'] = 4
# dataset.loc[((dataset['AGE'] >= 60) & (dataset['AGE'] < 70)), 'AgeCategory'] = 5
# dataset.loc[((dataset['AGE'] >= 70) & (dataset['AGE'] < 80)), 'AgeCategory'] = 6
#
# col_to_norm = ['LIMIT_BAL', 'BILL_AV_AMT', 'PAY_AMT_AV', 'AVAILABLE_CRED_PERCENT']
# dataset[col_to_norm] = dataset[col_to_norm].apply(lambda x: (x - np.mean(x)) / np.std(x))

    # Set the features we want to look at
features = [ 'LIMIT_BAL' , 'SEX' , 'EDUCATION' , 'MARRIAGE','AGE','PAY_MAX_SCORE','BILL_AV_AMT','PAY_AMT_AV','AVAILABLE_CRED_PERCENT']


    # Target assigned
y = dataset['default_payment_next_month'].copy()
    # Features assigned
X = dataset[features].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

X_train.shape
X_test.shape



filename = 'final_tree_model.sav'
imported_model = joblib.load(filename)

#Enter in tested values



# probability = .predict_proba([[creditA, new_sex, new_education, new_marriage,
#                                                new_age, new_credit_score, credit_bill, bill_payment, new_pay_to_bill,
#                                                new_credit_utilization]])

print(y_test)
result = imported_model.score(X_test, y_test)
print(result)

#enter what you expect

age = 25
gender = 1
education = 3
marriage = 2
credit_score = 10000
creditAmount = 10000
creditBalance = 7000
credit_bill = 10000
bill_payment = 4000

#calculates the expected credit
new_age = int(age)
new_sex = int(gender)
new_education = int(education)
new_marriage = int(marriage)
new_credit_score = int(credit_score)
new_pay_to_bill = int(bill_payment) / int(creditBalance)
new_credit_utilization = int(creditBalance) / int(creditAmount)


prediction = imported_model.predict([[creditAmount, new_sex, new_education, new_marriage, new_age,
                                     credit_bill, bill_payment, new_pay_to_bill,
                                    new_credit_utilization]])

probability = imported_model.predict_proba([[creditAmount, new_sex, new_education, new_marriage,
                                           new_age, credit_bill, bill_payment, new_pay_to_bill,
                                           new_credit_utilization]])

prediction = int(prediction[0])
probability = float(probability[0][1])

#features = [ 'LIMIT_BAL' , 'SEX' , 'EDUCATION' , 'MARRIAGE','AGE','PAY_MAX_SCORE','BILL_AV_AMT', 'PAY_AMT_AV', 'AVAILABLE_CRED_PERCENT']
#prints predictions
print(prediction)
print(probability)








