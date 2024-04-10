import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind

# reading data
raw = pd.read_csv("data/compas-scores-two-years.csv")
# selecting potential feature of interest
df = raw[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count',
               'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]

# filtering data to the same standard as the ones shown in the notebook
df = df[(df['days_b_screening_arrest'] <= 30) &
        (df['days_b_screening_arrest'] >= -30) &
        (df['is_recid'] != -1) &
        (df['c_charge_degree'] != 'O') &
        (df['race'].isin(['African-American', 'Caucasian'])) &
        (df['score_text'] != 'N/A')]
# creating the feature for time in jail
df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
df['jail_time'] = df['c_jail_out'] - df['c_jail_in']
df['jail_time_days'] = df['jail_time'].dt.days
df.drop(columns=['jail_time'], inplace=True)
df = pd.get_dummies(df)


# removing unecessary dummy variables and setting target variable
X = df.drop(columns=['two_year_recid','race_African-American', 'c_charge_degree_M',
                     'age_cat_Less than 25', 'sex_Female','score_text_Low','c_jail_in', 'c_jail_out'])
y = df['two_year_recid']
# split train & test. Cross validation is skipped due to data size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# dropping sensitive feature: race
X_train_trunc = X_train.drop(columns = ['race_Caucasian'])
X_test_trunc = X_test.drop(columns = ['race_Caucasian'])
# fitting the random forest model & get predictions
rfc = RandomForestClassifier()
rfc.fit(X_train_trunc, y_train)
y_pred = rfc.predict(X_test_trunc)

# calculating overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Total Validation Accuracy:", accuracy)
print("Total 2 year recid",sum(y_test), "out of", len(y_test)
      , "for a rate of", sum(y_test)/len(y_test))
print("Predicted 2 year recid", sum(y_pred), "out of", len(y_pred)
      , "for a rate of", sum(y_pred)/len(y_pred))

# backfitting race to check result by race
race_train = X_train['race_Caucasian']
race_test = X_test['race_Caucasian']

# calculating metrics for white convicts
accuracy_white = accuracy_score(y_test[race_test == True], y_pred[race_test == True])
print("White Validation Accuracy", accuracy_white, "actual is", sum(y_test[race_test == True]),
      "out of", len(y_test[race_test == True])
      , "for a rate of", sum(y_test[race_test == True])/len(y_test[race_test == True]))
print("Predicted", sum(y_pred[race_test == True]), "out of", len(y_pred[race_test == True])
      , "for a rate of", sum(y_pred[race_test == True]) / len(y_pred[race_test == True]))

# calculating metrics for black convicts
accuracy_black = accuracy_score(y_test[race_test == False], y_pred[race_test == False])
print("Black Validation Accuracy", accuracy_black, "actual is", sum(y_test[race_test == False]),
      "out of", len(y_test[race_test == False]), "for a rate of"
      , sum(y_test[race_test == False]) / len(y_test[race_test == False]))
print("Predicted", sum(y_pred[race_test == False]), "out of", len(y_pred[race_test == False])
      , "for a rate of", sum(y_pred[race_test == False]) / len(y_pred[race_test == False]))

# running t test to see if the treatment of the two groups are different, both actual and predicted
# it would seem that we do treat black and white convicts differently in two years, both in
# actual terms, also predicted terms when race is taken out
t_statistic, p_value = ttest_ind(y_pred[race_test == False], y_pred[race_test == True])
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There is a significant difference between"
          " the means for predicted conviction. T statistic is", t_statistic)
else:
    print("Fail to reject the null hypothesis. There is no significant difference between"
          " the means for predicted conviction. T statistic is", t_statistic)
t_statistic2, p_value2 = ttest_ind(y_test[race_test == False], y_test[race_test == True])
if p_value2 < alpha:
    print("Reject the null hypothesis. There is a significant difference between "
          "the means for actual conviction. T statstic is", t_statistic2)
else:
    print("Fail to reject the null hypothesis. There is no significant difference between"
          " the means for actual conviction. T statsitic is", t_statistic2)

# checking the importance of each features
# it seems that the most important features by far is is_recid, with all the other features
# being significantly less important. What we fail to capture here is the potential issues within
# the justice system itself. Since our data is based on court decisions, we are not able to control
# for the issues within the legal system itself. As a result, we have similar accuracy when
# predicting, but we can observe an issue within the treatment of the two groups of people
# even if we are not predicting more repeats
importances = rfc.feature_importances_
feature_names = X_train_trunc.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)