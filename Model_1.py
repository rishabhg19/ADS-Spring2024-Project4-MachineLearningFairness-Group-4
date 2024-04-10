import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


raw = pd.read_csv("data/compas-scores-two-years.csv")

df = raw[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count',
               'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid', 'c_jail_in', 'c_jail_out']]


df = df[(df['days_b_screening_arrest'] <= 30) &
        (df['days_b_screening_arrest'] >= -30) &
        (df['is_recid'] != -1) &
        (df['c_charge_degree'] != 'O') &
        (df['race'].isin(['African-American', 'Caucasian'])) &
        (df['score_text'] != 'N/A')]

df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
df['jail_time'] = df['c_jail_out'] - df['c_jail_in']
df['jail_time_days'] = df['jail_time'].dt.days
df.drop(columns=['jail_time'], inplace=True)
df = pd.get_dummies(df)

#pd.set_option('display.max_columns', None)
#print(df.head())


X = df.drop(columns=['two_year_recid', 'race_Caucasian','race_African-American', 'c_charge_degree_M','age_cat_Less than 25', 'sex_Female','score_text_Low','c_jail_in', 'c_jail_out'])
y = df['two_year_recid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print("Total Validation Accuracy:", accuracy)
print("Total 2 year recid",sum(y_test), "out of", len(y_test))
print("Predicted 2 year recid", sum(y_pred), "out of", len(y_pred))

race_train = df[df.index.isin(X_train.index)]['race_Caucasian']
race_test = df[df.index.isin(X_test.index)]['race_Caucasian']

accuracy_white = accuracy_score(y_test[race_test == True], y_pred[race_test == True])
print("White Validation Accuracy", accuracy_white, "total is", len(y_test[race_test == True]),
      "out of", sum(y_test[race_test == True]))
print("Predicted", sum(y_pred[race_test == True]), "out of", len(y_pred[race_test == True]))
white_true = list(y_test[race_test == True])
white_pred = list(y_pred[race_test == True])
diff_white = 0
for i in range(len(white_true)):
    if white_pred[i] == white_true[i]:
        diff_white += 1
print("manually calculated white accuracy",diff_white / len(white_pred))

accuracy_black = accuracy_score(y_test[race_test == False], y_pred[race_test == False])
print("Black Validation Accuracy", accuracy_black, "total is", len(y_test[race_test == False]),
      "out of", sum(y_test[race_test == False]))
print("Predicted", sum(y_pred[race_test == False]), "out of", len(y_pred[race_test == False]))


importances = rfc.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)

