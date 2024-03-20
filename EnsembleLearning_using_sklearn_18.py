import pandas as pd

# loading the data:
df = pd.read_csv('diabetes.csv')
print(df.head())

# Checking the null values:
print(df.isnull())

# describe all the values:
print(df.describe())

# outcomes values:
print(df.Outcome.value_counts())

# Dependent variables:
X = df.drop('Outcome', axis='columns')

# Independent variables:
y = df.Outcome

# Standard Scalar:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:3])

# Training and testing the data:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, random_state=10)

# Shape of train and test data:
print(X_train.shape)
print(X_test.shape)

# DecisionTreeClassifier:
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=5)
print(scores)
print(scores.mean())

# BaggingClassifier:
from sklearn.ensemble import BaggingClassifier
bag_model = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0
)
bag_model.fit(X_train, y_train)
print(bag_model.oob_score)
# Printing the accuracy of bagged model:
print(bag_model.score(X_test, y_test))

# Appling the cross_val_score on the bagged model:
scores_1 = cross_val_score(bag_model, X, y, cv=5)
print(scores_1.mean())

# RamdomForestClassifier:
from sklearn.ensemble import RandomForestClassifier
scores_2 = cross_val_score(RandomForestClassifier(), X, y, cv=5)
print(scores_2.mean())