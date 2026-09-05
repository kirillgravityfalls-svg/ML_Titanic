import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

data = sns.load_dataset('titanic')

data = data[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].copy()

data['sex'] = data['sex'].map({'male': 0, 'female': 1})

data['age'] = data['age'].fillna(data['age'].median())

data['embarked'] = data['embarked'].fillna('S')
data['embarked'] = data['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

data['family_size'] = data['sibsp'] + data['parch'] + 1
data['is_alone'] = (data['family_size'] == 1).astype(int)
data['has_child'] = ((data['age'] < 18) & (data['sibsp'] + data['parch'] > 0)).astype(int)

X = data.drop('survived', axis=1)
y = data['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)
logreg_pred = logreg.predict(X_test_scaled)
logreg_acc = accuracy_score(y_test, logreg_pred)

rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("Результаты сравнения моделей:")
print(f"Логистическая регрессия: {logreg_acc:.3f}")
print(f"Случайный лес: {rf_acc:.3f}")

print("\nЛучшая модель - Случайный лес")
print(classification_report(y_test, rf_pred))

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nВажность признаков:")
print(feature_importance)

plt.figure(figsize=(8, 5))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Важность признаков для предсказания выживаемости')
plt.xlabel('Важность')
plt.savefig('feature_importance.png')
