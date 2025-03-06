import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

df = pd.read_csv("../data/train.csv")

features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target = 'label'
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

knn = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

grid_search = GridSearchCV(knn, param_grid, scoring='f1_macro', cv=5, n_jobs=-1)
grid_search.fit(X_train_std, y_train)

print("Лучший кросс-валидационный F1-score: {:.5f}".format(grid_search.best_score_))

best_knn = grid_search.best_estimator_
y_test_pred = best_knn.predict(X_test_std)
f1_test = f1_score(y_test, y_test_pred, average='macro')
print("F1-score на валидационной выборке: {:.5f}".format(f1_test))

test_df = pd.read_csv("../data/test.csv")
X_test = test_df[features]
X_test_std = scaler.transform(X_test)

test_pred = best_knn.predict(X_test_std)

submission = pd.read_csv("../data/sample_submit.csv")

submission['label'] = test_pred

submission.to_csv("output/submission.csv", index=False)
print("submission.csv сохранен")
