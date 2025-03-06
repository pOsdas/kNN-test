import pandas as pd
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import catboost
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import StackingClassifier

# Фиксируем случайность для воспроизводимости
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# Загружаем данные
df = pd.read_csv("../data/train.csv")

# Определяем признаки и целевой столбец
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target = 'label'
X = df[features]
y = df[target]

# Разбиваем данные на обучающую и валидационную выборки (80/20)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)

# Определяем базовые модели с уменьшенным количеством итераций и использованием early stopping для CatBoost
cat_model = catboost.CatBoostClassifier(
    iterations=500,
    depth=8,
    learning_rate=0.03,
    loss_function='MultiClass',
    random_seed=seed,
    verbose=100,
    early_stopping_rounds=50
)

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=8,
    random_state=seed
)

xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=8,
    random_state=seed,
    objective='multi:softmax'
)

# Определяем финальный классификатор для ансамбля (Stacking)
final_estimator = catboost.CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.03,
    loss_function='MultiClass',
    random_seed=seed,
    verbose=0,
    early_stopping_rounds=50
)

# Создаем ансамбль с помощью StackingClassifier
stacking_model = StackingClassifier(
    estimators=[('cat', cat_model), ('lgb', lgb_model), ('xgb', xgb_model)],
    final_estimator=final_estimator,
    passthrough=True
)

# Обучаем ансамбль
stacking_model.fit(X_train, y_train)

# Оценка на валидационной выборке
y_val_pred = stacking_model.predict(X_val)
f1_val = f1_score(y_val, y_val_pred, average='macro')
print("F1-score на валидационной выборке: {:.5f}".format(f1_val))

# Загружаем тестовый набор данных
test_df = pd.read_csv("../data/test.csv")
X_test = test_df[features]

# Предсказания для тестового набора
test_pred = stacking_model.predict(X_test)

# Готовим файл для отправки (используя шаблон sample_submit.csv)
submission = pd.read_csv("../data/sample_submit.csv")
submission['label'] = test_pred

# Сохраняем итоговый файл
submission.to_csv("output/submission.csv", index=False)
print("Файл submission.csv сохранён.")
