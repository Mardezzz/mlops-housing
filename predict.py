import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib

# Загружаем данные
df = pd.read_csv("flats55.csv")

# Признаки и целевая переменная
X = df.drop("цена", axis=1)
y = df["цена"]

# Категориальные и числовые признаки
categorical = ["район"]
numeric = ["площадь", "комнаты", "этаж"]

# Преобразования
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
], remainder='passthrough')

# Модель
model = make_pipeline(preprocessor, LinearRegression())

# Обучение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Сохраняем модель
joblib.dump(model, "models/model.pkl")

# Проверка точности
print("R² на тесте:", model.score(X_test, y_test))
