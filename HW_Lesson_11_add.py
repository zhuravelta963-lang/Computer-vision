import pandas as pd #зчитує інфу з таблиці csv
import numpy as np #МАТЕМАТИЧНІ ОПЕРАЦІЇ
import tensorflow as tf #бібліотека для стор нейронки
from tensorflow import keras # keras працює з шарами
from tensorflow.keras import layers #інтерфейс, розширення для TensorFlow - побудова моделей
from sklearn.preprocessing import LabelEncoder #текстові мітки в числах
import matplotlib.pyplot as plt #бібліотке для побудови графіків, крута бібліотека

#2 зчитуємо інфу з цсв таблиці
#подивитися в документацію в пайтоні
df = pd.read_csv("data/figures.csv") #datafile
# print(df.head())

#3 перетворюємо назви фігур у цифри
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])
df['Area_Perimeter_Ratio'] = df['Area'] / df['Perimeter']
#fit_transform - перетворює назви на числа


# вибираємо стовбці для навчання (робимо мітки і ознаки)
X = df[["area", "perimeter", "corners", "Area_Perimeter_Ratio"]] #матриця ознак, чим навчається
y = df["label_enc"] #мітки, чим підписується

#4 створюємо модель
model = keras.Sequential([
    layers.Dense(16, activation = "relu", input_shape = (4,)),
    layers.Dense(16, activation = "relu"),
    layers.Dense(16, activation = "softmax")
])
#якщо кортеж з одного леемента, то ставиться кома

#5 компіляція модлейй - визначаємо як мережа буде навчатися
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
#adam - алгоритм, який обирає кращий алгоритм для навчання
#metrics - точність accuracy - значення у відсотках

#6 навчання

history = model.fit(X, y, epochs = 500, verbose = 0)
#verbose - вивід інформації в консоль. 0 щоб видалити це
#fit - навчання

#7 візуалізація навчання
plt.plot(history.history['loss'], label = "втрати")
plt.plot(history.history['accuracy'], label = "точність")
#plot щоб створити графікі
plt.xlabel('епоха')
plt.ylabel("значення")
plt.title("процес навчання моделі")
plt.legend()
plt.show()

#8 тестування
test = np.array([[25, 20, 0, 1.25]])
pred = model.predict(test)
print(f"імовірність кожного класу {pred}")
print(f"модель визначила {encoder.inverse_transform([np.argmax(pred)])}")