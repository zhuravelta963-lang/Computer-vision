import tensorflow as tf

from tensorflow.keras import layers, models
import numpy as np
from tensorflow.keras.preprocessing import image

#1 завантажуємо файли
train_ds = tf.keras.preprocessing.image_dataset_from_directory('dataset/train',
         image_size = (128, 128), batch_size = 30,
         label_mode = 'categorical')
test_ds = tf.keras.preprocessing.image_dataset_from_directory('dataset/test',
         image_size = (128, 128), batch_size = 30,
         label_mode = 'categorical')

#2 нормалізація зображення
normalization_layer = layers.Rescaling(1./255) #перевід усьогго на 0 і 1

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

#4 будуємо моdель
model = models.Sequential()

#згорткові нейронні мережі
#пroсті ознаки - краї, лінії і т.д.
model.add(layers.Conv2D(filters = 32, kernel_size = (3, 3),
                        activation='relu', input_shape=(128, 128, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#глибші ознаки - контури, структура
model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(filters = 64, activation='relu'))
model.add(layers.Dense(3, activation='softmax')) #якщо треба збільшити кількість класів, то замінити 3 на інше

#5 компіляція моделей
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
#6 навчання моделі
history = model.fit(train_ds, epochs=50, validation_data = test_ds)
test_loss, test_acc = model.evaluate(test_ds)
print(f'правдивість: {test_acc}%')

#7 робимо перевірку
class_name = ['cars', 'cats', 'dogs']

img = image.load_img("image/cars.jpg", target_size=(128, 128))

img_array = image.img_to_array(img)

#8 нормалізуємо зображення
img_array = img_array/255.0
img_array = np.expand_dims(img_array, axis=0)

#9 прогноз
pred = model.predict(img_array)

pred_index = np.argmax(pred[0])
print(f'імовірність по класам: {pred[0]}')
print(f'модель визначила: {class_name[pred_index]}')



