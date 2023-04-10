import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import csv

RANDOM_SEED = 37

dataset_path = r"202302103\model\data.csv"
dataset_y_path = r"202302103\model\data_y.csv"
model_path = r"202302103\model\model\training_relu"

NUM_CLASSES = 4

x_dataset = np.loadtxt(dataset_path, delimiter=",")
y_dataset = np.loadtxt(dataset_y_path, delimiter=",")

x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((75, )),
    tf.keras.layers.Dense(35, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(15, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(7, activation="relu"),
    tf.keras.layers.Dense(NUM_CLASSES,activation="softmax")
])

model.summary()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# モデルチェックポイントのコールバック
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_path, verbose=1, save_weights_only=False)
# 早期打ち切り用コールバック
es_callback = tf.keras.callbacks.EarlyStopping(patience=30, verbose=1)

model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=[cp_callback, es_callback]
)
# 保存したモデルのロード

model = tf.keras.models.load_model(model_path)

predict_result = model.predict(np.array([x_test[7]]))
print(np.squeeze(predict_result))
print("predicted: ",np.argmax(np.squeeze(predict_result)))

print("answer: ", y_test[7])