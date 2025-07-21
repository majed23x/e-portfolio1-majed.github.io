import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix

# Set seeds
np.random.seed(888)
tf.random.set_seed(112)

# Load data
(x_train_all, y_train_all), (x_test, y_test) = cifar10.load_data()

# Normalize data
x_train_all = x_train_all / 255.0
x_test = x_test / 255.0

# One-hot encoding
y_cat_train_all = to_categorical(y_train_all, 10)
y_cat_test = to_categorical(y_test, 10)

# Validation split
VALIDATION_SIZE = 10000
x_val = x_train_all[:VALIDATION_SIZE]
y_val_cat = y_cat_train_all[:VALIDATION_SIZE]

x_train = x_train_all[VALIDATION_SIZE:]
y_cat_train = y_cat_train_all[VALIDATION_SIZE:]

# Build model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=32, kernel_size=(4,4), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)

# Train
history = model.fit(x_train, y_cat_train, epochs=25, validation_data=(x_val, y_val_cat), callbacks=[early_stop])

# Plot metrics
metrics = pd.DataFrame(history.history)
metrics[['loss', 'val_loss']].plot(title='Loss')
metrics[['accuracy', 'val_accuracy']].plot(title='Accuracy')
plt.show()

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_cat_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Predictions
predictions = np.argmax(model.predict(x_test), axis=-1)
print(classification_report(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Predict single image
i = 16
plt.imshow(x_test[i])
plt.title("True label: " + str(y_test[i][0]))
plt.show()

predicted_label = np.argmax(model.predict(x_test[i].reshape(1, 32, 32, 3)))
print(f"Predicted label index: {predicted_label}")
