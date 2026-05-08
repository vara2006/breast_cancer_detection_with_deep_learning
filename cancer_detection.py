# IMPORT LIBRARIES

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# DOWNLOAD DATASET FROM KAGGLE

print("Downloading dataset...")

path = kagglehub.dataset_download(
    "uciml/breast-cancer-wisconsin-data"
)

print("Dataset downloaded at:")
print(path)


# LOAD CSV FILE

import os

csv_path = os.path.join(path, "data.csv")

df = pd.read_csv(csv_path)


# SHOW DATASET

print("\nFirst 5 Rows:\n")
print(df.head())


# DATASET INFO

print("\nDataset Info:\n")
print(df.info())


# CHECK NULL VALUES

print("\nMissing Values:\n")
print(df.isnull().sum())


# REMOVE UNNECESSARY COLUMNS

df = df.drop(columns=['id'])

if 'Unnamed: 32' in df.columns:
    df = df.drop(columns=['Unnamed: 32'])


# CONVERT LABELS

df['diagnosis'] = df['diagnosis'].map({
    'M': 1,
    'B': 0
})


# FEATURES AND LABELS

X = df.drop(columns='diagnosis')

y = df['diagnosis']


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# FEATURE SCALING

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# BUILD DEEP LEARNING MODEL

model = Sequential()

model.add(Dense(
    64,
    activation='relu',
    input_shape=(X_train.shape[1],)
))

model.add(Dropout(0.3))

model.add(Dense(
    32,
    activation='relu'
))

model.add(Dropout(0.3))

model.add(Dense(
    1,
    activation='sigmoid'
))


# COMPILE MODEL

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


# TRAIN MODEL

print("\nTraining Model...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2
)


# EVALUATE MODEL

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nTest Accuracy:")
print(accuracy * 100)


# PREDICTIONS

y_pred = model.predict(X_test)

y_pred = (y_pred > 0.5)


# CONFUSION MATRIX

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion matrix_graph.png")
plt.close()


# CLASSIFICATION REPORT

print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred
))


# ACCURACY GRAPH

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend(['Train', 'Validation'])

plt.savefig("accuracy_graph.png")
plt.close()


# LOSS GRAPH

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend(['Train', 'Validation'])

plt.savefig("model loss.png")
plt.close()


# SAVE MODEL

model.save("breast_cancer_model.h5")

print("\nModel Saved Successfully")
# ==============================
# PREDICT NEW PATIENT DATA
# ==============================

new_data = [[
    17.99,
    10.38,
    122.8,
    1001.0,
    0.1184,
    0.2776,
    0.3001,
    0.1471,
    0.2419,
    0.07871,
    1.095,
    0.9053,
    8.589,
    153.4,
    0.006399,
    0.04904,
    0.05373,
    0.01587,
    0.03003,
    0.006193,
    25.38,
    17.33,
    184.6,
    2019.0,
    0.1622,
    0.6656,
    0.7119,
    0.2654,
    0.4601,
    0.1189
]]

# SCALE DATA

new_data_scaled = scaler.transform(new_data)

# PREDICT

prediction = model.predict(new_data_scaled)

print("\nPrediction Value:")
print(prediction)

# RESULT

if prediction[0][0] > 0.5:
    print("\nCancer Detected")
else:
    print("\nNon Cancer")