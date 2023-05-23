import datetime
import os
from os import listdir
from os.path import join
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from time import time
from joblib import Parallel, delayed


START_DATE = "1999-12-31"
TRAIN_DATE = "2019-01-01"
VAL_DATE = "2021-01-01"
FEATURES = ["Low", "Open", "High", "Close", "Volume", "Adjusted Close"]
TARGET_FEATURE = "Adjusted Close"
TARGET = "Target"
WINDOW = 30
SANITY_CHECK = False


def get_x_y(df: pd.DataFrame):
    x_arr, y_arr = [], []
    for i in range(df.shape[0] - WINDOW):
        x = df.iloc[i : i + WINDOW][FEATURES].values
        x_arr.append(x)
        y = df.iloc[i + WINDOW][TARGET]
        y_arr.append(y)
    x_arr, y_arr = np.array(x_arr), np.array(y_arr).reshape(-1, 1)
    return x_arr, y_arr


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df[START_DATE:]
    volume = pd.Series(np.log(df["Volume"] + 1)).diff(1)[1:]
    df = df.pct_change(1)[1:]
    df["Volume"] = volume
    target = (df[TARGET_FEATURE] >= 0).astype(int)  # Add another class - 'const' when |change| < 0.2%
    scaler = StandardScaler()
    scaler = scaler.fit(df[:TRAIN_DATE][FEATURES].values)
    scaled_x = scaler.transform(df[FEATURES].values)
    df = pd.DataFrame(scaled_x, index=df.index, columns=df.columns)
    date_index = df.index.to_series()
    df["Weekday"] = (date_index.dt.day_of_week / 2) - 1
    df["Month Sin"] = np.sin(date_index.dt.month * (2 * np.pi / 12))
    df["Month Cos"] = np.cos(date_index.dt.month * (2 * np.pi / 12))
    df[TARGET] = target
    return df


def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_train = df[:TRAIN_DATE]
    if SANITY_CHECK:
        df_train = df_train[:100]
    df_val = df[TRAIN_DATE:VAL_DATE]
    df_test = df[VAL_DATE:]
    return df_train, df_val, df_test


def is_valid_csv(csv_path: str, df: pd.DataFrame) -> bool:
    min_len = WINDOW + 1
    if df.isnull().values.any():
        print(f"Skipping {csv_path}, nans detected")
        return False

    if len(df[START_DATE:]) < min_len:
        print(f"Skipping {csv_path}, not enough data at all")
        return False

    if len(df[START_DATE:TRAIN_DATE]) < min_len:
        print(f"Skipping {csv_path}, not enough train data")
        return False

    if len(df[TRAIN_DATE:VAL_DATE]) < min_len:
        print(f"Skipping {csv_path}, not enough val data")
        return False

    if len(df[VAL_DATE:]) < min_len:
        print(f"Skipping {csv_path}, not enough test data")
        return False

    return True


def process(idx: int, csv_path: str):
    print(f"Progress {csv_path} {idx}/1563", flush=True)
    df = pd.read_csv(csv_path, index_col="Date")[FEATURES]
    df.index = pd.to_datetime(df.index, format="%d-%m-%Y")
    if is_valid_csv(csv_path, df):
        df = preprocess(df)
        df.to_csv(f"data/preprocessed/{os.path.basename(csv_path)}")


dataset_path = "data\csv"
csv_paths = [join(dataset_path, f) for f in listdir(dataset_path)]
results = Parallel(n_jobs=-1)(delayed(process)(idx, csv_path) for idx, csv_path in enumerate(csv_paths))


df_train, df_val, df_test = split(df)
# Build datasets
x_train, y_train = get_x_y(df_train)
x_val, y_val = get_x_y(df_val)
x_test, y_test = get_x_y(df_test)
x_train, y_train, x_val, y_val, x_test, y_test

filtered = list(filter(lambda x: x != None, results))
x_train = np.concatenate([x[0] for x in filtered], axis=0)
y_train = np.concatenate([x[1] for x in filtered], axis=0)
x_val = np.concatenate([x[2] for x in filtered], axis=0)
y_val = np.concatenate([x[3] for x in filtered], axis=0)
x_test = np.concatenate([x[4] for x in filtered], axis=0)
y_test = np.concatenate([x[5] for x in filtered], axis=0)
# Print summary of data
print(f"Train data dimensions: {x_train.shape}, {y_train.shape}")
print(f"Validation data dimensions: {x_val.shape}, {y_val.shape}")
print(f"Test data dimensions: {x_test.shape}, {y_test.shape}")


# Let's make a list of CONSTANTS for modelling:
LAYERS = [200, 200, 200, 1]  # number of units in hidden and output layers
M_TRAIN = x_train.shape[0]  # number of training examples (2D)
M_VAL = x_val.shape[0]  # number of test examples (2D),full=X_test.shape[0]
N = x_train.shape[2]  # number of features
BATCH = 128  # batch size
EPOCH = 1000  # number of epochs
LR = 5e-2  # learning rate of the gradient descent
LAMBD = 3e-2  # lambda in L2 regularizaion
DP = 0.2  # dropout rate
RDP = 0.0  # recurrent dropout rate
print(f"layers={LAYERS}, train_examples={M_TRAIN}, test_examples={M_VAL}")
print(f"batch = {BATCH}, timesteps = {WINDOW}, features = {N}, epochs = {EPOCH}")
print(f"lr = {LR}, lambda = {LAMBD}, dropout = {DP}, recurr_dropout = {RDP}")

# Build the Model
model = Sequential()
model.add(
    LSTM(
        input_shape=(WINDOW, N),
        units=LAYERS[0],
        kernel_regularizer=l2(LAMBD),
        recurrent_regularizer=l2(LAMBD),
        dropout=DP,
        recurrent_dropout=RDP,
        return_sequences=True,
        return_state=False,
        stateful=False,
        unroll=False,
    )
)
model.add(BatchNormalization())
model.add(
    LSTM(
        units=LAYERS[1],
        kernel_regularizer=l2(LAMBD),
        recurrent_regularizer=l2(LAMBD),
        dropout=DP,
        recurrent_dropout=RDP,
        return_sequences=True,
        return_state=False,
        stateful=False,
        unroll=False,
    )
)
model.add(BatchNormalization())
model.add(
    LSTM(
        units=LAYERS[2],
        kernel_regularizer=l2(LAMBD),
        recurrent_regularizer=l2(LAMBD),
        dropout=DP,
        recurrent_dropout=RDP,
        return_sequences=False,
        return_state=False,
        stateful=False,
        unroll=False,
    )
)
model.add(BatchNormalization())
model.add(Dense(units=LAYERS[3], activation="sigmoid"))

# Compile the model with Adam optimizer
model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=Adam(lr=LR))
print(model.summary())

# Define a learning rate decay method:
lr_decay = ReduceLROnPlateau(monitor="loss", patience=8, verbose=0, factor=0.5, min_lr=1e-8)
# Define Early Stopping:
early_stop = EarlyStopping(
    monitor="val_accuracy",
    min_delta=0,
    patience=30 if SANITY_CHECK == False else 100,
    verbose=1,
    mode="auto",
    baseline=0,
    restore_best_weights=True,
)
# Train the model.
# The dataset is small for NN - let's use test_data for validation
start = time()
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = model.fit(
    x_train,
    y_train,
    epochs=EPOCH,
    batch_size=BATCH,
    validation_data=(x_val, y_val),
    shuffle=True,
    verbose=1,
    callbacks=[lr_decay, early_stop, tensorboard_callback],
)
print("-" * 65)
print(f"Training was completed in {time() - start:.2f} secs")
print("-" * 65)
# Evaluate the model:
train_loss, train_acc = model.evaluate(x_train, y_train, batch_size=BATCH, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=BATCH, verbose=0)
print("-" * 65)
print(f"train accuracy = {round(train_acc * 100, 4)}%")
print(f"test accuracy = {round(test_acc * 100, 4)}%")
print(f"test error = {round((1 - test_acc) * M_VAL)} out of {M_VAL} examples")

# Plot the loss and accuracy curves over epochs:
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
axs[0].plot(history.history["loss"], color="b", label="Training loss")
axs[0].plot(history.history["val_loss"], color="r", label="Validation loss")
axs[0].set_title("Loss curves")
axs[0].legend(loc="best", shadow=True)
axs[1].plot(history.history["accuracy"], color="b", label="Training accuracy")
axs[1].plot(history.history["val_accuracy"], color="r", label="Validation accuracy")
axs[1].set_title("Accuracy curves")
axs[1].legend(loc="best", shadow=True)
plt.show()


# target is prediction if next timestamp is increasing or decreasing
