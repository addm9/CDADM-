from Rpypots.optim import Adam
from Rpypots.imputation import SAITS
from Rpypots.imputation import Transformer
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow import keras
import tensorflow as tf
from pypots.utils.metrics import cal_mae
from pycorruptor import mcar, masked_fill
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from pycorruptor import mcar, masked_fill
def window_truncate(feature_vectors, seq_len, sliding_len=None):
    """ Generate time series samples, truncating windows from time-series data with a given sequence length.
    Parameters
    ----------
    feature_vectors: time series data, len(shape)=2, [total_length, feature_num]
    seq_len: sequence length
    sliding_len: size of the sliding window
    """
    sliding_len = seq_len if sliding_len is None else sliding_len
    total_len = feature_vectors.shape[0]
    start_indices = np.asarray(range(total_len // sliding_len)) * sliding_len
    if total_len - start_indices[-1] * sliding_len < seq_len:  # remove the last one if left length is not enough
        start_indices = start_indices[:-1]
    sample_collector = []
    for idx in start_indices:
        sample_collector.append(feature_vectors[idx: idx + seq_len])
    return np.asarray(sample_collector).astype('float32')

def add_artificial_mask(X, artificial_missing_rate, set_name):
    """Add artificial missing values.
    Parameters
    ----------
    X: feature vectors
    artificial_missing_rate: rate of artificial missing values that are going to be create
    set_name: dataset name, train/val/test
    """
    sample_num, seq_len, feature_num = X.shape
    if set_name == "train":
        # if this is train set, we don't need add artificial missing values right now.
        # If we want to apply MIT during training, dataloader will randomly mask out some values to generate X_hat

        # calculate empirical mean for model GRU-D, refer to paper
        mask = (~np.isnan(X)).astype(np.float32)
        X_filledWith0 = np.nan_to_num(X)
        empirical_mean_for_GRUD = np.sum(mask * X_filledWith0, axis=(0, 1)) / np.sum(
            mask, axis=(0, 1)
        )
        data_dict = {
            "X": X,
            "empirical_mean_for_GRUD": empirical_mean_for_GRUD,
        }
    else:
        # if this is val/test set, then we need to add artificial missing values right now,
        # because we need they are fixed
        X = X.reshape(-1)
        indices_for_holdout = random_mask(X, artificial_missing_rate)
        X_hat = np.copy(X)
        X_hat[indices_for_holdout] = np.nan  # X_hat contains artificial missing values
        missing_mask = (~np.isnan(X_hat)).astype(np.float32)
        # indicating_mask contains masks indicating artificial missing values
        indicating_mask = ((~np.isnan(X_hat)) ^ (~np.isnan(X))).astype(np.float32)

        data_dict = {
            "X": X.reshape([sample_num, seq_len, feature_num]),
            "X_hat": X_hat.reshape([sample_num, seq_len, feature_num]),
            "missing_mask": missing_mask.reshape([sample_num, seq_len, feature_num]),
            "indicating_mask": indicating_mask.reshape(
                [sample_num, seq_len, feature_num]
            ),
        }

    return data_dict

def random_mask(vector, artificial_missing_rate):
    """generate indices for random mask"""
    assert len(vector.shape) == 1
    indices = np.where(~np.isnan(vector))[0].tolist()
    indices = np.random.choice(indices, int(len(indices) * artificial_missing_rate))
    return indices

df = pd.read_csv("G:/会议/会议数据/LD2011_2014.txt", index_col=0, sep=";", decimal=",")
df.index = pd.to_datetime(df.index)
feature_names = df.columns.tolist()
feature_num = len(feature_names)
df["datetime"] = pd.to_datetime(df.index)
unique_months = df["datetime"].dt.to_period("M").unique()
selected_as_test = unique_months[:5]  # select first 10 months as test set
selected_as_train = unique_months[5:20]  # use left months as train set
test_set = df[df["datetime"].dt.to_period("M").isin(selected_as_test)]
train_set = df[df["datetime"].dt.to_period("M").isin(selected_as_train)]
scaler = StandardScaler()

train_set_X = scaler.fit_transform(train_set.loc[:, feature_names])
test_set_X = scaler.transform(test_set.loc[:, feature_names])

train_set_X = window_truncate(train_set_X, 100)
test_set_X = window_truncate(test_set_X, 100)

train1={'X':train_set_X}
X_intact_10, X_10, missing_mask_10, indicating_mask_10 = mcar(test_set_X,0.1) # hold out 10% observed values as ground truth
X_10 = masked_fill(X_10, 1 - missing_mask_10, np.nan)
test1_10 = {'X':X_10}
X_intact_20, X_20, missing_mask_20, indicating_mask_20 = mcar(test_set_X,0.2) # hold out 10% observed values as ground truth
X_20 = masked_fill(X_20, 1 - missing_mask_20, np.nan)
test1_20 = {'X':X_20}
X_intact_30, X_30, missing_mask_30, indicating_mask_30 = mcar(test_set_X,0.3) # hold out 10% observed values as ground truth
X_30 = masked_fill(X_30, 1 - missing_mask_30, np.nan)
test1_30 = {'X':X_30}
X_intact_40, X_40, missing_mask_40, indicating_mask_40 = mcar(test_set_X,0.4) # hold out 10% observed values as ground truth
X_40 = masked_fill(X_40, 1 - missing_mask_40, np.nan)
test1_40 = {'X':X_40}

#数据插补
transformer = Transformer(
    n_steps=100,
    n_features=370,
    n_layers=2,
    d_model=370,
    d_inner=128,
    n_heads=3,
    d_k=370,
    d_v=370,
    dropout=0.1,
    attn_dropout=0.1,
    ORT_weight=1,
    MIT_weight=1,
    batch_size=1,
    epochs=100,
    threshold_value=0.6,
    threshold_diff=0.4,
    mask_percentage=0.6,
    starttime=0,
    patience=100,
    optimizer=Adam(lr=1e-3),
    num_workers=0,
    device=['cuda:0'],
    model_saving_strategy="best",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer.fit(train_set=train1)

test = transformer.predict(test1_10)
test1_10 = test['imputation']
mae = cal_mae(test1_10, X_intact_10, indicating_mask_10)
current_memory = torch.cuda.memory_allocated(device)
print(mae)
print(current_memory)
test = transformer.predict(test1_20)
test1_20 = test['imputation']
mae = cal_mae(test1_20, X_intact_20, indicating_mask_20)
current_memory = torch.cuda.memory_allocated(device)
print(mae)
print(current_memory)
test = transformer.predict(test1_30)
test1_30 = test['imputation']
mae = cal_mae(test1_30, X_intact_30, indicating_mask_30)
current_memory = torch.cuda.memory_allocated(device)
print(mae)
print(current_memory)
test = transformer.predict(test1_40)
test1_40 = test['imputation']
mae = cal_mae(test1_40, X_intact_40, indicating_mask_40)
current_memory = torch.cuda.memory_allocated(device)
print(mae)
print(current_memory)
