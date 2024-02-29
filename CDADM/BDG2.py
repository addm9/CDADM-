import numpy as np
import pandas as pd

from CDADM.optim import Adam
from Rpypots.imputation import Transformer
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow import keras
import tensorflow as tf
from pypots.utils.metrics import cal_mae
from pycorruptor import mcar, masked_fill
import torch


## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

data1= pd.read_csv("G:/会议/会议数据/onemonth.csv")
data1 = reduce_mem_usage(data1)
data1 = data1.drop(["Unnamed: 0"],axis=1)
data1_train = data1[(data1["building_id"] <= 1610)]
data1_test = data1[(data1["building_id"] > 1610)]

#数据处理
num_samples_train= data1_train[data1_train["timestamp"] == "2016-01-01 01:00:00"].shape[0]
num_samples_test= data1_test [data1_test["timestamp"] == "2016-01-01 01:00:00"].shape[0]
train = data1_train.sort_values(by=["meter","timestamp"])
test = data1_test.sort_values(by=["meter","timestamp"])
train = train.drop(["timestamp"],axis=1).values
test= test.drop(["timestamp"],axis=1).values
train = StandardScaler().fit_transform(train)
test = StandardScaler().fit_transform(test)
train = train.reshape(num_samples_train,-1,93)
test = test.reshape(num_samples_test,-1,93)
train1={'X':train}
X_intact_10, X_10, missing_mask_10, indicating_mask_10 = mcar(test,0.1) # hold out 10% observed values as ground truth
X_10 = masked_fill(X_10, 1 - missing_mask_10, np.nan)
test1_10 = {'X':X_10}
X_intact_20, X_20, missing_mask_20, indicating_mask_20 = mcar(test,0.2) # hold out 10% observed values as ground truth
X_20 = masked_fill(X_20, 1 - missing_mask_20, np.nan)
test1_20 = {'X':X_20}
X_intact_30, X_30, missing_mask_30, indicating_mask_30 = mcar(test,0.3) # hold out 10% observed values as ground truth
X_30 = masked_fill(X_30, 1 - missing_mask_30, np.nan)
test1_30 = {'X':X_30}
X_intact_40, X_40, missing_mask_40, indicating_mask_40 = mcar(test,0.4) # hold out 10% observed values as ground truth
X_40 = masked_fill(X_40, 1 - missing_mask_40, np.nan)
test1_40 = {'X':X_40}
#数据插补
transformer = Transformer(
    n_steps=93,
    n_features=136,
    n_layers=2,
    d_model=136,
    d_inner=128,
    n_heads=3,
    d_k=136,
    d_v=136,
    dropout=0.1,
    attn_dropout=0.1,
    ORT_weight=1,
    MIT_weight=1,
    batch_size=1,
    epochs=200,
    threshold_value=0.4,
    threshold_diff=0.4,
    mask_percentage=0.2,
    starttime=40,
    patience=200,
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
