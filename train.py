import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data import MultiNormalizer
from pytorch_forecasting.metrics import MultiLoss, QuantileLoss
from lightning.pytorch import Trainer

climiate_df = pd.read_csv("./Data/Processed/Jeju_real.csv")
generation_df = pd.read_csv("./Data/Processed/supplyNdemand.csv")

climiate_st, climiate_ed = climiate_df["ts"].values[0], climiate_df["ts"].values[-1]
print(climiate_st, climiate_ed)

generation_df = generation_df[["ts", "solar", "wind", "renewable"]]
generation_df = generation_df[(generation_df["ts"] <= climiate_ed) & (generation_df["ts"] >= climiate_st)]
print(generation_df.head())


data = pd.merge_asof(climiate_df.sort_values("ts"),
                     generation_df.sort_values("ts"),
                     on="ts",
                     direction="nearest") 


data["hour"] = pd.to_datetime(climiate_df["ts"], format="%Y%m%d%H%M").dt.hour
data["day"] = pd.to_datetime(climiate_df["ts"], format="%Y%m%d%H%M").dt.day
data["month"] = pd.to_datetime(climiate_df["ts"], format="%Y%m%d%H%M").dt.month
data["weekday"] = pd.to_datetime(climiate_df["ts"], format="%Y%m%d%H%M").dt.weekday

data[["solar", "wind", "renewable"]] = data[["solar", "wind", "renewable"]].astype("Float32")
data[["cloud", "temp", "temp_max", "temp_min", "humidity", "ground_press", "wind_speed", 
          "wind_dir", "rain", "snow", "day", "month", "weekday"]] = data[["cloud", "temp", "temp_max", "temp_min", "humidity", "ground_press", "wind_speed", 
          "wind_dir", "rain", "snow", "day", "month", "weekday"]].astype("Float32")
          

scaler = StandardScaler()
data[["cloud", "temp", "temp_max", "temp_min", "humidity", "ground_press", "wind_speed", 
      "wind_dir", "rain", "snow"]] = scaler.fit_transform(
    data[["cloud", "temp", "temp_max", "temp_min", "humidity", "ground_press", "wind_speed", 
          "wind_dir", "rain", "snow"]]
)
      
ts_len = len(data["ts"].values)
data["ts"] = [i for i in range(ts_len)]
print(data.head())

max_encoder_length = 12  
max_prediction_length = 12  

train = data.iloc[:-36, :]
ts_len = len(train["ts"].values)
train["ts"] = [i for i in range(ts_len)]
print(data.head())

test = data.iloc[-36:-12, :]
ts_len = len(test["ts"].values)
test["ts"] = [i for i in range(ts_len)]

valid = data.iloc[-12:, :]
ts_len = len(valid["ts"].values)
valid["ts"] = [i for i in range(ts_len)]
print("###############")
print(train)
print(test)
print(valid)


train_dataset = TimeSeriesDataSet(
    train,
    time_idx="ts",
    target=["solar", "wind", "renewable"],
    group_ids=["location"],  
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["cloud", "temp", "temp_max", "temp_min", "humidity", "ground_press", "wind_speed", 
          "wind_dir", "rain", "snow"],
    time_varying_known_reals=["hour", "day", "month", "weekday"],
    target_normalizer=MultiNormalizer([NaNLabelEncoder() for _ in ["solar", "wind", "renewable"]])
)
test_dataset = TimeSeriesDataSet(
    test,
    time_idx="ts",
    target=["solar", "wind", "renewable"],
    group_ids=["location"],  
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["cloud", "temp", "temp_max", "temp_min", "humidity", "ground_press", "wind_speed", 
          "wind_dir", "rain", "snow"],
    time_varying_known_reals=["hour", "day", "month", "weekday"],
    target_normalizer=MultiNormalizer([NaNLabelEncoder() for _ in ["solar", "wind", "renewable"]])
)


train_dataloader = train_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
test_dataloader = test_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)


model = TemporalFusionTransformer.from_dataset(
    train_dataset,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    #output_size=[24, 24, 24], 
    loss=QuantileLoss()
)


trainer = Trainer(max_epochs=100, enable_model_summary=True, gradient_clip_val=0.1, accelerator="gpu", devices=1)
trainer.fit(model, train_dataloaders=train_dataloader)


predictions = model.predict(test_dataloader, return_index=True, return_decoder_lengths=True)
print(predictions)
