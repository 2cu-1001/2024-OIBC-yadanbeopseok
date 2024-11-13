import models
import preprocessing
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
import json

class DataCvt:
    def __init__(self, w1, w2, g1, g2):
        self.w1 = w1
        self.w2 = w2
        self.g1 = g1
        self.g2 = g2
        
        self.max_encoder_length = 12
        self.max_prediction_length = 12
        
    
    def cvt_weather_data(self):
        self.w1 = self.w1.get("actual_weather_1")
        self.w2 = self.w2.get("actual_weather_1")
        
        dfW1 = pd.DataFrame(self.w1)
        dfW2 = pd.DataFrame(self.w2)
        dfG1 = pd.DataFrame(self.g1)
        dfG2 = pd.DataFrame(self.g2)

        dfG1 = dfG1.drop(columns=["supply_power", "present_load", "supply_capacity", "operation_capacity"], axis=1)
        dfG2 = dfG2.drop(columns=["supply_power", "present_load", "supply_capacity", "operation_capacity"], axis=1)
        dfG1.columns = ["ts", "solar", "wind", "renewable"]
        dfG2.columns = ["ts", "solar", "wind", "renewable"]

        dfW1["ts"] = dfW1["ts"].apply(preprocessing.timestamp2datetime)
        dfW2["ts"] = dfW2["ts"].apply(preprocessing.timestamp2datetime)

        dfG1["ts"] = dfG1["ts"].apply(preprocessing.timestamp2datetime)
        dfG2["ts"] = dfG2["ts"].apply(preprocessing.timestamp2datetime)
        
        df = pd.concat([dfW1, dfW2], ignore_index=True)
        dfG = pd.concat([dfG1, dfG2], ignore_index=True)
        
        df.sort_values("ts")
        dfG.sort_values("ts")
        
        df_st, df_ed = df["ts"].values[0], df["ts"].values[-1]
        dfG = dfG[(dfG["ts"] <= df_ed) & (dfG["ts"] >= df_st)]

        df = pd.merge_asof(df.sort_values("ts"),
                     dfG.sort_values("ts"),
                     on="ts",
                     direction="nearest")
    
        df["hour"] = pd.to_datetime(df["ts"], format="%Y%m%d%H%M").dt.hour
        df["day"] = pd.to_datetime(df["ts"], format="%Y%m%d%H%M").dt.day
        df["month"] = pd.to_datetime(df["ts"], format="%Y%m%d%H%M").dt.month
        df["weekday"] = pd.to_datetime(df["ts"], format="%Y%m%d%H%M").dt.weekday
        
        df[["solar", "wind", "renewable"]] = df[["solar", "wind", "renewable"]].astype("Float32")
        df[["temp","real_feel_temp","real_feel_temp_shade","rel_hum","dew_point","wind_dir","wind_spd",
            "wind_gust_spd","uv_idx","vis","cld_cvr","ceiling","pressure","appr_temp","wind_chill_temp",
            "wet_bulb_temp","precip_1h"]] = df[["temp","real_feel_temp","real_feel_temp_shade","rel_hum",
                                                    "dew_point","wind_dir","wind_spd","wind_gust_spd","uv_idx",
                                                    "vis","cld_cvr","ceiling","pressure","appr_temp","wind_chill_temp",
                                                    "wet_bulb_temp","precip_1h"]].astype("Float32")
            
        scaler = StandardScaler()
        df[["temp","real_feel_temp","real_feel_temp_shade","rel_hum","dew_point","wind_dir","wind_spd",
            "wind_gust_spd","uv_idx","vis","cld_cvr","ceiling","pressure","appr_temp","wind_chill_temp",
            "wet_bulb_temp","precip_1h"]] = scaler.fit_transform(
            df[["temp","real_feel_temp","real_feel_temp_shade","rel_hum","dew_point","wind_dir","wind_spd",
            "wind_gust_spd","uv_idx","vis","cld_cvr","ceiling","pressure","appr_temp","wind_chill_temp",
            "wet_bulb_temp","precip_1h"]]
        )
        df["gruop"] = 0        

        ts_len = len(df["ts"].values)
        df["ts"] = [i for i in range(ts_len)] 

        dataset = TimeSeriesDataSet(
            df,
            time_idx="ts",
            target=["solar", "wind", "renewable"],
            group_ids=["gruop"],  
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_unknown_reals=["temp","real_feel_temp","real_feel_temp_shade","rel_hum","dew_point","wind_dir",
                                        "wind_spd","wind_gust_spd","uv_idx","vis","cld_cvr","ceiling","pressure","appr_temp",
                                        "wind_chill_temp","wet_bulb_temp","precip_1h"],
            time_varying_known_reals=["hour", "day", "month", "weekday"],
            target_normalizer=MultiNormalizer([NaNLabelEncoder() for _ in ["solar", "wind", "renewable"]]),
        )
        
        train_dataloader = dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
    
        return train_dataloader
        
        
    def cvt_generation_data(self):
        dfG1 = pd.DataFrame(self.g1)
        dfG2 = pd.DataFrame(self.g2)
        
        dfG1 = dfG1[["ts", "supply_power", "present_load"]]
        dfG2 = dfG2[["ts", "supply_power", "present_load"]]
        dfG1.columns ["ts", "supply", "demand"]
        dfG2.columns ["ts", "supply", "demand"]
        
        dfG1["ts"] = dfG1["ts"].apply(preprocessing.timestamp2datetime)
        dfG2["ts"] = dfG2["ts"].apply(preprocessing.timestamp2datetime)
        dfG = pd.concat([dfG1, dfG2], ignore_index=True)
        dfG.sort_values("ts")
        
        return dfG
        
    
    def cvt_generationSupplyDmand_data(self, generation ,supplyNdemand, nxt):
        dfG1 = pd.DataFrame(self.g1)
        dfG2 = pd.DataFrame(generation, columns=["solar", "wind", "renewable"])
        dfS = supplyNdemand

        dfG1 = dfG1.drop(columns=["supply_power", "present_load", "supply_capacity", "operation_capacity"], axis=1)
        dfG1.columns = ["ts", "solar", "wind", "renewable"]

        return None
