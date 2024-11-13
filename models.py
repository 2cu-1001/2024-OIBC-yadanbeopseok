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

class Climate2Generation:
    def _init_(self):
        self.climate_df_list = []
        self.climate_df_list.append(pd.read_csv("./Data/Processed/Bonggae-dong_real.csv"))
        # self.climate_df_list.append(pd.read_csv("./Data/Processed/Cheju-do_real.csv"))
        self.climate_df_list.append(pd.read_csv("./Data/Processed/Cheonji-dong_real.csv"))
        # self.climate_df_list.append(pd.read_csv("./Data/Processed/Gaigeturi_real.csv"))
        self.climate_df_list.append(pd.read_csv("./Data/Processed/Geumak-ri_real.csv"))
        self.climate_df_list.append(pd.read_csv("./Data/Processed/Gwangryeong-ri_real.csv"))
        self.climate_df_list.append(pd.read_csv("./Data/Processed/Hacheon-ri_real.csv"))
        self.climate_df_list.append(pd.read_csv("./Data/Processed/Ilgwa-ri_real.csv"))
        # self.climate_df_list.append(pd.read_csv("./Data/Processed/Jeju_real.csv"))
        self.climate_df_list.append(pd.read_csv("./Data/Processed/Sangmo-ri_real.csv"))
        self.climate_df_list.append(pd.read_csv("./Data/Processed/Songdang-ri_real.csv"))
        self.climate_df_list.append(pd.read_csv("./Data/Processed/Yongsu-ri_real.csv"))
        
        self.climate_df = pd.concat(self.climate_df_list, ignore_index=True)
        self.climate_df.sort_values("ts")
        self.climiate_st, self.climiate_ed = self.climate_df["ts"].values[0], self.climate_df["ts"].values[-1]
        
        self.generation_df = pd.read_csv("./Data/Processed/supplyNdemand.csv")
        self.generation_df = self.generation_df[["ts", "solar", "wind", "renewable"]]
        self.generation_df = self.generation_df[(self.generation_df["ts"] <= self.climiate_ed) 
                                                & (self.generation_df["ts"] >= self.climiate_st)]
        
        self.max_encoder_length = 12  
        self.max_prediction_length = 12  
        self.max_epoch = 10
    
    
    def make_dataframe(self):
        self.data = data = pd.merge_asof(self.climate_df.sort_values("ts"),
                     self.generation_df.sort_values("ts"),
                     on="ts",
                     direction="nearest") 
        
        self.data["hour"] = pd.to_datetime(self.climate_df["ts"], format="%Y%m%d%H%M").dt.hour
        self.data["day"] = pd.to_datetime(self.climate_df["ts"], format="%Y%m%d%H%M").dt.day
        self.data["month"] = pd.to_datetime(self.climate_df["ts"], format="%Y%m%d%H%M").dt.month
        self.data["weekday"] = pd.to_datetime(self.climate_df["ts"], format="%Y%m%d%H%M").dt.weekday

        self.data[["solar", "wind", "renewable"]] = self.data[["solar", "wind", "renewable"]].astype("Float32")
        self.data[["temp","real_feel_temp","real_feel_temp_shade","rel_hum","dew_point","wind_dir","wind_spd",
            "wind_gust_spd","uv_idx","vis","cld_cvr","ceiling","pressure","appr_temp","wind_chill_temp",
            "wet_bulb_temp","precip_1h"]] = self.data[["temp","real_feel_temp","real_feel_temp_shade","rel_hum",
                                                    "dew_point","wind_dir","wind_spd","wind_gust_spd","uv_idx",
                                                    "vis","cld_cvr","ceiling","pressure","appr_temp","wind_chill_temp",
                                                    "wet_bulb_temp","precip_1h"]].astype("Float32")
            
        self.scaler = StandardScaler()
        self.data[["temp","real_feel_temp","real_feel_temp_shade","rel_hum","dew_point","wind_dir","wind_spd",
            "wind_gust_spd","uv_idx","vis","cld_cvr","ceiling","pressure","appr_temp","wind_chill_temp",
            "wet_bulb_temp","precip_1h"]] = self.scaler.fit_transform(
            self.data[["temp","real_feel_temp","real_feel_temp_shade","rel_hum","dew_point","wind_dir","wind_spd",
            "wind_gust_spd","uv_idx","vis","cld_cvr","ceiling","pressure","appr_temp","wind_chill_temp",
            "wet_bulb_temp","precip_1h"]]
        )
        self.data["gruop"] = 0
    
    
    def make_dataloader(self):
        self.ts_len = len(self.train["ts"].values)
        self.train["ts"] = [i for i in range(self.ts_len)]    
        
        self.train_dataset = TimeSeriesDataSet(
            self.train,
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
        
        self.train_dataloader = self.train_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
    

    def train_model(self):
        self.model = TemporalFusionTransformer.from_dataset(
            self.train_dataset,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            #output_size=[24, 24, 24], 
            loss=QuantileLoss()
        )
        
        self.trainer = Trainer(max_epochs=self.max_epoch, enable_model_summary=True, gradient_clip_val=0.1, accelerator="gpu", devices=1)
        self.trainer.fit(self.model, train_dataloaders=self.train_dataloader)
        

    def save_model(self):
        torch.save(self.model.state_dict(), "c2g.pth")
    

    def predition(self, prediction_dataloader):
        self.predictions = self.model.predict(self.prediction_dataloader, return_index=True, return_decoder_lengths=True)
        self.print(self.predictions)
        
        return self.predictions
        
        
class Generation2Prrice:
    def __init__(self):
        self.generation_df = pd.read_csv("./Data/Processed/supplyNdemand.csv")
        self.generation_df = self.generation_df[["ts", "supply", "demand", "solar", "wind", "renewable"]]
        self.generation_df.sort_values("ts")
        self.generation_st, self.generation_ed = self.generation_df["ts"].values[0], self.generation_df["ts"].values[-1]

        self.price_df = pd.read_csv("./Data/Processed/real_price.csv")
        self.price_df = self.price_df[["ts", "real_confirmed_price"]]
        self.price_df.sort_values("ts")
        self.price_df = self.price_df[(self.rice_df["ts"] <= self.generation_ed) & 
                                      (self.price_df["ts"] >= self.generation_st)]
        
        self.max_encoder_length = 12  
        self.max_prediction_length = 12  
        self.max_epoch = 10
        
    
    def make_datafram(self):
        self.data = data = pd.merge_asof(self.generation_df.sort_values("ts"),
                     self.price_df.sort_values("ts"),
                     on="ts",
                     direction="nearest") 

        self.data["hour"] = pd.to_datetime(self.generation_df["ts"], format="%Y%m%d%H%M").dt.hour
        self.data["day"] = pd.to_datetime(self.generation_df["ts"], format="%Y%m%d%H%M").dt.day
        self.data["month"] = pd.to_datetime(self.generation_df["ts"], format="%Y%m%d%H%M").dt.month
        self.data["weekday"] = pd.to_datetime(self.eneration_df["ts"], format="%Y%m%d%H%M").dt.weekday

        self.data["real_confirmed_price"] = self.data["real_confirmed_price"].astype("Float32")
        self.data[["supply", "demand", "solar", "wind", "renewable"]] = self.data[["supply", "demand", "solar", "wind", "renewable"]].astype("Float32")
        
        self.scaler = StandardScaler()
        self.data[["solar", "wind", "renewable"]] = self.scaler.fit_transform(
            self.data[["solar", "wind", "renewable"]]
        )
        self.data["gruop"] = 0
        
    
    def make_loader(self):
        self.ts_len = len(self.train["ts"].values)
        self.train["ts"] = [i for i in range(self.ts_len)]
        
        self.train_dataset = TimeSeriesDataSet(
            self.train,
            time_idx="ts",
            target="real_confirmed_price",
            group_ids=["gruop"],  
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_unknown_reals=["supply", "demand", "solar", "wind", "renewable"],
            time_varying_known_reals=["hour", "day", "month", "weekday"],
            target_normalizer=NaNLabelEncoder(),
        )
        
        self.train_dataloader = self.train_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
        
    
    def train_model(self):
        self.model = TemporalFusionTransformer.from_dataset(
            self.train_dataset,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            #output_size=[24, 24, 24], 
            loss=QuantileLoss()
        )
        
        self.trainer = Trainer(max_epochs=1, enable_model_summary=True, gradient_clip_val=0.1, accelerator="gpu", devices=1)
        self.trainer.fit(self.model, train_dataloaders=self.train_dataloader)
        
    
    def save_model(self):
        torch.save(self.model.state_dict(), "g2p.pth")

    
    def predition(self, prediction_dataloader):
        self.predictions = self.model.predict(self.prediction_dataloader, return_index=True, return_decoder_lengths=True)
        self.print(self.predictions)
        
        return self.predictions