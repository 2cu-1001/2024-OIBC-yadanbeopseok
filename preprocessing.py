from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# dt_obj1 = datetime.fromtimestamp(1709222220)
# dt_obj2 = datetime.fromtimestamp(1709222580)
# dt_obj3 = datetime.fromtimestamp(1709226120)
# print("ts1 :", dt_obj1)
# print("ts2 :", dt_obj2)
# print("ts2 :", dt_obj3)

# #base ts일 때 ts의 기상을 예측함

def timestamp2datetime(timestamp):       
    timestamp = int(timestamp)
    dt = datetime.fromtimestamp(timestamp)

    if dt.minute < 15:
        dt = dt.replace(minute=0, second=0)
    elif dt.minute < 45:
        dt = dt.replace(minute=30, second=0)
    else:
        dt = (dt + timedelta(hours=1)).replace(minute=0, second=0)
    
    dt = dt.strftime("%Y%m%d%H%M")
    dt = int(dt)
    
    return dt
        

forecast_df1 = pd.read_csv("./Data/OIBC_2024_DATA/data/forecast1.csv")
forecast_df2 = pd.read_csv("./Data/OIBC_2024_DATA/data/forecast2.csv")
real_weather_df1 = pd.read_csv("./Data/OIBC_2024_DATA/data/real1.csv")
real_weather_df2 = pd.read_csv("./Data/OIBC_2024_DATA/data/real2.csv")

#하루 전 가격, 실시간 임시가격 배제함
real_price_df = pd.read_csv("./Data/OIBC_2024_DATA/data/real_time_price.csv")
state_df = pd.read_csv("./Data/OIBC_2024_DATA/data/state.csv")

#forecast1
#1 ~ 5926 Ilgwa-ri
#5928 ~ 11852 Geumak-ri
#11854 ~ 17778 Yongsu-ri
#17780 ~ 23704 Hacheon-ri
#23706 ~ 29630 Cheonji-dong
#29632 ~ 35556 Songdang-ri
#35558 ~ 41482 Bonggae-dong
#41484 ~ 47408 Gwangryeong-ri
#47410 ~ 53334 Sangmo-ri

#forecast2
#1 ~ 5926 Gaigeturi
#5928 ~ 11852 Cheju-do
#11854 ~ 17778 Jeju

#weather1
#1 ~ 7934 Ilgwa-ri
#7936 ~ 13960 Geumak-ri
#13962 ~ 21945 Yongsu-ri
#21947 ~ 28134 Hacheon-ri
#28136 ~ 36828 Cheonji-dong
#36830 ~ 42841 Songdang-ri
#42843 ~ 50576 Bonggae-dong
#50578 ~ 56811 Gwangryeong-ri
#56813 ~ 64667 Sangmo-ri

#weather2
#1 ~ 11333 Gaigeturi
#11335 ~ 22673 Cheju-do
#22675 ~ 34013 Jeju


forecast_df1["base_ts"][0:5925] = forecast_df1["base_ts"][0:5925].apply(timestamp2datetime)
forecast_df1["base_ts"][5927:11851] = forecast_df1["base_ts"][5927:11851].apply(timestamp2datetime)
forecast_df1["base_ts"][11853:17777] = forecast_df1["base_ts"][11853:17777].apply(timestamp2datetime)
forecast_df1["base_ts"][17779:23703] = forecast_df1["base_ts"][17779:23703].apply(timestamp2datetime)
forecast_df1["base_ts"][23705:29629] = forecast_df1["base_ts"][23705:29629].apply(timestamp2datetime)
forecast_df1["base_ts"][29631:35555] = forecast_df1["base_ts"][29631:35555].apply(timestamp2datetime)
forecast_df1["base_ts"][35557:41481] = forecast_df1["base_ts"][35557:41481].apply(timestamp2datetime)
forecast_df1["base_ts"][41483:47407] = forecast_df1["base_ts"][41483:47407].apply(timestamp2datetime)
forecast_df1["base_ts"][47409:53333] = forecast_df1["base_ts"][47409:53333].apply(timestamp2datetime)

forecast_df1["ts"][0:5925] = forecast_df1["ts"][0:5925].apply(timestamp2datetime)
forecast_df1["ts"][5927:11851] = forecast_df1["ts"][5927:11851].apply(timestamp2datetime)
forecast_df1["ts"][11853:17777] = forecast_df1["ts"][11853:17777].apply(timestamp2datetime)
forecast_df1["ts"][17779:23703] = forecast_df1["ts"][17779:23703].apply(timestamp2datetime)
forecast_df1["ts"][23705:29629] = forecast_df1["ts"][23705:29629].apply(timestamp2datetime)
forecast_df1["ts"][29631:35555] = forecast_df1["ts"][29631:35555].apply(timestamp2datetime)
forecast_df1["ts"][35557:41481] = forecast_df1["ts"][35557:41481].apply(timestamp2datetime)
forecast_df1["ts"][41483:47407] = forecast_df1["ts"][41483:47407].apply(timestamp2datetime)
forecast_df1["ts"][47409:53333] = forecast_df1["ts"][47409:53333].apply(timestamp2datetime)

forecast_df2["base_ts"][0:5925] = forecast_df2["base_ts"][0:5925].apply(timestamp2datetime)
forecast_df2["base_ts"][5927:11851] = forecast_df2["base_ts"][5927:11851].apply(timestamp2datetime)
forecast_df2["base_ts"][11853:17777] = forecast_df2["base_ts"][11853:17777].apply(timestamp2datetime)

forecast_df2["ts"][0:5925] = forecast_df2["ts"][0:5925].apply(timestamp2datetime)
forecast_df2["ts"][5927:11851] = forecast_df2["ts"][5927:11851].apply(timestamp2datetime)
forecast_df2["ts"][11853:17777] = forecast_df2["ts"][11853:17777].apply(timestamp2datetime)

real_weather_df1["ts"][0:7933] = real_weather_df1["ts"][0:7933].apply(timestamp2datetime)
real_weather_df1["ts"][7935:13959] = real_weather_df1["ts"][7935:13959].apply(timestamp2datetime)
real_weather_df1["ts"][13961:21944] = real_weather_df1["ts"][13961:21944].apply(timestamp2datetime)
real_weather_df1["ts"][21946:28133] = real_weather_df1["ts"][21946:28133].apply(timestamp2datetime)
real_weather_df1["ts"][28135:36827] = real_weather_df1["ts"][28135:36827].apply(timestamp2datetime)
real_weather_df1["ts"][36829:42840] = real_weather_df1["ts"][36829:42840].apply(timestamp2datetime)
real_weather_df1["ts"][42842:50575] = real_weather_df1["ts"][42842:50575].apply(timestamp2datetime)
real_weather_df1["ts"][50577:56810] = real_weather_df1["ts"][50577:56810].apply(timestamp2datetime)
real_weather_df1["ts"][56812:64666] = real_weather_df1["ts"][56812:64666].apply(timestamp2datetime)

real_weather_df2["ts"][0:11332] = real_weather_df2["ts"][0:11332].apply(timestamp2datetime)
real_weather_df2["ts"][11334:22672] = real_weather_df2["ts"][11334:22672].apply(timestamp2datetime)
real_weather_df2["ts"][22674:34012] = real_weather_df2["ts"][22674:34012].apply(timestamp2datetime)

real_price_df["ts"][:] = real_price_df["ts"][:].apply(timestamp2datetime)
state_df["ts"][:] = state_df["ts"][:].apply(timestamp2datetime)

real_price_df.to_csv("./Data/Processed/실제가격.csv", index=False)
state_df.to_csv("./Data/Processed/공급&수요.csv", index=False)

forecast_df1[0:5925].to_csv("./Data/Processed/Ilgwa-ri_forecast.csv", index=False)
forecast_df1[5927:11851].to_csv("./Data/Processed/Geumak-ri_forecast.csv", index=False)
forecast_df1[11853:17777].to_csv("./Data/Processed/Yongsu-ri_forecast.csv", index=False)
forecast_df1[17779:23703].to_csv("./Data/Processed/Hacheon-ri_forecast.csv", index=False)
forecast_df1[23705:29629].to_csv("./Data/Processed/Cheonji-dong_forecast.csv", index=False)
forecast_df1[29631:35555].to_csv("./Data/Processed/Songdang-ri_forecast.csv", index=False)
forecast_df1[35557:41481].to_csv("./Data/Processed/Bonggae-dong_forecast.csv", index=False)
forecast_df1[41483:47407].to_csv("./Data/Processed/Gwangryeong-ri_forecast.csv", index=False)
forecast_df1[47409:53333].to_csv("./Data/Processed/Sangmo-ri_forecast.csv")

forecast_df2[0:5925].to_csv("./Data/Processed/Gaigeturi_forecast.csv", index=False)
forecast_df2[5927:11851].to_csv("./Data/Processed/Cheju-do_forecast.csv", index=False)
forecast_df2[11853:17777].to_csv("./Data/Processed/Jeju_forecast.csv", index=False)

real_weather_df1[0:7933].to_csv("./Data/Processed/Ilgwa-ri_real.csv", index=False)
real_weather_df1[7935:13959].to_csv("./Data/Processed/Geumak-ri_real.csv", index=False)
real_weather_df1[13961:21944].to_csv("./Data/Processed/Yongsu-ri_real.csv", index=False)
real_weather_df1[21946:28133].to_csv("./Data/Processed/Hacheon-ri_real.csv", index=False)
real_weather_df1[28135:36827].to_csv("./Data/Processed/Cheonji-dong_real.csv", index=False)
real_weather_df1[36829:42840].to_csv("./Data/Processed/Songdang-ri_real.csv", index=False)
real_weather_df1[42842:50575].to_csv("./Data/Processed/Bonggae-dong_real.csv", index=False)
real_weather_df1[50577:56810].to_csv("./Data/Processed/Gwangryeong-ri_real.csv", index=False)
real_weather_df1[56812:64666].to_csv("./Data/Processed/Sangmo-ri_real.csv", index=False)

real_weather_df2[0:11332].to_csv("./Data/Processed/Gaigeturi_real.csv", index=False)
real_weather_df2[11334:22672].to_csv("./Data/Processed/Cheju-do_real.csv", index=False)
real_weather_df2[22674:34012].to_csv("./Data/Processed/Jeju_real.csv", index=False)