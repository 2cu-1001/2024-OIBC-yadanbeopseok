from data_requester import DataRequester
from train import OurTrainer
from make_json2pred import DataCvt

def main():
    data_requester = DataRequester()
    data_requester.API_KEY = "z"
    
    weather_forecast = data_requester.request_actual_weather(date="2024-11-14")
    weather_real =data_requester.request_weather_forecast(date="2024-11-15")
    
    generation_yesterday = data_requester.request_elec_supply(date="2024-11-14")
    
    dataCvt = DataCvt(weather_forecast, weather_real, generation_yesterday)
    
    weather_prediction = dataCvt.cvt_weather_data()
    supplyNdemand = dataCvt.cvt_generation_data()
    
    OurTrainer.train_c2g()
    OurTrainer.train_g2p()
    
    predicted_generation = OurTrainer.predict_c2g(weather_prediction)
    
    genSupplyDemand_prediction = dataCvt.cvt_generationSupplyDmand_data(predicted_generation, supplyNdemand)
    
    predicted_price = OurTrainer.predict_g2p(genSupplyDemand_prediction)
    
    print(predicted_price)

if __name__ == "__main__":
    main()