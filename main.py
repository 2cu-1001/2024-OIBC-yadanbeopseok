from data_requester import DataRequester
from train import OurTrainer
from make_json2pred import DataCvt

def main():
    print("#############################################################################")
    data_requester = DataRequester()
    data_requester.API_KEY = ""
    
    cur_day = "2024-11-12"
    pre_day = "2024-11-13"
    nxt = "20241115"
    
    weather_cur =data_requester.request_actual_weather(date=cur_day)
    weather_pre = data_requester.request_actual_weather(date=pre_day)
    
    generation_cur = data_requester.request_elec_supply(date=cur_day)
    generation_pre = data_requester.request_elec_supply(date=pre_day)
    
    dataCvt = DataCvt(weather_cur, weather_pre, generation_cur, generation_pre)
    trainer = OurTrainer()
    trainer.c2g.max_epoch = 1
    trainer.g2p.max_epoch = 1
    
    weather_prediction = dataCvt.cvt_weather_data()
    
    trainer.train_c2g()
    
    predicted_generation = trainer.predict_c2g(weather_prediction)
    print(predicted_generation)
    
    supplyNdemand = dataCvt.cvt_generation_data()
    trainer.train_g2p()
    
    genSupplyDemand_prediction = dataCvt.cvt_generationSupplyDmand_data(predicted_generation, supplyNdemand, nxt)
    
    predicted_price = trainer.predict_g2p(genSupplyDemand_prediction)
    print(predicted_price)

if __name__ == "__main__":
    main()
