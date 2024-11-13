import models
import preprocessing

class DataCvt:
    def __init__(self, w1, w2, g1):
        self.w1 = w1
        self.w2 = w2
        self.g1 = g1
        
    
    def cvt_weather_data(self):
        self.w1 = self.w1["actual_weather_1"]
        self.w2 = self.w2["actual_weather_1"]
        
        return None
        
        
    def cvt_generation_data(self):
        self.w1 = self.w1["actual_weather_1"]
        self.w2 = self.w2["actual_weather_1"]
        
        return None
        
    
    def cvt_generationSupplyDmand_data(self, generation ,supplyNdemand):

        
        return None