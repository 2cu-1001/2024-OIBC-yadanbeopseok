import json
import requests

class DataRequester:
    def __init__(self):
        self.base_url = "https://research-api.solarkim.com/data/cmpt-2024/"
        self.API_KEY = ""
        
    
    #제주 기상 실측 데이터 조회
    def request_actual_weather(self, date):
        actual_weather = requests.get(f'https://research-api.solarkim.com/data/cmpt-2024/actual-weather/{date}', headers={
                            'Authorization': f'Bearer {self.API_KEY}'
                        }).json()
        return actual_weather
    
    
    #제주 기상 예측 데이터 조회
    def request_weather_forecast(self, date):
        weather_forecast = requests.get(f'https://research-api.solarkim.com/data/cmpt-2024/weather-forecast/{date}', headers={
                            'Authorization': f'Bearer {self.API_KEY}'
                        }).json()
        return weather_forecast
        
    
    #제주 전력시장 시장전기가격 하루 전 가격 조회
    def request_smp_da(self, date):
        smp_da = requests.get(f'https://research-api.solarkim.com/data/cmpt-2024/smp-da/{date}', headers={
                                'Authorization': f'Bearer {self.API_KEY}'
                            }).json()
        return smp_da
    
    
    #제주 전력시장 시장전기가격 실시간 임시가격, 확정가격 조회
    def request_smp_rt_rc(self, date):
        smp_rt_rc = requests.get(f'https://research-api.solarkim.com/data/cmpt-2024/smp-rt-rc/{date}', headers={
                            'Authorization': f'Bearer {self.API_KEY}'
                        }).json()
        return smp_rt_rc
    
    
    #제주 전력시장 현황 데이터 조회
    def request_elec_supply(self, date):
        elec_supply = requests.get(f'https://research-api.solarkim.com/data/cmpt-2024/elec-supply/{date}', headers={
                            'Authorization': f'Bearer {self.API_KEY}'
                        }).json()
        return elec_supply
    
