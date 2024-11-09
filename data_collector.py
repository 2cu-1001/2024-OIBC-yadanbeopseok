from data_requester import DataRequester
import json


data_requester = DataRequester()
data_requester.API_KEY = ""

tmp = data_requester.request_actual_weather(date="2024-10-28")

with open("./tmp.json", 'w') as f:
    json.dump(tmp, f)

