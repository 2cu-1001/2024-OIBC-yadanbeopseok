from data_requester import DataRequester
import json

def main():
    data_requester = DataRequester()
    data_requester.API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzd"+\
        "WIiOiJuajd6OXNKSE1EdWpCSEx6QTZpUWhpIiwiaWF0Ijo"+\
        "xNzMwNjc4MjcxLCJleHAiOjE3MzE1OTY0MDAsInR5cGUiOi"+\
        "JhcGlfa2V5In0.UWFj9Fw0nALOLQOMQ-n7SBzRRqug542dF1BkyCGDmR0"
    
    tmp = data_requester.request_actual_weather(date="2024-10-28")
    
    with open("./tmp.json", 'w') as f:
        json.dump(tmp, f)
    

if __name__ == "__main__":
    main()