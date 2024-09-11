import yfinance as yf
from prophet import Prophet
import pandas as pd
import pickle

class Model:
    def __init__(self):
        self.START = "2015-01-01"
        self.model = None
    
    def load_data(self, ticker, end) -> pd.DataFrame:
        data = yf.download(ticker, self.START, end)
        data.reset_index(inplace=True)
        return data
    
    def train(self, data, saved_model) -> bool: # return True if new model is trained
        df_train = data[['Date','Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
        print(f"Loading saved model for {saved_model}")
        self.load_model(f'{saved_model}.pkl')
        if self.model is None:
            print(f"No saved model for {saved_model}. Training new model")
            self.model = Prophet()
            self.model.fit(df_train)
            return True
        return False

    def predict(self, period=60) -> pd.DataFrame:
        future = self.model.make_future_dataframe(periods=period)
        forecast = self.model.predict(future)
        # print(forecast.tail(period))
        return forecast[['ds','yhat']].tail(period)
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)
            
    def load_model(self, filename):
        try:
            with open(filename, 'rb') as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            self.model = None
            print("Model not found")