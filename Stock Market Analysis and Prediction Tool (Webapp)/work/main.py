from flask import Flask,render_template,request,redirect,session,url_for,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import bcrypt
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime,date
import requests
import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
import base64
import yfinance as yf
import model 
import warnings
warnings.filterwarnings('ignore')


app = Flask(__name__,template_folder='template')
CORS(app)  # Enable CORS

m = model.Model()

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/bullbear'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'
API_KEY = '0Y980T1R7F9RY7V2'
BASE_URL = 'https://www.alphavantage.co/query'


# List of stock symbols
stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'META', 'TSLA', 'BX', 'HDB', 'BAJFINANCE.NS', 'TCS', 'NFLX', 'NVDA', 'PYPL', 'INTC', 'CSCO', 'PEP', 'KO', 'PFE', 'JNJ', 'V', 'MA', 'DIS', 'ADBE', 'CRM', 'ORCL', 'IBM', 'UBER', 'LYFT', 'SNAP', 'SQ', 'SPOT', 'ZM', 'DOCU', 'BABA', 'JD', 'SHOP', 'WMT', 'TGT', 'MCD', 'SBUX', 'NKE', 'LULU', 'BA', 'GE', 'F', 'GM', 'TM', 'HMC', 'UBS', 'DB', 'HSBC']

cache = {}

#update stock data 

def update_stock_data():
    print("Updating stock data")
    data = yf.download(stock_symbols, period="1d", group_by='ticker')
    stocks = []
    for symbol in stock_symbols:
        if symbol in data and not data[symbol].empty:
            stock_data = data[symbol]
            close_price = stock_data['Close'].iloc[0]
            open_price = stock_data['Open'].iloc[0]
            # Check if close_price or open_price is NaN
            if close_price != close_price:  # Check for NaN
                close_price = 0
            if open_price != open_price:  # Check for NaN
                open_price = 0
            change = close_price - open_price
            stocks.append({
                "symbol": symbol,
                "name": yf.Ticker(symbol).info.get('shortName', 'N/A'),
                "price": close_price,
                "change": change
            })
    cache['stocks'] = stocks
    cache['timestamp'] = datetime.now()

#For fetching stock news 

def fetch_stock_news(api_key, query='stock market'):
    url = f'https://newsapi.org/v2/everything?q={query}&apiKey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        return []
    
 # Function to load data from Yfinance
def load_data(stock_symbol):
    stock_data = yf.download(stock_symbol)
    stock_data = stock_data['Close'].dropna()
    stock_data.index = pd.to_datetime(stock_data.index)
    return stock_data   


#contact table in dtabase

class Contacts(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(20), nullable=False)
    phone_num = db.Column(db.String(12), nullable=False)
    msg = db.Column(db.String(120), nullable=False)
    date = db.Column(db.String(12), nullable=True)

#register table in database

class Register(db.Model): 
    sno = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(50), unique = True , index=True ,nullable = False)
    password = db.Column(db.String(500))
    confirm_pass = db.Column(db.String(500))
    
    
    def __init__(self ,email,password,username,confirm_pass):
        self.username = username
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8') , bcrypt.gensalt()).decode('utf-8')
        self.confirm_pass = bcrypt.hashpw(confirm_pass.encode('utf-8') , bcrypt.gensalt()).decode('utf-8')
         
        
    
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8') , self.password.encode('utf-8'))
        
with app.app_context():
    db.create_all()
        
# Home page 

@app.route("/")

def home():
    return render_template('index.html')


@app.route("/index")

def Home():
    return render_template('index.html')

# contact page

@app.route("/contact",methods = ['GET','POST'])

def contact():
    if (request.method == 'POST'):
        name =request.form.get('name')
        email =request.form.get('email')
        phone =request.form.get('phone')
        message =request.form.get('message')
        
        entry = Contacts(name=name , email=email ,date= datetime.now() ,phone_num=phone ,msg=message )
        db.session.add(entry)
        db.session.commit()        
    return render_template('contact.html')

# About page

@app.route("/about")

def About():
    return render_template('about.html')

# News page 

@app.route("/news")
def post():
    api_key = 'd1f3dd71bc4b42819f26d3d004161891'
    articles = fetch_stock_news(api_key)
    articles.sort(key=lambda x: datetime.strptime(x['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'), reverse=True)


    return render_template('news.html', articles=articles)

# Login page 

@app.route("/login", methods=['GET', 'POST'])
def login():
    error = None 
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = Register.query.filter_by(email=email).first()
        
        if user:
            if bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
                session['username'] = user.username
                session['email'] = user.email
                print("User authenticated successfully")
                return redirect('/dashboard')
            else:
                error = 'Invalid password'
                print("Invalid password")
        else:
            error = 'User not found'
            print("User not found")
    
    return render_template('login.html', error=error)

# Register page 

@app.route("/register", methods=['GET', 'POST'])
def Registration():
    error = None 
    if (request.method == 'POST'):
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password'] 
        
        existing_user = Register.query.filter_by(email=email).first()
        if existing_user:
            error = 'Email already exists. Please use a different email.'
        else:
            new_user = Register(username=username, email=email, password=password, confirm_pass=confirm_password)
            db.session.add(new_user)
            db.session.commit()
            return redirect('/login')
    
    return render_template('register.html', error=error)

# Dashboard page

@app.route("/dashboard")

def Dashboard():
        if 'username' in session and  session['username']: 
            return render_template('dashboard.html')
    
        return render_template('login.html')

# Index in dashboard 

@app.route("/indexAL")

def indexAL():
    
    
    return render_template('indexAL.html')

# About in dashboard 

@app.route("/aboutAL")

def AboutAL():
    return render_template('aboutAL.html')

# News in dashboard 

@app.route("/newsAL")
def postAL():
    api_key = 'd1f3dd71bc4b42819f26d3d004161891'
    articles = fetch_stock_news(api_key)
    articles.sort(key=lambda x: datetime.strptime(x['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'), reverse=True)


    return render_template('newsAL.html', articles=articles)

# contact in dashboard

@app.route("/contactAL",methods = ['GET','POST'])

def contactAL():
    if (request.method == 'POST'):
        name =request.form.get('name')
        email =request.form.get('email')
        phone =request.form.get('phone')
        message =request.form.get('message')
        
        entry = Contacts(name=name , email=email ,date= datetime.now() ,phone_num=phone ,msg=message )
        db.session.add(entry)
        db.session.commit()        
    return render_template('contactAL.html')


# Profile 

@app.route('/Profile')
def Profile():
    return render_template('Profile.html')



#company list


@app.route("/companylist")
def companylist():
    return render_template('companylist.html')

# Fetch stock data from Yfinance

@app.route('/stock', methods=['POST'])
def stock():
    symbol = request.form['symbol']
    data = get_stock_data(symbol)
    if data:
        return render_template('stock.html', data=data, symbol=symbol)
    else:
        return render_template('error.html', message="Could not retrieve stock data.", symbol=symbol)

def get_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(interval='30m', period='2d')  # Fetching 1 day data with 30-minute interval
        if data.empty:
            print(f"No data found for {symbol}")
            return None
       
        data = data.reset_index()
        data_dict = data.to_dict(orient='records')
        for entry in data_dict:
            entry['Datetime'] = entry['Datetime'].strftime("%d/%m/%Y, %H:%M")
            entry['Open'] = f"{entry['Open']:.2f}"
            entry['High'] = f"{entry['High']:.2f}"
            entry['Low'] = f"{entry['Low']:.2f}"
            entry['Close'] = f"{entry['Close']:.2f}"
        return data_dict
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None

# Exchange Currency

@app.route('/exchange', methods=['POST'])
def exchange():    
    open_price = float(request.form['open'])
    high_price = float(request.form['high'])
    low_price = float(request.form['low'])
    close_price = float(request.form['close'])
    volume = request.form['volume']
    time = request.form['time']
    currency = request.form['currency']
    
    # Fetch the exchange rate
    exchange_rate = get_exchange_rate(currency)
    if exchange_rate:
        converted_open = open_price * exchange_rate
        converted_high = high_price * exchange_rate
        converted_low = low_price * exchange_rate
        converted_close = close_price * exchange_rate

        # Formatting converted values to two decimal places
        converted_open = f"{converted_open:.2f}"
        converted_high = f"{converted_high:.2f}"
        converted_low = f"{converted_low:.2f}"
        converted_close = f"{converted_close:.2f}"
        return render_template('exchange.html', 
                               original_open=open_price, original_high=high_price, original_low=low_price, original_close=close_price, original_volume=volume, time=time,
                               converted_open=converted_open, converted_high=converted_high, converted_low=converted_low, converted_close=converted_close,
                               currency=currency, exchange_rate=exchange_rate)
    else:
        return render_template('error.html', message="Could not retrieve exchange rate.", symbol='')

def get_exchange_rate(currency):
    mock_exchange_rates = {
        'INR': 83.55,
        'EUR': 0.93,
        'GBP': 0.79,
        'KWD': 0.31,
        'CHF': 0.89,
        'YEN': 157.40,
        'CNY': 7.25,
    }
    return mock_exchange_rates.get(currency, None)


#Prediction

@app.route('/prediction', methods=['GET'])
def Prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol').upper()
    period = int(request.args.get('period'))

    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="7d")  # Fetch last week of data

    # Include additional details if available
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

    # Convert the DataFrame to HTML
    stock_table = hist.to_html(classes='data', header="true", index=False)

    # Fetch additional company information
    info = ticker.info

    # Load stock data for ARIMA model
    price_of_stock = ticker.history(period="max")['Close'].dropna()
    price_of_stock.index = pd.to_datetime(price_of_stock.index)
    price_of_stock = price_of_stock.loc['2024-01-01':]

    # Fitting the ARIMA model
    model = ARIMA(price_of_stock, order=(0, 1, 0))
    fitted = model.fit()

    # Last date in the stock data
    last_date = price_of_stock.index[-1]

    # Creating a DateTime index for the next month
    next_month_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=period, freq='B')

    predictions = fitted.predict(0, len(next_month_index))
    predictions = predictions.iloc[1:]
    predictions.index = next_month_index
    diff = price_of_stock[-1] - predictions[0]
    predictions = predictions + diff

    # Plotting the predicted stock data
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(price_of_stock, label='Actual Stock Price')
    plt.plot(predictions, color='orange', label='Forecasted Period')
    plt.title(f'{symbol} Stock Price with Forecast Period')
    plt.xlabel('Time')
    plt.ylabel(f'{symbol} Stock Price')
    plt.legend(loc='upper left', fontsize=8)

    # Save the prediction plot to a BytesIO object
    buffer4 = BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    image_base64_8 = base64.b64encode(buffer4.getvalue()).decode('utf-8')
    plt.close()

    forecast_list = [[date.strftime('%Y-%m-%d'), price] for date, price in predictions.items()]

    return render_template('plot.html', 
                           stock_table=stock_table, 
                           forecast_list=forecast_list, 
                           image_base64=image_base64_8, 
                           info=info)


#Logout

@app.route('/logout')
def Logout():
    return render_template('index.html')

#explore_more 
@app.route('/explore_more')
def Explore():
    return render_template('explore_more.html')

@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    if 'stocks' not in cache:
        update_stock_data()
    print('Stocks in cache:', cache['stocks'])  # Add this line for debugging
    return jsonify(cache['stocks'])

# AAPL GRAPH 




@app.route('/AAPL')
def AAPL():
    # Data for the first plot
    years_revenue = [2019, 2020, 2021, 2022, 2023]
    revenue = [200, 220, 250, 280, 300]  # Values in billions
    net_income = [50, 55, 60, 70, 75]    # Values in billions

    # Bar width
    bar_width = 0.35

    # Positions of the bars on the x-axis
    r1 = np.arange(len(years_revenue))
    r2 = [x + bar_width for x in r1]

    # Create a bar chart for revenue and net income
    plt.figure(figsize=(10, 6))
    plt.bar(r1, revenue, color='blue', width=bar_width, label='Revenue')
    plt.bar(r2, net_income, color='orange', width=bar_width, label='Net income')

    # Add titles and labels for the first plot
    plt.title('Revenue and Net Income')
    plt.xlabel('Year')
    plt.ylabel('Amount (in billions)')
    plt.xticks([r + bar_width / 2 for r in range(len(years_revenue))], years_revenue)
    plt.legend()

    # Save the first plot to a BytesIO object
    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    image_base64_1 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    plt.close()

    # Data for the second plot 
    years_assets = ['2019', '2020', '2021', '2022', '2023']
    total_assets = [300, 280, 290, 310, 300]  # Example data, replace with actual values if known
    total_liabilities = [180, 200, 220, 240, 230]  # Example data, replace with actual values if known

    # Position of the bars on the x-axis for the second plot
    x = np.arange(len(years_assets))

    # Width of a bar for the second plot
    width = 0.35

    # Create the second plot
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, total_assets, width, label='Total assets', color='royalblue')
    bars2 = plt.bar(x + width/2, total_liabilities, width, label='Total liabilities', color='goldenrod')

    # Add some text for labels, title and axes ticks for the second plot
    plt.ylabel('Amount (in billions)')
    plt.title('Total assets and total liabilities by year')
    plt.xticks(x, years_assets)
    plt.legend()

    # Save the second plot to a BytesIO object
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    image_base64_2 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    plt.close()

    # Data for the third plot
    years = [2019, 2020, 2021, 2022, 2023]
    net_change_in_cash = [20, -10, -5, -10, 5]  # Values in billions

    # Create a bar chart for net change in cash
    plt.figure(figsize=(10, 6))
    plt.bar(years, net_change_in_cash, color='blue')

    # Add titles and labels for the third plot
    plt.title('Net Change in Cash')
    plt.xlabel('Year')
    plt.ylabel('Net Change in Cash (in billions)')
    plt.axhline(0, color='black', linewidth=0.5)

    # Save the third plot to a BytesIO object
    buffer3 = BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    image_base64_3 = base64.b64encode(buffer3.getvalue()).decode('utf-8')
    plt.close()

    # Fetch stock data for Apple Inc. (AAPL)
    ticker = 'AAPL'
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="7d")  # Fetch last week of data

    # Include additional details if available
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

    # Fetch additional company information
    info = stock_data.info

    # Convert the DataFrame to HTML
    stock_table = hist.to_html(classes='data', header="true", index=False)

    # Load stock data for ARIMA model
    price_of_stock = stock_data.history(period="max")['Close'].dropna()
    price_of_stock.index = pd.to_datetime(price_of_stock.index)
    price_of_stock = price_of_stock.loc['2024-01-01':]

    # Fitting the ARIMA model
    model = ARIMA(price_of_stock, order=(0, 1, 0))
    fitted = model.fit()

    # Last date in the stock data
    last_date = price_of_stock.index[-1]

    # Creating a DateTime index for the next month
    next_month_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

    predictions = fitted.predict(0, len(next_month_index))
    predictions = predictions.iloc[1:]
    predictions.index = next_month_index
    diff = price_of_stock[-1] - predictions[0]
    predictions = predictions + diff

    # Plotting the predicted stock data
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(price_of_stock, label='Actual Stock Price')
    plt.plot(predictions, color='orange', label='Forecasted Period')
    plt.title(f'{ticker} Stock Price with Forecast Period')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Stock Price')
    plt.legend(loc='upper left', fontsize=8)

    # Save the prediction plot to a BytesIO object
    buffer4 = BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    image_base64_4 = base64.b64encode(buffer4.getvalue()).decode('utf-8')
    plt.close()

    return render_template('AAPL.html', 
                           image1=image_base64_1, 
                           image2=image_base64_2, 
                           image3=image_base64_3, 
                           image4=image_base64_4, 
                           stock_table=stock_table, 
                           info=info)




# GOOGLE graphs


@app.route('/GOOGL')
def GOOGL():
    # Data for Total assets and liabilities over time plot
    years = ['2019', '2020', '2021', '2022', '2023']
    total_assets = [200, 300, 350, 375, 400]  # Example values in billions
    total_liabilities = [50, 75, 100, 125, 150]  # Example values in billions

    # Generate Total assets and liabilities over time plot
    x1 = np.arange(len(years))
    width = 0.35
    fig1, ax1 = plt.subplots()
    bars1 = ax1.bar(x1 - width/2, total_assets, width, label='Total assets', color='blue')
    bars2 = ax1.bar(x1 + width/2, total_liabilities, width, label='Total liabilities', color='orange')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Amount (in billions)')
    ax1.set_title('Total Assets and Total Liabilities over Years')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(years)
    ax1.legend()

    # Save Total assets and liabilities plot to a BytesIO object
    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    image_base64_5 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    plt.close(fig1)

    # Data for Revenue and Net Income over the years plot
    revenue = [150, 200, 250, 275, 300]  # Example values in billions
    net_income = [30, 40, 50, 60, 70]  # Example values in billions

    # Generate Revenue and Net Income over the years plot
    fig2, ax2 = plt.subplots()
    bars3 = ax2.bar(x1 - width/2, revenue, width, label='Revenue', color='blue')
    bars4 = ax2.bar(x1 + width/2, net_income, width, label='Net income', color='orange')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Amount (in billions)')
    ax2.set_title('Revenue and Net Income over Years')
    ax2.set_xticks(x1)
    ax2.set_xticklabels(years)
    ax2.legend()

    # Save Revenue and Net Income plot to a BytesIO object
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    image_base64_6 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    plt.close(fig2)

    # Data for Net Change in Cash over time plot
    net_change_in_cash = [1, 5, -3, 0.5, 2]  # Example values in billions

    # Generate Net Change in Cash over time plot
    fig3, ax3 = plt.subplots()
    bars5 = ax3.bar(x1, net_change_in_cash, width, label='Net change in cash', color='blue')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Amount (in billions)')
    ax3.set_title('Net Change in Cash over Years')
    ax3.set_xticks(x1)
    ax3.set_xticklabels(years)
    ax3.legend()

    # Save Net Change in Cash plot to a BytesIO object
    buffer3 = BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    image_base64_7 = base64.b64encode(buffer3.getvalue()).decode('utf-8')
    plt.close(fig3)

    # Fetch stock data for Google
    ticker = 'GOOG'
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="7d")  # Fetch last week of data

    # Include additional details if available
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

    # Fetch additional company information
    info = stock_data.info

    # Convert the DataFrame to HTML
    stock_table = hist.to_html(classes='data', header="true", index=False)

    # Load stock data for ARIMA model
    price_of_stock = stock_data.history(period="max")['Close'].dropna()
    price_of_stock.index = pd.to_datetime(price_of_stock.index)
    price_of_stock = price_of_stock.loc['2024-01-01':]

    # Fitting the ARIMA model
    model = ARIMA(price_of_stock, order=(0, 1, 0))
    fitted = model.fit()

    # Last date in the stock data
    last_date = price_of_stock.index[-1]

    # Creating a DateTime index for the next month
    next_month_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

    predictions = fitted.predict(0, len(next_month_index))
    predictions = predictions.iloc[1:]
    predictions.index = next_month_index
    diff = price_of_stock[-1] - predictions[0]
    predictions = predictions + diff

    # Plotting the predicted stock data
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(price_of_stock, label='Actual Stock Price')
    plt.plot(predictions, color='orange', label='Forecasted Period')
    plt.title(f'{ticker} Stock Price with Forecast Period')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Stock Price')
    plt.legend(loc='upper left', fontsize=8)

    # Save the prediction plot to a BytesIO object
    buffer4 = BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    image_base64_8 = base64.b64encode(buffer4.getvalue()).decode('utf-8')
    plt.close()

    # Render HTML template with plot images
    return render_template('GOOGL.html',
                           image_base64_5=image_base64_5,
                           image_base64_6=image_base64_6,
                           image_base64_7=image_base64_7,
                           image_base64_8=image_base64_8,
                           stock_table=stock_table,
                           info=info)




# MSFT graph



@app.route('/MSFT')
def MSFT():
    # Data for Revenue and Net Income plot
    years = ['2019', '2020', '2021', '2022', '2023']
    revenue = [100, 130, 160, 200, 180]  # Example data based on visual estimation
    net_income = [30, 40, 50, 60, 70]  # Example data based on visual estimation

    x = np.arange(len(years))  # the label locations
    width = 0.35  # the width of the bars

    # Plot 1: Revenue and Net Income
    fig1, ax1 = plt.subplots()
    bars1 = ax1.bar(x - width/2, revenue, width, label='Revenue')
    bars2 = ax1.bar(x + width/2, net_income, width, label='Net income')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Amount (in billions)')
    ax1.set_title('Company Revenue and Net Income by Year')
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.legend()

    # Save the plot to a BytesIO object
    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    image_base64_9 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    plt.close(fig1)

    # Data for Total Assets and Total Liabilities plot
    total_assets = [250, 280, 300, 350, 400]  # Example data based on visual estimation
    total_liabilities = [150, 180, 200, 220, 250]  # Example data based on visual estimation

    # Plot 2: Total Assets and Total Liabilities
    fig2, ax2 = plt.subplots()
    bars3 = ax2.bar(x - width/2, total_assets, width, label='Total assets')
    bars4 = ax2.bar(x + width/2, total_liabilities, width, label='Total liabilities')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Amount (in billions)')
    ax2.set_title('Company Total Assets and Total Liabilities by Year')
    ax2.set_xticks(x)
    ax2.set_xticklabels(years)
    ax2.legend()

    # Save the plot to a BytesIO object
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    image_base64_10 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    plt.close(fig2)

    # Data for Net Change in Cash plot
    net_change_in_cash = [-1, 2, 0.5, -0.1, 20]  # Example data based on visual estimation

    # Plot 3: Net Change in Cash
    fig3, ax3 = plt.subplots()
    bars5 = ax3.bar(x, net_change_in_cash, width, label='Net change in cash')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Amount (in billions)')
    ax3.set_title('Company Net Change in Cash by Year')
    ax3.set_xticks(x)
    ax3.set_xticklabels(years)
    ax3.legend()

    # Save the plot to a BytesIO object
    buffer3 = BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    image_base64_11 = base64.b64encode(buffer3.getvalue()).decode('utf-8')
    plt.close(fig3)
    
      # Fetch stock data for Microsoft 
    ticker = 'MSFT'
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="7d")  # Fetch last week of data

    # Include additional details if available
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

    # Fetch additional company information
    info = stock_data.info

    # Convert the DataFrame to HTML
    stock_table = hist.to_html(classes='data', header="true", index=False)

    # Load stock data for ARIMA model
    price_of_stock = stock_data.history(period="max")['Close'].dropna()
    price_of_stock.index = pd.to_datetime(price_of_stock.index)
    price_of_stock = price_of_stock.loc['2024-01-01':]

    # Fitting the ARIMA model
    model = ARIMA(price_of_stock, order=(0, 1, 0))
    fitted = model.fit()

    # Last date in the stock data
    last_date = price_of_stock.index[-1]

    # Creating a DateTime index for the next month
    next_month_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

    predictions = fitted.predict(0, len(next_month_index))
    predictions = predictions.iloc[1:]
    predictions.index = next_month_index
    diff = price_of_stock[-1] - predictions[0]
    predictions = predictions + diff

    # Plotting the predicted stock data
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(price_of_stock, label='Actual Stock Price')
    plt.plot(predictions, color='orange', label='Forecasted Period')
    plt.title(f'{ticker} Stock Price with Forecast Period')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Stock Price')
    plt.legend(loc='upper left', fontsize=8)

    # Save the prediction plot to a BytesIO object
    buffer4 = BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    image_base64_12 = base64.b64encode(buffer4.getvalue()).decode('utf-8')
    plt.close()


    # Render HTML template with plot images
    return render_template('MSFT.html', 
                           image_base64_9=image_base64_9,
                           image_base64_10=image_base64_10,
                           image_base64_11=image_base64_11,
                           image_base64_12=image_base64_12,
                           stock_table=stock_table,
                           info=info)



# TSLA graph



@app.route('/TSLA')
def TSLA():
    # Data for Revenue and Net Income plot
    years = [2019, 2020, 2021, 2022, 2023]
    revenue = [10, 20, 30, 45, 50]  # Example data for revenue in billions
    net_income = [1, 2, 3, 4, 5]  # Example data for net income in billions

    # Bar width and positions
    bar_width = 0.35
    r1 = np.arange(len(years))
    r2 = [x + bar_width for x in r1]

    # Plot 1: Revenue and Net Income
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(r1, revenue, color='blue', width=bar_width, edgecolor='grey', label='Revenue')
    ax1.bar(r2, net_income, color='orange', width=bar_width, edgecolor='grey', label='Net income')
    ax1.set_xlabel('Year', fontweight='bold')
    ax1.set_ylabel('Billions (B)', fontweight='bold')
    ax1.set_title('Revenue and Net Income over Years', fontweight='bold')
    ax1.set_xticks([r + bar_width/2 for r in range(len(years))])
    ax1.set_xticklabels(years)
    ax1.legend()

    # Save plot 1 to a BytesIO object
    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    image_base64_13 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    plt.close(fig1)

    # Data for Total Assets and Total Liabilities plot
    total_assets = [30, 50, 45, 60, 100]
    total_liabilities = [20, 25, 25, 40, 50]

    # Plot 2: Total Assets and Total Liabilities
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars1 = ax2.bar(years, total_assets, width=bar_width, label='Total assets', color='blue')
    bars2 = ax2.bar(years, total_liabilities, width=bar_width, label='Total liabilities', color='orange')
    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_ylabel('Amount (in billions)', fontweight='bold')
    ax2.set_title('Total Assets and Liabilities by Year', fontweight='bold')
    ax2.legend()

    # Annotate bars with values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate('{}'.format(height),
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

    # Save plot 2 to a BytesIO object
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    image_base64_14 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    plt.close(fig2)

    # Data for Net Change in Cash plot
    net_change_in_cash = [1.5, 10, -2, -1, 0.5]  # Example data for net change in cash

    # Plot 3: Net Change in Cash
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.bar(years, net_change_in_cash, color='blue')
    ax3.set_xlabel('Year', fontweight='bold')
    ax3.set_ylabel('Net Change in Cash (in Billions)', fontweight='bold')
    ax3.set_title('Net Change in Cash by Year', fontweight='bold')

    # Save plot 3 to a BytesIO object
    buffer3 = BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    image_base64_15 = base64.b64encode(buffer3.getvalue()).decode('utf-8')
    plt.close(fig3)
    
      # Fetch stock data for Tesla 
    ticker = 'TSLA'
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="7d")  # Fetch last week of data

    # Include additional details if available
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

    # Fetch additional company information
    info = stock_data.info

    # Convert the DataFrame to HTML
    stock_table = hist.to_html(classes='data', header="true", index=False)

    # Load stock data for ARIMA model
    price_of_stock = stock_data.history(period="max")['Close'].dropna()
    price_of_stock.index = pd.to_datetime(price_of_stock.index)
    price_of_stock = price_of_stock.loc['2024-01-01':]

    # Fitting the ARIMA model
    model = ARIMA(price_of_stock, order=(0, 1, 0))
    fitted = model.fit()

    # Last date in the stock data
    last_date = price_of_stock.index[-1]

    # Creating a DateTime index for the next month
    next_month_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

    predictions = fitted.predict(0, len(next_month_index))
    predictions = predictions.iloc[1:]
    predictions.index = next_month_index
    diff = price_of_stock[-1] - predictions[0]
    predictions = predictions + diff

    # Plotting the predicted stock data
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(price_of_stock, label='Actual Stock Price')
    plt.plot(predictions, color='orange', label='Forecasted Period')
    plt.title(f'{ticker} Stock Price with Forecast Period')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Stock Price')
    plt.legend(loc='upper left', fontsize=8)

    # Save the prediction plot to a BytesIO object
    buffer4 = BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    image_base64_16 = base64.b64encode(buffer4.getvalue()).decode('utf-8')
    plt.close()

    # Render HTML template with plot images
    return render_template('TSLA.html', 
                           image_base64_13=image_base64_13,
                           image_base64_14=image_base64_14,
                           image_base64_15=image_base64_15,
                           image_base64_16=image_base64_16,
                           stock_table=stock_table,
                           info=info)



#Bajaj Finance Graph 


@app.route('/BAJFINANCE.NS')
def BAJFINANCE():
    # Data for the first plot
    years_revenue = [2020, 2021, 2022, 2023, 2024]
    revenue = [100, 50, 150, 250, 300]  # Assumed values in billions
    net_income = [50, 25, 75, 125, 150]  # Assumed values in billions

    # Bar width
    bar_width = 0.35

    # Positions of the bars on the x-axis
    r1 = np.arange(len(years_revenue))
    r2 = [x + bar_width for x in r1]

    # Create a bar chart for revenue and net income
    plt.figure(figsize=(10, 6))
    plt.bar(r1, revenue, color='blue', width=bar_width, label='Revenue')
    plt.bar(r2, net_income, color='orange', width=bar_width, label='Net income')

    # Add titles and labels for the first plot
    plt.title('Revenue and Net Income')
    plt.xlabel('Year')
    plt.ylabel('Amount (in billions)')
    plt.xticks([r + bar_width / 2 for r in range(len(years_revenue))], years_revenue)
    plt.legend()

    # Save the first plot to a BytesIO object
    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    image_base64_17 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    plt.close()

    # Data for the second plot 
    years_assets = ['2020', '2021', '2022', '2023', '2024']
    total_assets = [1, 1.1, 1.2, 2.5, 3]  # Assumed values in trillions
    total_liabilities = [0.7, 0.8, 0.9, 2, 2.5]  # Assumed values in trillions

    # Position of the bars on the x-axis for the second plot
    x = np.arange(len(years_assets))

    # Width of a bar for the second plot
    width = 0.35

    # Create the second plot
    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width/2, total_assets, width, label='Total assets', color='royalblue')
    bars2 = plt.bar(x + width/2, total_liabilities, width, label='Total liabilities', color='goldenrod')

    # Add some text for labels, title and axes ticks for the second plot
    plt.ylabel('Amount (in trillions)')
    plt.title('Total assets and total liabilities by year')
    plt.xticks(x, years_assets)
    plt.legend()

    # Save the second plot to a BytesIO object
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    image_base64_18 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    plt.close()

    # Data for the third plot
    years = [2020, 2021, 2022, 2023, 2024]
    net_change_in_cash = [5, 2, 10, -10, 20]  # Example values for net change in cash in billions

    # Create a bar chart for net change in cash
    plt.figure(figsize=(10, 6))
    plt.bar(years, net_change_in_cash, color='blue')

    # Add titles and labels for the third plot
    plt.title('Net Change in Cash')
    plt.xlabel('Year')
    plt.ylabel('Net Change in Cash (in billions)')
    plt.axhline(0, color='black', linewidth=0.5)

    # Save the third plot to a BytesIO object
    buffer3 = BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    image_base64_19 = base64.b64encode(buffer3.getvalue()).decode('utf-8')
    plt.close()

    # Fetch stock data for Bajaj Finance (BAJFINANCE)
    ticker = 'BAJFINANCE.NS'
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="7d")  # Fetch last week of data

    # Include additional details if available
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

    # Fetch additional company information
    info = stock_data.info

    # Convert the DataFrame to HTML
    stock_table = hist.to_html(classes='data', header="true", index=False)

    # Load stock data for ARIMA model
    price_of_stock = stock_data.history(period="max")['Close'].dropna()
    price_of_stock.index = pd.to_datetime(price_of_stock.index)
    price_of_stock = price_of_stock.loc['2024-01-01':]

    # Fitting the ARIMA model
    model = ARIMA(price_of_stock, order=(0, 1, 0))
    fitted = model.fit()

    # Last date in the stock data
    last_date = price_of_stock.index[-1]

    # Creating a DateTime index for the next month
    next_month_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

    predictions = fitted.predict(0, len(next_month_index))
    predictions = predictions.iloc[1:]
    predictions.index = next_month_index
    diff = price_of_stock[-1] - predictions[0]
    predictions = predictions + diff

    # Plotting the predicted stock data
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(price_of_stock, label='Actual Stock Price')
    plt.plot(predictions, color='orange', label='Forecasted Period')
    plt.title(f'{ticker} Stock Price with Forecast Period')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Stock Price')
    plt.legend(loc='upper left', fontsize=8)

    # Save the prediction plot to a BytesIO object
    buffer4 = BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    image_base64_20 = base64.b64encode(buffer4.getvalue()).decode('utf-8')
    plt.close()

    return render_template('BAJFINANCE.NS.html', 
                           image_base64_17=image_base64_17, 
                           image_base64_18=image_base64_18, 
                           image_base64_19=image_base64_19, 
                           image_base64_20=image_base64_20, 
                           stock_table=stock_table, 
                           info=info)



# TCS Graph



@app.route('/TCS')
def TCS():
    # Data for Total assets and liabilities over time plot
    years = np.array([2020, 2021, 2022, 2023, 2024])
    total_assets = np.array([1.0, 1.2, 1.4, 1.6, 1.8])  # Values in trillions (T)
    total_liabilities = np.array([0.4, 0.45, 0.5, 0.55, 0.6])  # Values in trillions (T)

    # Generate Total assets and liabilities over time plot
    bar_width = 0.35
    index = np.arange(len(years))
    fig1, ax1 = plt.subplots()
    bars1 = ax1.bar(index, total_assets, bar_width, color='blue', label='Total assets')
    bars2 = ax1.bar(index + bar_width, total_liabilities, bar_width, color='orange', label='Total liabilities')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Amount (T)')
    ax1.set_title('Total Assets and Total Liabilities Over Years')
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(years)
    ax1.legend()

    # Save Total assets and liabilities plot to a BytesIO object
    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    image_base64_21 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    plt.close(fig1)

    # Data for Revenue and Net Income over the years plot
    revenue = np.array([1.2, 1.4, 1.6, 1.8, 2.0])  # Values in trillions (T)
    net_income = np.array([0.3, 0.35, 0.4, 0.45, 0.5])  # Values in trillions (T)

    # Generate Revenue and Net Income over the years plot
    fig2, ax2 = plt.subplots()
    bars3 = ax2.bar(index, revenue, bar_width, color='blue', label='Revenue')
    bars4 = ax2.bar(index + bar_width, net_income, bar_width, color='orange', label='Net income')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Amount (T)')
    ax2.set_title('Revenue and Net Income Over Years')
    ax2.set_xticks(index + bar_width / 2)
    ax2.set_xticklabels(years)
    ax2.legend()

    # Save Revenue and Net Income plot to a BytesIO object
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    image_base64_22 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    plt.close(fig2)

    # Data for Net Change in Cash over time plot
    net_change_in_cash = [15, -25, 45, -35, 20]  # Example values in billions

    # Generate Net Change in Cash over time plot
    fig3, ax3 = plt.subplots()
    bars5 = ax3.bar(years, net_change_in_cash, color='blue')
    for bar in bars5:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Net change in cash (B)')
    ax3.set_title('Net Change in Cash Over Years')
    ax3.set_ylim(-50, 50)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    # Save Net Change in Cash plot to a BytesIO object
    buffer3 = BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    image_base64_23 = base64.b64encode(buffer3.getvalue()).decode('utf-8')
    plt.close(fig3)

    # Fetch stock data for TCS
    ticker = 'TCS'
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="7d")  # Fetch last week of data

    # Include additional details if available
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

    # Fetch additional company information
    info = stock_data.info

    # Convert the DataFrame to HTML
    stock_table = hist.to_html(classes='data', header="true", index=False)

    # Load stock data for ARIMA model
    price_of_stock = stock_data.history(period="max")['Close'].dropna()
    price_of_stock.index = pd.to_datetime(price_of_stock.index)
    price_of_stock = price_of_stock.loc['2024-01-01':]

    # Fitting the ARIMA model
    model = ARIMA(price_of_stock, order=(0, 1, 0))
    fitted = model.fit()

    # Last date in the stock data
    last_date = price_of_stock.index[-1]

    # Creating a DateTime index for the next month
    next_month_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

    predictions = fitted.predict(0, len(next_month_index))
    predictions = predictions.iloc[1:]
    predictions.index = next_month_index
    diff = price_of_stock[-1] - predictions[0]
    predictions = predictions + diff

    # Plotting the predicted stock data
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(price_of_stock, label='Actual Stock Price')
    plt.plot(predictions, color='orange', label='Forecasted Period')
    plt.title(f'{ticker} Stock Price with Forecast Period')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Stock Price')
    plt.legend(loc='upper left', fontsize=8)

    # Save the prediction plot to a BytesIO object
    buffer4 = BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    image_base64_24 = base64.b64encode(buffer4.getvalue()).decode('utf-8')
    plt.close()

    # Render HTML template with plot images
    return render_template('TCS.html',
                           image_base64_21=image_base64_21,
                           image_base64_22=image_base64_22,
                           image_base64_23=image_base64_23,
                           image_base64_24=image_base64_24,
                           stock_table=stock_table,
                           info=info)


# BX Graph




@app.route('/BX')
def BX():
    # Data for Annual Income Statement plot
    years = [2019, 2020, 2021, 2022, 2023]
    revenue = [10, 5, 20, 10, 10]  # Sample values based on the pattern observed
    net_income = [2, 1, 8, 3, 2]  # Sample values based on the pattern observed

    # Generate Annual Income Statement plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(years))
    bars1 = ax1.bar(x - width / 2, revenue, width, label='Revenue', color='blue')
    bars2 = ax1.bar(x + width / 2, net_income, width, label='Net income', color='orange')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Billion Dollars')
    ax1.set_title('Revenue and Net Income Over Years')
    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.legend()
    ax1.axhline(0, color='black', linewidth=0.5)
    for bar in bars1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom' if yval < 0 else 'top', ha='center')
    for bar in bars2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom' if yval < 0 else 'top', ha='center')
    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    image_base64_25 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    plt.close(fig1)

    # Data for Annual Balance Sheet plot
    total_assets = np.array([30, 20, 35, 40, 40])
    total_liabilities = np.array([20, 10, 25, 25, 25])

    # Generate Annual Balance Sheet plot
    fig2, ax2 = plt.subplots()
    bar_width = 0.35
    index = np.arange(len(years))
    bar1 = ax2.bar(index, total_assets, bar_width, label='Total assets', color='blue')
    bar2 = ax2.bar(index + bar_width, total_liabilities, bar_width, label='Total liabilities', color='orange')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Amount (in Billions)')
    ax2.set_title('Total Assets and Liabilities (2019-2023)')
    ax2.set_xticks(index + bar_width / 2)
    ax2.set_xticklabels(years)
    ax2.legend()
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    image_base64_26 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    plt.close(fig2)

    # Data for Annual Cash Flow plot
    net_change_in_cash = np.array([0, -0.5, 0.2, 2, -1])

    # Generate Annual Cash Flow plot
    fig3, ax3 = plt.subplots()
    bars = ax3.bar(index, net_change_in_cash, bar_width, label='Net change in cash', color='blue')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Amount (in Billions)')
    ax3.set_title('Net Change in Cash (2019-2023)')
    ax3.set_xticks(index)
    ax3.set_xticklabels(years)
    ax3.legend()
    buffer3 = BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    image_base64_27 = base64.b64encode(buffer3.getvalue()).decode('utf-8')
    plt.close(fig3)

    # Fetch stock data for Blackstone
    ticker = 'BX'
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="7d")  # Fetch last week of data

    # Include additional details if available
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

    # Fetch additional company information
    info = stock_data.info

    # Convert the DataFrame to HTML
    stock_table = hist.to_html(classes='data', header="true", index=False)

    # Load stock data for ARIMA model
    price_of_stock = stock_data.history(period="max")['Close'].dropna()
    price_of_stock.index = pd.to_datetime(price_of_stock.index)
    price_of_stock = price_of_stock.loc['2024-01-01':]

    # Fitting the ARIMA model
    model = ARIMA(price_of_stock, order=(0, 1, 0))
    fitted = model.fit()

    # Last date in the stock data
    last_date = price_of_stock.index[-1]

    # Creating a DateTime index for the next month
    next_month_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

    predictions = fitted.predict(0, len(next_month_index))
    predictions = predictions.iloc[1:]
    predictions.index = next_month_index
    diff = price_of_stock[-1] - predictions[0]
    predictions = predictions + diff

    # Plotting the predicted stock data
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(price_of_stock, label='Actual Stock Price')
    plt.plot(predictions, color='orange', label='Forecasted Period')
    plt.title(f'{ticker} Stock Price with Forecast Period')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Stock Price')
    plt.legend(loc='upper left', fontsize=8)

    # Save the prediction plot to a BytesIO object
    buffer4 = BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    image_base64_28 = base64.b64encode(buffer4.getvalue()).decode('utf-8')
    plt.close()

    # Render HTML template with plot images
    return render_template('BX.html',
                           image_base64_25=image_base64_25,
                           image_base64_26=image_base64_26,
                           image_base64_27=image_base64_27,
                           image_base64_28=image_base64_28,
                           stock_table=stock_table,
                           info=info)


# HBD Graph 


@app.route('/HDB')
def HDB():
    # Plot 1: Revenue and Net Income (2020-2024)
    years = ['2020', '2021', '2022', '2023', '2024']
    revenue = [0.5, 0.6, 0.8, 0.9, 2.0]  # Example values for revenue in trillions
    net_income = [0.1, 0.2, 0.3, 0.4, 0.6]  # Example values for net income in trillions

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(years))

    bar1 = ax1.bar(index, revenue, bar_width, label='Revenue', color='blue')
    bar2 = ax1.bar(index + bar_width, net_income, bar_width, label='Net income', color='orange')

    ax1.set_title('Revenue and Net Income (2020-2024)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Amount (in trillions)')
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(years)
    ax1.axhline(0, color='black', linewidth=0.8)
    for bar in bar1 + bar2:
        height = bar.get_height()
        ax1.annotate(f'{height}T', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom')
    ax1.legend()

    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    image_base64_29 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    plt.close(fig1)

    # Plot 2: Total Assets and Total Liabilities (2020-2024)
    total_assets = [15, 17, 20, 25, 35]  # Example values for total assets in trillions
    total_liabilities = [10, 12, 15, 20, 30]  # Example values for total liabilities in trillions

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bar1 = ax2.bar(index, total_assets, bar_width, label='Total assets', color='blue')
    bar2 = ax2.bar(index + bar_width, total_liabilities, bar_width, label='Total liabilities', color='orange')

    ax2.set_title('Total Assets and Total Liabilities (2020-2024)')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Amount (in trillions)')
    ax2.set_xticks(index + bar_width / 2)
    ax2.set_xticklabels(years)
    ax2.axhline(0, color='black', linewidth=0.8)
    for bar in bar1 + bar2:
        height = bar.get_height()
        ax2.annotate(f'{height}T', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom')
    ax2.legend()

    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    image_base64_30 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    plt.close(fig2)

    # Plot 3: Net Change in Cash (2020-2024)
    net_change_in_cash = np.array([50, 300, 350, 400, 320])  # Values in billions (B)

    fig3, ax3 = plt.subplots()
    bars = ax3.bar(years, net_change_in_cash, color='blue', label='Net change in cash')

    ax3.set_xlabel('Year')
    ax3.set_ylabel('Net change in cash (B)')
    ax3.set_title('Net Change in Cash Over Years')
    ax3.set_xticks(years)
    ax3.legend()

    buffer3 = BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    image_base64_31 = base64.b64encode(buffer3.getvalue()).decode('utf-8')
    plt.close(fig3)
    
    # Fetch stock data for HDB
    ticker = 'HDB'
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="7d")  # Fetch last week of data

    # Include additional details if available
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

    # Fetch additional company information
    info = stock_data.info

    # Convert the DataFrame to HTML
    stock_table = hist.to_html(classes='data', header="true", index=False)

    # Load stock data for ARIMA model
    price_of_stock = stock_data.history(period="max")['Close'].dropna()
    price_of_stock.index = pd.to_datetime(price_of_stock.index)
    price_of_stock = price_of_stock.loc['2024-01-01':]

    # Fitting the ARIMA model
    model = ARIMA(price_of_stock, order=(0, 1, 0))
    fitted = model.fit()

    # Last date in the stock data
    last_date = price_of_stock.index[-1]

    # Creating a DateTime index for the next month
    next_month_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

    predictions = fitted.predict(0, len(next_month_index))
    predictions = predictions.iloc[1:]
    predictions.index = next_month_index
    diff = price_of_stock[-1] - predictions[0]
    predictions = predictions + diff

    # Plotting the predicted stock data
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(price_of_stock, label='Actual Stock Price')
    plt.plot(predictions, color='orange', label='Forecasted Period')
    plt.title(f'{ticker} Stock Price with Forecast Period')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Stock Price')
    plt.legend(loc='upper left', fontsize=8)

    # Save the prediction plot to a BytesIO object
    buffer4 = BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    image_base64_32 = base64.b64encode(buffer4.getvalue()).decode('utf-8')
    plt.close()
    
    


    # Render HTML template with plot images
    return render_template('HDB.html',
                           image_base64_29=image_base64_29,
                           image_base64_30=image_base64_30,
                           image_base64_31=image_base64_31,
                           image_base64_32=image_base64_32,
                           stock_table=stock_table,
                           info=info)



# AMZN Graph



@app.route('/AMZN')
def AMZN():
    # Data for Annual Income Statement
    years = ['2019', '2020', '2021', '2022', '2023']
    revenue = [150, 200, 250, 300, 350]  # Example revenue data in billions
    net_income = [10, 20, 25, 30, 5]  # Example net income data in billions

    # Create Annual Income Statement plot
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    bar_width = 0.4
    r1 = np.arange(len(years))
    r2 = [x + bar_width for x in r1]

    ax1.bar(r1, revenue, color='blue', width=bar_width, edgecolor='grey', label='Revenue')
    ax1.bar(r2, net_income, color='orange', width=bar_width, edgecolor='grey', label='Net Income')

    ax1.set_xlabel('Year', fontweight='bold')
    ax1.set_ylabel('Amount (in billions)', fontweight='bold')
    ax1.set_xticks([r + bar_width/2 for r in range(len(years))])
    ax1.set_xticklabels(years)
    ax1.set_yticks(np.arange(0, 401, 100))
    ax1.set_yticklabels(['0', '100B', '200B', '300B', '400B'])
    ax1.legend()

    # Convert plot to base64 encoding
    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    image_base64_33 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    plt.close(fig1)

    # Data for Annual Balance Sheet
    total_assets = [100, 150, 200, 250, 300]  # Example total assets data in billions
    total_liabilities = [50, 100, 150, 200, 250]  # Example total liabilities data in billions

    # Create Annual Balance Sheet plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.bar(r1, total_assets, color='blue', width=bar_width, edgecolor='grey', label='Total assets')
    ax2.bar(r2, total_liabilities, color='orange', width=bar_width, edgecolor='grey', label='Total liabilities')

    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_ylabel('Amount (in billions)', fontweight='bold')
    ax2.set_xticks([r + bar_width/2 for r in range(len(years))])
    ax2.set_xticklabels(years)
    ax2.set_yticks(np.arange(0, 401, 100))
    ax2.set_yticklabels(['0', '100B', '200B', '300B', '400B'])
    ax2.legend()

    # Convert plot to base64 encoding
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    image_base64_34 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    plt.close(fig2)

    # Data for Annual Cash Flow
    net_change_in_cash = [2, 4, -1, 8, 9]  # Example net change in cash data in billions

    # Create Annual Cash Flow plot
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.bar(r1, net_change_in_cash, color='blue', edgecolor='grey', label='Net change in cash')

    ax3.set_xlabel('Year', fontweight='bold')
    ax3.set_ylabel('Amount (in billions)', fontweight='bold')
    ax3.set_xticks(r1)
    ax3.set_xticklabels(years)
    ax3.set_yticks(np.arange(0, 11, 2))
    ax3.set_yticklabels(['0', '2B', '4B', '6B', '8B', '10B'])
    ax3.legend()

    # Convert plot to base64 encoding
    buffer3 = BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    image_base64_35 = base64.b64encode(buffer3.getvalue()).decode('utf-8')
    plt.close(fig3)
    
       # Fetch stock data for AMZN
    ticker = 'AMZN'
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="7d")  # Fetch last week of data

    # Include additional details if available
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

    # Fetch additional company information
    info = stock_data.info

    # Convert the DataFrame to HTML
    stock_table = hist.to_html(classes='data', header="true", index=False)

    # Load stock data for ARIMA model
    price_of_stock = stock_data.history(period="max")['Close'].dropna()
    price_of_stock.index = pd.to_datetime(price_of_stock.index)
    price_of_stock = price_of_stock.loc['2024-01-01':]

    # Fitting the ARIMA model
    model = ARIMA(price_of_stock, order=(0, 1, 0))
    fitted = model.fit()

    # Last date in the stock data
    last_date = price_of_stock.index[-1]

    # Creating a DateTime index for the next month
    next_month_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

    predictions = fitted.predict(0, len(next_month_index))
    predictions = predictions.iloc[1:]
    predictions.index = next_month_index
    diff = price_of_stock[-1] - predictions[0]
    predictions = predictions + diff

    # Plotting the predicted stock data
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(price_of_stock, label='Actual Stock Price')
    plt.plot(predictions, color='orange', label='Forecasted Period')
    plt.title(f'{ticker} Stock Price with Forecast Period')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Stock Price')
    plt.legend(loc='upper left', fontsize=8)

    # Save the prediction plot to a BytesIO object
    buffer4 = BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    image_base64_36 = base64.b64encode(buffer4.getvalue()).decode('utf-8')
    plt.close()
    

    # Render HTML template with embedded plots
    return render_template('AMZN.html', image_base64_33=image_base64_33,
                                              image_base64_34=image_base64_34,
                                              image_base64_35=image_base64_35,
                                              image_base64_36=image_base64_36,
                                              stock_table=stock_table,
                                              info=info)



# Meta Graph


@app.route('/META')
def META():
    # Data for Annual Income Statement
    years_income = ['2019', '2020', '2021', '2022', '2023']
    revenue = [50, 70, 90, 80, 100]
    net_income = [10, 20, 30, 20, 40]

    # Create Annual Income Statement plot
    fig1, ax1 = plt.subplots()
    width = 0.35
    x1 = np.arange(len(years_income))
    bars1 = ax1.bar(x1 - width/2, revenue, width, label='Revenue', color='blue')
    bars2 = ax1.bar(x1 + width/2, net_income, width, label='Net income', color='orange')

    ax1.set_ylabel('Amount')
    ax1.set_title('Revenue and Net Income by Year')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(years_income)
    ax1.legend()

    def autolabel(bars):
        """Attach a text label above each bar in bars, displaying its height."""
        for bar in bars:
            height = bar.get_height()
            ax1.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)

    fig1.tight_layout()

    # Convert plot to base64 encoding
    buffer1 = BytesIO()
    plt.savefig(buffer1, format='png')
    buffer1.seek(0)
    image_base64_37 = base64.b64encode(buffer1.getvalue()).decode('utf-8')
    plt.close(fig1)

    # Data for Annual Balance Sheet
    years_balance = ['2019', '2020', '2021', '2022', '2023']
    total_assets = [100, 110, 120, 130, 200]
    total_liabilities = [40, 50, 60, 70, 100]

    # Create Annual Balance Sheet plot
    fig2, ax2 = plt.subplots()
    x2 = np.arange(len(years_balance))
    bars1 = ax2.bar(x2 - width/2, total_assets, width, label='Total assets', color='blue')
    bars2 = ax2.bar(x2 + width/2, total_liabilities, width, label='Total liabilities', color='orange')

    ax2.set_ylabel('Amount (in B)')
    ax2.set_title('Total Assets and Total Liabilities by Year')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(years_balance)
    ax2.legend()

    autolabel(bars1)
    autolabel(bars2)

    fig2.tight_layout()

    # Convert plot to base64 encoding
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png')
    buffer2.seek(0)
    image_base64_38 = base64.b64encode(buffer2.getvalue()).decode('utf-8')
    plt.close(fig2)

    # Data for Annual Cash Flow
    years_cash_flow = ['2019', '2020', '2021', '2022', '2023']
    net_change_in_cash = [3, -0.5, -0.5, -0.5, 20]

    # Create Annual Cash Flow plot
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.bar(years_cash_flow, net_change_in_cash, color='blue', label='Net change in cash')

    ax3.set_xlabel('Year')
    ax3.set_ylabel('Amount (in B)')
    ax3.set_title('Net Change in Cash')
    ax3.legend(loc='lower left')
    ax3.grid(axis='y', linestyle='--', alpha=0.7)

    fig3.tight_layout()

    # Convert plot to base64 encoding
    buffer3 = BytesIO()
    plt.savefig(buffer3, format='png')
    buffer3.seek(0)
    image_base64_39 = base64.b64encode(buffer3.getvalue()).decode('utf-8')
    plt.close(fig3)
    
    # Fetch stock data for Meta
    ticker = 'META'
    stock_data = yf.Ticker(ticker)
    hist = stock_data.history(period="7d")  # Fetch last week of data

    # Include additional details if available
    hist.reset_index(inplace=True)
    hist['Date'] = hist['Date'].dt.strftime('%Y-%m-%d')
    hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

    # Fetch additional company information
    info = stock_data.info

    # Convert the DataFrame to HTML
    stock_table = hist.to_html(classes='data', header="true", index=False)

    # Load stock data for ARIMA model
    price_of_stock = stock_data.history(period="max")['Close'].dropna()
    price_of_stock.index = pd.to_datetime(price_of_stock.index)
    price_of_stock = price_of_stock.loc['2024-01-01':]

    # Fitting the ARIMA model
    model = ARIMA(price_of_stock, order=(0, 1, 0))
    fitted = model.fit()

    # Last date in the stock data
    last_date = price_of_stock.index[-1]

    # Creating a DateTime index for the next month
    next_month_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='B')

    predictions = fitted.predict(0, len(next_month_index))
    predictions = predictions.iloc[1:]
    predictions.index = next_month_index
    diff = price_of_stock[-1] - predictions[0]
    predictions = predictions + diff

    # Plotting the predicted stock data
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(price_of_stock, label='Actual Stock Price')
    plt.plot(predictions, color='orange', label='Forecasted Period')
    plt.title(f'{ticker} Stock Price with Forecast Period')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Stock Price')
    plt.legend(loc='upper left', fontsize=8)

    # Save the prediction plot to a BytesIO object
    buffer4 = BytesIO()
    plt.savefig(buffer4, format='png')
    buffer4.seek(0)
    image_base64_40 = base64.b64encode(buffer4.getvalue()).decode('utf-8')
    plt.close()
    

    # Render HTML template with embedded plots
    return render_template('META.html',
                           image_base64_37=image_base64_37,
                           image_base64_38=image_base64_38,
                           image_base64_39=image_base64_39,
                           image_base64_40=image_base64_40,
                           stock_table=stock_table,
                           info=info)

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(update_stock_data, 'interval', minutes=5)
    scheduler.start()
    update_stock_data()  # Initial load
    try:
        app.run(debug=True)
    finally:
        scheduler.shutdown()

app.run(debug=True)


