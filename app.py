from flask import Flask, render_template, request, redirect, url_for, session, flash, g, jsonify
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import plotly
import plotly.graph_objects as go
import json
import sqlite3
import hashlib
import os
from functools import wraps
import time
from tenacity import retry, stop_after_attempt, wait_exponential

app = Flask(__name__)
app.secret_key = "your_secret_key_123"
app.config['DATABASE'] = 'stocksense.db'
app.config['PREDICTION_CACHE_DAYS'] = 3
app.config['THROTTLE_DELAY'] = 0.5

# Global variable for request throttling
LAST_REQUEST_TIME = 0

# Default stock symbols
DEFAULT_STOCKS = {
    'Indian Stocks': {
        'RELIANCE.NS': 'Reliance Industries',
        'TCS.NS': 'Tata Consultancy Services',
        'HDFCBANK.NS': 'HDFC Bank',
        'INFY.NS': 'Infosys',
        'BHARTIARTL.NS': 'Bharti Airtel'
    },
    'US Stocks': {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft',
        'GOOGL': 'Alphabet (Google)',
        'AMZN': 'Amazon',
        'TSLA': 'Tesla'
    }
}

CURRENCY_SYMBOLS = {
    'INR': '₹',
    'USD': '$',
    'KRW': '₩',
    'JPY': '¥',
    'GBP': '£'
}

# Database initialization
def init_db():
    with app.app_context():
        db = get_db()
        db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        db.execute("""
        CREATE TABLE IF NOT EXISTS predictions_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            predictions TEXT NOT NULL,
            company_info TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, start_date, end_date)
        );
        """)
        db.commit()

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Initialize database if it doesn't exist
if not os.path.exists(app.config['DATABASE']):
    init_db()

# Template filters
@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d'):
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.strptime(value, '%Y-%m-%d')
        except ValueError:
            return value
    return value.strftime(format)

@app.template_filter('format_number')
def format_number(value):
    try:
        if value is None:
            return "0"
        value = float(value)
        if abs(value) >= 1_000_000_000:
            return f"{value/1_000_000_000:.2f}B"
        elif abs(value) >= 1_000_000:
            return f"{value/1_000_000:.2f}M"
        elif abs(value) >= 1_000:
            return f"{value:,.0f}"
        return f"{value:,.2f}"
    except (ValueError, TypeError):
        return "0"

# Request throttling decorator
def throttle_requests(min_interval=1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            global LAST_REQUEST_TIME
            elapsed = time.time() - LAST_REQUEST_TIME
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            result = func(*args, **kwargs)
            LAST_REQUEST_TIME = time.time()
            return result
        return wrapper
    return decorator

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# Stock validation
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def is_valid_ticker(ticker):
    if not ticker or ticker.upper() == 'NONE':
        return False
    try:
        stock = yf.Ticker(ticker)
        if not stock.info.get('symbol'):
            return False
        return True
    except Exception:
        return False

# Data functions
@throttle_requests(min_interval=0.5)
def fetch_stock_data(ticker, period):
    try:
        data = yf.download(ticker, period=period, progress=False)
        return data
    except Exception as e:
        app.logger.error(f"Download failed for {ticker}: {str(e)}")
        return pd.DataFrame()

def get_company_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'symbol': ticker,
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'marketCap': info.get('marketCap'),
            'currency': info.get('currency', 'USD')
        }
    except Exception as e:
        app.logger.warning(f"Couldn't fetch company info for {ticker}: {str(e)}")
        return {
            'name': ticker,
            'symbol': ticker,
            'currency': 'USD'
        }

def get_currency_symbol(currency_code):
    return CURRENCY_SYMBOLS.get(currency_code, '$')

# Model functions
def prepare_training_data(data, sequence_length=30, prediction_days=7):
    scaler = MinMaxScaler()
    prices = data[['Open', 'Close']].dropna()
    scaled_data = scaler.fit_transform(prices)
    X, y = [], []
    for i in range(sequence_length, len(scaled_data) - prediction_days):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i:i+prediction_days].flatten())
    return np.array(X), np.array(y), scaler

def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(14)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Cache functions
def get_cached_prediction(ticker, start_date, end_date):
    db = get_db()
    try:
        result = db.execute(
            """SELECT predictions, company_info FROM predictions_cache 
            WHERE ticker = ? AND start_date = ? AND end_date = ? 
            AND created_at > datetime('now', ?)""",
            (ticker, start_date, end_date, f"-{app.config['PREDICTION_CACHE_DAYS']} days")
        ).fetchone()
        if result:
            return {
                'predictions': json.loads(result['predictions']),
                'company_info': json.loads(result['company_info'])
            }
    except Exception as e:
        app.logger.error(f"Error checking cache: {str(e)}")
    return None

def cache_prediction(ticker, start_date, end_date, predictions, company_info):
    db = get_db()
    try:
        db.execute(
            """INSERT OR REPLACE INTO predictions_cache 
            (ticker, start_date, end_date, predictions, company_info) 
            VALUES (?, ?, ?, ?, ?)""",
            (ticker, start_date, end_date, json.dumps(predictions), json.dumps(company_info))
        )
        db.commit()
    except Exception as e:
        app.logger.error(f"Failed to cache prediction: {str(e)}")

# Visualization function
def create_plot(predictions, ticker, currency):
    if not predictions:
        app.logger.error("No prediction data available for graph")
        return None
    
    try:
        fig = go.Figure()
        dates = [p['date'] for p in predictions]
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=[p['open'] for p in predictions],
            name=f'Open ({currency})',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=dates,
            y=[p['close'] for p in predictions],
            name=f'Close ({currency})',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f"{ticker} Price Predictions",
            xaxis_title="Date",
            yaxis_title=f"Price ({currency})",
            template="plotly_white"
        )
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    except Exception as e:
        app.logger.error(f"Error creating plot: {str(e)}")
        return None

# Routes
@app.route('/')
@login_required
def landing():
    today = datetime.now().strftime('%Y-%m-%d')
    next_week = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    return render_template('app.html', 
                         current_date=today,
                         current_end_date=next_week,
                         default_stocks=DEFAULT_STOCKS,
                         predictions=[],
                         graphJSON=None,
                         company=None,
                         ticker=None,
                         currency_symbol='$')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    ticker = request.form.get('ticker', '').upper().strip()
    
    if not ticker or not is_valid_ticker(ticker):
        flash('Invalid stock symbol. Please enter a valid ticker.', 'error')
        return redirect(url_for('landing'))
    
    try:
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        period = request.form.get('period', '3mo')
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_dt > end_dt:
            raise ValueError("End date must be after start date")
        
        # Check cache first
        cached = get_cached_prediction(ticker, start_date, end_date)
        if cached:
            company_info = cached['company_info']
            currency_symbol = get_currency_symbol(company_info.get('currency', 'USD'))
            graphJSON = create_plot(cached['predictions'], ticker, currency_symbol)
            
            return render_template('app.html',
                                predictions=cached['predictions'],
                                graphJSON=graphJSON,
                                company=company_info,
                                ticker=ticker,
                                currency_symbol=currency_symbol,
                                from_cache=True)
        
        # No cache found, make new prediction
        company_info = get_company_info(ticker)
        currency_symbol = get_currency_symbol(company_info.get('currency', 'USD'))
        
        data = fetch_stock_data(ticker, period)
        if data.empty:
            raise ValueError("No data available for this period")
        
        X, y, scaler = prepare_training_data(data)
        if len(X) < 1:
            raise ValueError("Insufficient historical data for prediction")
        
        model = create_model(X.shape[1:])
        model.fit(X, y, epochs=5, batch_size=32, verbose=0)
        
        predictions = []
        window = data[['Open', 'Close']].iloc[-30:].values
        window = scaler.transform(window)
        
        pred_dates = pd.date_range(start_date, end_date, freq='B')
        
        for date in pred_dates:
            pred = model.predict(np.array([window]), verbose=0)[0]
            prices = scaler.inverse_transform([pred[:2]])[0]
            
            predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': float(prices[0]),
                'close': float(prices[1])
            })
            
            window = np.vstack([window[1:], [pred[0], pred[1]]])
        
        graphJSON = create_plot(predictions, ticker, currency_symbol)
        if not graphJSON:
            flash('Error generating prediction graph', 'error')
            return redirect(url_for('landing'))
        
        cache_prediction(ticker, start_date, end_date, predictions, company_info)
        
        return render_template('app.html',
                             predictions=predictions,
                             graphJSON=graphJSON,
                             company=company_info,
                             ticker=ticker,
                             currency_symbol=currency_symbol)
    
    except ValueError as e:
        flash(str(e), 'error')
        return redirect(url_for('landing'))
    
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}", exc_info=True)
        flash("An error occurred during prediction. Please try again.", 'error')
        return redirect(url_for('landing'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            flash('Please fill in all fields', 'error')
            return render_template('login.html')
        
        db = get_db()
        user = db.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        
        if user and user['password'] == hash_password(password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('landing'))
        
        flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        email = request.form.get('email', '').strip().lower()
        
        if not all([username, password, email]):
            flash('Please fill in all fields', 'error')
            return render_template('register.html')
        
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
                (username, hash_password(password), email)
            )
            db.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError as e:
            if 'username' in str(e):
                flash('Username already exists', 'error')
            else:
                flash('Email already exists', 'error')
        except Exception as e:
            app.logger.error(f"Registration error: {str(e)}")
            flash('Registration failed. Please try again.', 'error')
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)