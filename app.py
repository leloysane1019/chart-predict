from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = joblib.load("stock_up_model.pkl")

def compute_features(df):
    df = df.copy()
    df['ma5'] = df['Close'].rolling(window=5).mean()
    df['ma25'] = df['Close'].rolling(window=25).mean()
    df['dis_ma5'] = (df['Close'] - df['ma5']) / df['ma5']
    df['dis_ma25'] = (df['Close'] - df['ma25']) / df['ma25']
    rolling_std = df['Close'].rolling(window=20).std()
    df['bb_width'] = (rolling_std * 2) / df['Close']
    df['price_range'] = (df['High'].rolling(window=20).max() - df['Low'].rolling(window=20).min()) / df['Low'].rolling(window=20).min()
    df['vol_ratio'] = df['Volume'].rolling(window=10).mean() / df['Volume'].rolling(window=20).mean()
    if len(df) > 1:
        X = np.arange(len(df)).reshape(-1, 1)
        y = df['Close'].values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        df['trend_slope'] = reg.coef_[0][0]
    else:
        df['trend_slope'] = np.nan
    return df

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
def form_post(request: Request, ticker: str = Form(...)):
    try:
        df = yf.download(ticker, period="60d", interval="1d")[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = compute_features(df).dropna()
        X_input = df[['trend_slope', 'dis_ma5', 'dis_ma25', 'bb_width', 'price_range', 'vol_ratio']].iloc[-1:]
        prob = model.predict_proba(X_input)[0][1]
        result = {
            "ticker": ticker,
            "probability": round(prob * 100, 2),
            "message": "上がる可能性が高いです！" if prob >= 0.5 else "慎重に判断しましょう。"
        }
    except Exception as e:
        result = {"error": str(e)}
    return templates.TemplateResponse("index.html", {"request": request, "result": result})