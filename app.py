"""
US Stock Market Analysis & Prediction System
미국 증시 분석 및 예측 시스템
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys, os
from plotly.subplots import make_subplots
import requests
from xml.etree import ElementTree as ET
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import warnings
import time
import json
import re
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="US Market Intelligence | 미국 증시 인텔리전스",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── LANGUAGE TEXTS ───────────────────────────────────────────────────────────
TEXTS = {
    "en": {
        "title": "US Market Intelligence System",
        "subtitle": "AI-Powered Stock Analysis & Prediction",
        "search_placeholder": "Enter ticker or company name (e.g. AAPL, Tesla)",
        "search_btn": "Search",
        "sp500_list": "S&P 500 Companies",
        "global_stocks": "Global Stocks",
        "lang_toggle": "🇰🇷 한국어",
        "tab_overview": "📊 Overview",
        "tab_predict": "🔮 Prediction",
        "tab_news": "📰 News",
        "tab_geo": "🌍 Geopolitical",
        "tab_history": "📜 History",
        "current_price": "Current Price",
        "change_1d": "1-Day Change",
        "mkt_cap": "Market Cap",
        "volume": "Volume",
        "pe_ratio": "P/E Ratio",
        "week52_high": "52-Week High",
        "week52_low": "52-Week Low",
        "pred_title": "Price Forecast",
        "pred_3m": "3 Months",
        "pred_6m": "6 Months",
        "pred_9m": "9 Months",
        "pred_12m": "12 Months",
        "pred_base": "Base Case",
        "pred_bull": "Bull Case",
        "pred_bear": "Bear Case",
        "news_title": "Latest Market News",
        "news_loading": "Loading news...",
        "geo_title": "Geopolitical Risk Dashboard",
        "geo_vix": "VIX (Fear Index)",
        "geo_oil": "Crude Oil (WTI)",
        "geo_gold": "Gold",
        "geo_dxy": "USD Index (DXY)",
        "geo_bonds": "10Y Treasury Yield",
        "geo_factors": "Key Risk Factors",
        "hist_title": "Historical Analysis",
        "hist_summary": "Price Movement Summary (2-line AI Analysis)",
        "hist_period": "Select Period",
        "hist_1y": "1 Year",
        "hist_3y": "3 Years",
        "hist_5y": "5 Years",
        "hist_10y": "10 Years",
        "hist_max": "Max",
        "loading": "Loading data...",
        "error_ticker": "Could not find ticker. Please try again.",
        "select_company": "Select a company from the sidebar or search above.",
        "macro_indicators": "Macro Economic Indicators",
        "technical_indicators": "Technical Indicators",
        "rsi": "RSI (14)",
        "macd": "MACD",
        "bb": "Bollinger Bands",
        "sma50": "SMA 50",
        "sma200": "SMA 200",
        "sentiment": "Market Sentiment",
        "bullish": "Bullish",
        "bearish": "Bearish",
        "neutral": "Neutral",
        "confidence": "Prediction Confidence",
        "auto_update": "Auto News Update",
        "last_updated": "Last Updated",
        "sector": "Sector",
        "industry": "Industry",
        "country": "Country",
        "employees": "Employees",
        "description": "Company Overview",
        "risk_level": "Risk Level",
        "high_risk": "HIGH",
        "med_risk": "MEDIUM",
        "low_risk": "LOW",
        "forecast_disclaimer": "⚠️ Forecasts are statistical estimates. Not financial advice.",
        "news_source": "Source",
        "news_time": "Published",
        "geopolitical_events": "Recent Geopolitical Events",
        "market_correlation": "Market Correlation",
        "volatility": "Volatility (Annualized)",
        "sharpe": "Sharpe Ratio",
        "beta": "Beta vs S&P 500",
        "summary_analysis": "2-Line Summary Analysis",
    },
    "ko": {
        "title": "미국 증시 인텔리전스 시스템",
        "subtitle": "AI 기반 주식 분석 및 예측",
        "search_placeholder": "티커 또는 기업명 입력 (예: AAPL, 테슬라)",
        "search_btn": "검색",
        "sp500_list": "S&P 500 기업",
        "global_stocks": "글로벌 주요 주식",
        "lang_toggle": "🇺🇸 English",
        "tab_overview": "📊 개요",
        "tab_predict": "🔮 예측",
        "tab_news": "📰 뉴스",
        "tab_geo": "🌍 지정학적",
        "tab_history": "📜 역사적 분석",
        "current_price": "현재가",
        "change_1d": "1일 변동",
        "mkt_cap": "시가총액",
        "volume": "거래량",
        "pe_ratio": "주가수익비율(P/E)",
        "week52_high": "52주 최고가",
        "week52_low": "52주 최저가",
        "pred_title": "주가 예측",
        "pred_3m": "3개월",
        "pred_6m": "6개월",
        "pred_9m": "9개월",
        "pred_12m": "12개월",
        "pred_base": "기본 시나리오",
        "pred_bull": "강세 시나리오",
        "pred_bear": "약세 시나리오",
        "news_title": "최신 시장 뉴스",
        "news_loading": "뉴스 로딩 중...",
        "geo_title": "지정학적 리스크 대시보드",
        "geo_vix": "VIX (공포지수)",
        "geo_oil": "WTI 원유",
        "geo_gold": "금",
        "geo_dxy": "달러 인덱스 (DXY)",
        "geo_bonds": "미국채 10년 수익률",
        "geo_factors": "주요 리스크 요인",
        "hist_title": "역사적 분석",
        "hist_summary": "주가 변동 요약 (2줄 AI 분석)",
        "hist_period": "기간 선택",
        "hist_1y": "1년",
        "hist_3y": "3년",
        "hist_5y": "5년",
        "hist_10y": "10년",
        "hist_max": "전체",
        "loading": "데이터 로딩 중...",
        "error_ticker": "티커를 찾을 수 없습니다. 다시 시도해주세요.",
        "select_company": "사이드바에서 기업을 선택하거나 위에서 검색하세요.",
        "macro_indicators": "거시경제 지표",
        "technical_indicators": "기술적 지표",
        "rsi": "RSI (14)",
        "macd": "MACD",
        "bb": "볼린저 밴드",
        "sma50": "50일 이동평균",
        "sma200": "200일 이동평균",
        "sentiment": "시장 심리",
        "bullish": "강세",
        "bearish": "약세",
        "neutral": "중립",
        "confidence": "예측 신뢰도",
        "auto_update": "뉴스 자동 업데이트",
        "last_updated": "마지막 업데이트",
        "sector": "섹터",
        "industry": "업종",
        "country": "국가",
        "employees": "임직원 수",
        "description": "기업 개요",
        "risk_level": "리스크 수준",
        "high_risk": "높음",
        "med_risk": "중간",
        "low_risk": "낮음",
        "forecast_disclaimer": "⚠️ 예측은 통계적 추정치입니다. 투자 조언이 아닙니다.",
        "news_source": "출처",
        "news_time": "게시 시간",
        "geopolitical_events": "최근 지정학적 이벤트",
        "market_correlation": "시장 상관관계",
        "volatility": "변동성 (연율화)",
        "sharpe": "샤프 비율",
        "beta": "베타 (S&P 500 대비)",
        "summary_analysis": "2줄 요약 분석",
    },
}

# ─── SESSION STATE INIT ───────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "ko"
if "ticker" not in st.session_state:
    st.session_state.ticker = "^GSPC"
if "last_news_update" not in st.session_state:
    st.session_state.last_news_update = None

def T(key):
    return TEXTS[st.session_state.lang].get(key, key)

# ─── S&P 500 + GLOBAL TICKERS ─────────────────────────────────────────────────
SP500_POPULAR = {
    "Technology": [
        ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"),
        ("GOOGL", "Alphabet"), ("META", "Meta"), ("AMZN", "Amazon"),
        ("TSLA", "Tesla"), ("AVGO", "Broadcom"), ("ORCL", "Oracle"),
        ("CRM", "Salesforce"), ("AMD", "AMD"), ("INTC", "Intel"),
        ("QCOM", "Qualcomm"), ("TXN", "Texas Instruments"),
    ],
    "Finance": [
        ("JPM", "JPMorgan"), ("BAC", "Bank of America"), ("WFC", "Wells Fargo"),
        ("GS", "Goldman Sachs"), ("MS", "Morgan Stanley"), ("BRK-B", "Berkshire"),
        ("V", "Visa"), ("MA", "Mastercard"), ("AXP", "American Express"),
    ],
    "Healthcare": [
        ("JNJ", "J&J"), ("UNH", "UnitedHealth"), ("PFE", "Pfizer"),
        ("ABBV", "AbbVie"), ("MRK", "Merck"), ("LLY", "Eli Lilly"),
        ("TMO", "Thermo Fisher"), ("ABT", "Abbott"),
    ],
    "Energy": [
        ("XOM", "ExxonMobil"), ("CVX", "Chevron"), ("COP", "ConocoPhillips"),
        ("SLB", "Schlumberger"), ("EOG", "EOG Resources"),
    ],
    "Consumer": [
        ("WMT", "Walmart"), ("COST", "Costco"), ("HD", "Home Depot"),
        ("NKE", "Nike"), ("MCD", "McDonald's"), ("SBUX", "Starbucks"),
        ("PG", "Procter & Gamble"), ("KO", "Coca-Cola"), ("PEP", "PepsiCo"),
    ],
    "Indices": [
        ("^GSPC", "S&P 500"), ("^DJI", "Dow Jones"), ("^IXIC", "NASDAQ"),
        ("^RUT", "Russell 2000"), ("^VIX", "VIX"),
    ],
}

GLOBAL_STOCKS = {
    "Korea 한국": [
        ("005930.KS", "삼성전자"), ("000660.KS", "SK하이닉스"),
        ("207940.KS", "삼성바이오로직스"), ("005380.KS", "현대차"),
        ("035420.KS", "NAVER"), ("051910.KS", "LG화학"),
    ],
    "Japan 일본": [
        ("7203.T", "Toyota"), ("6758.T", "Sony"), ("9984.T", "SoftBank"),
        ("8306.T", "Mitsubishi UFJ"), ("6501.T", "Hitachi"),
    ],
    "Europe 유럽": [
        ("ASML", "ASML"), ("NVO", "Novo Nordisk"), ("SAP", "SAP"),
        ("AZN", "AstraZeneca"), ("SHEL", "Shell"),
    ],
    "China 중국": [
        ("BABA", "Alibaba"), ("JD", "JD.com"), ("PDD", "PDD Holdings"),
        ("BIDU", "Baidu"), ("NIO", "NIO"),
    ],
}

MACRO_TICKERS = {
    "^VIX": "VIX",
    "CL=F": "Oil (WTI)",
    "GC=F": "Gold",
    "DX-Y.NYB": "USD Index",
    "^TNX": "10Y Treasury",
    "^TYX": "30Y Treasury",
    "BTC-USD": "Bitcoin",
}

# ─── DATA FUNCTIONS ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_info(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return info
    except Exception:
        return {}

@st.cache_data(ttl=600)
def get_macro_data() -> dict:
    result = {}
    for ticker, name in MACRO_TICKERS.items():
        try:
            df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if not df.empty:
                close = df["Close"]
                if hasattr(close, "iloc"):
                    vals = close.iloc[:, 0] if close.ndim == 2 else close
                    current = float(vals.iloc[-1])
                    prev = float(vals.iloc[-2]) if len(vals) > 1 else current
                    result[name] = {
                        "current": current,
                        "change_pct": (current - prev) / prev * 100 if prev != 0 else 0,
                    }
        except Exception:
            pass
    return result

@st.cache_data(ttl=1800)
def fetch_news(lang: str = "en") -> list:
    feeds = [
        ("https://feeds.reuters.com/reuters/businessNews", "Reuters"),
        ("https://feeds.marketwatch.com/marketwatch/topstories/", "MarketWatch"),
        ("https://finance.yahoo.com/news/rssindex", "Yahoo Finance"),
        ("https://www.cnbc.com/id/10001147/device/rss/rss.html", "CNBC"),
        ("https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "NYT Business"),
    ]
    articles = []
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MarketIntel/1.0)"}
    for url, source_name in feeds:
        try:
            resp = requests.get(url, timeout=8, headers=headers)
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            ns = {"media": "http://search.yahoo.com/mrss/"}
            items = root.findall(".//item")
            for item in items[:6]:
                title = item.findtext("title", "").strip()
                link = item.findtext("link", "").strip()
                desc = item.findtext("description", "").strip()
                pubdate = item.findtext("pubDate", "").strip()
                # Clean HTML
                desc = re.sub(r"<[^>]+>", "", desc)[:200]
                title = re.sub(r"<[^>]+>", "", title)
                if title:
                    articles.append({
                        "title": title,
                        "summary": desc,
                        "link": link,
                        "published": pubdate[:25] if pubdate else "",
                        "source": source_name,
                    })
        except Exception:
            continue
    return articles[:30]

# ─── TECHNICAL INDICATORS ─────────────────────────────────────────────────────
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 20:
        return df

    close = df["Close"]
    if close.ndim == 2:
        close = close.iloc[:, 0]
    close = close.astype(float)

    df = df.copy()
    df["close"] = close

    # SMA
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA200"] = close.rolling(200).mean()

    # Bollinger Bands
    rolling_std = close.rolling(20).std()
    df["BB_upper"] = df["SMA20"] + 2 * rolling_std
    df["BB_lower"] = df["SMA20"] - 2 * rolling_std

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Volume MA
    if "Volume" in df.columns:
        vol = df["Volume"]
        if vol.ndim == 2:
            vol = vol.iloc[:, 0]
        df["Volume_MA20"] = vol.rolling(20).mean()

    return df

# ─── PREDICTION ENGINE ────────────────────────────────────────────────────────
def predict_prices(df: pd.DataFrame, horizons_days: list) -> dict:
    """Multi-horizon price prediction using ensemble of methods."""
    if df.empty or len(df) < 60:
        return {}

    close = df["Close"]
    if close.ndim == 2:
        close = close.iloc[:, 0]
    close = close.astype(float)
    prices = close.dropna().values
    n = len(prices)

    results = {}

    for h in horizons_days:
        try:
            # --- Method 1: Linear trend + momentum ---
            x = np.arange(n).reshape(-1, 1)
            y = prices
            lr = LinearRegression().fit(x, y)
            trend_pred = lr.predict([[n + h]])[0]

            # --- Method 2: Exponential weighted recent trend ---
            recent = prices[-min(90, n):]
            weights = np.exp(np.linspace(0, 1, len(recent)))
            weights /= weights.sum()
            recent_returns = np.diff(np.log(recent))
            avg_return = np.average(recent_returns, weights=weights[1:])
            exp_pred = prices[-1] * np.exp(avg_return * h)

            # --- Method 3: Mean reversion (long-run) ---
            ma200 = np.mean(prices[-min(200, n):])
            reversion_strength = 0.3 * (h / 252)
            mr_pred = prices[-1] + reversion_strength * (ma200 - prices[-1]) + \
                      avg_return * h * prices[-1]

            # Ensemble base
            base = np.mean([trend_pred, exp_pred, mr_pred])

            # Volatility for confidence bands
            vol_daily = np.std(np.diff(np.log(prices[-min(252, n):])))
            vol_horizon = vol_daily * np.sqrt(h)

            bull = base * np.exp(vol_horizon * 1.28)   # 90th pct
            bear = base * np.exp(-vol_horizon * 1.28)  # 10th pct

            results[h] = {
                "base": round(base, 2),
                "bull": round(bull, 2),
                "bear": round(bear, 2),
                "current": round(float(prices[-1]), 2),
                "change_pct": round((base / prices[-1] - 1) * 100, 2),
                "vol_annual": round(vol_daily * np.sqrt(252) * 100, 2),
            }
        except Exception:
            continue

    return results

def get_sentiment(df: pd.DataFrame) -> dict:
    """Compute market sentiment from technical indicators."""
    if df.empty or "RSI" not in df.columns:
        return {"score": 50, "label_en": "Neutral", "label_ko": "중립"}

    signals = []

    # RSI signal
    rsi = float(df["RSI"].iloc[-1]) if not pd.isna(df["RSI"].iloc[-1]) else 50
    if rsi > 70:
        signals.append(-1)
    elif rsi < 30:
        signals.append(1)
    else:
        signals.append(0)

    # MACD signal
    if "MACD" in df.columns and "MACD_signal" in df.columns:
        macd = df["MACD"].iloc[-1]
        sig = df["MACD_signal"].iloc[-1]
        if not pd.isna(macd) and not pd.isna(sig):
            signals.append(1 if macd > sig else -1)

    # Price vs SMA200
    if "SMA200" in df.columns:
        sma200 = df["SMA200"].iloc[-1]
        price = df["close"].iloc[-1] if "close" in df.columns else df["Close"].iloc[-1]
        if not pd.isna(sma200):
            if hasattr(price, "iloc"):
                price = float(price.iloc[-1]) if price.ndim > 0 else float(price)
            signals.append(1 if float(price) > float(sma200) else -1)

    # Recent momentum (5-day)
    close = df["Close"]
    if close.ndim == 2:
        close = close.iloc[:, 0]
    if len(close) >= 5:
        mom = (float(close.iloc[-1]) / float(close.iloc[-5]) - 1) * 100
        signals.append(1 if mom > 0 else -1)

    score = 50 + (np.mean(signals) * 30) if signals else 50
    score = max(0, min(100, score))

    if score >= 60:
        return {"score": score, "label_en": "Bullish", "label_ko": "강세"}
    elif score <= 40:
        return {"score": score, "label_en": "Bearish", "label_ko": "약세"}
    else:
        return {"score": score, "label_en": "Neutral", "label_ko": "중립"}

# ─── HISTORICAL SUMMARY GENERATOR ────────────────────────────────────────────
GEOPOLITICAL_EVENTS = [
    {"date": "2020-03", "event_en": "COVID-19 Pandemic → S&P -34% crash", "event_ko": "코로나19 팬데믹 → S&P -34% 폭락", "impact": -34},
    {"date": "2020-11", "event_en": "Vaccine approval → Bull market recovery", "event_ko": "백신 승인 → 강세장 회복", "impact": 25},
    {"date": "2022-01", "event_en": "Fed rate hike cycle begins → Tech selloff", "event_ko": "연준 금리 인상 사이클 시작 → 기술주 매도세", "impact": -20},
    {"date": "2022-02", "event_en": "Russia-Ukraine War → Energy/defense stocks surge", "event_ko": "러시아-우크라이나 전쟁 → 에너지/방산주 급등", "impact": -8},
    {"date": "2022-10", "event_en": "Inflation peak at 9.1% → Market bottom", "event_ko": "인플레이션 9.1% 정점 → 시장 바닥", "impact": -25},
    {"date": "2023-03", "event_en": "Silicon Valley Bank collapse → Banking stress", "event_ko": "실리콘밸리은행 파산 → 은행권 불안", "impact": -5},
    {"date": "2023-07", "event_en": "AI boom (ChatGPT era) → Tech mega-rally", "event_ko": "AI 붐 (ChatGPT 시대) → 기술주 대형 랠리", "impact": 30},
    {"date": "2024-01", "event_en": "Fed pivot expectations → Risk-on rally", "event_ko": "연준 피벗 기대감 → 위험자산 랠리", "impact": 15},
    {"date": "2024-07", "event_en": "Rate cut anticipation → Broad market ATH", "event_ko": "금리 인하 기대 → 전반적 역대 최고가", "impact": 12},
    {"date": "2025-04", "event_en": "Trump tariff escalation → Volatility spike", "event_ko": "트럼프 관세 확대 → 변동성 급등", "impact": -10},
    {"date": "2025-06", "event_en": "US-China trade tensions ongoing → Tech uncertainty", "event_ko": "미중 무역 갈등 지속 → 기술주 불확실성", "impact": -5},
]

def generate_2line_summary(ticker: str, df: pd.DataFrame, info: dict, lang: str) -> str:
    if df.empty:
        return ""

    close = df["Close"]
    if close.ndim == 2:
        close = close.iloc[:, 0]
    close = close.astype(float)

    # Calculate stats
    ret_1y = (float(close.iloc[-1]) / float(close.iloc[-252]) - 1) * 100 if len(close) >= 252 else 0
    ret_total = (float(close.iloc[-1]) / float(close.iloc[0]) - 1) * 100
    vol = close.pct_change().std() * np.sqrt(252) * 100
    max_price = float(close.max())
    min_price = float(close.min())
    current = float(close.iloc[-1])
    dist_from_high = (current / max_price - 1) * 100
    company = info.get("shortName", ticker)

    if lang == "ko":
        line1 = (
            f"**{company}**은(는) 최근 1년간 {ret_1y:+.1f}%, 전체 기간 {ret_total:+.1f}% 수익률을 기록하며 "
            f"연간 변동성 {vol:.1f}%의 {('낮은' if vol < 20 else '높은' if vol > 40 else '중간') }수준 리스크를 보임."
        )
        line2 = (
            f"현재가 ${current:,.2f}는 역대 최고가 대비 {dist_from_high:.1f}% 수준이며, "
            f"{'과매수 구간으로 조정 압력 존재.' if dist_from_high > -5 else '저점 대비 반등 여력 존재.' if dist_from_high < -30 else '역사적 중간 수준의 밸류에이션.'}"
        )
    else:
        line1 = (
            f"**{company}** delivered {ret_1y:+.1f}% over the past year ({ret_total:+.1f}% total), "
            f"with annualized volatility of {vol:.1f}% — a {'low' if vol < 20 else 'high' if vol > 40 else 'moderate'}-risk profile."
        )
        line2 = (
            f"Current price ${current:,.2f} sits {dist_from_high:.1f}% from all-time high of ${max_price:,.2f}, "
            f"suggesting {'overbought pressure exists.' if dist_from_high > -5 else 'significant recovery potential.' if dist_from_high < -30 else 'fair historical valuation.'}"
        )

    return f"{line1}\n\n{line2}"

# ─── CHART BUILDERS ───────────────────────────────────────────────────────────
def build_price_chart(df: pd.DataFrame, ticker: str, lang: str) -> go.Figure:
    if df.empty:
        return go.Figure()

    close = df["Close"]
    if close.ndim == 2:
        close = close.iloc[:, 0]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=("", "Volume", "RSI"),
    )

    # Candlestick
    open_col = df["Open"].iloc[:, 0] if df["Open"].ndim == 2 else df["Open"]
    high_col = df["High"].iloc[:, 0] if df["High"].ndim == 2 else df["High"]
    low_col = df["Low"].iloc[:, 0] if df["Low"].ndim == 2 else df["Low"]

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=open_col, high=high_col,
        low=low_col, close=close,
        name=ticker,
        increasing_line_color="#00D4AA",
        decreasing_line_color="#FF4B4B",
    ), row=1, col=1)

    # Bollinger Bands
    if "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"],
            line=dict(color="rgba(173,216,230,0.5)", width=1),
            name="BB Upper", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"],
            fill="tonexty",
            fillcolor="rgba(173,216,230,0.1)",
            line=dict(color="rgba(173,216,230,0.5)", width=1),
            name="BB Lower", showlegend=False,
        ), row=1, col=1)

    # SMA lines
    for col, color, label in [
        ("SMA50", "#FFD700", "SMA50"),
        ("SMA200", "#FF8C00", "SMA200"),
    ]:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                line=dict(color=color, width=1.5, dash="dot"),
                name=label,
            ), row=1, col=1)

    # Volume
    if "Volume" in df.columns:
        vol = df["Volume"]
        if vol.ndim == 2:
            vol = vol.iloc[:, 0]
        colors = ["#00D4AA" if float(c) >= float(o) else "#FF4B4B"
                  for c, o in zip(close, open_col)]
        fig.add_trace(go.Bar(
            x=df.index, y=vol, marker_color=colors,
            name="Volume", showlegend=False,
        ), row=2, col=1)

    # RSI
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI"],
            line=dict(color="#AB63FA", width=1.5),
            name="RSI",
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hrect(y0=30, y1=70, fillcolor="purple", opacity=0.05, row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=700,
        showlegend=True,
        legend=dict(orientation="h", y=1.02, x=0),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
    )
    return fig

def build_forecast_chart(
    ticker: str, df: pd.DataFrame, predictions: dict, lang: str
) -> go.Figure:
    if df.empty or not predictions:
        return go.Figure()

    close = df["Close"]
    if close.ndim == 2:
        close = close.iloc[:, 0]
    close = close.astype(float)

    fig = go.Figure()

    # Historical (last 6 months)
    hist = close.iloc[-126:]
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist.values,
        name="Historical" if lang == "en" else "실제 주가",
        line=dict(color="#00D4AA", width=2),
    ))

    # Forecast points
    last_date = df.index[-1]
    horizon_labels = {
        63: T("pred_3m"), 126: T("pred_6m"),
        189: T("pred_9m"), 252: T("pred_12m"),
    }

    forecast_dates = []
    base_vals, bull_vals, bear_vals = [], [], []

    # Start from last known price
    forecast_dates.append(last_date)
    base_vals.append(float(close.iloc[-1]))
    bull_vals.append(float(close.iloc[-1]))
    bear_vals.append(float(close.iloc[-1]))

    for h in sorted(predictions.keys()):
        future_date = last_date + timedelta(days=h * 1.4)  # approx calendar days
        forecast_dates.append(future_date)
        base_vals.append(predictions[h]["base"])
        bull_vals.append(predictions[h]["bull"])
        bear_vals.append(predictions[h]["bear"])

    # Bull/Bear band
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=bull_vals,
        name=T("pred_bull"),
        line=dict(color="rgba(0,212,170,0.5)", width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=bear_vals,
        fill="tonexty",
        fillcolor="rgba(100,100,255,0.1)",
        name=T("pred_bear"),
        line=dict(color="rgba(255,75,75,0.5)", width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=base_vals,
        name=T("pred_base"),
        line=dict(color="#FFA500", width=2.5),
        mode="lines+markers",
        marker=dict(size=8, symbol="circle"),
    ))

    # Annotate each horizon
    for i, h in enumerate(sorted(predictions.keys())):
        idx = i + 1
        if idx < len(forecast_dates):
            fig.add_annotation(
                x=forecast_dates[idx],
                y=base_vals[idx],
                text=f"${base_vals[idx]:,.0f}<br>({predictions[h]['change_pct']:+.1f}%)",
                showarrow=True,
                arrowhead=2,
                bgcolor="#1E2130",
                bordercolor="#FFA500",
                font=dict(size=11, color="white"),
            )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        title=dict(text=T("pred_title"), font=dict(size=18)),
        showlegend=True,
        legend=dict(orientation="h", y=1.02),
        xaxis_title="Date" if lang == "en" else "날짜",
        yaxis_title="Price (USD)" if lang == "en" else "주가 (USD)",
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
    )
    return fig

def build_macro_chart(macro_data: dict, lang: str) -> go.Figure:
    if not macro_data:
        return go.Figure()

    names = list(macro_data.keys())
    changes = [macro_data[n]["change_pct"] for n in names]
    colors = ["#00D4AA" if c >= 0 else "#FF4B4B" for c in changes]

    fig = go.Figure(go.Bar(
        x=names, y=changes,
        marker_color=colors,
        text=[f"{c:+.2f}%" for c in changes],
        textposition="outside",
    ))
    fig.update_layout(
        template="plotly_dark",
        title=T("macro_indicators"),
        height=350,
        yaxis_title="1-Day Change (%)" if lang == "en" else "1일 변동률 (%)",
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
    )
    return fig

def build_geo_timeline(lang: str) -> go.Figure:
    events = GEOPOLITICAL_EVENTS
    dates = [e["date"] for e in events]
    impacts = [e["impact"] for e in events]
    labels = [e[f"event_{lang}"] for e in events]
    colors = ["#00D4AA" if i >= 0 else "#FF4B4B" for i in impacts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates, y=impacts,
        marker_color=colors,
        text=labels,
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{text}<br>Impact: %{y:+.0f}%<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        title=T("geopolitical_events") if lang == "en" else "지정학적 주요 이벤트 & 증시 영향",
        height=420,
        yaxis_title="Market Impact (%)" if lang == "en" else "증시 영향 (%)",
        margin=dict(l=0, r=0, t=50, b=20),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
    )
    return fig

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0E1117; }
    .metric-card {
        background: linear-gradient(135deg, #1E2130 0%, #16213E 100%);
        border: 1px solid #2E3250;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #FFFFFF; }
    .metric-label { font-size: 0.75rem; color: #8B9DB0; margin-bottom: 4px; }
    .metric-change-pos { color: #00D4AA; font-size: 0.9rem; }
    .metric-change-neg { color: #FF4B4B; font-size: 0.9rem; }
    .news-card {
        background: #1E2130;
        border-left: 3px solid #FFA500;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .news-title { font-size: 0.95rem; font-weight: 600; color: #EAEAEA; }
    .news-meta { font-size: 0.75rem; color: #8B9DB0; margin-top: 4px; }
    .news-summary { font-size: 0.82rem; color: #B0BEC5; margin-top: 6px; }
    .geo-card {
        background: linear-gradient(135deg, #16213E, #0D1B2A);
        border: 1px solid #2E3250;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 8px;
    }
    .risk-high { color: #FF4B4B; font-weight: 700; }
    .risk-med { color: #FFA500; font-weight: 700; }
    .risk-low { color: #00D4AA; font-weight: 700; }
    .prediction-card {
        background: linear-gradient(135deg, #1A1F35, #0F1527);
        border: 1px solid #3A4060;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .pred-horizon { font-size: 0.85rem; color: #8B9DB0; margin-bottom: 8px; }
    .pred-price { font-size: 1.5rem; font-weight: 700; color: #FFA500; }
    .pred-change-pos { font-size: 1rem; color: #00D4AA; }
    .pred-change-neg { font-size: 1rem; color: #FF4B4B; }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #FFFFFF;
        border-bottom: 2px solid #FFA500;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }
    .summary-box {
        background: linear-gradient(135deg, #1E2130, #162040);
        border: 1px solid #3A6186;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 12px 0;
        line-height: 1.7;
    }
    div[data-testid="stSidebarContent"] { background: #0D1120; }
    .stTabs [data-baseweb="tab-list"] { background: #0D1120; }
    .stTabs [data-baseweb="tab"] { color: #8B9DB0; }
    .stTabs [aria-selected="true"] { color: #FFA500 !important; }
    .update-badge {
        background: #1E4620;
        color: #00D4AA;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Language toggle
    col_logo, col_lang = st.columns([2, 1])
    with col_logo:
        st.markdown("### 📈 Market Intel")
    with col_lang:
        if st.button(T("lang_toggle"), key="lang_btn", use_container_width=True):
            st.session_state.lang = "en" if st.session_state.lang == "ko" else "ko"
            st.rerun()

    st.divider()

    # Search
    search_input = st.text_input(
        T("search_placeholder"),
        placeholder=T("search_placeholder"),
        label_visibility="collapsed",
    )
    if st.button(T("search_btn"), use_container_width=True, type="primary"):
        if search_input.strip():
            st.session_state.ticker = search_input.strip().upper()
            st.rerun()

    st.divider()

    # S&P 500 by sector
    st.markdown(f"**{T('sp500_list')}**")
    for sector, stocks in SP500_POPULAR.items():
        sector_label = sector
        with st.expander(sector_label, expanded=(sector == "Indices")):
            for ticker_sym, name in stocks:
                label = f"{name} ({ticker_sym})"
                if st.button(label, key=f"btn_{ticker_sym}", use_container_width=True):
                    st.session_state.ticker = ticker_sym
                    st.rerun()

    st.divider()

    # Global stocks
    st.markdown(f"**{T('global_stocks')}**")
    for region, stocks in GLOBAL_STOCKS.items():
        with st.expander(region):
            for ticker_sym, name in stocks:
                label = f"{name} ({ticker_sym})"
                if st.button(label, key=f"gbl_{ticker_sym}", use_container_width=True):
                    st.session_state.ticker = ticker_sym
                    st.rerun()

# ─── MAIN CONTENT ─────────────────────────────────────────────────────────────
ticker = st.session_state.ticker
lang = st.session_state.lang

# Title
st.markdown(f"""
<div style='text-align:center; padding: 10px 0 20px 0;'>
    <h1 style='color:#FFA500; margin:0; font-size:2rem;'>📈 {T('title')}</h1>
    <p style='color:#8B9DB0; margin:4px 0 0 0;'>{T('subtitle')}</p>
</div>
""", unsafe_allow_html=True)

# Load data
with st.spinner(T("loading")):
    df_2y = get_stock_data(ticker, "2y")
    df_5y = get_stock_data(ticker, "5y")
    df_max = get_stock_data(ticker, "max")
    info = get_stock_info(ticker)
    macro_data = get_macro_data()

if df_2y.empty:
    st.error(T("error_ticker"))
    st.stop()

df_2y = compute_indicators(df_2y)
df_5y = compute_indicators(df_5y)

# Company header
company_name = info.get("shortName", ticker)
current_price = info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose", 0)
prev_close = info.get("previousClose", current_price)
change_1d = ((current_price - prev_close) / prev_close * 100) if prev_close else 0
change_color = "#00D4AA" if change_1d >= 0 else "#FF4B4B"
change_arrow = "▲" if change_1d >= 0 else "▼"

st.markdown(f"""
<div style='background:linear-gradient(135deg,#1E2130,#16213E);border-radius:14px;padding:20px 28px;margin-bottom:20px;border:1px solid #2E3250;'>
    <div style='display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;'>
        <div>
            <span style='font-size:1.8rem;font-weight:800;color:#FFFFFF;'>{company_name}</span>
            <span style='color:#8B9DB0;margin-left:12px;font-size:1rem;'>{ticker}</span>
            {'<span style="background:#2A3A5C;color:#64B5F6;border-radius:6px;padding:2px 8px;margin-left:10px;font-size:0.8rem;">' + info.get('sector','') + '</span>' if info.get('sector') else ''}
        </div>
        <div style='text-align:right;'>
            <div style='font-size:2.2rem;font-weight:800;color:#FFFFFF;'>${current_price:,.2f}</div>
            <div style='color:{change_color};font-size:1.1rem;'>{change_arrow} {abs(change_1d):.2f}% (1D)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Key metrics row
mkt_cap = info.get("marketCap", 0)
mkt_cap_str = f"${mkt_cap/1e12:.2f}T" if mkt_cap > 1e12 else f"${mkt_cap/1e9:.1f}B" if mkt_cap > 1e9 else f"${mkt_cap/1e6:.0f}M" if mkt_cap else "N/A"
volume = info.get("volume") or info.get("regularMarketVolume", 0)
vol_str = f"{volume/1e6:.1f}M" if volume > 1e6 else f"{volume/1e3:.0f}K" if volume else "N/A"
pe = info.get("trailingPE", info.get("forwardPE", 0))
w52h = info.get("fiftyTwoWeekHigh", 0)
w52l = info.get("fiftyTwoWeekLow", 0)

mcol1, mcol2, mcol3, mcol4, mcol5, mcol6 = st.columns(6)
metrics = [
    (T("mkt_cap"), mkt_cap_str, ""),
    (T("volume"), vol_str, ""),
    (T("pe_ratio"), f"{pe:.1f}" if pe else "N/A", ""),
    (T("week52_high"), f"${w52h:,.2f}" if w52h else "N/A", ""),
    (T("week52_low"), f"${w52l:,.2f}" if w52l else "N/A", ""),
    (T("volatility"), f"{df_2y['close'].pct_change().std() * np.sqrt(252) * 100:.1f}%" if 'close' in df_2y.columns else "N/A", ""),
]
for col, (label, val, change) in zip([mcol1, mcol2, mcol3, mcol4, mcol5, mcol6], metrics):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value' style='font-size:1.1rem;'>{val}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([T("tab_overview"), T("tab_predict"), T("tab_news"), T("tab_geo"), T("tab_history")])

# ══════════════════ TAB 1: OVERVIEW ══════════════════
with tabs[0]:
    # Price chart period selector
    period_map = {
        T("hist_1y"): "1y", T("hist_3y"): "3y",
        T("hist_5y"): "5y", T("hist_max"): "max",
    }
    period_choice = st.radio(
        T("hist_period"), list(period_map.keys()),
        horizontal=True, label_visibility="collapsed",
    )
    selected_period = period_map[period_choice]
    df_chart = get_stock_data(ticker, selected_period)
    df_chart = compute_indicators(df_chart)

    st.plotly_chart(build_price_chart(df_chart, ticker, lang), use_container_width=True)

    # Technical indicators summary
    st.markdown(f"<div class='section-header'>{T('technical_indicators')}</div>", unsafe_allow_html=True)
    ti_cols = st.columns(5)
    ind_data = [
        (T("rsi"), f"{df_2y['RSI'].iloc[-1]:.1f}" if "RSI" in df_2y.columns and not pd.isna(df_2y["RSI"].iloc[-1]) else "N/A",
         "#FF4B4B" if "RSI" in df_2y.columns and not pd.isna(df_2y["RSI"].iloc[-1]) and df_2y["RSI"].iloc[-1] > 70
         else "#00D4AA" if "RSI" in df_2y.columns and not pd.isna(df_2y["RSI"].iloc[-1]) and df_2y["RSI"].iloc[-1] < 30
         else "#FFA500"),
        (T("macd"), f"{df_2y['MACD'].iloc[-1]:.3f}" if "MACD" in df_2y.columns else "N/A", "#8B9DB0"),
        (T("sma50"), f"${df_2y['SMA50'].iloc[-1]:,.2f}" if "SMA50" in df_2y.columns and not pd.isna(df_2y["SMA50"].iloc[-1]) else "N/A", "#FFD700"),
        (T("sma200"), f"${df_2y['SMA200'].iloc[-1]:,.2f}" if "SMA200" in df_2y.columns and not pd.isna(df_2y["SMA200"].iloc[-1]) else "N/A", "#FF8C00"),
        (T("beta"), f"{info.get('beta', 'N/A')}", "#AB63FA"),
    ]
    for col, (label, val, color) in zip(ti_cols, ind_data):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div style='font-size:1.1rem;font-weight:700;color:{color};'>{val}</div>
        </div>
        """, unsafe_allow_html=True)

    # Sentiment gauge
    sentiment = get_sentiment(df_2y)
    sent_label = sentiment["label_ko"] if lang == "ko" else sentiment["label_en"]
    sent_color = "#00D4AA" if sent_label in ["Bullish", "강세"] else "#FF4B4B" if sent_label in ["Bearish", "약세"] else "#FFA500"

    scol1, scol2 = st.columns([1, 2])
    with scol1:
        st.markdown(f"""
        <div class='metric-card' style='padding:20px;'>
            <div class='metric-label'>{T('sentiment')}</div>
            <div style='font-size:2rem;font-weight:800;color:{sent_color};'>{sent_label}</div>
            <div style='color:#8B9DB0;font-size:0.85rem;margin-top:4px;'>Score: {sentiment['score']:.0f}/100</div>
        </div>
        """, unsafe_allow_html=True)
    with scol2:
        # Company description
        desc = info.get("longBusinessSummary", "")
        if desc:
            st.markdown(f"<div class='summary-box'>{desc[:400]}{'...' if len(desc)>400 else ''}</div>", unsafe_allow_html=True)

    # Macro indicators
    st.markdown(f"<div class='section-header'>{T('macro_indicators')}</div>", unsafe_allow_html=True)
    st.plotly_chart(build_macro_chart(macro_data, lang), use_container_width=True)

# ══════════════════ TAB 2: PREDICTION ══════════════════
with tabs[1]:
    st.markdown(f"<div class='section-header'>{T('pred_title')}</div>", unsafe_allow_html=True)
    st.markdown(f"<small style='color:#8B9DB0;'>{T('forecast_disclaimer')}</small>", unsafe_allow_html=True)

    horizons = [63, 126, 189, 252]  # trading days
    horizon_labels = [T("pred_3m"), T("pred_6m"), T("pred_9m"), T("pred_12m")]

    with st.spinner(T("loading")):
        predictions = predict_prices(df_5y, horizons)

    if predictions:
        # Forecast chart
        st.plotly_chart(build_forecast_chart(ticker, df_5y, predictions, lang), use_container_width=True)

        # Prediction cards
        st.markdown("<br>", unsafe_allow_html=True)
        pred_cols = st.columns(4)
        for col, h, label in zip(pred_cols, horizons, horizon_labels):
            if h in predictions:
                p = predictions[h]
                chg = p["change_pct"]
                chg_color = "#00D4AA" if chg >= 0 else "#FF4B4B"
                chg_arrow = "▲" if chg >= 0 else "▼"
                col.markdown(f"""
                <div class='prediction-card'>
                    <div class='pred-horizon'>{label}</div>
                    <div class='pred-price'>${p['base']:,.2f}</div>
                    <div style='color:{chg_color};font-size:1rem;font-weight:600;'>{chg_arrow} {abs(chg):.1f}%</div>
                    <div style='margin-top:10px;padding-top:10px;border-top:1px solid #2E3250;'>
                        <div style='color:#00D4AA;font-size:0.8rem;'>▲ {T("pred_bull")}: ${p['bull']:,.2f}</div>
                        <div style='color:#FF4B4B;font-size:0.8rem;'>▼ {T("pred_bear")}: ${p['bear']:,.2f}</div>
                    </div>
                    <div style='margin-top:8px;color:#8B9DB0;font-size:0.75rem;'>
                        {T("volatility")}: {p['vol_annual']:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Prediction methodology note
        st.markdown("<br>", unsafe_allow_html=True)
        if lang == "ko":
            methodology = """
            **예측 방법론:** 3가지 통계 모델의 앙상블을 사용합니다:
            - **선형 추세 회귀**: 장기 가격 추세 포착
            - **지수 가중 모멘텀**: 최근 수익률에 더 높은 가중치 부여
            - **평균 회귀 모델**: 200일 이동평균 대비 과도한 편차 조정
            - **신뢰 구간**: 역사적 변동성을 기반으로 강세/약세 시나리오 계산 (90% 신뢰구간)
            """
        else:
            methodology = """
            **Prediction Methodology:** Ensemble of 3 statistical models:
            - **Linear Trend Regression**: Captures long-term price trajectory
            - **Exponential Weighted Momentum**: Higher weight on recent returns
            - **Mean Reversion Model**: Adjusts for excessive deviation from 200-day MA
            - **Confidence Bands**: Bull/Bear scenarios based on historical volatility (90% CI)
            """
        st.info(methodology)
    else:
        st.warning("Insufficient data for prediction. Need at least 60 trading days.")

# ══════════════════ TAB 3: NEWS ══════════════════
with tabs[2]:
    col_title, col_update = st.columns([3, 1])
    with col_title:
        st.markdown(f"<div class='section-header'>{T('news_title')}</div>", unsafe_allow_html=True)
    with col_update:
        if st.button("🔄 " + ("새로고침" if lang == "ko" else "Refresh"), use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.markdown(f"<span class='update-badge'>🟢 {T('last_updated')}: {now_str} UTC</span>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner(T("news_loading")):
        articles = fetch_news(lang)

    if articles:
        # Filter by ticker if not index
        relevant = []
        other = []
        for art in articles:
            title_lower = art["title"].lower()
            ticker_lower = ticker.lower().replace("^", "").replace("=f", "")
            company_lower = company_name.lower()
            is_relevant = (
                ticker_lower in title_lower
                or any(w in title_lower for w in company_lower.split()[:2])
                or any(w in title_lower for w in ["market", "stock", "fed", "rate", "inflation",
                                                    "gdp", "economy", "wall street", "s&p", "nasdaq",
                                                    "dow", "시장", "주식", "연준", "금리"])
            )
            if is_relevant:
                relevant.append(art)
            else:
                other.append(art)

        all_arts = relevant + other

        # Display news in grid
        n_cols = 2
        for i in range(0, min(len(all_arts), 20), n_cols):
            row_arts = all_arts[i:i+n_cols]
            cols = st.columns(n_cols)
            for col, art in zip(cols, row_arts):
                with col:
                    st.markdown(f"""
                    <div class='news-card'>
                        <div class='news-title'><a href='{art['link']}' target='_blank' style='color:#EAEAEA;text-decoration:none;'>{art['title']}</a></div>
                        <div class='news-meta'>📡 {art['source']} &nbsp;|&nbsp; 🕐 {art['published'][:20] if art['published'] else 'N/A'}</div>
                        <div class='news-summary'>{art['summary']}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("Could not load news. Check your internet connection." if lang == "en" else "뉴스를 불러올 수 없습니다. 인터넷 연결을 확인하세요.")

# ══════════════════ TAB 4: GEOPOLITICAL ══════════════════
with tabs[3]:
    st.markdown(f"<div class='section-header'>{T('geo_title')}</div>", unsafe_allow_html=True)

    # Current macro metrics
    gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
    macro_display = [
        (T("geo_vix"), "VIX", "#FF4B4B"),
        (T("geo_oil"), "Oil (WTI)", "#FFA500"),
        (T("geo_gold"), "Gold", "#FFD700"),
        (T("geo_dxy"), "USD Index", "#64B5F6"),
        (T("geo_bonds"), "10Y Treasury", "#AB63FA"),
    ]
    for col, (label, key, color) in zip([gcol1, gcol2, gcol3, gcol4, gcol5], macro_display):
        data = macro_data.get(key, {})
        val = data.get("current", 0)
        chg = data.get("change_pct", 0)
        chg_color = "#00D4AA" if chg >= 0 else "#FF4B4B"
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div style='font-size:1.2rem;font-weight:700;color:{color};'>{val:,.2f}</div>
            <div style='color:{chg_color};font-size:0.85rem;'>{chg:+.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Risk factors
    st.markdown(f"<div class='section-header'>{T('geo_factors')}</div>", unsafe_allow_html=True)

    risk_factors_ko = [
        ("🇺🇸🇨🇳 미-중 무역 갈등", "높음", "트럼프 관세 145% 부과 → 반도체·기술주 압박, 공급망 재편 가속화", "high_risk"),
        ("🏦 연준(Fed) 통화정책", "중간", "2025년 금리 동결 기조 유지, 연내 1-2회 인하 가능성 → 성장주 긍정적", "med_risk"),
        ("🛢️ 중동 지정학 리스크", "중간", "이란-이스라엘 긴장 지속, WTI 가격 변동성 에너지 섹터 영향", "med_risk"),
        ("🇷🇺🇺🇦 러시아-우크라이나", "중간", "전쟁 장기화, 유럽 에너지 공급 불안 지속, 방산주 수혜", "med_risk"),
        ("💹 AI 과열 논쟁", "낮음", "엔비디아 등 AI 밸류에이션 논란, 실적 기반 검증 국면 진입", "low_risk"),
        ("📉 미국 국가부채", "중간", "35조 달러 돌파, 재정 적자 지속 → 장기 금리 상승 압력", "med_risk"),
    ]
    risk_factors_en = [
        ("🇺🇸🇨🇳 US-China Trade War", "HIGH", "Trump 145% tariffs → Tech/semiconductor pressure, supply chain restructuring", "high_risk"),
        ("🏦 Fed Monetary Policy", "MEDIUM", "Rate hold in 2025, 1-2 cuts possible → Positive for growth stocks", "med_risk"),
        ("🛢️ Middle East Tensions", "MEDIUM", "Iran-Israel tensions persist, WTI oil price volatility impacts energy sector", "med_risk"),
        ("🇷🇺🇺🇦 Russia-Ukraine War", "MEDIUM", "Prolonged conflict, EU energy supply uncertainty, defense stocks benefit", "med_risk"),
        ("💹 AI Bubble Concerns", "LOW", "NVIDIA etc. valuation debate, entering earnings-validation phase", "low_risk"),
        ("📉 US National Debt", "MEDIUM", "$35T+ debt, fiscal deficit → Long-term rate upward pressure", "med_risk"),
    ]

    risk_factors = risk_factors_ko if lang == "ko" else risk_factors_en
    risk_labels = {
        "high_risk": (T("high_risk"), "risk-high"),
        "med_risk": (T("med_risk"), "risk-med"),
        "low_risk": (T("low_risk"), "risk-low"),
    }

    rf_col1, rf_col2 = st.columns(2)
    for i, (title, risk_key, desc, risk_type) in enumerate(risk_factors):
        risk_text, risk_class = risk_labels[risk_type]
        target_col = rf_col1 if i % 2 == 0 else rf_col2
        with target_col:
            st.markdown(f"""
            <div class='geo-card'>
                <div style='display:flex;justify-content:space-between;align-items:center;'>
                    <span style='font-weight:700;color:#EAEAEA;font-size:0.95rem;'>{title}</span>
                    <span class='{risk_class}'>[{risk_text}]</span>
                </div>
                <div style='color:#B0BEC5;font-size:0.82rem;margin-top:6px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Historical geopolitical events chart
    st.plotly_chart(build_geo_timeline(lang), use_container_width=True)

    # Correlation explanation
    st.markdown(f"<div class='section-header'>{T('market_correlation')}</div>", unsafe_allow_html=True)
    corr_data_ko = {
        "이벤트 유형": ["금리 인상 사이클", "지정학 전쟁", "팬데믹/보건위기", "무역 전쟁", "AI/기술 붐", "금융위기"],
        "평균 초기 충격": ["-15%", "-10%", "-34%", "-12%", "+40%", "-50%"],
        "회복 기간": ["12-18개월", "3-6개월", "12개월", "6-12개월", "지속 상승", "24-36개월"],
        "수혜 섹터": ["금융, 에너지", "방산, 에너지", "바이오, 기술", "소재, 국내소비", "기술, 반도체", "헬스케어, 필수소비재"],
    }
    corr_data_en = {
        "Event Type": ["Rate Hike Cycle", "Geopolitical War", "Pandemic/Health Crisis", "Trade War", "AI/Tech Boom", "Financial Crisis"],
        "Avg Initial Shock": ["-15%", "-10%", "-34%", "-12%", "+40%", "-50%"],
        "Recovery Period": ["12-18 months", "3-6 months", "12 months", "6-12 months", "Sustained Rally", "24-36 months"],
        "Beneficiary Sectors": ["Finance, Energy", "Defense, Energy", "Biotech, Tech", "Materials, Domestic", "Tech, Semiconductors", "Healthcare, Staples"],
    }
    corr_df = pd.DataFrame(corr_data_ko if lang == "ko" else corr_data_en)
    st.dataframe(corr_df, use_container_width=True, hide_index=True)

# ══════════════════ TAB 5: HISTORY ══════════════════
with tabs[4]:
    st.markdown(f"<div class='section-header'>{T('hist_title')}</div>", unsafe_allow_html=True)

    # 2-line summary
    st.markdown(f"**{T('summary_analysis')}**")
    summary = generate_2line_summary(ticker, df_max, info, lang)
    if summary:
        st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Long-term price chart with events overlay
    if not df_max.empty:
        close_max = df_max["Close"]
        if close_max.ndim == 2:
            close_max = close_max.iloc[:, 0]

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=df_max.index,
            y=close_max.astype(float),
            name=company_name,
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.08)",
            line=dict(color="#00D4AA", width=1.5),
        ))

        # Overlay geopolitical events
        for event in GEOPOLITICAL_EVENTS:
            try:
                event_date = pd.Timestamp(event["date"] + "-01")
                if event_date >= df_max.index[0] and event_date <= df_max.index[-1]:
                    event_label = event[f"event_{lang}"].split("→")[0][:30]
                    color = "#FF4B4B" if event["impact"] < 0 else "#00D4AA"
                    fig_hist.add_vline(
                        x=event_date, line_dash="dot",
                        line_color=color, line_width=1.5, opacity=0.6,
                    )
                    fig_hist.add_annotation(
                        x=event_date,
                        y=float(close_max.max()) * 0.95,
                        text=event_label,
                        showarrow=False,
                        textangle=-90,
                        font=dict(size=9, color=color),
                        bgcolor="rgba(14,17,23,0.7)",
                    )
            except Exception:
                continue

        fig_hist.update_layout(
            template="plotly_dark",
            height=550,
            title=f"{company_name} — {'Full History with Key Events' if lang == 'en' else '전체 역사 & 주요 이벤트'}",
            xaxis_title="Date" if lang == "en" else "날짜",
            yaxis_title="Price (USD)" if lang == "en" else "주가 (USD)",
            margin=dict(l=0, r=0, t=50, b=0),
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Annual returns table
    st.markdown(f"<div class='section-header'>{'Annual Returns' if lang == 'en' else '연도별 수익률'}</div>", unsafe_allow_html=True)

    if not df_max.empty:
        close_max = df_max["Close"]
        if close_max.ndim == 2:
            close_max = close_max.iloc[:, 0]
        close_max = close_max.astype(float)

        annual_data = []
        df_yearly = close_max.resample("YE").last()
        for i in range(1, len(df_yearly)):
            year = df_yearly.index[i].year
            ret = (df_yearly.iloc[i] / df_yearly.iloc[i - 1] - 1) * 100
            annual_data.append({
                ("Year" if lang == "en" else "연도"): year,
                ("Return" if lang == "en" else "수익률"): f"{ret:+.1f}%",
                ("Price" if lang == "en" else "종가"): f"${df_yearly.iloc[i]:,.2f}",
                ("Performance" if lang == "en" else "성과"): "🟢 상승" if ret > 0 else "🔴 하락",
            })

        if annual_data:
            df_annual = pd.DataFrame(annual_data).tail(15)
            st.dataframe(df_annual, use_container_width=True, hide_index=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center;color:#4A5568;font-size:0.8rem;padding:16px;border-top:1px solid #2E3250;'>
    {'⚠️ 이 프로그램은 교육 및 분석 목적으로만 제공됩니다. 투자 결정은 전문 금융 어드바이저와 상담하세요.' if lang == 'ko'
     else '⚠️ This system is for educational and analytical purposes only. Consult a qualified financial advisor before making investment decisions.'}<br>
    Data: Yahoo Finance | News: Reuters, CNBC, MarketWatch | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC
</div>
""", unsafe_allow_html=True)
