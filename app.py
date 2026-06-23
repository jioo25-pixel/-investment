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
        "tab_company": "🏛️ Company History",
        "tab_relations": "🔗 Relationships",
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
        "tab_company": "🏛️ 기업 역사",
        "tab_relations": "🔗 이해관계",
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
tabs = st.tabs([T("tab_overview"), T("tab_predict"), T("tab_news"), T("tab_geo"), T("tab_history"), T("tab_company"), T("tab_relations")])

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

        # Overlay geopolitical events — vertical text, no rotation
        price_max = float(close_max.max())
        for idx_e, event in enumerate(GEOPOLITICAL_EVENTS):
            try:
                event_date = pd.Timestamp(event["date"] + "-01")
                if event_date >= df_max.index[0] and event_date <= df_max.index[-1]:
                    raw_label = event[f"event_{lang}"].split("→")[0].strip()
                    # Build top-to-bottom text: each character on its own line
                    vertical_text = "<br>".join(list(raw_label))
                    color = "#FF4B4B" if event["impact"] < 0 else "#00D4AA"
                    # Alternate y positions to prevent overlap
                    y_pos = price_max * (0.92 - (idx_e % 3) * 0.10)
                    fig_hist.add_vline(
                        x=event_date, line_dash="dot",
                        line_color=color, line_width=1.5, opacity=0.6,
                    )
                    fig_hist.add_annotation(
                        x=event_date,
                        y=y_pos,
                        text=vertical_text,
                        showarrow=False,
                        textangle=0,
                        font=dict(size=8, color=color, family="monospace"),
                        bgcolor="rgba(14,17,23,0.75)",
                        bordercolor=color,
                        borderwidth=1,
                        borderpad=2,
                        align="center",
                        xanchor="center",
                        yanchor="top",
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

# ══════════════════ TAB 6 + 7 RENDERED BELOW AFTER DB DEFINITIONS ══════════════════
# (see end of file after DB defs)

# ══════════════════ TAB 6: COMPANY HISTORY DB ══════════════════
COMPANY_HISTORY_DB = {
    "AAPL": {
        "name": "Apple Inc.",
        "founded": "1976",
        "founders": "Steve Jobs, Steve Wozniak, Ronald Wayne",
        "en": [
            ("1976", "🍎 Founded", "Apple Computer Co. founded in Jobs' garage. First product: Apple I personal computer."),
            ("1980", "📈 IPO", "Apple goes public at $22/share. Biggest US IPO since Ford Motor in 1956."),
            ("1984", "💻 Macintosh", "Iconic '1984' Super Bowl ad. First mass-market GUI computer launched."),
            ("1985", "🚪 Jobs Exits", "Steve Jobs forced out by board. Company struggles through late 80s/90s."),
            ("1997", "🔄 Jobs Returns", "Apple acquires NeXT for $429M, bringing Jobs back. Company near bankruptcy."),
            ("1998", "🖥️ iMac", "Colorful all-in-one iMac launches. Design-led revival begins."),
            ("2001", "🎵 iPod + iTunes", "iPod changes the music industry. iTunes Store follows in 2003."),
            ("2007", "📱 iPhone", "Steve Jobs unveils iPhone. Smartphone revolution begins."),
            ("2008", "📲 App Store", "App Store launches with 500 apps. Transforms software distribution."),
            ("2010", "📺 iPad", "iPad creates the modern tablet category."),
            ("2011", "💔 Jobs Passes", "Steve Jobs passes away. Tim Cook becomes CEO."),
            ("2014", "⌚ Apple Watch", "Wearables division begins. Watch + Health ecosystem expands."),
            ("2016", "🔧 Services Era", "Services (iCloud, Apple Music, App Store) become key revenue driver."),
            ("2020", "💰 $2T Valuation", "First US company to reach $2 trillion market cap."),
            ("2021", "🔬 Apple Silicon", "M1 chip — Apple's own ARM-based processor. Breaks Intel dependency."),
            ("2023", "🥽 Vision Pro", "Apple Vision Pro spatial computing headset announced at $3,499."),
            ("2024", "🤖 Apple Intelligence", "On-device AI features. Partnership with OpenAI for Siri enhancement."),
        ],
        "ko": [
            ("1976", "🍎 창업", "잡스의 차고에서 애플컴퓨터 설립. 첫 제품: Apple I 개인용 컴퓨터."),
            ("1980", "📈 상장", "주당 22달러로 IPO. 포드모터 이후 최대 규모 미국 IPO."),
            ("1984", "💻 매킨토시", "전설적인 '1984' 슈퍼볼 광고. 최초 대중용 GUI 컴퓨터 출시."),
            ("1985", "🚪 잡스 퇴출", "이사회에 의해 잡스 축출. 80-90년대 암흑기 시작."),
            ("1997", "🔄 잡스 복귀", "NeXT 4억2900만 달러에 인수하며 잡스 복귀. 회사는 파산 위기."),
            ("1998", "🖥️ iMac", "컬러풀한 일체형 iMac 출시. 디자인 중심 부활의 시작."),
            ("2001", "🎵 iPod + iTunes", "iPod으로 음악 산업 판도 변경. 2003년 iTunes 스토어 오픈."),
            ("2007", "📱 아이폰", "스티브 잡스 아이폰 공개. 스마트폰 혁명 시작."),
            ("2008", "📲 앱스토어", "500개 앱으로 앱스토어 오픈. 소프트웨어 유통 혁신."),
            ("2010", "📺 아이패드", "현대적 태블릿 카테고리 창조."),
            ("2011", "💔 잡스 별세", "스티브 잡스 별세. 팀 쿡 CEO 취임."),
            ("2014", "⌚ 애플워치", "웨어러블 사업부 시작. 건강 생태계 확장."),
            ("2016", "🔧 서비스 시대", "iCloud·애플뮤직·앱스토어 등 서비스가 핵심 수익원으로."),
            ("2020", "💰 시총 2조 달러", "미국 최초 시가총액 2조 달러 돌파."),
            ("2021", "🔬 애플 실리콘", "자체 ARM 기반 M1 칩 출시. 인텔 의존도 탈피."),
            ("2023", "🥽 비전 프로", "공간 컴퓨팅 헤드셋 애플 비전 프로 3499달러에 발표."),
            ("2024", "🤖 애플 인텔리전스", "온디바이스 AI 기능. OpenAI와 파트너십으로 시리 강화."),
        ],
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "founded": "1975",
        "founders": "Bill Gates, Paul Allen",
        "en": [
            ("1975", "🖥️ Founded", "Gates and Allen found Microsoft in Albuquerque, NM. First product: BASIC interpreter for Altair 8800."),
            ("1981", "💾 MS-DOS", "IBM licenses MS-DOS for its PC. Microsoft retains rights — pivotal decision."),
            ("1985", "🪟 Windows 1.0", "First Windows OS launched. GUI interface for IBM-compatible PCs."),
            ("1986", "📈 IPO", "Microsoft goes public at $21/share. Gates becomes a billionaire at 31."),
            ("1990", "📦 Office Suite", "Microsoft Office (Word+Excel+PowerPoint) bundles become dominant."),
            ("1995", "🌐 Windows 95 + IE", "Windows 95 massive launch. Internet Explorer bundled — browser wars begin."),
            ("2000", "⚖️ Antitrust", "DOJ antitrust case. Judge orders breakup (overturned on appeal). Stock peaks."),
            ("2001", "🎮 Xbox", "Xbox console launched, entering gaming market against Sony PlayStation."),
            ("2008", "☁️ Azure", "Microsoft Azure cloud platform launches. Cloud-first pivot begins."),
            ("2014", "🔄 Nadella CEO", "Satya Nadella becomes CEO. Open-source pivot, cloud-first strategy."),
            ("2016", "💼 LinkedIn $26B", "Acquires LinkedIn for $26.2 billion."),
            ("2018", "🐙 GitHub $7.5B", "Acquires GitHub for $7.5 billion. Developer community trust rebuilt."),
            ("2020", "🎮 Activision Deal", "Gaming push: $68.7B Activision Blizzard deal announced (closed 2023)."),
            ("2023", "🤖 OpenAI $10B", "$10B investment in OpenAI. Copilot AI integrated across all products."),
            ("2024", "👑 $3T Valuation", "Briefly surpasses Apple as world's most valuable company at $3T+."),
        ],
        "ko": [
            ("1975", "🖥️ 창업", "게이츠와 앨런, 뉴멕시코 주 앨버커키에서 마이크로소프트 설립. 첫 제품: Altair 8800용 BASIC 인터프리터."),
            ("1981", "💾 MS-DOS", "IBM이 MS-DOS 라이선스 취득. MS는 권리 보유 — 역사적 결정."),
            ("1985", "🪟 윈도우 1.0", "첫 윈도우 OS 출시. IBM 호환 PC용 GUI 인터페이스."),
            ("1986", "📈 상장", "주당 21달러 IPO. 게이츠, 31세에 억만장자."),
            ("1990", "📦 오피스 제품군", "마이크로소프트 오피스(워드+엑셀+파워포인트) 번들이 시장 지배."),
            ("1995", "🌐 윈도우95 + IE", "윈도우95 대대적 출시. IE 번들 — 브라우저 전쟁 시작."),
            ("2000", "⚖️ 독점금지 소송", "DOJ 반독점 소송. 분할 명령(항소심 취소). 주가 고점."),
            ("2001", "🎮 엑스박스", "엑스박스 콘솔 출시. 소니 플레이스테이션과 게임 시장 경쟁."),
            ("2008", "☁️ 애저", "마이크로소프트 애저 클라우드 플랫폼 출시. 클라우드 전환 시작."),
            ("2014", "🔄 나델라 CEO", "사티아 나델라 CEO 취임. 오픈소스 전환, 클라우드 퍼스트 전략."),
            ("2016", "💼 링크드인 260억불", "링크드인 262억 달러에 인수."),
            ("2018", "🐙 깃허브 75억불", "깃허브 75억 달러에 인수. 개발자 커뮤니티 신뢰 회복."),
            ("2020", "🎮 액티비전", "게임 강화: 687억 달러 액티비전블리자드 인수 발표 (2023년 완료)."),
            ("2023", "🤖 오픈AI 100억불", "오픈AI 100억 달러 투자. 코파일럿 AI 전 제품 통합."),
            ("2024", "👑 시총 3조 달러", "잠시 애플 제치고 세계 최고 시총 3조 달러+ 달성."),
        ],
    },
    "NVDA": {
        "name": "NVIDIA Corporation",
        "founded": "1993",
        "founders": "Jensen Huang, Chris Malachowsky, Curtis Priem",
        "en": [
            ("1993", "🟩 Founded", "Jensen Huang, Chris Malachowsky, and Curtis Priem found NVIDIA in Sunnyvale, CA."),
            ("1995", "🎮 NV1 GPU", "First product NV1 launched. Early 3D graphics for gaming."),
            ("1999", "💎 GeForce 256", "Coined term 'GPU'. GeForce 256 is world's first GPU. Nvidia goes public."),
            ("2006", "⚡ CUDA", "CUDA parallel computing platform launched. Unlocks GPU for general computing beyond graphics."),
            ("2012", "🧠 AlexNet Moment", "Deep learning breakthrough — AlexNet trained on NVIDIA GPUs wins ImageNet. AI era begins."),
            ("2016", "🚗 Autonomous Driving", "NVIDIA Drive PX platform for self-driving cars. Partners with Tesla, Toyota."),
            ("2018", "🔬 RTX Ray Tracing", "RTX 20 series — real-time ray tracing for photorealistic gaming graphics."),
            ("2019", "🔴 Mellanox $6.9B", "Acquires Mellanox for $6.9B. Enters data center networking."),
            ("2020", "💰 ARM Deal", "$40B Arm acquisition announced (blocked by regulators in 2022)."),
            ("2022", "🤖 ChatGPT Era", "ChatGPT launches. H100 GPU demand explodes. Nvidia supply shortages begin."),
            ("2023", "🚀 $1T Club", "Joins $1 trillion market cap club. H100 becomes the 'gold of AI'."),
            ("2024", "👑 #1 Most Valuable", "Briefly becomes world's most valuable company. Blackwell B200 GPU announced."),
            ("2025", "🌐 Sovereign AI", "Nations building AI infrastructure — NVIDIA at center of global AI arms race."),
        ],
        "ko": [
            ("1993", "🟩 창업", "젠슨 황, 크리스 말라초프스키, 커티스 프리엠이 캘리포니아 서니베일에서 엔비디아 설립."),
            ("1995", "🎮 NV1 GPU", "첫 제품 NV1 출시. 게임용 초기 3D 그래픽."),
            ("1999", "💎 지포스 256", "'GPU' 용어 창안. 지포스 256은 세계 최초 GPU. 나스닥 상장."),
            ("2006", "⚡ CUDA", "CUDA 병렬 컴퓨팅 플랫폼 출시. 그래픽 외 범용 컴퓨팅으로 GPU 영역 확장."),
            ("2012", "🧠 알렉스넷 모멘트", "딥러닝 혁신 — 알렉스넷이 NVIDIA GPU로 훈련하여 ImageNet 우승. AI 시대 시작."),
            ("2016", "🚗 자율주행", "자율주행용 NVIDIA Drive PX 플랫폼. 테슬라·토요타와 파트너십."),
            ("2018", "🔬 RTX 레이트레이싱", "RTX 20 시리즈 — 사실적 게임 그래픽용 실시간 레이트레이싱."),
            ("2019", "🔴 멜라녹스 69억불", "멜라녹스 69억 달러 인수. 데이터센터 네트워킹 진출."),
            ("2020", "💰 ARM 인수 시도", "400억 달러 ARM 인수 발표 (규제 당국에 의해 2022년 무산)."),
            ("2022", "🤖 챗GPT 시대", "챗GPT 출시. H100 GPU 수요 폭발. 엔비디아 공급 부족 시작."),
            ("2023", "🚀 시총 1조 달러", "시가총액 1조 달러 클럽 합류. H100은 'AI의 금'으로 불림."),
            ("2024", "👑 세계 1위 기업", "잠시 세계 최고 시가총액 기업 등극. 블랙웰 B200 GPU 발표."),
            ("2025", "🌐 소버린 AI", "각국이 AI 인프라 구축 — 엔비디아가 글로벌 AI 군비경쟁의 중심."),
        ],
    },
    "TSLA": {
        "name": "Tesla, Inc.",
        "founded": "2003",
        "founders": "Martin Eberhard, Marc Tarpenning (Elon Musk joined 2004)",
        "en": [
            ("2003", "⚡ Founded", "Martin Eberhard and Marc Tarpenning found Tesla Motors in San Carlos, CA."),
            ("2004", "💼 Musk Invests", "Elon Musk leads Series A funding of $7.5M. Becomes chairman."),
            ("2008", "🚗 Roadster", "First Tesla Roadster delivered. World's first highway-legal electric sports car. Musk becomes CEO."),
            ("2010", "📈 IPO", "Tesla IPO at $17/share. First US automaker IPO since Ford in 1956."),
            ("2012", "🚘 Model S", "Model S sedan launched. Named Motor Trend Car of the Year. Supercharger network begins."),
            ("2015", "🔋 Powerwall", "Tesla Energy division. Powerwall home battery + utility-scale Powerpack."),
            ("2016", "🤖 Autopilot", "Autopilot hardware 2.0. Acquires SolarCity for $2.6B."),
            ("2017", "🏭 Gigafactory 1", "Nevada Gigafactory operational. Battery production at scale begins."),
            ("2019", "🛻 Cybertruck Reveal", "Cybertruck unveiled. 'Armored glass' incident becomes viral moment."),
            ("2020", "💰 S&P 500 Entry", "Tesla added to S&P 500. Stock rises 743% in 2020. Joins $1T club briefly."),
            ("2021", "⚡ 4680 Battery", "Structural battery pack + 4680 cells. New architecture for cost reduction."),
            ("2022", "🤖 Optimus Robot", "Tesla Bot (Optimus) humanoid robot revealed. AI Day showcase."),
            ("2023", "📉 Price Wars", "Tesla cuts prices aggressively. Margin pressure. Cybertruck finally delivered."),
            ("2024", "🚕 Robotaxi", "FSD v12 neural net driving. Robotaxi event showcasing autonomous future."),
        ],
        "ko": [
            ("2003", "⚡ 창업", "마틴 에버하드와 마크 타페닝이 캘리포니아 산카를로스에서 테슬라모터스 설립."),
            ("2004", "💼 머스크 투자", "일론 머스크가 750만 달러 시리즈A 투자 주도. 이사회 의장 취임."),
            ("2008", "🚗 로드스터", "첫 테슬라 로드스터 인도. 세계 최초 고속도로 주행 가능 전기 스포츠카. 머스크 CEO."),
            ("2010", "📈 상장", "주당 17달러 IPO. 1956년 포드 이후 최초 미국 자동차기업 IPO."),
            ("2012", "🚘 모델 S", "모델 S 세단 출시. 모터트렌드 올해의 차 선정. 슈퍼차저 네트워크 시작."),
            ("2015", "🔋 파워월", "테슬라 에너지 사업부. 가정용 파워월 + 산업용 파워팩."),
            ("2016", "🤖 오토파일럿", "오토파일럿 하드웨어 2.0. 26억 달러에 솔라시티 인수."),
            ("2017", "🏭 기가팩토리 1", "네바다 기가팩토리 가동. 배터리 대량 생산 시작."),
            ("2019", "🛻 사이버트럭 공개", "사이버트럭 공개. '강화유리' 사고가 바이럴 명장면으로."),
            ("2020", "💰 S&P 500 편입", "테슬라 S&P 500 편입. 2020년 주가 743% 상승. 잠시 1조 달러 클럽."),
            ("2021", "⚡ 4680 배터리", "구조용 배터리 팩 + 4680 셀. 원가 절감을 위한 새 아키텍처."),
            ("2022", "🤖 옵티머스 로봇", "테슬라봇(옵티머스) 인간형 로봇 공개. AI 데이 쇼케이스."),
            ("2023", "📉 가격 전쟁", "테슬라 공격적 가격 인하. 마진 압박. 사이버트럭 드디어 인도."),
            ("2024", "🚕 로보택시", "FSD v12 신경망 자율주행. 자율주행 미래 선보이는 로보택시 이벤트."),
        ],
    },
}

COMPANY_RELATIONS_DB = {
    "AAPL": {
        "suppliers_en": [
            ("TSMC (Taiwan)", "A-series / M-series chip fabrication", "Critical", "#FF4B4B"),
            ("Samsung (Korea)", "OLED displays, NAND flash memory", "High", "#FFA500"),
            ("Foxconn (Taiwan/China)", "iPhone assembly — 70% of production", "Critical", "#FF4B4B"),
            ("Corning (USA)", "Gorilla Glass for all iPhone screens", "High", "#FFA500"),
            ("Broadcom (USA)", "Wi-Fi / Bluetooth chips", "Medium", "#FFD700"),
            ("Murata (Japan)", "Capacitors, wireless components", "Medium", "#FFD700"),
            ("LG Energy (Korea)", "Battery cells for MacBook / iPad", "Medium", "#FFD700"),
            ("Skyworks (USA)", "RF chips for cellular connectivity", "Medium", "#FFD700"),
        ],
        "competitors_en": [
            ("Samsung", "Smartphones, tablets, wearables — direct global rival"),
            ("Google / Alphabet", "Android OS ecosystem, Pixel phones, AI assistant"),
            ("Microsoft", "PC/laptop market, cloud services (Azure vs iCloud)"),
            ("Meta", "VR/AR headsets — Vision Pro vs Quest"),
            ("Spotify", "Music streaming vs Apple Music"),
            ("Amazon", "Smart home, voice assistant (Alexa vs Siri)"),
        ],
        "customers_en": [
            ("Consumer (Direct)", "~60% revenue — iPhones sold via Apple Store, carriers"),
            ("Enterprise", "Corporate Mac/iPad deployments, MDM ecosystem"),
            ("Education", "iPad in Education program — millions of devices"),
            ("Developers", "App Store ecosystem — 30M+ registered developers"),
        ],
        "resources_en": [
            ("Rare Earth Metals", "Neodymium (magnets), Terbium — sourced from China/Australia"),
            ("Cobalt", "Battery cathode — primarily DRC (Congo). ESG risk."),
            ("Aluminum", "MacBook/iPhone chassis — global commodity"),
            ("Silicon Wafers", "Semiconductor base — TSMC processes"),
            ("Lithium", "Battery anodes — Chile, Australia sourcing"),
        ],
        "subsidiaries_en": [
            ("Beats Electronics", "Acquired 2014 for $3B — headphones & audio"),
            ("Shazam", "Acquired 2018 for $400M — music recognition"),
            ("Intel Modem Division", "Acquired 2019 for $1B — 5G modem tech"),
            ("AuthenTec", "Acquired 2012 for $356M — Touch ID fingerprint tech"),
        ],
        "suppliers_ko": [
            ("TSMC (대만)", "A시리즈/M시리즈 칩 파운드리", "핵심", "#FF4B4B"),
            ("삼성 (한국)", "OLED 디스플레이, NAND 플래시 메모리", "높음", "#FFA500"),
            ("폭스콘 (대만/중국)", "아이폰 조립 — 생산의 70%", "핵심", "#FF4B4B"),
            ("코닝 (미국)", "전 아이폰 화면용 고릴라 글라스", "높음", "#FFA500"),
            ("브로드컴 (미국)", "Wi-Fi / 블루투스 칩", "중간", "#FFD700"),
            ("무라타 (일본)", "커패시터, 무선 부품", "중간", "#FFD700"),
            ("LG에너지솔루션 (한국)", "맥북/아이패드용 배터리 셀", "중간", "#FFD700"),
            ("스카이웍스 (미국)", "셀룰러 연결용 RF 칩", "중간", "#FFD700"),
        ],
        "competitors_ko": [
            ("삼성", "스마트폰, 태블릿, 웨어러블 — 직접 글로벌 경쟁자"),
            ("구글/알파벳", "안드로이드 OS 생태계, 픽셀폰, AI 어시스턴트"),
            ("마이크로소프트", "PC/노트북 시장, 클라우드 서비스 (애저 vs iCloud)"),
            ("메타", "VR/AR 헤드셋 — 비전 프로 vs 퀘스트"),
            ("스포티파이", "음악 스트리밍 vs 애플뮤직"),
            ("아마존", "스마트홈, 음성 어시스턴트 (알렉사 vs 시리)"),
        ],
        "customers_ko": [
            ("일반 소비자 (직접)", "매출 약 60% — 애플스토어·통신사 통해 아이폰 판매"),
            ("기업", "법인 맥/아이패드 도입, MDM 생태계"),
            ("교육기관", "아이패드 교육 프로그램 — 수백만 기기"),
            ("개발자", "앱스토어 생태계 — 3000만+ 등록 개발자"),
        ],
        "resources_ko": [
            ("희토류 금속", "네오디뮴(자석), 테르븀 — 중국/호주 소싱"),
            ("코발트", "배터리 양극재 — 주로 DRC(콩고). ESG 리스크."),
            ("알루미늄", "맥북/아이폰 하우징 — 글로벌 원자재"),
            ("실리콘 웨이퍼", "반도체 기반 — TSMC 가공"),
            ("리튬", "배터리 음극재 — 칠레, 호주 소싱"),
        ],
        "subsidiaries_ko": [
            ("비츠 일렉트로닉스", "2014년 30억 달러 인수 — 헤드폰 및 오디오"),
            ("샤잠", "2018년 4억 달러 인수 — 음악 인식"),
            ("인텔 모뎀 사업부", "2019년 10억 달러 인수 — 5G 모뎀 기술"),
            ("오센텍", "2012년 3억5600만 달러 인수 — 터치ID 지문 기술"),
        ],
    },
    "NVDA": {
        "suppliers_en": [
            ("TSMC (Taiwan)", "All GPU fabrication — 4nm/3nm nodes", "Critical", "#FF4B4B"),
            ("Samsung (Korea)", "HBM memory (High Bandwidth Memory) for H100/H200", "Critical", "#FF4B4B"),
            ("SK Hynix (Korea)", "HBM3E memory for Blackwell B200 GPUs", "Critical", "#FF4B4B"),
            ("Micron (USA)", "GDDR6X memory for consumer GPUs", "High", "#FFA500"),
            ("ASE Group (Taiwan)", "Advanced chip packaging / CoWoS", "High", "#FFA500"),
            ("Synopsys / Cadence (USA)", "EDA tools for chip design", "Medium", "#FFD700"),
        ],
        "competitors_en": [
            ("AMD", "MI300X GPUs — main AI competitor. 'The #2 AI chip'"),
            ("Intel", "Gaudi 3 AI accelerators. Battling for data center"),
            ("Google (TPU)", "Custom TPUs power all Google AI — not for sale"),
            ("Amazon (Trainium)", "AWS custom AI chips — reducing Nvidia dependency"),
            ("Microsoft (Maia)", "Azure Maia AI accelerator — Microsoft custom silicon"),
            ("Qualcomm", "AI inference chips for edge/mobile devices"),
        ],
        "customers_en": [
            ("Microsoft / Azure", "Largest single cloud buyer of H100s. Billions in orders."),
            ("Meta", "350,000+ H100s for Llama AI training"),
            ("Google", "Large H100 buyer alongside own TPUs"),
            ("Amazon AWS", "H100 instances. Also building own chips."),
            ("Tesla", "D1 training cluster + FSD compute"),
            ("OpenAI", "Primary training compute partner"),
        ],
        "resources_en": [
            ("TSMC Capacity", "Allocation-limited — TSMC CoWoS packaging bottleneck"),
            ("HBM Supply", "Samsung/SK Hynix HBM production constrained through 2025"),
            ("Rare Earths", "Neodymium, Tantalum for chip components"),
            ("Power Infrastructure", "H100 DGX racks require 10kW+ per unit — data center power a limit"),
        ],
        "subsidiaries_en": [
            ("Mellanox (InfiniBand)", "Acquired 2020 $6.9B — AI cluster networking backbone"),
            ("Cumulus Networks", "Acquired 2020 — network OS software"),
            ("Arm (Stake)", "$40B acquisition blocked; still ecosystem partner"),
            ("DeepMind partnership", "Research collaboration (not owned)"),
        ],
        "suppliers_ko": [
            ("TSMC (대만)", "전 GPU 파운드리 — 4nm/3nm 공정", "핵심", "#FF4B4B"),
            ("삼성 (한국)", "H100/H200용 HBM (고대역폭 메모리)", "핵심", "#FF4B4B"),
            ("SK하이닉스 (한국)", "블랙웰 B200 GPU용 HBM3E 메모리", "핵심", "#FF4B4B"),
            ("마이크론 (미국)", "소비자 GPU용 GDDR6X 메모리", "높음", "#FFA500"),
            ("ASE그룹 (대만)", "고급 칩 패키징 / CoWoS", "높음", "#FFA500"),
            ("시놉시스/캐던스 (미국)", "칩 설계용 EDA 툴", "중간", "#FFD700"),
        ],
        "competitors_ko": [
            ("AMD", "MI300X GPU — 주요 AI 경쟁자. 'AI 칩 2위'"),
            ("인텔", "가우디3 AI 가속기. 데이터센터 경쟁"),
            ("구글 (TPU)", "자체 TPU로 전 구글 AI 구동 — 판매 안 함"),
            ("아마존 (Trainium)", "AWS 자체 AI 칩 — 엔비디아 의존도 축소 시도"),
            ("마이크로소프트 (Maia)", "애저 마이아 AI 가속기 — MS 자체 실리콘"),
            ("퀄컴", "엣지/모바일 기기용 AI 추론 칩"),
        ],
        "customers_ko": [
            ("마이크로소프트/애저", "H100 최대 단일 구매 클라우드. 수십억 달러 주문."),
            ("메타", "라마 AI 학습용 H100 35만+ 대 구매"),
            ("구글", "자체 TPU와 함께 대규모 H100 구매"),
            ("아마존 AWS", "H100 인스턴스 제공. 자체 칩도 개발 중."),
            ("테슬라", "D1 학습 클러스터 + FSD 컴퓨팅"),
            ("오픈AI", "주요 학습 컴퓨팅 파트너"),
        ],
        "resources_ko": [
            ("TSMC 생산 용량", "할당 제한 — TSMC CoWoS 패키징 병목"),
            ("HBM 공급", "삼성/SK하이닉스 HBM 생산 2025년까지 제한"),
            ("희토류", "칩 부품용 네오디뮴, 탄탈럼"),
            ("전력 인프라", "H100 DGX 랙 당 10kW+ 필요 — 데이터센터 전력이 한계"),
        ],
        "subsidiaries_ko": [
            ("멜라녹스 (인피니밴드)", "2020년 69억 달러 인수 — AI 클러스터 네트워킹 backbone"),
            ("큐물러스 네트웍스", "2020년 인수 — 네트워크 OS 소프트웨어"),
            ("Arm (지분)", "400억 달러 인수 무산; 여전히 생태계 파트너"),
            ("딥마인드 파트너십", "연구 협력 (지분 소유 아님)"),
        ],
    },
    "TSLA": {
        "suppliers_en": [
            ("Panasonic (Japan)", "2170 battery cells for Model 3/Y at Nevada Gigafactory", "Critical", "#FF4B4B"),
            ("CATL (China)", "LFP battery cells for Standard Range models", "Critical", "#FF4B4B"),
            ("LG Energy (Korea)", "Cylindrical cells for Model S/X/Cybertruck", "High", "#FFA500"),
            ("Samsung SDI (Korea)", "Battery cells for energy storage products", "Medium", "#FFD700"),
            ("NVIDIA", "Drive PX chips for early Autopilot (now custom HW4)", "Low", "#00D4AA"),
            ("Mobileye (Intel)", "Early Autopilot sensor processing (ended 2016)", "Historical", "#8B9DB0"),
        ],
        "competitors_en": [
            ("BYD (China)", "#1 EV seller globally in 2023. Aggressive on price."),
            ("GM / Chevy Bolt, Silverado EV", "US legacy automaker going electric"),
            ("Ford (F-150 Lightning, Mustang Mach-E)", "Strong brand + dealer network"),
            ("Hyundai/Kia (Ioniq 6, EV6)", "Top-rated EVs, ICCU tech advantage"),
            ("Rivian", "EV trucks/SUVs, Amazon delivery van partnership"),
            ("Lucid Motors", "Premium long-range EV — targets Model S"),
        ],
        "customers_en": [
            ("Consumer Direct", "No dealers — all sales via Tesla.com and stores"),
            ("Enterprise Fleet", "Corporate EV fleets, taxi services"),
            ("Utilities / Grid", "Megapack utility-scale battery storage"),
            ("Homeowners", "Powerwall + Solar Roof ecosystem"),
        ],
        "resources_en": [
            ("Lithium", "#1 battery input — Chile, Australia, Nevada. Price volatile."),
            ("Cobalt", "Reduced in LFP cells but still in NCA batteries"),
            ("Nickel", "High-nickel cathode for energy density — Russian/Indonesian supply"),
            ("Copper", "Motors, wiring harness — extensive per vehicle"),
            ("Rare Earths", "Permanent magnets in motors — China supply risk"),
        ],
        "subsidiaries_en": [
            ("Tesla Energy", "Powerwall, Megapack, Solar Roof products"),
            ("Tesla Insurance", "Direct auto insurance using driving behavior data"),
            ("SolarCity (merged)", "Acquired 2016 $2.6B — solar panels, now Tesla Solar"),
            ("The Boring Company (related)", "Elon Musk venture — Tesla vehicles in tunnels"),
        ],
        "suppliers_ko": [
            ("파나소닉 (일본)", "네바다 기가팩토리에서 모델3/Y용 2170 배터리 셀", "핵심", "#FF4B4B"),
            ("CATL (중국)", "스탠다드 레인지 모델용 LFP 배터리 셀", "핵심", "#FF4B4B"),
            ("LG에너지솔루션 (한국)", "모델S/X/사이버트럭용 원통형 셀", "높음", "#FFA500"),
            ("삼성SDI (한국)", "에너지 저장 제품용 배터리 셀", "중간", "#FFD700"),
            ("엔비디아", "초기 오토파일럿용 Drive PX 칩 (현재 자체 HW4)", "낮음", "#00D4AA"),
            ("모빌아이 (인텔)", "초기 오토파일럿 센서 처리 (2016년 종료)", "과거", "#8B9DB0"),
        ],
        "competitors_ko": [
            ("BYD (중국)", "2023년 글로벌 EV 판매 1위. 공격적인 가격 정책."),
            ("GM / 쉐보레 볼트, 실버라도 EV", "미국 전통 자동차기업의 전기차 전환"),
            ("포드 (F-150 라이트닝, 머스탱 맥-E)", "강력한 브랜드 + 딜러 네트워크"),
            ("현대/기아 (아이오닉6, EV6)", "최고 평가 EV, ICCU 기술 우위"),
            ("리비안", "EV 트럭/SUV, 아마존 배달 밴 파트너십"),
            ("루시드 모터스", "프리미엄 장거리 EV — 모델S 타겟"),
        ],
        "customers_ko": [
            ("소비자 직판", "딜러 없음 — Tesla.com과 직영점에서만 판매"),
            ("기업 차량", "법인 EV 차량대, 택시 서비스"),
            ("전력 회사/그리드", "메가팩 유틸리티 스케일 배터리 저장"),
            ("홈오너", "파워월 + 솔라루프 생태계"),
        ],
        "resources_ko": [
            ("리튬", "배터리 1위 원료 — 칠레, 호주, 네바다. 가격 변동성 큼."),
            ("코발트", "LFP 셀에서는 감소했으나 NCA 배터리에는 여전히 필요"),
            ("니켈", "에너지 밀도용 고니켈 양극재 — 러시아/인도네시아 공급"),
            ("구리", "모터, 와이어링 하네스 — 차량당 대량 사용"),
            ("희토류", "모터 영구 자석 — 중국 공급 리스크"),
        ],
        "subsidiaries_ko": [
            ("테슬라 에너지", "파워월, 메가팩, 솔라루프 제품군"),
            ("테슬라 인슈어런스", "주행 데이터 기반 직접 자동차 보험"),
            ("솔라시티 (합병)", "2016년 26억 달러 인수 — 태양광, 현재 테슬라 솔라"),
            ("더 보링 컴퍼니 (관련)", "일론 머스크 벤처 — 터널 내 테슬라 차량 운행"),
        ],
    },
}

def get_company_history(ticker: str, lang: str) -> list:
    key = ticker.upper().replace("^", "")
    db = COMPANY_HISTORY_DB.get(key)
    if not db:
        return []
    return db.get(lang, db.get("en", []))

def get_company_meta(ticker: str) -> dict:
    key = ticker.upper().replace("^", "")
    return COMPANY_HISTORY_DB.get(key, {})

def get_company_relations(ticker: str) -> dict:
    key = ticker.upper().replace("^", "")
    return COMPANY_RELATIONS_DB.get(key, {})

def build_history_timeline(history: list, company_name: str, lang: str) -> go.Figure:
    if not history:
        return go.Figure()

    years = [h[0] for h in history]
    titles = [h[1] for h in history]
    descs = [h[2] for h in history]

    # Alternate above/below to avoid overlap
    y_pos = [1 if i % 2 == 0 else -1 for i in range(len(history))]
    y_text = [1.15 if y > 0 else -1.15 for y in y_pos]

    fig = go.Figure()

    # Timeline spine
    fig.add_shape(type="line", x0=years[0], x1=years[-1], y0=0, y1=0,
                  line=dict(color="#3A4060", width=2))

    # Event dots and connectors
    for i, (yr, title, desc, yp, yt) in enumerate(zip(years, titles, descs, y_pos, y_text)):
        color = "#FFA500" if i % 3 == 0 else "#00D4AA" if i % 3 == 1 else "#AB63FA"
        # Connector line
        fig.add_shape(type="line", x0=yr, x1=yr, y0=0, y1=yp * 0.9,
                      line=dict(color=color, width=1.5, dash="dot"))
        # Dot on spine
        fig.add_trace(go.Scatter(
            x=[yr], y=[0],
            mode="markers",
            marker=dict(size=12, color=color, line=dict(color="white", width=2)),
            hovertext=f"<b>{yr} {title}</b><br>{desc}",
            hoverinfo="text",
            showlegend=False,
        ))
        # Label
        fig.add_annotation(
            x=yr, y=yt,
            text=f"<b>{yr}</b><br>{title}",
            showarrow=False,
            font=dict(size=9, color=color),
            bgcolor="rgba(20,25,45,0.85)",
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
            align="center",
        )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        title=f"{company_name} — {'Corporate History Timeline' if lang == 'en' else '기업 역사 타임라인'}",
        xaxis=dict(showgrid=False, zeroline=False, tickmode="array",
                   tickvals=years, ticktext=years, tickangle=45),
        yaxis=dict(visible=False, range=[-1.8, 1.8]),
        margin=dict(l=0, r=0, t=50, b=60),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        hovermode="closest",
    )
    return fig

# ══════════════════ TAB 6: COMPANY HISTORY TIMELINE ══════════════════
with tabs[5]:
    meta = get_company_meta(ticker)
    history_list = get_company_history(ticker, lang)

    if meta:
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1E2130,#16213E);border-radius:14px;
                    padding:20px 28px;margin-bottom:20px;border:1px solid #2E3250;'>
            <div style='font-size:1.5rem;font-weight:800;color:#FFA500;'>{meta.get("name","")}</div>
            <div style='color:#B0BEC5;margin-top:6px;font-size:0.9rem;'>
                {'창업연도' if lang=='ko' else 'Founded'}: <b style='color:#FFFFFF;'>{meta.get("founded","")}</b>
                &nbsp;|&nbsp;
                {'창업자' if lang=='ko' else 'Founders'}: <b style='color:#FFFFFF;'>{meta.get("founders","")}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if history_list:
            st.plotly_chart(
                build_history_timeline(history_list, meta.get("name", ticker), lang),
                use_container_width=True,
            )
            st.markdown(f"<div class='section-header'>{'Key Milestones' if lang=='en' else '주요 이정표 상세'}</div>", unsafe_allow_html=True)
            n_cols = 3
            for i in range(0, len(history_list), n_cols):
                row = history_list[i:i+n_cols]
                cols_h = st.columns(n_cols)
                for col_h, (yr, title, desc) in zip(cols_h, row):
                    with col_h:
                        st.markdown(f"""
                        <div style='background:#1E2130;border-left:3px solid #FFA500;
                                    border-radius:8px;padding:12px 14px;margin-bottom:10px;min-height:100px;'>
                            <div style='color:#FFA500;font-size:1rem;font-weight:700;'>{yr}</div>
                            <div style='color:#FFFFFF;font-size:0.88rem;font-weight:600;margin:4px 0;'>{title}</div>
                            <div style='color:#B0BEC5;font-size:0.78rem;line-height:1.5;'>{desc}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("이 기업의 상세 역사 데이터가 준비 중입니다. AAPL, MSFT, NVDA, TSLA는 지원됩니다." if lang=="ko"
                    else "Detailed history not yet available. Try AAPL, MSFT, NVDA, or TSLA.")
    else:
        desc_text = info.get("longBusinessSummary","")
        if desc_text:
            st.markdown(f"<div class='summary-box'>{desc_text}</div>", unsafe_allow_html=True)
        facts = [
            ("Sector" if lang=="en" else "섹터", info.get("sector","N/A")),
            ("Industry" if lang=="en" else "업종", info.get("industry","N/A")),
            ("Country" if lang=="en" else "국가", info.get("country","N/A")),
            ("Employees" if lang=="en" else "임직원", f"{info.get('fullTimeEmployees',0):,}" if info.get("fullTimeEmployees") else "N/A"),
        ]
        fa, fb = st.columns(2)
        for j, (k, v) in enumerate(facts):
            (fa if j%2==0 else fb).markdown(f"""
            <div class='metric-card' style='text-align:left;padding:12px 16px;'>
                <span style='color:#8B9DB0;font-size:0.8rem;'>{k}</span><br>
                <span style='color:#FFFFFF;font-weight:600;'>{v}</span>
            </div>""", unsafe_allow_html=True)
        st.info("📌 " + ("AAPL·MSFT·NVDA·TSLA 검색 시 상세 역사 타임라인을 볼 수 있습니다." if lang=="ko"
                          else "Search AAPL, MSFT, NVDA, or TSLA for full history timeline."))

# ══════════════════ TAB 7: RELATIONSHIPS ══════════════════
with tabs[6]:
    st.markdown(f"<div class='section-header'>{'Corporate Relationships & Dependencies' if lang=='en' else '기업 이해관계 및 의존성 분석'}</div>",
                unsafe_allow_html=True)
    relations = get_company_relations(ticker)

    if relations:
        suffix = "_ko" if lang=="ko" else "_en"
        rel_tabs = st.tabs([
            "🏭 " + ("Suppliers" if lang=="en" else "공급망·협력사"),
            "⚔️ " + ("Competitors" if lang=="en" else "경쟁사"),
            "👥 " + ("Customers" if lang=="en" else "주요 고객"),
            "⛏️ " + ("Resources" if lang=="en" else "핵심 자원·원자재"),
            "🏢 " + ("Subsidiaries" if lang=="en" else "자회사·인수기업"),
        ])
        with rel_tabs[0]:
            for sup_name, role, importance, color in relations.get("suppliers"+suffix, []):
                imp_width = {"Critical":100,"핵심":100,"High":75,"높음":75,"Medium":50,"중간":50,"Low":25,"낮음":25,"Historical":15,"과거":15}
                w = imp_width.get(importance, 50)
                st.markdown(f"""
                <div style='background:#1A1F35;border-radius:10px;padding:14px 18px;margin-bottom:10px;border:1px solid {color}40;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <span style='font-weight:700;color:#FFFFFF;'>🏭 {sup_name}</span>
                        <span style='color:{color};font-weight:700;background:{color}20;padding:2px 10px;border-radius:12px;font-size:0.8rem;'>{importance}</span>
                    </div>
                    <div style='color:#B0BEC5;font-size:0.82rem;margin:6px 0 8px 0;'>{role}</div>
                    <div style='background:#0D1120;border-radius:4px;height:6px;'>
                        <div style='background:{color};width:{w}%;height:6px;border-radius:4px;'></div>
                    </div>
                </div>""", unsafe_allow_html=True)
        with rel_tabs[1]:
            cc1, cc2 = st.columns(2)
            for i, (comp_name, comp_desc) in enumerate(relations.get("competitors"+suffix, [])):
                (cc1 if i%2==0 else cc2).markdown(f"""
                <div class='geo-card' style='border-left:3px solid #FF4B4B;'>
                    <div style='font-weight:700;color:#FF6B6B;'>⚔️ {comp_name}</div>
                    <div style='color:#B0BEC5;font-size:0.82rem;margin-top:6px;'>{comp_desc}</div>
                </div>""", unsafe_allow_html=True)
        with rel_tabs[2]:
            for cust_name, cust_desc in relations.get("customers"+suffix, []):
                st.markdown(f"""
                <div class='geo-card' style='border-left:3px solid #00D4AA;'>
                    <div style='font-weight:700;color:#00D4AA;'>👥 {cust_name}</div>
                    <div style='color:#B0BEC5;font-size:0.82rem;margin-top:6px;'>{cust_desc}</div>
                </div>""", unsafe_allow_html=True)
        with rel_tabs[3]:
            for res_name, res_desc in relations.get("resources"+suffix, []):
                st.markdown(f"""
                <div class='geo-card' style='border-left:3px solid #FFD700;'>
                    <div style='font-weight:700;color:#FFD700;'>⛏️ {res_name}</div>
                    <div style='color:#B0BEC5;font-size:0.82rem;margin-top:6px;'>{res_desc}</div>
                </div>""", unsafe_allow_html=True)
        with rel_tabs[4]:
            for sub_name, sub_desc in relations.get("subsidiaries"+suffix, []):
                st.markdown(f"""
                <div class='geo-card' style='border-left:3px solid #AB63FA;'>
                    <div style='font-weight:700;color:#AB63FA;'>🏢 {sub_name}</div>
                    <div style='color:#B0BEC5;font-size:0.82rem;margin-top:6px;'>{sub_desc}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-header'>{'Dependency Overview' if lang=='en' else '의존성 종합 현황'}</div>", unsafe_allow_html=True)
        suffix2 = "_ko" if lang=="ko" else "_en"
        categories_r = (["공급망","경쟁사","고객","핵심자원","자회사"] if lang=="ko"
                        else ["Suppliers","Competitors","Customers","Resources","Subsidiaries"])
        counts_r = [
            len(relations.get("suppliers"+suffix2,[])),
            len(relations.get("competitors"+suffix2,[])),
            len(relations.get("customers"+suffix2,[])),
            len(relations.get("resources"+suffix2,[])),
            len(relations.get("subsidiaries"+suffix2,[])),
        ]
        fig_rel = go.Figure(go.Bar(x=categories_r, y=counts_r,
            marker_color=["#FFA500","#FF4B4B","#00D4AA","#FFD700","#AB63FA"],
            text=counts_r, textposition="outside"))
        fig_rel.update_layout(template="plotly_dark", height=300,
            title=f"{company_name} — {'Mapped Relationships' if lang=='en' else '매핑된 이해관계 수'}",
            margin=dict(l=0,r=0,t=50,b=0), plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
        st.plotly_chart(fig_rel, use_container_width=True)
    else:
        st.info("📌 " + ("현재 AAPL·MSFT·NVDA·TSLA의 이해관계 데이터가 제공됩니다." if lang=="ko"
                          else "Relationship data available for AAPL, MSFT, NVDA, TSLA."))

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center;color:#4A5568;font-size:0.8rem;padding:16px;border-top:1px solid #2E3250;'>
    {'⚠️ 이 프로그램은 교육 및 분석 목적으로만 제공됩니다. 투자 결정은 전문 금융 어드바이저와 상담하세요.' if lang == 'ko'
     else '⚠️ This system is for educational and analytical purposes only. Consult a qualified financial advisor before making investment decisions.'}<br>
    Data: Yahoo Finance | News: Reuters, CNBC, MarketWatch | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC
</div>
""", unsafe_allow_html=True)
