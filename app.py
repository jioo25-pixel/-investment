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

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:
    _HAS_AUTOREFRESH = False

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
        "tab_semi": "💾 Semiconductors",
        "tab_sectors": "🗂️ Sector Explorer",
        "tab_invest": "📈 Investment Analysis",
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
        "tab_semi": "💾 반도체 생태계",
        "tab_sectors": "🗂️ 섹터 탐색",
        "tab_invest": "📈 투자 분석",
    },
}

# ─── SESSION STATE INIT ───────────────────────────────────────────────────────
if "lang" not in st.session_state:
    st.session_state.lang = "ko"
if "ticker" not in st.session_state:
    st.session_state.ticker = "^GSPC"
if "last_news_update" not in st.session_state:
    st.session_state.last_news_update = None
if "news_translated" not in st.session_state:
    st.session_state.news_translated = False
if "geo_translated" not in st.session_state:
    st.session_state.geo_translated = False
if "sidebar_view" not in st.session_state:
    st.session_state.sidebar_view = None  # None = normal tabs, "semi" or "sectors"
if "data_fetched_at" not in st.session_state:
    st.session_state.data_fetched_at = None

def T(key):
    return TEXTS[st.session_state.lang].get(key, key)

# ─── SEARCH MAPPING (Korean / English / Ticker aliases) ───────────────────────
SEARCH_MAP = {
    # Indices
    "s&p500":"^GSPC","s&p 500":"^GSPC","sp500":"^GSPC","에스앤피500":"^GSPC",
    "s&p":"^GSPC","snp":"^GSPC","snp500":"^GSPC",
    "다우":"^DJI","dow":"^DJI","dow jones":"^DJI","다우존스":"^DJI",
    "나스닥":"^IXIC","nasdaq":"^IXIC",
    "러셀":"^RUT","russell":"^RUT","russell2000":"^RUT",
    "vix":"^VIX","공포지수":"^VIX","변동성지수":"^VIX",
    # US Big Tech
    "애플":"AAPL","apple":"AAPL",
    "마이크로소프트":"MSFT","microsoft":"MSFT",
    "구글":"GOOGL","google":"GOOGL","알파벳":"GOOGL","alphabet":"GOOGL",
    "메타":"META","meta":"META","페이스북":"META","facebook":"META",
    "아마존":"AMZN","amazon":"AMZN",
    "넷플릭스":"NFLX","netflix":"NFLX",
    "테슬라":"TSLA","tesla":"TSLA",
    "엔비디아":"NVDA","nvidia":"NVDA","엔비":"NVDA",
    "어도비":"ADBE","adobe":"ADBE",
    "세일즈포스":"CRM","salesforce":"CRM",
    "오라클":"ORCL","oracle":"ORCL",
    "인텔":"INTC","intel":"INTC",
    "amd":"AMD","에이엠디":"AMD",
    "퀄컴":"QCOM","qualcomm":"QCOM",
    "브로드컴":"AVGO","broadcom":"AVGO",
    "arm":"ARM","암홀딩스":"ARM","arm홀딩스":"ARM",
    "팔란티어":"PLTR","palantir":"PLTR",
    "스노우플레이크":"SNOW","snowflake":"SNOW",
    "데이터독":"DDOG","datadog":"DDOG",
    "서비스나우":"NOW","servicenow":"NOW",
    "우버":"UBER","uber":"UBER",
    "에어비앤비":"ABNB","airbnb":"ABNB",
    "도어대시":"DASH","doordash":"DASH",
    "스냅":"SNAP","snap":"SNAP","스냅챗":"SNAP","snapchat":"SNAP",
    "핀터레스트":"PINS","pinterest":"PINS",
    "트위터":"X","twitter":"X",
    "줌":"ZM","zoom":"ZM",
    "쇼피파이":"SHOP","shopify":"SHOP",
    "스포티파이":"SPOT","spotify":"SPOT",
    "트윌리오":"TWLO","twilio":"TWLO",
    "ui패스":"PATH","uipath":"PATH",
    "몽고디비":"MDB","mongodb":"MDB",
    "컨플루언트":"CFLT","confluent":"CFLT",
    "센티넬원":"S","sentinelone":"S",
    # Semiconductors
    "tsmc":"TSM","대만반도체":"TSM","타이완반도체":"TSM",
    "마이크론":"MU","micron":"MU",
    "램리서치":"LRCX","lam research":"LRCX","램":"LRCX",
    "kla":"KLAC","kla코퍼레이션":"KLAC",
    "어플라이드머티리얼스":"AMAT","applied materials":"AMAT","어플라이드":"AMAT",
    "asml":"ASML","에이에스엠엘":"ASML",
    "시놉시스":"SNPS","synopsys":"SNPS",
    "캐던스":"CDNS","cadence":"CDNS",
    "텍사스인스트루먼트":"TXN","texas instruments":"TXN","ti":"TXN",
    "마블":"MRVL","marvell":"MRVL",
    "엔테그리스":"ENTG","entegris":"ENTG",
    "글로벌파운드리":"GFS","globalfoundries":"GFS",
    "모놀리식파워":"MPWR","monolithic power":"MPWR",
    "온세미":"ON","on semiconductor":"ON",
    "nxp":"NXPI","엔엑스피":"NXPI",
    "울프스피드":"WOLF","wolfspeed":"WOLF",
    "스카이웍스":"SWKS","skyworks":"SWKS",
    "마이크로칩":"MCHP","microchip":"MCHP",
    # Korean Stocks
    "삼성전자":"005930.KS","삼성":"005930.KS","samsung":"005930.KS",
    "sk하이닉스":"000660.KS","하이닉스":"000660.KS","skhynix":"000660.KS","sk hynix":"000660.KS",
    "삼성바이오로직스":"207940.KS","삼성바이오":"207940.KS",
    "현대차":"005380.KS","현대자동차":"005380.KS","hyundai":"005380.KS",
    "기아":"000270.KS","kia":"000270.KS",
    "네이버":"035420.KS","naver":"035420.KS",
    "카카오":"035720.KS","kakao":"035720.KS",
    "lg화학":"051910.KS","lgchem":"051910.KS","lg chem":"051910.KS",
    "lg에너지솔루션":"373220.KS","lg에너지":"373220.KS","lges":"373220.KS",
    "삼성sdi":"006400.KS","samsung sdi":"006400.KS",
    "sk이노베이션":"096770.KS","sk innovation":"096770.KS",
    "셀트리온":"068270.KS","celltrion":"068270.KS",
    "포스코":"005490.KS","posco":"005490.KS",
    "한국전력":"015760.KS","kepco":"015760.KS",
    "kb금융":"105560.KS","kb":"105560.KS",
    "신한금융":"055550.KS","shinhan":"055550.KS",
    "현대모비스":"012330.KS","hyundai mobis":"012330.KS",
    "카카오뱅크":"323410.KS","kakaobank":"323410.KS",
    "크래프톤":"259960.KS","krafton":"259960.KS","배틀그라운드":"259960.KS",
    "하이브":"352820.KS","hybe":"352820.KS","방탄소년단":"352820.KS","bts":"352820.KS",
    "엔씨소프트":"036570.KS","ncsoft":"036570.KS",
    "넥슨":"3659.T","nexon":"3659.T",
    "두산에너빌리티":"034020.KS","두산":"034020.KS",
    "한화에어로스페이스":"012450.KS","한화에어로":"012450.KS",
    # Japan
    "토요타":"TM","toyota":"TM",
    "소니":"SONY","sony":"SONY",
    "소프트뱅크":"9984.T","softbank":"9984.T",
    "파나소닉":"PCRFY","panasonic":"PCRFY",
    "도쿄일렉트론":"TOELY","tokyo electron":"TOELY","tel":"TOELY",
    "신에츠화학":"SIEGY","shin-etsu":"SIEGY","신에츠":"SIEGY",
    "히타치":"HTHIY","hitachi":"HTHIY",
    "무라타":"MRAAY","murata":"MRAAY",
    "키엔스":"KYCCF","keyence":"KYCCF",
    # China
    "알리바바":"BABA","alibaba":"BABA",
    "jd닷컴":"JD","jd.com":"JD",
    "바이두":"BIDU","baidu":"BIDU",
    "니오":"NIO","nio":"NIO",
    "리오토":"LI","li auto":"LI",
    "샤오펑":"XPEV","xpeng":"XPEV",
    "pdd":"PDD","핀둬둬":"PDD",
    # Finance
    "제이피모건":"JPM","jpmorgan":"JPM","jp모건":"JPM",
    "뱅크오브아메리카":"BAC","bank of america":"BAC","뱅오아":"BAC",
    "웰스파고":"WFC","wells fargo":"WFC",
    "골드만삭스":"GS","goldman sachs":"GS","골드만":"GS",
    "모건스탠리":"MS","morgan stanley":"MS",
    "버크셔해서웨이":"BRK-B","berkshire":"BRK-B","버크셔":"BRK-B",
    "비자":"V","visa":"V",
    "마스터카드":"MA","mastercard":"MA",
    "페이팔":"PYPL","paypal":"PYPL",
    "블록":"SQ","block":"SQ","스퀘어":"SQ","square":"SQ",
    "코인베이스":"COIN","coinbase":"COIN",
    "블랙스톤":"BX","blackstone":"BX",
    "kkr":"KKR",
    "찰스슈왑":"SCHW","charles schwab":"SCHW",
    # Healthcare & Pharma
    "존슨앤존슨":"JNJ","johnson & johnson":"JNJ",
    "유나이티드헬스":"UNH","unitedhealth":"UNH",
    "일라이릴리":"LLY","eli lilly":"LLY","릴리":"LLY",
    "노보노디스크":"NVO","novo nordisk":"NVO","노보":"NVO",
    "애브비":"ABBV","abbvie":"ABBV",
    "머크":"MRK","merck":"MRK",
    "화이자":"PFE","pfizer":"PFE",
    "써모피셔":"TMO","thermo fisher":"TMO",
    "모더나":"MRNA","moderna":"MRNA",
    "바이오엔텍":"BNTX","biontech":"BNTX",
    "길리어드":"GILD","gilead":"GILD",
    "바이오젠":"BIIB","biogen":"BIIB",
    "일루미나":"ILMN","illumina":"ILMN",
    "인튜이티브서지컬":"ISRG","intuitive surgical":"ISRG",
    "덱스컴":"DXCM","dexcom":"DXCM",
    "애보트":"ABT","abbott":"ABT",
    # Energy & Oil
    "엑슨모빌":"XOM","exxonmobil":"XOM","exxon":"XOM",
    "쉐브론":"CVX","chevron":"CVX",
    "코노코필립스":"COP","conocophillips":"COP",
    "할리버튼":"HAL","halliburton":"HAL",
    "슐럼버거":"SLB","schlumberger":"SLB",
    "쉘":"SHEL","shell":"SHEL",
    "bp":"BP",
    "토탈에너지스":"TTE","totalenergies":"TTE",
    "쉐니어에너지":"LNG","cheniere":"LNG",
    "넥스트에라":"NEE","nextera":"NEE",
    "컨스텔레이션에너지":"CEG","constellation energy":"CEG",
    "비스트라":"VST","vistra":"VST",
    # Solar
    "앤페이즈":"ENPH","enphase":"ENPH",
    "퍼스트솔라":"FSLR","first solar":"FSLR",
    "솔라엣지":"SEDG","solaredge":"SEDG",
    "선런":"RUN","sunrun":"RUN",
    # Automotive
    "포드":"F","ford":"F",
    "제네럴모터스":"GM","general motors":"GM","지엠":"GM",
    "리비안":"RIVN","rivian":"RIVN",
    "루시드":"LCID","lucid":"LCID",
    "스텔란티스":"STLA","stellantis":"STLA",
    # Defense
    "록히드마틴":"LMT","lockheed martin":"LMT","록히드":"LMT",
    "레이시온":"RTX","raytheon":"RTX",
    "노스롭그루만":"NOC","northrop grumman":"NOC","노스롭":"NOC",
    "제네럴다이내믹스":"GD","general dynamics":"GD",
    "보잉":"BA","boeing":"BA",
    "크라토스":"KTOS","kratos":"KTOS",
    # Battery & Materials
    "퀀텀스케이프":"QS","quantumscape":"QS",
    "에노빅스":"ENVX","enovix":"ENVX",
    "알버마를":"ALB","albemarle":"ALB",
    "sqm":"SQM",
    # Quantum
    "아이온큐":"IONQ","ionq":"IONQ",
    "리게티":"RGTI","rigetti":"RGTI",
    "디웨이브":"QBTS","d-wave":"QBTS",
    # Space
    "로켓랩":"RKLB","rocket lab":"RKLB",
    "ast스페이스모바일":"ASTS","ast spacemobile":"ASTS",
    "인튜이티브머신스":"LUNR","intuitive machines":"LUNR",
    "버진갤럭틱":"SPCE","virgin galactic":"SPCE",
    # Consumer
    "월마트":"WMT","walmart":"WMT",
    "코스트코":"COST","costco":"COST",
    "홈디포":"HD","home depot":"HD",
    "나이키":"NKE","nike":"NKE",
    "맥도날드":"MCD","mcdonald's":"MCD","맥도":"MCD",
    "스타벅스":"SBUX","starbucks":"SBUX",
    "코카콜라":"KO","coca-cola":"KO","코카":"KO",
    "펩시콜라":"PEP","pepsi":"PEP","펩시":"PEP",
    "프록터앤갬블":"PG","procter & gamble":"PG","p&g":"PG",
    "룰루레몬":"LULU","lululemon":"LULU",
    "치폴레":"CMG","chipotle":"CMG",
    "타겟":"TGT","target":"TGT",
    "로우스":"LOW","lowe's":"LOW",
    # Crypto
    "비트코인":"BTC-USD","bitcoin":"BTC-USD","btc":"BTC-USD",
    "이더리움":"ETH-USD","ethereum":"ETH-USD","eth":"ETH-USD",
    "솔라나":"SOL-USD","solana":"SOL-USD","sol":"SOL-USD",
    "마이크로스트래티지":"MSTR","microstrategy":"MSTR",
    # Nuclear
    "카메코":"CCJ","cameco":"CCJ",
    "옥로":"OKLO","oklo":"OKLO",
    "뉴스케일":"SMR","nuscale":"SMR",
    # Commodities
    "원유":"CL=F","wti":"CL=F","crude oil":"CL=F",
    "천연가스":"NG=F","natural gas":"NG=F",
    "금":"GC=F","gold":"GC=F",
    "은":"SI=F","silver":"SI=F",
    "구리":"HG=F","copper":"HG=F",
    # ETFs
    "qqq":"QQQ","나스닥etf":"QQQ",
    "spy":"SPY","s&p500etf":"SPY",
    "ark":"ARKK","아크":"ARKK",
    "반도체etf":"SOXX","soxx":"SOXX",
    "배터리etf":"LIT","리튬etf":"LIT",
    "우라늄etf":"URA",
    "태양광etf":"TAN",
    "방산etf":"ITA",
}

def resolve_ticker(query: str) -> str:
    """Resolve Korean/English name or ticker to Yahoo Finance symbol."""
    q = query.strip()
    q_lower = q.lower()
    if q_lower in SEARCH_MAP:
        return SEARCH_MAP[q_lower]
    # Partial match
    for key, val in SEARCH_MAP.items():
        if q_lower in key or key in q_lower:
            return val
    return q.upper()

# ─── TERM GLOSSARY (hover tooltips) ──────────────────────────────────────────
GLOSSARY = {
    # Technical Indicators
    "RSI":       "RSI(상대강도지수): 0~100 숫자로 주가가 너무 많이 올랐는지(70 이상=과매수) 너무 많이 떨어졌는지(30 이하=과매도) 알려주는 지표",
    "RSI (14)":  "RSI(14일): 최근 14일 기준으로 주가가 과열인지 침체인지 0~100으로 표시. 70↑ 과열, 30↓ 침체",
    "MACD":      "MACD: 단기와 장기 평균 가격의 차이로 '지금 주가가 오르는 힘인지, 내리는 힘인지' 방향을 알려주는 지표",
    "SMA50":     "SMA50(50일 이동평균): 최근 50거래일 평균 가격. 이 선 위에 있으면 단기 상승 추세",
    "SMA200":    "SMA200(200일 이동평균): 최근 200거래일 평균 가격. 이 선 위에 있으면 장기 상승 추세. 황금선이라 불림",
    "50일 이동평균": "최근 50거래일(약 2.5개월) 주가의 평균선. 단기 추세를 파악할 때 사용",
    "200일 이동평균": "최근 200거래일(약 10개월) 주가의 평균선. 장기 추세 파악용. 이 선 위=강세, 아래=약세",
    "볼린저밴드":  "볼린저밴드: 주가가 움직이는 정상적인 범위(위-중간-아래 3선). 상단선 근처=과열, 하단선 근처=과냉각",
    "BB Upper":  "볼린저밴드 상단선: 주가가 여기 근처면 너무 많이 올라 조정 가능성 있음",
    "BB Lower":  "볼린저밴드 하단선: 주가가 여기 근처면 너무 많이 떨어져 반등 가능성 있음",
    "베타 (S&P 500 대비)": "베타: 시장 전체가 1% 움직일 때 이 주식이 얼마나 움직이는지. 1.5이면 시장보다 1.5배 더 크게 움직임(고위험·고수익)",
    "Beta vs S&P 500": "Beta: When the market moves 1%, this stock moves by this amount. 1.5 = 50% more volatile than market",
    "변동성 (연율화)": "변동성: 주가가 1년 동안 얼마나 들쭉날쭉 움직이는지를 %로 표시. 20% 이하=안정, 40% 이상=고위험",
    "Volatility (Annualized)": "Annualized Volatility: How much the stock price swings up/down in a year. Under 20%=stable, over 40%=high risk",
    "샤프 비율":   "샤프 비율: 위험을 감수한 것 대비 얼마나 많은 수익을 냈는지. 1 이상=좋음, 2 이상=매우 좋음",
    "Sharpe Ratio": "Sharpe Ratio: Return earned per unit of risk. Above 1 = good, above 2 = excellent",
    "P/E Ratio":   "P/E(주가수익비율): 주가가 1년 이익의 몇 배인지. 20이면 '지금 이익의 20년치를 주고 사는 것'. 낮을수록 저평가",
    "주가수익비율(P/E)": "주가수익비율: 이 주식이 1년 벌어들이는 돈의 몇 배로 거래되는지. 낮으면 저렴, 높으면 비싼 편",
    "시가총액":    "시가총액: 이 회사의 총 가치(주가 × 발행 주식 수). 클수록 대형주",
    "Market Cap":  "Market Cap: Total value of all shares = share price × total shares. Larger = bigger company",
    "거래량":      "거래량: 오늘 이 주식이 얼마나 많이 사고팔렸는지. 거래량이 갑자기 늘면 중요한 사건이 있다는 신호",
    "Volume":      "Volume: Number of shares traded today. A sudden spike usually signals an important event",
    "52주 최고가": "52주 최고가: 최근 1년 중 가장 높았던 주가. 현재가가 여기 근처면 역대 최고 수준",
    "52주 최저가": "52주 최저가: 최근 1년 중 가장 낮았던 주가. 현재가가 여기 근처면 크게 하락한 상태",
    "52-Week High": "Highest price in the past 52 weeks. Current price near here = near all-time recent high",
    "52-Week Low":  "Lowest price in the past 52 weeks. Current price near here = significantly down from peak",
    # Macro terms
    "VIX":        "VIX(공포지수): 투자자들이 얼마나 두려워하는지를 나타내는 지수. 20 이하=안정, 30 이상=공포, 40 이상=극심한 공포",
    "VIX (공포지수)": "공포지수(VIX): 미국 증시의 불안감 척도. 20 이하=안정, 30 이상=투자자 두려움, 40↑=극심한 패닉",
    "WTI":        "WTI(서부텍사스원유): 미국 기준 원유 가격. 전 세계 에너지·물가·운송비용에 영향을 줌",
    "원유 (WTI)":  "서부텍사스원유: 미국산 원유의 기준 가격. 오르면 물가 상승·에너지주 강세, 내리면 그 반대",
    "WTI 원유":   "서부텍사스원유: 국제 원유 가격의 기준. 배럴당 가격으로 표시. 에너지·항공·운송 업종에 직접 영향",
    "DXY":        "달러인덱스(DXY): 미국 달러가 다른 주요 통화들 대비 얼마나 강한지. 오르면 달러 강세=신흥국 압박, 내리면 달러 약세",
    "달러 인덱스 (DXY)": "달러인덱스: 달러 가치를 유로·엔 등 6개 통화 대비 측정. 오르면 미국 수출 기업에 불리",
    "USD Index (DXY)": "Dollar Index: Measures USD strength vs 6 major currencies. Rising = stronger dollar = headwind for US exporters",
    "10Y Treasury": "10-Year Treasury Yield: The interest rate on US government 10-year bonds. Rising yield = higher borrowing costs = headwind for stocks",
    "미국채 10년 수익률": "미국 10년 국채금리: 미국 정부가 10년짜리 국채를 발행할 때 지급하는 금리. 오르면 주식시장에 부담",
    "10년 국채":   "미국 10년물 국채금리: 가장 중요한 기준금리. 오르면 대출비용↑, 주식 매력↓. 시장 전체의 나침반",
    "Gold":        "Gold (금): 불확실성이 커질 때 안전자산으로 오르는 경향. 달러가 약해지거나 전쟁·위기 시 상승",
    "금":          "안전자산 금: 전쟁·경제위기·달러 약세 시 가격이 오르는 대표적 안전자산",
    "Bitcoin":     "비트코인: 세계 최대 암호화폐. 위험자산으로 분류되어 주식시장과 함께 움직이는 경향",
    "비트코인":    "세계 최대 암호화폐. 발행량 2100만개로 제한. 디지털 금이라 불리며 인플레이션 헤지 수단으로도 활용",
    # Prediction terms
    "Bull Case":   "Bull Case(강세 시나리오): 모든 것이 잘 풀릴 때의 낙관적 주가 예측값",
    "Base Case":   "Base Case(기본 시나리오): 현재 추세가 그대로 이어질 때의 주가 예측값",
    "Bear Case":   "Bear Case(약세 시나리오): 악재가 터졌을 때의 비관적 주가 예측값",
    "강세 시나리오": "모든 상황이 유리하게 흘러갈 경우(호실적·금리인하 등)의 낙관적 목표주가",
    "기본 시나리오": "현재 추세가 그대로 유지될 경우의 주가 예측. 가장 가능성 높은 시나리오",
    "약세 시나리오": "악재(경기침체·금리급등 등)가 발생할 경우의 비관적 예측 주가",
    "IPO":         "IPO(기업공개): 회사가 처음으로 주식시장에 상장하여 일반인에게 주식을 파는 것",
    "EPS":         "EPS(주당순이익): 회사가 주식 1주당 얼마를 벌었는지. 높을수록 회사 실적이 좋다는 의미",
    "ETF":         "ETF(상장지수펀드): 여러 주식을 묶어 하나처럼 거래하는 상품. 분산투자 효과로 개별 주식보다 안전",
    "CAGR":        "CAGR(연평균성장률): 매년 평균 몇 %씩 성장했는지. 복리로 계산한 성장 속도",
    "시장 심리":   "시장 심리(Sentiment): 투자자들이 지금 낙관적인지(강세) 비관적인지(약세) 나타내는 분위기 지표",
    "Market Sentiment": "Sentiment: Whether investors feel optimistic (bullish) or pessimistic (bearish) about the market",
    "HBM":         "HBM(고대역폭메모리): AI 서버·GPU에 들어가는 초고속 메모리. 엔비디아 AI칩 핵심 부품. SK하이닉스·삼성이 주요 공급",
    "EUV":         "EUV(극자외선 노광): 머리카락 1/10000 굵기의 초미세 반도체 회로를 그리는 ASML만의 기술. 이 장비 없이 최첨단 칩 불가",
    "GPU":         "GPU(그래픽처리장치): 원래 게임 그래픽용이었으나 AI 학습에 필수적인 연산 칩. 엔비디아가 세계 1위",
    "DRAM":        "DRAM(디램): 컴퓨터/스마트폰의 임시 기억장치(RAM). 전원 끄면 지워짐. 삼성·SK하이닉스·마이크론이 세계 3대 제조사",
    "NAND":        "NAND 플래시: SSD·USB·스마트폰에 쓰이는 저장 장치. 전원 꺼도 데이터 유지",
    "LFP":         "LFP(인산철 배터리): 전기차에 쓰이는 배터리. 코발트 없어 저렴하고 안전하지만 에너지 밀도가 낮음. 테슬라·BYD 사용",
    "GLP-1":       "GLP-1(비만치료제): 오젬픽·위고비 등 식욕 억제 당뇨·비만 치료제. 일라이릴리·노보노디스크의 핵심 의약품",
    "SMR":         "SMR(소형모듈원자로): 기존 원자력보다 작고 경제적인 차세대 원전. AI 데이터센터 전력 공급원으로 주목",
    "OSAT":        "OSAT(반도체 후공정): 완성된 칩을 포장하고 테스트하는 공정. ASE·암코가 세계 1·2위",
    "EDA":         "EDA(전자설계자동화): 반도체 설계에 쓰이는 소프트웨어 툴. 시놉시스·캐던스가 양분",
    "CoWoS":       "CoWoS(칩 패키징 기술): 여러 칩을 하나처럼 붙이는 TSMC의 첨단 패키징. AI GPU 생산의 병목 구간",
    "REIT":        "REIT(부동산투자신탁): 건물·토지 등에 투자하고 임대수익을 주주에게 나눠주는 펀드 형태의 주식",
    "CRISPR":      "CRISPR(유전자가위): DNA를 정밀하게 잘라 편집하는 기술. 암·유전병 치료 혁명을 이끄는 바이오 기술",
    "mRNA":        "mRNA(메신저RNA): 코로나 백신으로 유명해진 기술. 세포에게 특정 단백질 만드는 법을 알려줘 면역력 형성",
    "SoC":         "SoC(시스템온칩): CPU·GPU·메모리 등 여러 기능을 칩 하나에 집약한 것. 스마트폰 두뇌 역할",
    "LNG":         "LNG(액화천연가스): 천연가스를 -162도로 냉각해 액체로 만든 것. 선박으로 수출 가능해 전 세계 에너지 무역의 핵심",
    "셰일가스":    "셰일가스: 암석층에 갇혀있는 천연가스. 미국이 수평 시추 기술로 세계 최대 생산국. 에너지 독립의 핵심",
    "ESG":         "ESG(환경·사회·지배구조): 기업이 환경(E)·사회(S)·윤리경영(G)을 잘 하는지 평가하는 기준. 기관투자자의 투자 기준",
    "MCU":         "MCU(마이크로컨트롤러): 가전제품·자동차·산업기기 내부에 들어가는 소형 두뇌 칩. TI·마이크로칩이 주요 생산",
    "P/B Ratio":   "P/B(주가순자산비율): 주가를 회사의 순자산 대비 몇 배로 사는지. 1 미만이면 청산 가치보다 싸게 거래",
    "ROE":         "ROE(자기자본이익률): 주주 돈으로 얼마를 벌었는지. 15% 이상이면 우수한 수익성",
    "FCF":         "FCF(잉여현금흐름): 영업활동 후 실제로 남은 현금. 배당·자사주 매입에 쓰이는 진짜 이익",
    # Valuation
    "PER":          "PER(주가수익비율): 현재 주가 ÷ 주당순이익. 낮을수록 저평가. 같은 업종 평균과 비교하는 것이 중요",
    "PBR":          "PBR(주가순자산비율): 주가 ÷ 주당순자산. 1배 미만이면 자산가치보다 싸게 거래되는 저평가 주식",
    "PEG":          "PEG(주가수익성장비율): PER ÷ EPS 성장률. 1 미만=저평가, 1 초과=고평가. 성장성까지 고려한 밸류에이션",
    "P/S Ratio":    "P/S(주가매출비율): 주가 ÷ 주당매출액. 이익이 없는 성장기업 평가에 유용. 낮을수록 매출 대비 저평가",
    "EV/EBITDA":    "EV/EBITDA: 기업 전체 가치(시총+부채) ÷ 세전·이자·감가상각 전 이익. 기업 인수합병 시 주로 사용하는 지표",
    # Profitability
    "ROA":          "ROA(총자산이익률): 순이익 ÷ 총자산 × 100. 회사 전체 자산으로 얼마를 버는지. 높을수록 효율적 경영",
    "ROIC":         "ROIC(투자자본수익률): 순영업이익 ÷ 투자자본 × 100. 부채·자본 모두 포함한 자본 효율성. WACC보다 높으면 가치 창출",
    "영업이익률":   "영업이익률: 매출에서 영업비용을 빼고 남은 비율. 본업의 수익성. 높을수록 핵심 사업이 경쟁우위 보유",
    "Operating Margin": "Operating Margin: Operating profit ÷ revenue. Shows how profitable the core business is. Higher = stronger competitive moat",
    "Gross Margin": "Gross Margin(매출총이익률): 매출에서 원가만 뺀 이익 비율. 브랜드 파워와 가격결정력을 보여주는 핵심 지표",
    "Net Margin":   "Net Margin(순이익률): 매출 중 최종적으로 남는 순이익 비율. 모든 비용·세금 차감 후 실질 수익성",
    # Growth
    "Revenue Growth": "매출 성장률: 전년 대비 매출이 얼마나 늘었는지. 기업의 사업 확장 속도를 측정하는 핵심 성장 지표",
    "EPS Growth":   "EPS 성장률: 주당순이익이 전년 대비 얼마나 증가했는지. 이익 성장 속도를 보여줌. 지속 성장 시 주가 상승 동력",
    # Stability
    "부채비율":     "부채비율: 총부채 ÷ 자기자본 × 100. 낮을수록 재무 안정. 100% 이하가 안전권. 200% 이상이면 주의 필요",
    "Debt/Equity":  "Debt-to-Equity: Total debt ÷ shareholders equity. Lower = more financially stable. Under 1.0 is generally safe",
    "유동비율":     "유동비율: 유동자산 ÷ 유동부채 × 100. 1년 이내 갚아야 할 빚을 1년 이내 현금화 가능 자산으로 충당 가능한지. 150% 이상이 안전",
    "Current Ratio": "Current Ratio: Current assets ÷ current liabilities. Measures short-term solvency. Above 1.5 is healthy",
    "Interest Coverage": "이자보상배율: 영업이익 ÷ 이자비용. 이자를 영업이익으로 몇 배나 갚을 수 있는지. 3배 이상이면 안전",
    # Cash Flow
    "FCF Yield":    "FCF 수익률: FCF ÷ 시가총액 × 100. 내가 투자한 돈 대비 얼마의 잉여현금이 나오는지. 높을수록 주주 환원 여력 큼",
    "Operating CF": "영업현금흐름: 실제 영업에서 들어온 현금. 순이익은 회계 처리로 조작 가능하지만 현금흐름은 더 정직한 지표",
    "CAPEX":        "CAPEX(자본적지출): 공장·설비·기술 등에 투자하는 비용. 미래 성장을 위한 투자. FCF = 영업현금흐름 - CAPEX",
    # Models
    "DCF":          "DCF(현금흐름할인법): 미래에 벌어들일 현금을 현재 가치로 환산해 적정 주가를 구하는 대표적 내재가치 평가 모델",
    "WACC":         "WACC(가중평균자본비용): 기업이 자금을 조달하는 데 드는 평균 비용. DCF 계산 시 미래 현금흐름을 할인하는 데 사용",
    "Graham Number": "그레이엄 넘버: √(22.5 × EPS × BPS). 가치투자의 아버지 벤저민 그레이엄이 제시한 적정 주가 상한선",
    "Altman Z-Score": "알트만 Z-스코어: 재무비율로 기업 부도 가능성을 예측하는 모델. 1.8 미만=위험, 3 이상=안전",
    "Piotroski F-Score": "피오트로스키 F-스코어: 재무 건전성 9개 항목 점수화(0~9점). 8~9점=강한 매수, 0~1점=강한 매도 신호",
    "Magic Formula": "매직 포뮬러: 조엘 그린블라트의 투자법. 높은 ROIC + 낮은 EV/EBIT 기업을 동시에 선별하는 퀀트 전략",
}

def gl(term: str, display: str = None) -> str:
    """Wrap a term with a hover tooltip using the glossary."""
    explanation = GLOSSARY.get(term, GLOSSARY.get(display or "", ""))
    label = display or term
    if explanation:
        # Escape quotes in explanation
        safe_exp = explanation.replace('"', '&quot;').replace("'", "&#39;")
        return (
            f'<span class="gl-term" title="{safe_exp}">'
            f'{label}'
            f'</span>'
        )
    return label

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

# ─── TRANSLATION (Google Translate free API) ─────────────────────────────────
@st.cache_data(ttl=3600)
def translate_to_korean(text: str) -> str:
    """Translate text to Korean using Google Translate free endpoint."""
    if not text or not text.strip():
        return text
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": "auto",
            "tl": "ko",
            "dt": "t",
            "q": text[:500],
        }
        resp = requests.get(url, params=params, timeout=5)
        result = resp.json()
        translated = "".join(part[0] for part in result[0] if part[0])
        return translated
    except Exception:
        return text

# ─── DATA FUNCTIONS ───────────────────────────────────────────────────────────
@st.cache_data(ttl=300)   # 5분 — 주가 데이터
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

@st.cache_data(ttl=300)
def fetch_company_news(ticker: str, company_name: str) -> list:
    """Fetch news specifically about the given ticker/company. Uses yfinance first, then RSS."""
    articles = []

    # 1. yfinance built-in news (most directly relevant)
    try:
        yf_news = yf.Ticker(ticker).news or []
        for item in yf_news[:20]:
            content = item.get("content", {})
            title   = content.get("title", "") or item.get("title", "")
            link    = content.get("canonicalUrl", {}).get("url", "") or item.get("link", "")
            summary = content.get("summary", "") or item.get("summary", "")
            publisher = content.get("provider", {}).get("displayName", "") or item.get("publisher", "")
            pub_ts  = content.get("pubDate", "") or ""
            if not pub_ts:
                raw_ts = item.get("providerPublishTime", 0)
                pub_ts = datetime.utcfromtimestamp(raw_ts).strftime("%Y-%m-%d %H:%M") if raw_ts else ""
            title   = re.sub(r"<[^>]+>", "", str(title)).strip()
            summary = re.sub(r"<[^>]+>", "", str(summary))[:250].strip()
            if title:
                articles.append({
                    "title": title, "summary": summary,
                    "link": link, "published": str(pub_ts)[:16],
                    "source": publisher, "relevance": "direct",
                })
    except Exception:
        pass

    # 2. RSS feeds — strictly filter to company mentions only
    _rss_feeds = [
        ("https://feeds.reuters.com/reuters/businessNews", "Reuters"),
        ("https://feeds.marketwatch.com/marketwatch/topstories/", "MarketWatch"),
        ("https://finance.yahoo.com/news/rssindex", "Yahoo Finance"),
        ("https://www.cnbc.com/id/10001147/device/rss/rss.html", "CNBC"),
    ]
    ticker_clean   = ticker.lower().replace("^", "").replace("=f", "")
    company_words  = [w.lower() for w in re.split(r"[\s,\.]+", company_name) if len(w) > 3][:4]
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MarketIntel/1.0)"}
    for url, src in _rss_feeds:
        try:
            resp = requests.get(url, timeout=8, headers=headers)
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:15]:
                title   = re.sub(r"<[^>]+>", "", item.findtext("title",  "")).strip()
                desc    = re.sub(r"<[^>]+>", "", item.findtext("description", ""))[:250].strip()
                link    = item.findtext("link", "").strip()
                pubdate = item.findtext("pubDate", "")[:25]
                tl = (title + " " + desc).lower()
                if ticker_clean in tl or any(w in tl for w in company_words):
                    articles.append({
                        "title": title, "summary": desc,
                        "link": link, "published": pubdate,
                        "source": src, "relevance": "direct",
                    })
        except Exception:
            continue

    # Deduplicate by title prefix
    seen, unique = set(), []
    for art in articles:
        key = art["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(art)
    return unique[:25]


@st.cache_data(ttl=300)
def fetch_geo_news(ticker: str, sector: str, industry: str) -> list:
    """Fetch recent geopolitical/macro news relevant to company sector."""
    # Sector → search keywords
    SECTOR_KW = {
        "Technology":              ["china tariff", "semiconductor export", "AI regulation", "chip ban", "antitrust tech"],
        "Communication Services":  ["china tariff", "AI regulation", "antitrust", "streaming", "social media"],
        "Consumer Discretionary":  ["tariff consumer", "china trade", "inflation retail", "consumer spending"],
        "Consumer Staples":        ["inflation food", "supply chain", "tariff consumer", "agricultural"],
        "Energy":                  ["oil price", "OPEC", "Russia Ukraine energy", "Iran sanctions", "crude oil"],
        "Financials":              ["fed rate", "interest rate", "federal reserve", "banking regulation", "inflation"],
        "Health Care":             ["FDA regulation", "drug pricing", "healthcare reform", "biotech FDA"],
        "Industrials":             ["tariff manufacturing", "supply chain", "infrastructure", "defense spending"],
        "Materials":               ["china tariff metals", "supply chain materials", "mining regulation"],
        "Real Estate":             ["interest rate housing", "fed rate real estate", "mortgage rate"],
        "Utilities":               ["energy regulation", "interest rate utility", "renewable energy policy"],
        "Defense":                 ["Russia Ukraine", "Middle East military", "defense budget", "NATO"],
    }
    keywords = SECTOR_KW.get(sector, ["trade war", "fed rate", "inflation", "geopolitical"])
    # Also add industry-specific terms
    ind_lower = industry.lower()
    if "semiconductor" in ind_lower or "chip" in ind_lower:
        keywords += ["semiconductor export", "chip war", "TSMC", "ASML export"]
    if "oil" in ind_lower or "gas" in ind_lower:
        keywords += ["OPEC", "crude oil", "LNG price"]
    if "bank" in ind_lower or "financ" in ind_lower:
        keywords += ["federal reserve", "bank regulation", "credit"]

    _rss_feeds = [
        ("https://feeds.reuters.com/reuters/businessNews", "Reuters"),
        ("https://www.cnbc.com/id/10001147/device/rss/rss.html", "CNBC"),
        ("https://feeds.marketwatch.com/marketwatch/topstories/", "MarketWatch"),
        ("https://finance.yahoo.com/news/rssindex", "Yahoo Finance"),
    ]
    articles = []
    headers  = {"User-Agent": "Mozilla/5.0 (compatible; MarketIntel/1.0)"}
    for url, src in _rss_feeds:
        try:
            resp = requests.get(url, timeout=8, headers=headers)
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item")[:20]:
                title   = re.sub(r"<[^>]+>", "", item.findtext("title",  "")).strip()
                desc    = re.sub(r"<[^>]+>", "", item.findtext("description", ""))[:250].strip()
                link    = item.findtext("link", "").strip()
                pubdate = item.findtext("pubDate", "")[:25]
                tl = (title + " " + desc).lower()
                if any(kw.lower() in tl for kw in keywords):
                    articles.append({
                        "title": title, "summary": desc,
                        "link": link, "published": pubdate,
                        "source": src,
                    })
        except Exception:
            continue

    seen, unique = set(), []
    for art in articles:
        key = art["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(art)
    return unique[:15]


# Sector → relevant geo risk factor keys
_SECTOR_GEO_KEYS = {
    "Technology":             ["us_china", "ai_bubble", "fed", "us_debt"],
    "Communication Services": ["us_china", "ai_bubble", "fed"],
    "Consumer Discretionary": ["us_china", "fed", "us_debt"],
    "Consumer Staples":       ["fed", "russia_ukraine", "middle_east"],
    "Energy":                 ["middle_east", "russia_ukraine", "fed"],
    "Financials":             ["fed", "us_debt", "russia_ukraine"],
    "Health Care":            ["fed", "us_debt"],
    "Industrials":            ["us_china", "russia_ukraine", "fed"],
    "Materials":              ["us_china", "russia_ukraine", "middle_east"],
    "Real Estate":            ["fed", "us_debt"],
    "Utilities":              ["fed", "us_debt", "middle_east"],
    "Defense":                ["russia_ukraine", "middle_east", "us_china"],
}
_GEO_FACTORS_EN = {
    "us_china":      ("🇺🇸🇨🇳 US-China Trade War",    "HIGH",   "Trump 145% tariffs → Tech/semiconductor pressure, supply chain restructuring",      "high_risk"),
    "fed":           ("🏦 Fed Monetary Policy",         "MEDIUM", "Rate hold in 2025, 1-2 cuts possible → Positive for growth stocks",                 "med_risk"),
    "middle_east":   ("🛢️ Middle East Tensions",        "MEDIUM", "Iran-Israel tensions persist, WTI oil price volatility impacts energy sector",      "med_risk"),
    "russia_ukraine":("🇷🇺🇺🇦 Russia-Ukraine War",      "MEDIUM", "Prolonged conflict, EU energy supply uncertainty, defense stocks benefit",          "med_risk"),
    "ai_bubble":     ("💹 AI Valuation Debate",         "LOW",    "NVIDIA etc. valuation debate, entering earnings-validation phase",                  "low_risk"),
    "us_debt":       ("📉 US National Debt",            "MEDIUM", "$35T+ debt, fiscal deficit → Long-term rate upward pressure",                       "med_risk"),
}
_GEO_FACTORS_KO = {
    "us_china":      ("🇺🇸🇨🇳 미-중 무역 갈등",         "높음",   "트럼프 관세 145% 부과 → 반도체·기술주 압박, 공급망 재편 가속화",              "high_risk"),
    "fed":           ("🏦 연준(Fed) 통화정책",           "중간",   "2025년 금리 동결 기조 유지, 연내 1-2회 인하 가능성 → 성장주 긍정적",            "med_risk"),
    "middle_east":   ("🛢️ 중동 지정학 리스크",           "중간",   "이란-이스라엘 긴장 지속, WTI 가격 변동성 에너지 섹터 영향",                    "med_risk"),
    "russia_ukraine":("🇷🇺🇺🇦 러시아-우크라이나",         "중간",   "전쟁 장기화, 유럽 에너지 공급 불안 지속, 방산주 수혜",                          "med_risk"),
    "ai_bubble":     ("💹 AI 밸류에이션 논쟁",           "낮음",   "엔비디아 등 AI 밸류에이션 논란, 실적 기반 검증 국면 진입",                       "low_risk"),
    "us_debt":       ("📉 미국 국가부채",                "중간",   "35조 달러 돌파, 재정 적자 지속 → 장기 금리 상승 압력",                           "med_risk"),
}


def get_relevant_geo_factors(sector: str, lang: str) -> list:
    """Return geo risk factors relevant to the given sector."""
    keys = _SECTOR_GEO_KEYS.get(sector, list(_GEO_FACTORS_EN.keys()))
    db   = _GEO_FACTORS_KO if lang == "ko" else _GEO_FACTORS_EN
    return [db[k] for k in keys if k in db]


def summarize_news_impact(articles: list, company_name: str, ticker: str,
                          current_price: float, change_1d: float, lang: str) -> str:
    """Rule-based analysis: what do these news headlines mean for the stock?"""
    if not articles:
        return ""

    titles = " ".join(a["title"].lower() for a in articles[:10])
    summaries = " ".join(a.get("summary","").lower() for a in articles[:10])
    text = titles + " " + summaries

    # Sentiment scoring
    pos_kw = ["beat", "record", "surge", "rally", "upgrade", "buy", "strong",
               "growth", "profit", "revenue", "partnership", "launch", "expand",
               "exceed", "raise", "positive", "outperform", "contract", "win"]
    neg_kw = ["miss", "decline", "fall", "drop", "cut", "downgrade", "sell",
               "loss", "lawsuit", "fine", "investigation", "recall", "layoff",
               "weak", "concern", "risk", "warn", "lower", "negative", "underperform"]
    topic_kw = {
        "earnings": ["earn", "eps", "revenue", "profit", "quarter", "result"],
        "regulation": ["regulation", "antitrust", "lawsuit", "fine", "sec", "ftc", "doj"],
        "macro": ["fed", "rate", "inflation", "recession", "economy", "gdp", "tariff"],
        "product": ["product", "launch", "release", "model", "chip", "device", "service"],
        "analyst": ["analyst", "target", "price target", "upgrade", "downgrade", "rating"],
        "trade": ["china", "tariff", "trade", "export", "ban", "supply chain"],
    }

    pos_score = sum(1 for w in pos_kw if w in text)
    neg_score = sum(1 for w in neg_kw if w in text)
    topics_found = [t for t, kws in topic_kw.items() if any(k in text for k in kws)]

    net = pos_score - neg_score
    if net >= 3:
        sentiment_label = ("📈 긍정적" if lang=="ko" else "📈 Positive")
        sentiment_color = "#FF4040"
        price_impact    = ("주가 상승 압력 우세. 단기 모멘텀 강화 가능성." if lang=="ko"
                           else "Upside price pressure dominant. Short-term momentum may strengthen.")
    elif net >= 1:
        sentiment_label = ("📊 다소 긍정" if lang=="ko" else "📊 Mildly Positive")
        sentiment_color = "#FFA500"
        price_impact    = ("뉴스 흐름은 소폭 긍정적. 현 주가 수준 지지 예상." if lang=="ko"
                           else "News flow mildly supportive. Expect current price level to hold.")
    elif net <= -3:
        sentiment_label = ("📉 부정적" if lang=="ko" else "📉 Negative")
        sentiment_color = "#4488FF"
        price_impact    = ("하락 압력 우세. 단기 변동성 확대 및 조정 가능성." if lang=="ko"
                           else "Downside pressure dominant. Short-term volatility and pullback likely.")
    elif net <= -1:
        sentiment_label = ("📊 다소 부정" if lang=="ko" else "📊 Mildly Negative")
        sentiment_color = "#4488FF"
        price_impact    = ("일부 부정적 뉴스 혼재. 단기 주가 약세 가능성 주시 필요." if lang=="ko"
                           else "Some negative news in the mix. Watch for short-term weakness.")
    else:
        sentiment_label = ("📊 중립" if lang=="ko" else "📊 Neutral")
        sentiment_color = "#8B9DB0"
        price_impact    = ("뉴스 영향 중립적. 실적·거시 지표가 방향성을 결정할 것." if lang=="ko"
                           else "News impact neutral. Earnings and macro data will determine direction.")

    topic_labels_ko = {
        "earnings": "📊 실적/수익", "regulation": "⚖️ 규제/법률",
        "macro": "🏦 거시경제", "product": "🚀 제품/서비스",
        "analyst": "🔍 애널리스트", "trade": "🌐 무역/관세",
    }
    topic_labels_en = {
        "earnings": "📊 Earnings", "regulation": "⚖️ Regulation",
        "macro": "🏦 Macro", "product": "🚀 Product/Service",
        "analyst": "🔍 Analyst", "trade": "🌐 Trade",
    }
    tl = topic_labels_ko if lang=="ko" else topic_labels_en
    topic_chips = " ".join(
        f"<span style='background:#1A2744;border:1px solid #3A4060;border-radius:12px;"
        f"padding:2px 9px;font-size:0.75rem;color:#B0BEC5;margin-right:4px;'>{tl[t]}</span>"
        for t in topics_found if t in tl
    )

    n = len(articles)
    label_n   = ("관련 기사" if lang=="ko" else "related articles")
    label_sum = ("뉴스 종합 판단" if lang=="ko" else "News Summary")
    label_imp = ("주가 영향 전망" if lang=="ko" else "Price Impact Outlook")

    return f"""
<div style='background:linear-gradient(135deg,#0F1527,#1A1F35);border:1px solid #2E3250;
            border-radius:12px;padding:16px 20px;margin-top:16px;'>
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
        <span style='font-size:1rem;font-weight:800;color:{sentiment_color};'>{sentiment_label}</span>
        <span style='color:#4A5568;font-size:0.8rem;'>|</span>
        <span style='color:#8B9DB0;font-size:0.8rem;'>{n} {label_n}</span>
        <span style='color:#4A5568;font-size:0.8rem;'>|</span>
        <span style='color:#8B9DB0;font-size:0.8rem;'>{company_name} ({ticker})</span>
    </div>
    {f"<div style='margin-bottom:8px;'>{topic_chips}</div>" if topic_chips else ""}
    <div style='color:#8B9DB0;font-size:0.78rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.5px;margin-bottom:4px;'>{label_sum}</div>
    <div style='color:#EAEAEA;font-size:0.88rem;line-height:1.6;margin-bottom:10px;'>
        {_news_narrative(pos_score, neg_score, topics_found, company_name, ticker, lang)}
    </div>
    <div style='border-top:1px solid #2E3250;padding-top:10px;margin-top:4px;'>
        <span style='color:#8B9DB0;font-size:0.78rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.5px;'>{label_imp}&nbsp;</span>
        <span style='color:{sentiment_color};font-size:0.88rem;'>{price_impact}</span>
    </div>
</div>
"""


def _news_narrative(pos: int, neg: int, topics: list, company: str, ticker: str, lang: str) -> str:
    parts = []
    topic_set = set(topics)
    if "earnings" in topic_set:
        parts.append(("최근 실적 관련 뉴스가 다수 보도됨." if lang=="ko"
                       else "Earnings-related coverage is prominent."))
    if "analyst" in topic_set:
        parts.append(("애널리스트 투자의견 변동 관련 보도 확인됨." if lang=="ko"
                       else "Analyst rating changes reported."))
    if "regulation" in topic_set:
        parts.append(("규제·법적 리스크 관련 뉴스 존재. 불확실성 요인." if lang=="ko"
                       else "Regulatory/legal risk news present — adds uncertainty."))
    if "trade" in topic_set:
        parts.append(("무역/관세 이슈가 기업 공급망·마진에 영향 가능." if lang=="ko"
                       else "Trade/tariff issues may affect supply chain and margins."))
    if "macro" in topic_set:
        parts.append(("거시경제 변수(금리·인플레이션)가 밸류에이션에 영향을 미칠 수 있음." if lang=="ko"
                       else "Macro variables (rates/inflation) may weigh on valuation."))
    if "product" in topic_set:
        parts.append(("신제품·서비스 관련 뉴스가 중장기 성장 기대를 자극할 수 있음." if lang=="ko"
                       else "New product/service news may drive mid-term growth expectations."))
    if not parts:
        parts.append(("현재 주요 이슈는 전반적 시장 흐름과 연동되어 있음." if lang=="ko"
                       else "Current issues are broadly tied to overall market trends."))
    return " ".join(parts)


def summarize_geo_impact(risk_factors: list, geo_news: list, company_name: str,
                         sector: str, current_price: float, lang: str) -> str:
    """Rule-based geo impact summary for the stock."""
    text = " ".join(
        (a["title"] + " " + a.get("summary","")).lower()
        for a in geo_news[:8]
    )

    # Risk level tally
    high_risks = [r for r in risk_factors if r[3] == "high_risk"]
    med_risks  = [r for r in risk_factors if r[3] == "med_risk"]
    low_risks  = [r for r in risk_factors if r[3] == "low_risk"]

    if high_risks:
        overall = ("⚠️ 고위험 요인 존재" if lang=="ko" else "⚠️ High-Risk Factors Present")
        color   = "#FF4B4B"
        price_view = ("지정학 리스크가 주가에 하방 압력을 가할 수 있음. 변동성 확대 구간." if lang=="ko"
                      else "Geopolitical risks may add downside pressure. Expect elevated volatility.")
    elif len(med_risks) >= 2:
        overall = ("🔶 중간 위험 복합" if lang=="ko" else "🔶 Multiple Medium Risks")
        color   = "#FFA500"
        price_view = ("복수의 중간 리스크가 단기 주가 변동성을 키울 수 있음. 방어적 포지션 고려." if lang=="ko"
                      else "Multiple medium risks may amplify short-term volatility. Consider defensive posture.")
    else:
        overall = ("🟢 리스크 제한적" if lang=="ko" else "🟢 Limited Risk Exposure")
        color   = "#00D4AA"
        price_view = ("현 지정학 환경에서 이 섹터 노출도는 낮음. 안정적 주가 흐름 기대." if lang=="ko"
                      else "Low sector exposure to current geo environment. Stable price action expected.")

    # Sector-specific impact comment
    sec_comment_map_ko = {
        "Technology":             f"{company_name}는 미중 무역갈등·반도체 수출 규제의 직접 영향권. AI 수요는 지속적 상승 동력.",
        "Energy":                 f"중동 긴장·OPEC 정책이 {company_name}의 유가·마진에 직결. 지정학 불안 시 수혜 가능.",
        "Financials":             f"연준 금리 경로가 {company_name} 이자마진·대출 성장에 핵심 변수.",
        "Consumer Discretionary": f"인플레·금리가 소비자 지출 여력을 제약, {company_name} 매출 성장에 영향.",
        "Industrials":            f"관세·공급망 재편이 {company_name} 원가 구조에 영향. 방산 수요는 긍정 요인.",
        "Health Care":            f"규제·약가 정책이 {company_name} 수익성의 핵심 리스크.",
        "Defense":                f"지정학 긴장 지속은 {company_name} 방산 수요 증가로 직결. 분쟁 장기화 시 수혜.",
    }
    sec_comment_map_en = {
        "Technology":             f"{company_name} is directly exposed to US-China trade tensions and semiconductor export controls. AI demand remains a structural tailwind.",
        "Energy":                 f"Middle East tensions and OPEC policy directly affect {company_name}'s oil pricing and margins. Geopolitical risk can be a tailwind.",
        "Financials":             f"Fed rate path is the key variable for {company_name}'s net interest margin and loan growth.",
        "Consumer Discretionary": f"Inflation and rates constrain consumer spending power, impacting {company_name}'s revenue growth.",
        "Industrials":            f"Tariffs and supply chain restructuring affect {company_name}'s cost structure. Defense demand is a positive.",
        "Health Care":            f"Drug pricing regulation and FDA policy are core risks to {company_name}'s profitability.",
        "Defense":                f"Sustained geopolitical tensions directly translate to higher defense demand benefiting {company_name}.",
    }
    sc_map = sec_comment_map_ko if lang=="ko" else sec_comment_map_en
    sec_comment = sc_map.get(sector, (
        f"현재 지정학 환경이 {company_name}에 미치는 간접적 영향을 모니터링 중."
        if lang=="ko" else
        f"Monitoring indirect geopolitical impact on {company_name}."
    ))

    n_geo = len(geo_news)
    label_geo = ("관련 지정학 뉴스" if lang=="ko" else "relevant geo news items")
    label_sum = ("지정학 종합 판단" if lang=="ko" else "Geopolitical Summary")
    label_imp = ("주가 영향 전망" if lang=="ko" else "Price Impact Outlook")

    return f"""
<div style='background:linear-gradient(135deg,#0F1527,#1A1F35);border:1px solid #2E3250;
            border-radius:12px;padding:16px 20px;margin-top:16px;'>
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
        <span style='font-size:1rem;font-weight:800;color:{color};'>{overall}</span>
        <span style='color:#4A5568;font-size:0.8rem;'>|</span>
        <span style='color:#8B9DB0;font-size:0.8rem;'>{n_geo} {label_geo}</span>
        <span style='color:#4A5568;font-size:0.8rem;'>|</span>
        <span style='color:#8B9DB0;font-size:0.8rem;'>{sector or "N/A"}</span>
    </div>
    <div style='color:#8B9DB0;font-size:0.78rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.5px;margin-bottom:4px;'>{label_sum}</div>
    <div style='color:#EAEAEA;font-size:0.88rem;line-height:1.6;margin-bottom:10px;'>
        {sec_comment}
    </div>
    <div style='border-top:1px solid #2E3250;padding-top:10px;'>
        <span style='color:#8B9DB0;font-size:0.78rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.5px;'>{label_imp}&nbsp;</span>
        <span style='color:{color};font-size:0.88rem;'>{price_view}</span>
    </div>
</div>
"""


def summarize_prediction_rationale(predictions: dict, df: pd.DataFrame,
                                    ticker: str, company_name: str, lang: str) -> str:
    """Explain WHY the model produced these prediction values."""
    if not predictions or df.empty:
        return ""

    close = df["Close"]
    if close.ndim == 2:
        close = close.iloc[:, 0]
    prices = close.dropna().astype(float).values
    if len(prices) < 60:
        return ""

    current = float(prices[-1])

    # Compute key signals
    ma50  = float(np.mean(prices[-50:]))  if len(prices) >= 50  else current
    ma200 = float(np.mean(prices[-200:])) if len(prices) >= 200 else current
    ret_1y = float((prices[-1] / prices[-min(252,len(prices))]) - 1) * 100
    ret_3m = float((prices[-1] / prices[-min(63, len(prices))]) - 1) * 100
    vol_daily = float(np.std(np.diff(np.log(prices[-min(252,len(prices)):]))))
    vol_annual = vol_daily * np.sqrt(252) * 100

    trend_up   = current > ma50 > ma200
    trend_down = current < ma50 < ma200
    above_ma200 = current > ma200
    high_vol = vol_annual > 40

    # Collect 12-month prediction
    p12 = predictions.get(252, predictions.get(max(predictions.keys()), {}))
    chg_pct = p12.get("change_pct", 0)

    # Build rationale points
    if lang == "ko":
        points = []
        # Trend
        if trend_up:
            points.append(f"📈 <b>추세 정배열</b>: 현재가({current:,.0f}) > SMA50({ma50:,.0f}) > SMA200({ma200:,.0f}) — 장기 상승 추세 유효. 선형회귀 모델이 상향 방향 채택.")
        elif trend_down:
            points.append(f"📉 <b>추세 역배열</b>: 현재가({current:,.0f}) < SMA50({ma50:,.0f}) < SMA200({ma200:,.0f}) — 장기 하락 추세. 선형회귀 모델이 하향 방향 반영.")
        else:
            if above_ma200:
                points.append(f"📊 <b>혼조 추세</b>: 200일선 위에 있으나 단기 모멘텀 약화. 모델이 중립~소폭 상승 방향 반영.")
            else:
                points.append(f"📊 <b>혼조 추세</b>: 200일선 아래 위치. 모멘텀 모델이 하방 편향 반영.")

        # Momentum
        if ret_3m > 10:
            points.append(f"🚀 <b>강한 단기 모멘텀</b>: 최근 3개월 수익률 {ret_3m:+.1f}%. 지수가중 모멘텀 모델이 상승 연속성 반영.")
        elif ret_3m < -10:
            points.append(f"⬇️ <b>약한 단기 모멘텀</b>: 최근 3개월 {ret_3m:+.1f}%. 모멘텀 모델이 단기 하방 편향 반영.")
        else:
            points.append(f"➡️ <b>중립 모멘텀</b>: 최근 3개월 {ret_3m:+.1f}%. 모멘텀 모델 기여도 낮음.")

        # Mean reversion
        diff_pct = (current - ma200) / ma200 * 100
        if diff_pct > 20:
            points.append(f"🔄 <b>평균회귀 압력</b>: 현재가가 200일 이동평균 대비 {diff_pct:+.1f}% 고평가. 평균회귀 모델이 하방 조정 기여.")
        elif diff_pct < -20:
            points.append(f"🔄 <b>평균회귀 지지</b>: 현재가가 200일 이동평균 대비 {diff_pct:+.1f}% 저평가. 평균회귀 모델이 상향 복귀 기여.")
        else:
            points.append(f"🔄 <b>평균 근처</b>: 200일선 대비 {diff_pct:+.1f}%. 평균회귀 영향 중립.")

        # Volatility
        if high_vol:
            points.append(f"⚡ <b>높은 변동성</b>: 연환산 {vol_annual:.1f}% — 강세/약세 구간 폭이 넓음. 예측 불확실성 높음.")
        else:
            points.append(f"✅ <b>낮은 변동성</b>: 연환산 {vol_annual:.1f}% — 비교적 안정적. 예측 신뢰도 상대적으로 높음.")

        conclusion = (f"3개 모델 앙상블 기준 12개월 기본 예측: <b style='color:{'#FF4040' if chg_pct>=0 else '#4488FF'};'>{chg_pct:+.1f}%</b> "
                      f"({'상승' if chg_pct >= 0 else '하락'} 방향). "
                      f"과거 1년 실적({ret_1y:+.1f}%)이 모델의 기저 추세로 반영됨.")

    else:
        points = []
        if trend_up:
            points.append(f"📈 <b>Bullish alignment</b>: Price({current:,.0f}) > SMA50({ma50:,.0f}) > SMA200({ma200:,.0f}) — Long-term uptrend intact. Linear regression assigns upward trajectory.")
        elif trend_down:
            points.append(f"📉 <b>Bearish alignment</b>: Price({current:,.0f}) < SMA50({ma50:,.0f}) < SMA200({ma200:,.0f}) — Long-term downtrend. Regression reflects downward slope.")
        else:
            if above_ma200:
                points.append(f"📊 <b>Mixed trend</b>: Above 200-day MA but short-term momentum weakening. Models lean mildly positive.")
            else:
                points.append(f"📊 <b>Mixed trend</b>: Below 200-day MA. Momentum model reflects downward bias.")

        if ret_3m > 10:
            points.append(f"🚀 <b>Strong short-term momentum</b>: +{ret_3m:.1f}% over 3M. Exponential-weighted model reinforces upward continuation.")
        elif ret_3m < -10:
            points.append(f"⬇️ <b>Weak short-term momentum</b>: {ret_3m:+.1f}% over 3M. Momentum model adds downward bias.")
        else:
            points.append(f"➡️ <b>Neutral momentum</b>: {ret_3m:+.1f}% over 3M. Low momentum contribution to forecast.")

        diff_pct = (current - ma200) / ma200 * 100
        if diff_pct > 20:
            points.append(f"🔄 <b>Mean reversion headwind</b>: Price is {diff_pct:+.1f}% above 200-day MA. Reversion model pulls forecast downward.")
        elif diff_pct < -20:
            points.append(f"🔄 <b>Mean reversion tailwind</b>: Price is {diff_pct:+.1f}% below 200-day MA. Reversion model lifts the forecast.")
        else:
            points.append(f"🔄 <b>Near mean</b>: {diff_pct:+.1f}% vs 200-day MA. Mean reversion effect is neutral.")

        if high_vol:
            points.append(f"⚡ <b>High volatility</b>: Annualized {vol_annual:.1f}% — wide bull/bear bands. Higher forecast uncertainty.")
        else:
            points.append(f"✅ <b>Low volatility</b>: Annualized {vol_annual:.1f}% — relatively stable. Forecast confidence is higher.")

        conclusion = (f"3-model ensemble 12M base forecast: <b style='color:{'#FF4040' if chg_pct>=0 else '#4488FF'};'>{chg_pct:+.1f}%</b> "
                      f"({'upside' if chg_pct >= 0 else 'downside'}). "
                      f"Prior 1Y return ({ret_1y:+.1f}%) is embedded as the trend baseline.")

    bullet_html = "".join(
        f"<div style='padding:5px 0;border-bottom:1px solid #1E2130;font-size:0.87rem;color:#D0D8E8;'>{p}</div>"
        for p in points
    )
    label_rat   = ("📋 예측 근거 요약" if lang=="ko" else "📋 Prediction Rationale")
    label_con   = ("종합 결론" if lang=="ko" else "Conclusion")

    return f"""
<div style='background:linear-gradient(135deg,#0F1527,#1A1F35);border:1px solid #2E3250;
            border-radius:12px;padding:16px 20px;margin-top:20px;'>
    <div style='font-size:1rem;font-weight:800;color:#FFA500;margin-bottom:12px;'>{label_rat}</div>
    {bullet_html}
    <div style='margin-top:12px;padding-top:10px;border-top:1px solid #2E3250;'>
        <span style='color:#8B9DB0;font-size:0.78rem;font-weight:700;text-transform:uppercase;
                     letter-spacing:0.5px;'>{label_con}&nbsp;</span>
        <span style='font-size:0.88rem;color:#EAEAEA;'>{conclusion}</span>
    </div>
</div>
"""

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
def _rangeselector(lang: str = "en") -> dict:
    """Standard time-range selector buttons for all date-axis charts."""
    return dict(
        buttons=[
            dict(count=1,  label="1D", step="day",   stepmode="backward"),
            dict(count=1,  label="1M", step="month", stepmode="backward"),
            dict(count=6,  label="6M", step="month", stepmode="backward"),
            dict(count=1,  label="1Y", step="year",  stepmode="backward"),
            dict(count=3,  label="3Y", step="year",  stepmode="backward"),
            dict(count=5,  label="5Y", step="year",  stepmode="backward"),
            dict(step="all", label="ALL"),
        ],
        bgcolor="#1E2130",
        activecolor="#FFA500",
        font=dict(color="#FFFFFF", size=11),
        x=0, y=1.02,
    )

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
        increasing_line_color="#FF4040",
        decreasing_line_color="#4488FF",
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
        colors = ["#FF4040" if float(c) >= float(o) else "#4488FF"
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
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1,  label="1D",  step="day",   stepmode="backward"),
                    dict(count=1,  label="1M",  step="month", stepmode="backward"),
                    dict(count=6,  label="6M",  step="month", stepmode="backward"),
                    dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                    dict(count=3,  label="3Y",  step="year",  stepmode="backward"),
                    dict(count=5,  label="5Y",  step="year",  stepmode="backward"),
                    dict(step="all", label="ALL"),
                ],
                bgcolor="#1E2130",
                activecolor="#FFA500",
                bordercolor="#3A4060",
                borderwidth=1,
                font=dict(color="#FFFFFF", size=12),
                x=0, y=1.02,
            ),
            type="date",
        ),
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
        legend=dict(orientation="h", y=1.12),
        xaxis=dict(
            rangeselector=_rangeselector(lang),
            rangeslider=dict(visible=False),
            type="date",
            title="Date" if lang == "en" else "날짜",
        ),
        yaxis_title="Price (USD)" if lang == "en" else "주가 (USD)",
        margin=dict(l=0, r=0, t=70, b=0),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
    )
    return fig

def build_macro_chart(macro_data: dict, lang: str) -> go.Figure:
    if not macro_data:
        return go.Figure()

    names = list(macro_data.keys())
    changes = [macro_data[n]["change_pct"] for n in names]
    colors = ["#FF4040" if c >= 0 else "#4488FF" for c in changes]

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
    colors = ["#FF4040" if i >= 0 else "#4488FF" for i in impacts]

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
        height=450,
        xaxis=dict(
            rangeselector=_rangeselector(lang),
            rangeslider=dict(visible=False),
            type="date",
        ),
        yaxis_title="Market Impact (%)" if lang == "en" else "증시 영향 (%)",
        margin=dict(l=0, r=0, t=70, b=20),
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
    )
    return fig

# ─── AUTO REFRESH ────────────────────────────────────────────────────────────
_REFRESH_INTERVAL_MS = 5 * 60 * 1000  # 5 minutes
if _HAS_AUTOREFRESH:
    st_autorefresh(interval=_REFRESH_INTERVAL_MS, limit=None, key="global_autorefresh")

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
    .metric-change-pos { color: #FF4040; font-size: 0.9rem; }
    .metric-change-neg { color: #4488FF; font-size: 0.9rem; }
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
    .pred-change-pos { font-size: 1rem; color: #FF4040; }
    .pred-change-neg { font-size: 1rem; color: #4488FF; }
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
    .gl-term {
        cursor: help;
        border-bottom: 1px dotted #FFA500;
        color: inherit;
    }
    .data-timestamp {
        position: fixed;
        top: 56px;
        right: 18px;
        z-index: 9999;
        background: rgba(14,17,23,0.92);
        border: 1px solid #2E3250;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.72rem;
        color: #8B9DB0;
        backdrop-filter: blur(6px);
        display: flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
    }
    .data-timestamp .dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        background: #00D4AA;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%,100% { opacity: 1; }
        50%      { opacity: 0.3; }
    }
    .translate-btn {
        background: #1E2130;
        border: 1px solid #FFA500;
        border-radius: 8px;
        color: #FFA500;
        padding: 4px 12px;
        font-size: 0.8rem;
        cursor: pointer;
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
        key="search_box",
    )
    if st.button(T("search_btn"), use_container_width=True, type="primary"):
        if search_input.strip():
            resolved = resolve_ticker(search_input.strip())
            st.session_state.ticker = resolved
            st.rerun()

    # Search hint
    if search_input.strip():
        preview = resolve_ticker(search_input.strip())
        if preview.upper() != search_input.strip().upper():
            st.markdown(
                f"<div style='color:#FFA500;font-size:0.75rem;margin-top:-8px;'>"
                f"→ {preview}</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # Quick-access: Semiconductor & Sector Explorer
    semi_label = "💾 반도체 생태계" if st.session_state.lang == "ko" else "💾 Semiconductor Ecosystem"
    sect_label = "🗂️ 섹터 탐색기" if st.session_state.lang == "ko" else "🗂️ Sector Explorer"
    sb_col1, sb_col2 = st.columns(2)
    with sb_col1:
        semi_active = st.session_state.sidebar_view == "semi"
        if st.button(semi_label, use_container_width=True,
                     type="primary" if semi_active else "secondary",
                     key="sb_semi"):
            st.session_state.sidebar_view = None if semi_active else "semi"
            st.rerun()
    with sb_col2:
        sect_active = st.session_state.sidebar_view == "sectors"
        if st.button(sect_label, use_container_width=True,
                     type="primary" if sect_active else "secondary",
                     key="sb_sectors"):
            st.session_state.sidebar_view = None if sect_active else "sectors"
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

# ── Record fetch timestamp ───────────────────────────────────────────────────
_now_utc = datetime.utcnow()
st.session_state.data_fetched_at = _now_utc

# Determine market data as-of date (last trading day in price history)
_market_date_str = ""
if not df_2y.empty:
    _last_idx = df_2y.index[-1]
    _market_date_str = pd.Timestamp(_last_idx).strftime("%Y-%m-%d")

_fetch_str = _now_utc.strftime("%Y-%m-%d %H:%M") + " UTC"
_auto_label = ("5분마다 자동갱신" if lang == "ko" else "auto-refresh 5 min") if _HAS_AUTOREFRESH else ("수동 새로고침" if lang == "ko" else "manual refresh")
_ts_label   = ("주가 기준일" if lang == "ko" else "Price date")
_ts_fetched = ("조회 시각" if lang == "ko" else "Fetched")

st.markdown(f"""
<div class="data-timestamp">
    <span class="dot"></span>
    <span>{_ts_label}: <b style='color:#EAEAEA;'>{_market_date_str}</b></span>
    <span style='color:#4A5568;'>|</span>
    <span>{_ts_fetched}: <b style='color:#EAEAEA;'>{_fetch_str}</b></span>
    <span style='color:#4A5568;'>|</span>
    <span style='color:#00D4AA;'>{_auto_label}</span>
</div>
""", unsafe_allow_html=True)

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
change_color = "#FF4040" if change_1d >= 0 else "#4488FF"
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
    (gl("Market Cap" if lang=="en" else "시가총액", T("mkt_cap")), mkt_cap_str, ""),
    (gl("Volume" if lang=="en" else "거래량", T("volume")), vol_str, ""),
    (gl("P/E Ratio" if lang=="en" else "주가수익비율(P/E)", T("pe_ratio")), f"{pe:.1f}" if pe else "N/A", ""),
    (gl("52-Week High" if lang=="en" else "52주 최고가", T("week52_high")), f"${w52h:,.2f}" if w52h else "N/A", ""),
    (gl("52-Week Low" if lang=="en" else "52주 최저가", T("week52_low")), f"${w52l:,.2f}" if w52l else "N/A", ""),
    (gl("Volatility (Annualized)" if lang=="en" else "변동성 (연율화)", T("volatility")), f"{df_2y['close'].pct_change().std() * np.sqrt(252) * 100:.1f}%" if 'close' in df_2y.columns else "N/A", ""),
]
for col, (label, val, change) in zip([mcol1, mcol2, mcol3, mcol4, mcol5, mcol6], metrics):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value' style='font-size:1.1rem;'>{val}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── TABS (only shown when no sidebar panel is active) ────────────────────────
from contextlib import nullcontext as _nctx
_show_tabs = st.session_state.sidebar_view is None
if _show_tabs:
    tabs = st.tabs([T("tab_overview"), T("tab_predict"), T("tab_news"), T("tab_geo"), T("tab_history"), T("tab_company"), T("tab_relations"), T("tab_invest")])

# ══════════════════ TAB 1: OVERVIEW ══════════════════
if _show_tabs:
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
        _rsi_label = gl("RSI (14)", T("rsi"))
        _macd_label = gl("MACD", T("macd"))
        _sma50_label = gl("SMA50", T("sma50"))
        _sma200_label = gl("SMA200", T("sma200"))
        _beta_label = gl("Beta vs S&P 500", T("beta"))
        ind_data = [
            (_rsi_label, f"{df_2y['RSI'].iloc[-1]:.1f}" if "RSI" in df_2y.columns and not pd.isna(df_2y["RSI"].iloc[-1]) else "N/A",
             "#FF4B4B" if "RSI" in df_2y.columns and not pd.isna(df_2y["RSI"].iloc[-1]) and df_2y["RSI"].iloc[-1] > 70
             else "#00D4AA" if "RSI" in df_2y.columns and not pd.isna(df_2y["RSI"].iloc[-1]) and df_2y["RSI"].iloc[-1] < 30
             else "#FFA500"),
            (_macd_label, f"{df_2y['MACD'].iloc[-1]:.3f}" if "MACD" in df_2y.columns else "N/A", "#8B9DB0"),
            (_sma50_label, f"${df_2y['SMA50'].iloc[-1]:,.2f}" if "SMA50" in df_2y.columns and not pd.isna(df_2y["SMA50"].iloc[-1]) else "N/A", "#FFD700"),
            (_sma200_label, f"${df_2y['SMA200'].iloc[-1]:,.2f}" if "SMA200" in df_2y.columns and not pd.isna(df_2y["SMA200"].iloc[-1]) else "N/A", "#FF8C00"),
            (_beta_label, f"{info.get('beta', 'N/A')}", "#AB63FA"),
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
        sent_color = "#FF4040" if sent_label in ["Bullish", "강세"] else "#4488FF" if sent_label in ["Bearish", "약세"] else "#FFA500"

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
                    chg_color = "#FF4040" if chg >= 0 else "#4488FF"
                    chg_arrow = "▲" if chg >= 0 else "▼"
                    col.markdown(f"""
                    <div class='prediction-card'>
                        <div class='pred-horizon'>{label}</div>
                        <div class='pred-price'>${p['base']:,.2f}</div>
                        <div style='color:{chg_color};font-size:1rem;font-weight:600;'>{chg_arrow} {abs(chg):.1f}%</div>
                        <div style='margin-top:10px;padding-top:10px;border-top:1px solid #2E3250;'>
                            <div style='color:#FF4040;font-size:0.8rem;'>▲ {gl("Bull Case", T("pred_bull"))}: ${p['bull']:,.2f}</div>
                            <div style='color:#4488FF;font-size:0.8rem;'>▼ {gl("Bear Case", T("pred_bear"))}: ${p['bear']:,.2f}</div>
                        </div>
                        <div style='margin-top:8px;color:#8B9DB0;font-size:0.75rem;'>
                            {gl("Volatility (Annualized)", T("volatility"))}: {p['vol_annual']:.1f}%
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

            # ── Prediction rationale summary ──────────────────────────────
            _pred_rationale = summarize_prediction_rationale(predictions, df_5y, ticker, company_name, lang)
            if _pred_rationale:
                st.markdown(_pred_rationale, unsafe_allow_html=True)
        else:
            st.warning("Insufficient data for prediction. Need at least 60 trading days.")

    # ══════════════════ TAB 3: NEWS ══════════════════
    with tabs[2]:
        col_title, col_trans, col_update = st.columns([3, 1, 1])
        with col_title:
            st.markdown(f"<div class='section-header'>{T('news_title')}: {company_name} ({ticker})</div>",
                        unsafe_allow_html=True)
        with col_trans:
            trans_label = ("🌐 영어로 보기" if st.session_state.news_translated else "🌐 한글 번역")
            if st.button(trans_label, use_container_width=True, key="news_trans_btn"):
                st.session_state.news_translated = not st.session_state.news_translated
                st.rerun()
        with col_update:
            if st.button("🔄 " + ("새로고침" if lang == "ko" else "Refresh"), use_container_width=True, key="news_refresh_btn"):
                st.cache_data.clear()
                st.rerun()

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.markdown(f"<span class='update-badge'>🟢 {T('last_updated')}: {now_str} UTC &nbsp;|&nbsp; 5분마다 자동갱신</span>",
                    unsafe_allow_html=True)
        if st.session_state.news_translated:
            st.markdown("<span class='update-badge' style='background:#1E3A5F;color:#64B5F6;margin-left:8px;'>🌐 한글 번역 중</span>",
                        unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        with st.spinner(T("news_loading")):
            articles = fetch_company_news(ticker, company_name)

        if articles:
            n_cols = 2
            for i in range(0, min(len(articles), 20), n_cols):
                row_arts = articles[i:i+n_cols]
                cols_n = st.columns(n_cols)
                for col_n, art in zip(cols_n, row_arts):
                    with col_n:
                        title_text   = art["title"]
                        summary_text = art.get("summary", "")
                        if st.session_state.news_translated:
                            with st.spinner("번역 중..."):
                                title_text   = translate_to_korean(title_text)
                                summary_text = translate_to_korean(summary_text)
                        st.markdown(f"""
                        <div class='news-card'>
                            <div class='news-title'><a href='{art["link"]}' target='_blank'
                                style='color:#EAEAEA;text-decoration:none;'>{title_text}</a></div>
                            <div class='news-meta'>📡 {art["source"]} &nbsp;|&nbsp;
                                🕐 {art["published"][:16] if art["published"] else "N/A"}</div>
                            <div class='news-summary'>{summary_text}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info(("관련 뉴스를 찾지 못했습니다. 잠시 후 새로고침 해주세요."
                     if lang == "ko" else
                     "No relevant news found for this ticker. Try refreshing."))

        # ── News impact analysis summary ──────────────────────────────────
        _news_summary_html = summarize_news_impact(
            articles, company_name, ticker, change_1d, change_1d, lang
        )
        if _news_summary_html:
            st.markdown(_news_summary_html, unsafe_allow_html=True)

    # ══════════════════ TAB 4: GEOPOLITICAL ══════════════════
    with tabs[3]:
        _geo_hdr_col, _geo_btn_col, _geo_ref_col = st.columns([3, 1, 1])
        with _geo_hdr_col:
            st.markdown(f"<div class='section-header'>{T('geo_title')}: {company_name}</div>",
                        unsafe_allow_html=True)
        with _geo_btn_col:
            _geo_btn_label = "🌐 영어로 보기" if st.session_state.geo_translated else "🌐 한글 번역"
            if st.button(_geo_btn_label, key="geo_trans_btn", use_container_width=True):
                st.session_state.geo_translated = not st.session_state.geo_translated
                st.rerun()
        with _geo_ref_col:
            if st.button("🔄 " + ("새로고침" if lang == "ko" else "Refresh"), key="geo_refresh_btn", use_container_width=True):
                st.cache_data.clear()
                st.rerun()

        # Current macro metrics (always relevant)
        gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
        macro_display = [
            (gl("VIX (공포지수)" if lang == "ko" else "VIX", T("geo_vix")), "VIX", "#FF4B4B"),
            (gl("WTI 원유" if lang == "ko" else "WTI", T("geo_oil")), "Oil (WTI)", "#FFA500"),
            (gl("금" if lang == "ko" else "Gold", T("geo_gold")), "Gold", "#FFD700"),
            (gl("달러 인덱스 (DXY)" if lang == "ko" else "USD Index (DXY)", T("geo_dxy")), "USD Index", "#64B5F6"),
            (gl("미국채 10년 수익률" if lang == "ko" else "10Y Treasury", T("geo_bonds")), "10Y Treasury", "#AB63FA"),
        ]
        for col, (label, key, color) in zip([gcol1, gcol2, gcol3, gcol4, gcol5], macro_display):
            data = macro_data.get(key, {})
            val = data.get("current", 0)
            chg = data.get("change_pct", 0)
            chg_color = "#FF4040" if chg >= 0 else "#4488FF"
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div style='font-size:1.2rem;font-weight:700;color:{color};'>{val:,.2f}</div>
                <div style='color:{chg_color};font-size:0.85rem;'>{chg:+.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Sector-filtered risk factors
        _co_sector   = info.get("sector", "") or ""
        _co_industry = info.get("industry", "") or ""
        _eff_lang_geo = "ko" if (lang == "ko" or st.session_state.geo_translated) else "en"
        risk_factors  = get_relevant_geo_factors(_co_sector, _eff_lang_geo)

        if _co_sector:
            _sec_badge = f"<span style='background:#1A2744;border:1px solid #FFA500;border-radius:12px;padding:2px 10px;font-size:0.78rem;color:#FFA500;'>{_co_sector}</span>"
            st.markdown(
                f"<div class='section-header'>{T('geo_factors')} {_sec_badge}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"<div class='section-header'>{T('geo_factors')}</div>", unsafe_allow_html=True)

        risk_labels = {
            "high_risk": (T("high_risk"), "risk-high"),
            "med_risk":  (T("med_risk"),  "risk-med"),
            "low_risk":  (T("low_risk"),  "risk-low"),
        }
        rf_col1, rf_col2 = st.columns(2)
        for i, (title_rf, risk_key, desc_rf, risk_type) in enumerate(risk_factors):
            risk_text, risk_class = risk_labels[risk_type]
            with (rf_col1 if i % 2 == 0 else rf_col2):
                st.markdown(f"""
                <div class='geo-card'>
                    <div style='display:flex;justify-content:space-between;align-items:center;'>
                        <span style='font-weight:700;color:#EAEAEA;font-size:0.95rem;'>{title_rf}</span>
                        <span class='{risk_class}'>[{risk_text}]</span>
                    </div>
                    <div style='color:#B0BEC5;font-size:0.82rem;margin-top:6px;'>{desc_rf}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Live geopolitical/macro news relevant to this company's sector
        _geo_news_label = ("🌐 관련 최신 지정학·거시경제 뉴스" if lang == "ko"
                           else "🌐 Latest Relevant Geopolitical & Macro News")
        st.markdown(f"<div class='section-header'>{_geo_news_label}</div>", unsafe_allow_html=True)

        with st.spinner("뉴스 로딩 중..." if lang == "ko" else "Loading news..."):
            geo_news = fetch_geo_news(ticker, _co_sector, _co_industry)

        if geo_news:
            for art in geo_news:
                title_g   = art["title"]
                summary_g = art.get("summary", "")
                if st.session_state.geo_translated:
                    with st.spinner("번역 중..."):
                        title_g   = translate_to_korean(title_g)
                        summary_g = translate_to_korean(summary_g)
                st.markdown(f"""
                <div class='news-card' style='border-left:3px solid #AB63FA;'>
                    <div class='news-title'><a href='{art["link"]}' target='_blank'
                        style='color:#EAEAEA;text-decoration:none;'>{title_g}</a></div>
                    <div class='news-meta'>📡 {art["source"]} &nbsp;|&nbsp;
                        🕐 {art["published"][:16] if art["published"] else "N/A"}</div>
                    <div class='news-summary'>{summary_g}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("관련 지정학 뉴스를 찾지 못했습니다." if lang == "ko" else "No relevant geopolitical news found.")

        # ── Geo impact analysis summary ───────────────────────────────────
        _geo_summary_html = summarize_geo_impact(
            risk_factors, geo_news, company_name, _co_sector, 0.0, lang
        )
        if _geo_summary_html:
            st.markdown(_geo_summary_html, unsafe_allow_html=True)

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
                        color = "#4488FF" if event["impact"] < 0 else "#FF4040"
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
                height=580,
                title=f"{company_name} — {'Full History with Key Events' if lang == 'en' else '전체 역사 & 주요 이벤트'}",
                xaxis=dict(
                    rangeselector=_rangeselector(lang),
                    rangeslider=dict(visible=False),
                    type="date",
                    title="Date" if lang == "en" else "날짜",
                ),
                yaxis_title="Price (USD)" if lang == "en" else "주가 (USD)",
                margin=dict(l=0, r=0, t=70, b=0),
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
                    _clean = sup_name.split("(")[0].strip()
                    _sym = resolve_ticker(_clean)
                    _btn_key = f"rel_sup_{ticker}_{sup_name[:20]}"
                    if st.button(f"📈 {sup_name}", key=_btn_key, use_container_width=True):
                        st.session_state.ticker = _sym
                        st.session_state.sidebar_view = None
                        st.rerun()
                    st.markdown(f"""
                    <div style='background:#1A1F35;border-radius:10px;padding:14px 18px;margin:-8px 0 10px 0;border:1px solid {color}40;'>
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
                    _clean = comp_name.split("/")[0].split("(")[0].strip()
                    _sym = resolve_ticker(_clean)
                    _btn_key = f"rel_comp_{ticker}_{comp_name[:20]}"
                    target_col = cc1 if i%2==0 else cc2
                    with target_col:
                        if st.button(f"📈 {comp_name}", key=_btn_key, use_container_width=True):
                            st.session_state.ticker = _sym
                            st.session_state.sidebar_view = None
                            st.rerun()
                        st.markdown(f"""
                        <div class='geo-card' style='border-left:3px solid #FF4B4B;margin:-8px 0 8px 0;'>
                            <div style='font-weight:700;color:#FF6B6B;'>⚔️ {comp_name}</div>
                            <div style='color:#B0BEC5;font-size:0.82rem;margin-top:6px;'>{comp_desc}</div>
                        </div>""", unsafe_allow_html=True)
            with rel_tabs[2]:
                for cust_name, cust_desc in relations.get("customers"+suffix, []):
                    _clean = cust_name.split("(")[0].strip()
                    _sym = resolve_ticker(_clean)
                    _is_generic = _sym == _clean.upper() and len(_sym) > 6
                    _btn_key = f"rel_cust_{ticker}_{cust_name[:20]}"
                    if not _is_generic:
                        if st.button(f"📈 {cust_name}", key=_btn_key, use_container_width=True):
                            st.session_state.ticker = _sym
                            st.session_state.sidebar_view = None
                            st.rerun()
                    st.markdown(f"""
                    <div class='geo-card' style='border-left:3px solid #00D4AA;{'margin:-8px 0 8px 0;' if not _is_generic else ''}'>
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
                    _clean = sub_name.split("(")[0].strip()
                    _sym = resolve_ticker(_clean)
                    _is_generic = _sym == _clean.upper() and len(_sym) > 8
                    _btn_key = f"rel_sub_{ticker}_{sub_name[:20]}"
                    if not _is_generic:
                        if st.button(f"📈 {sub_name}", key=_btn_key, use_container_width=True):
                            st.session_state.ticker = _sym
                            st.session_state.sidebar_view = None
                            st.rerun()
                    st.markdown(f"""
                    <div class='geo-card' style='border-left:3px solid #AB63FA;{'margin:-8px 0 8px 0;' if not _is_generic else ''}'>
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

    # ══════════════════ TAB 8: SEMICONDUCTOR ECOSYSTEM ══════════════════
SEMI_UNIVERSE = {
    "fabless_en": {
        "label": "🧠 Fabless Chip Designers",
        "desc": "Design chips but outsource fabrication to foundries",
        "color": "#FFA500",
        "companies": [
            ("NVDA",  "NVIDIA",     "AI GPU / Data Center",        "USA",    4.5),
            ("AMD",   "AMD",        "CPU / GPU / AI Chips",         "USA",    3.2),
            ("QCOM",  "Qualcomm",   "Mobile SoC / 5G Modem",       "USA",    2.8),
            ("AVGO",  "Broadcom",   "Networking / Wi-Fi Chips",     "USA",    3.8),
            ("MRVL",  "Marvell",    "Data Center / 5G",            "USA",    1.2),
            ("MCHP",  "Microchip",  "MCU / Analog",                "USA",    0.8),
            ("SWKS",  "Skyworks",   "RF Chips for Mobile",         "USA",    0.7),
            ("MPWR",  "Monolithic Power","Power Management",       "USA",    0.9),
            ("AMAT",  "Applied Mat","(Equipment — see below)",     "USA",    None),
            ("ARM",   "Arm Holdings","CPU Architecture Licensor",  "UK",     1.5),
            ("MDIA",  "MediaTek",   "Mobile SoC (budget/mid)",     "Taiwan", 1.1),
            ("BRCM",  "Broadcom",   "Networking ASIC",             "USA",    3.8),
        ],
    },
    "idm_en": {
        "label": "🏭 IDM (Integrated Device Manufacturers)",
        "desc": "Design AND manufacture their own chips",
        "color": "#00D4AA",
        "companies": [
            ("INTC",      "Intel",           "CPU / Data Center / Foundry", "USA",    1.8),
            ("005930.KS", "Samsung",         "Logic / DRAM / NAND / Foundry","Korea",  4.5),
            ("TXN",       "Texas Instruments","Analog / Embedded",          "USA",    1.6),
            ("STM",       "STMicroelectronics","MCU / Power / Automotive",  "Europe", 0.7),
            ("NXPI",      "NXP Semiconductors","Automotive / IoT",          "Netherlands",0.6),
            ("ON",        "ON Semiconductor","Power / Automotive",          "USA",    0.5),
            ("WOLF",      "Wolfspeed",       "Silicon Carbide (SiC) EV",    "USA",    0.2),
        ],
    },
    "foundry_en": {
        "label": "🔬 Pure-Play Foundries",
        "desc": "Manufacture chips designed by others (contract fab)",
        "color": "#AB63FA",
        "companies": [
            ("TSM",       "TSMC",            "World's #1 Foundry — 2nm/3nm/5nm", "Taiwan", 8.5),
            ("GFS",       "GlobalFoundries", "Mature nodes, US/EU security supply", "USA", 0.9),
            ("000660.KS", "SK Hynix",        "Memory + Foundry services",       "Korea", 1.2),
            ("UMC",       "UMC",             "Mature node foundry",             "Taiwan", 0.4),
            ("SMICY",     "SMIC",            "China's largest foundry (N+2 node)","China", 0.6),
            ("VX",        "Semtech",         "Specialized analog foundry",      "USA",    0.2),
        ],
    },
    "memory_en": {
        "label": "💾 Memory Manufacturers",
        "desc": "DRAM, NAND Flash, HBM production",
        "color": "#64B5F6",
        "companies": [
            ("MU",        "Micron",          "DRAM / NAND / HBM — US memory giant", "USA",   0.9),
            ("005930.KS", "Samsung Memory",  "World #1 DRAM + NAND + HBM3E",    "Korea",  4.5),
            ("000660.KS", "SK Hynix",        "HBM3E leader for NVIDIA AI chips", "Korea",  1.2),
            ("WDC",       "Western Digital", "NAND / SSD storage",              "USA",    0.5),
            ("STX",       "Seagate",         "HDD + Enterprise storage",        "USA",    0.4),
        ],
    },
    "equipment_en": {
        "label": "⚙️ Semiconductor Equipment",
        "desc": "Machines that make chips — critical chokepoint",
        "color": "#FF6B6B",
        "companies": [
            ("ASML",  "ASML",              "EUV Lithography — sole global supplier", "Netherlands", 3.8),
            ("AMAT",  "Applied Materials", "CVD/PVD/Etch/CMP tools",            "USA",  1.6),
            ("LRCX",  "Lam Research",      "Etch & Deposition systems",          "USA",  1.1),
            ("KLAC",  "KLA Corporation",   "Process Control & Inspection",       "USA",  0.9),
            ("ONTO",  "Onto Innovation",   "Metrology & Inspection",             "USA",  0.2),
            ("ACMR",  "ACM Research",      "Wafer Cleaning — China alt.",        "USA",  0.2),
            ("TOELY", "Tokyo Electron",    "CVD / Etch / Coater systems",        "Japan",1.8),
            ("HIMX",  "Himax",             "Display driver ICs",                 "Taiwan",0.1),
            ("COHU",  "Cohu",              "Semiconductor test handlers",        "USA",  0.1),
            ("FORM",  "FormFactor",        "Wafer probe cards",                  "USA",  0.1),
            ("ENTG",  "Entegris",          "Materials delivery systems",         "USA",  0.3),
            ("CCMP",  "CMC Materials",     "CMP slurries & pads",               "USA",  0.2),
        ],
    },
    "materials_en": {
        "label": "⛏️ Semiconductor Materials & Chemicals",
        "desc": "Silicon wafers, gases, photoresist, specialty chemicals",
        "color": "#FFD700",
        "companies": [
            ("SIEGY", "Shin-Etsu Chemical","#1 silicon wafer maker globally",   "Japan", 1.2),
            ("SUMCF", "SUMCO",             "#2 silicon wafer maker",            "Japan", 0.3),
            ("SOLV",  "Solvay",            "Ultra-pure chemicals for fabs",     "Belgium",0.2),
            ("APD",   "Air Products",      "Ultra-pure gases (N2, H2, Ar, O2)", "USA",   0.6),
            ("LIN",   "Linde",             "Specialty gases for semiconductor", "Ireland",1.8),
            ("AZPN",  "AspenTech",         "Process optimization software",     "USA",   0.2),
            ("CMC",   "CMC Materials",     "Polishing slurries (CMP)",          "USA",   0.2),
            ("FSM",   "Ferroglobe",        "Silicon metal — solar/semi raw mat","Spain", 0.1),
            ("TROX",  "Tronox",            "Titanium dioxide specialty chems",  "USA",   0.1),
        ],
    },
    "packaging_en": {
        "label": "📦 Advanced Packaging & Testing",
        "desc": "OSAT, CoWoS, HBM stacking, chip-on-wafer",
        "color": "#00BFA5",
        "companies": [
            ("ASX",   "ASE Technology",   "World's #1 OSAT packaging & test",  "Taiwan",0.5),
            ("AMKR",  "Amkor Technology", "#2 OSAT — advanced packaging",       "USA",   0.3),
            ("MX",    "Magnachip",        "Display driver, OLED IC",            "Korea", 0.1),
            ("IMOS",  "ChipMOS",          "Memory test & packaging",            "Taiwan",0.1),
            ("SPIL",  "SPIL",             "Siliconware Precision packaging",    "Taiwan",0.2),
        ],
    },
    "eda_ip_en": {
        "label": "🖥️ EDA Software & IP",
        "desc": "Design tools and IP blocks that enable chip design",
        "color": "#CE93D8",
        "companies": [
            ("SNPS",  "Synopsys",          "EDA tools + IP (acquired Ansys)",   "USA",   0.9),
            ("CDNS",  "Cadence Design",    "EDA tools — PCB, IC, system sim",   "USA",   0.8),
            ("MENT",  "Siemens EDA",       "Mentor Graphics — part of Siemens", "Germany",None),
            ("ARM",   "Arm Holdings",      "CPU IP cores — licensed to all",    "UK",    1.5),
            ("AMBA",  "Ambarella",         "Vision AI / SoC IP",                "USA",   0.2),
            ("IMPV",  "Imperva",           "Security IP",                       "USA",   0.1),
        ],
    },
}

# Korean version (same tickers, translated labels)
SEMI_UNIVERSE_KO = {
    "fabless": {
        "label": "🧠 팹리스 (설계 전문)",
        "desc": "칩을 설계하되 생산은 파운드리에 외주",
        "color": "#FFA500",
    },
    "idm": {
        "label": "🏭 종합반도체기업 (IDM)",
        "desc": "설계·생산을 모두 자체 수행",
        "color": "#00D4AA",
    },
    "foundry": {
        "label": "🔬 파운드리 (위탁생산)",
        "desc": "다른 기업이 설계한 칩을 수탁 생산",
        "color": "#AB63FA",
    },
    "memory": {
        "label": "💾 메모리 반도체",
        "desc": "DRAM, NAND 플래시, HBM 생산",
        "color": "#64B5F6",
    },
    "equipment": {
        "label": "⚙️ 반도체 장비",
        "desc": "반도체 제조 장비 — 공급망의 핵심 병목",
        "color": "#FF6B6B",
    },
    "materials": {
        "label": "⛏️ 소재·화학·가스",
        "desc": "실리콘 웨이퍼, 특수가스, 포토레지스트, 슬러리",
        "color": "#FFD700",
    },
    "packaging": {
        "label": "📦 패키징·테스트 (OSAT)",
        "desc": "CoWoS, HBM 스태킹, 칩온웨이퍼 등 고급 패키징",
        "color": "#00BFA5",
    },
    "eda_ip": {
        "label": "🖥️ EDA 소프트웨어·IP",
        "desc": "칩 설계를 가능하게 하는 툴 및 IP 블록",
        "color": "#CE93D8",
    },
}

SEMI_SUPPLY_CHAIN_KO = [
    ("⛏️ 소재·가스", "SIEGY, SUMCF, APD, Linde\n실리콘 웨이퍼, 특수가스"),
    ("⚙️ 장비", "ASML, AMAT, LRCX, KLAC, TEL\nEUV 노광·식각·증착 장비"),
    ("🔬 파운드리", "TSMC, 삼성, GlobalFoundries\n웨이퍼 위탁 생산"),
    ("🧠 설계 (팹리스)", "NVDA, AMD, Qualcomm, Broadcom\n칩 아키텍처 설계"),
    ("📦 패키징·테스트", "ASE, Amkor\n최종 패키징·검수"),
    ("🖥️ 완제품", "서버, PC, 스마트폰, 자동차\n최종 고객"),
]

SEMI_SUPPLY_CHAIN_EN = [
    ("⛏️ Materials & Gases", "SIEGY, SUMCF, APD, Linde\nSilicon wafers, specialty gases"),
    ("⚙️ Equipment", "ASML, AMAT, LRCX, KLAC, TEL\nEUV litho, etch, deposition tools"),
    ("🔬 Foundry / Fab", "TSMC, Samsung, GlobalFoundries\nWafer contract manufacturing"),
    ("🧠 Chip Design (Fabless)", "NVDA, AMD, Qualcomm, Broadcom\nChip architecture & design"),
    ("📦 Packaging & Test", "ASE, Amkor\nFinal packaging & quality test"),
    ("🖥️ End Products", "Servers, PCs, Smartphones, Autos\nEnd customers"),
]

@st.cache_data(ttl=600)
def get_semi_prices(tickers: list) -> dict:
    result = {}
    for t in tickers:
        try:
            info_d = yf.Ticker(t).fast_info
            price = getattr(info_d, "last_price", 0) or 0
            prev  = getattr(info_d, "previous_close", price) or price
            chg   = (price - prev) / prev * 100 if prev else 0
            mktcap= getattr(info_d, "market_cap", 0) or 0
            result[t] = {"price": price, "chg": chg, "mktcap": mktcap}
        except Exception:
            result[t] = {"price": 0, "chg": 0, "mktcap": 0}
    return result

if st.session_state.sidebar_view == "semi":
    lang_s = lang
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0D1B2A,#1A2744);border-radius:14px;
                padding:18px 28px;margin-bottom:20px;border:1px solid #2E3250;'>
        <div style='font-size:1.4rem;font-weight:800;color:#FFA500;'>
            {'💾 반도체 생태계 완전 분석' if lang_s=='ko' else '💾 Semiconductor Ecosystem'}
        </div>
        <div style='color:#8B9DB0;font-size:0.85rem;margin-top:6px;'>
            {'설계 → 소재 → 장비 → 파운드리 → 패키징까지 전체 밸류체인 커버'
             if lang_s=='ko' else
             'Full value chain: Design → Materials → Equipment → Foundry → Packaging'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Supply chain flow diagram ──
    st.markdown(f"<div class='section-header'>{'반도체 공급망 흐름도' if lang_s=='ko' else 'Supply Chain Flow'}</div>",
                unsafe_allow_html=True)
    chain = SEMI_SUPPLY_CHAIN_KO if lang_s == "ko" else SEMI_SUPPLY_CHAIN_EN
    chain_cols = st.columns(len(chain))
    for idx, (step_title, step_desc) in enumerate(chain):
        arrow = "→" if idx < len(chain) - 1 else ""
        chain_cols[idx].markdown(f"""
        <div style='background:#1A1F35;border:1px solid #3A4060;border-radius:10px;
                    padding:12px 10px;text-align:center;min-height:100px;position:relative;'>
            <div style='font-size:0.85rem;font-weight:700;color:#FFA500;'>{step_title}</div>
            <div style='font-size:0.72rem;color:#B0BEC5;margin-top:6px;white-space:pre-line;line-height:1.5;'>{step_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Category tabs ──
    cat_keys = ["fabless_en","idm_en","foundry_en","memory_en","equipment_en","materials_en","packaging_en","eda_ip_en"]
    cat_labels_ko = SEMI_UNIVERSE_KO
    cat_label_map = {
        "fabless_en":   (SEMI_UNIVERSE["fabless_en"]["label"],   SEMI_UNIVERSE_KO["fabless"]["label"]),
        "idm_en":       (SEMI_UNIVERSE["idm_en"]["label"],       SEMI_UNIVERSE_KO["idm"]["label"]),
        "foundry_en":   (SEMI_UNIVERSE["foundry_en"]["label"],   SEMI_UNIVERSE_KO["foundry"]["label"]),
        "memory_en":    (SEMI_UNIVERSE["memory_en"]["label"],    SEMI_UNIVERSE_KO["memory"]["label"]),
        "equipment_en": (SEMI_UNIVERSE["equipment_en"]["label"], SEMI_UNIVERSE_KO["equipment"]["label"]),
        "materials_en": (SEMI_UNIVERSE["materials_en"]["label"], SEMI_UNIVERSE_KO["materials"]["label"]),
        "packaging_en": (SEMI_UNIVERSE["packaging_en"]["label"], SEMI_UNIVERSE_KO["packaging"]["label"]),
        "eda_ip_en":    (SEMI_UNIVERSE["eda_ip_en"]["label"],    SEMI_UNIVERSE_KO["eda_ip"]["label"]),
    }
    semi_tabs = st.tabs([v[1] if lang_s=="ko" else v[0] for v in cat_label_map.values()])

    for tab_idx, (cat_key, semi_tab) in enumerate(zip(cat_keys, semi_tabs)):
        with semi_tab:
            cat_data = SEMI_UNIVERSE[cat_key]
            color = cat_data["color"]
            desc  = (list(SEMI_UNIVERSE_KO.values())[tab_idx]["desc"]
                     if lang_s=="ko" else cat_data["desc"])

            st.markdown(f"<div style='color:#8B9DB0;font-size:0.85rem;margin-bottom:14px;'>{desc}</div>",
                        unsafe_allow_html=True)

            # Fetch live prices for this category
            tickers_in_cat = [c[0] for c in cat_data["companies"] if c[4] is not None]
            with st.spinner("Loading prices..." if lang_s=="en" else "시세 로딩 중..."):
                price_data = get_semi_prices(tickers_in_cat)

            # Company cards grid
            cols_per_row = 3
            companies = cat_data["companies"]
            for row_start in range(0, len(companies), cols_per_row):
                row_companies = companies[row_start:row_start+cols_per_row]
                row_cols = st.columns(cols_per_row)
                for col_s, (t_sym, t_name, t_role, t_country, t_mktcap_ref) in zip(row_cols, row_companies):
                    pd_live = price_data.get(t_sym, {})
                    live_price = pd_live.get("price", 0)
                    live_chg   = pd_live.get("chg", 0)
                    chg_color  = "#FF4040" if live_chg >= 0 else "#4488FF"
                    chg_arrow  = "▲" if live_chg >= 0 else "▼"
                    price_str  = f"${live_price:,.2f}" if live_price else "—"

                    flag_map = {"USA":"🇺🇸","Korea":"🇰🇷","Taiwan":"🇹🇼","Japan":"🇯🇵",
                                "Netherlands":"🇳🇱","UK":"🇬🇧","Germany":"🇩🇪",
                                "Belgium":"🇧🇪","Ireland":"🇮🇪","China":"🇨🇳",
                                "Spain":"🇪🇸","France":"🇫🇷","Europe":"🇪🇺"}
                    flag = flag_map.get(t_country, "🌐")

                    with col_s:
                        if st.button(
                            f"📈 {t_name}  {price_str}  {chg_arrow}{abs(live_chg):.1f}%",
                            key=f"semi_{cat_key}_{t_sym}",
                            use_container_width=True,
                            help=f"{t_role} | {flag} {t_country} | 클릭하면 분석으로 이동",
                        ):
                            st.session_state.ticker = t_sym
                            st.session_state.sidebar_view = None
                            st.rerun()
                        st.markdown(f"""
                        <div style='background:#1A1F35;border:1px solid {color}40;border-radius:0 0 10px 10px;
                                    padding:8px 14px 12px 14px;margin-bottom:12px;margin-top:-8px;'>
                            <div style='display:flex;justify-content:space-between;align-items:center;'>
                                <span style='color:{color};font-weight:700;font-size:0.82rem;'>{t_sym}</span>
                                <span style='font-size:0.78rem;color:#8B9DB0;'>{flag} {t_country}</span>
                            </div>
                            <div style='color:#B0BEC5;font-size:0.75rem;margin-top:3px;'>{t_role}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Market cap comparison chart for this category
            valid_cos = [(c[0], c[1], c[4]) for c in companies if c[4] is not None]
            if valid_cos:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#8B9DB0;font-size:0.82rem;'>"
                            f"{'시가총액 규모 비교 (조 달러 기준 추정치, 실시간 아님)' if lang_s=='ko' else 'Market Cap Reference (approx. $T — not real-time)'}"
                            f"</div>", unsafe_allow_html=True)
                names_c  = [c[1] for c in valid_cos]
                mktcaps_c= [c[2] for c in valid_cos]
                fig_semi = go.Figure(go.Bar(
                    x=names_c, y=mktcaps_c,
                    marker_color=color,
                    text=[f"${v}T" for v in mktcaps_c],
                    textposition="outside",
                ))
                fig_semi.update_layout(
                    template="plotly_dark", height=280,
                    margin=dict(l=0,r=0,t=20,b=0),
                    yaxis_title="Market Cap (Approx. $T)" if lang_s=="en" else "시총 (추정, 조 달러)",
                    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                )
                st.plotly_chart(fig_semi, use_container_width=True)

    # ── ASML / TSMC spotlight ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-header'>{'🔑 Key Chokepoints' if lang_s=='en' else '🔑 글로벌 핵심 병목 기업'}</div>",
                unsafe_allow_html=True)
    choke_en = [
        ("ASML", "EUV Lithography", "🇳🇱", "#FF4B4B",
         "Sole supplier of EUV machines globally. Without ASML, no sub-7nm chip can be made. "
         "Each machine costs $150-380M. Export-controlled to China. "
         "Key leverage point in US-China tech war."),
        ("TSMC", "Leading-Edge Foundry", "🇹🇼", "#AB63FA",
         "Manufactures chips for Apple, NVIDIA, AMD, Qualcomm, and more. "
         "Controls ~90% of sub-5nm production globally. Taiwan geopolitical risk = "
         "global semiconductor supply risk. Building fabs in Arizona, Japan, Germany."),
        ("Samsung", "Memory + Foundry", "🇰🇷", "#00D4AA",
         "World #1 in DRAM and NAND. Critical HBM supplier for NVIDIA AI chips. "
         "Also competes with TSMC in advanced foundry (GAA 3nm). "
         "Korea's dominant tech export — makes up ~20% of Korean GDP."),
        ("SK Hynix", "HBM Leader", "🇰🇷", "#FFD700",
         "Supplies HBM3E — the memory inside NVIDIA H200/B200 AI GPUs. "
         "Without SK Hynix HBM, NVIDIA cannot build AI data center chips. "
         "Capacity constrained through 2025-2026."),
    ]
    choke_ko = [
        ("ASML", "EUV 리소그래피", "🇳🇱", "#FF4B4B",
         "EUV 장비 세계 유일 공급업체. ASML 없이는 7nm 이하 칩 생산 불가. "
         "장비 1대 가격 1500억~4000억원. 중국 수출 통제 대상. "
         "미중 기술 전쟁의 핵심 레버리지 포인트."),
        ("TSMC", "최첨단 파운드리", "🇹🇼", "#AB63FA",
         "애플·엔비디아·AMD·퀄컴 등의 칩을 위탁 생산. "
         "5nm 이하 글로벌 생산의 약 90% 장악. 대만 지정학 리스크 = "
         "글로벌 반도체 공급 리스크. 미국·일본·독일에 팹 건설 중."),
        ("삼성전자", "메모리 + 파운드리", "🇰🇷", "#00D4AA",
         "DRAM·NAND 세계 1위. 엔비디아 AI 칩용 핵심 HBM 공급사. "
         "GAA 3nm로 TSMC와 파운드리 경쟁 중. "
         "한국 핵심 수출 기업 — 한국 GDP의 약 20% 차지."),
        ("SK하이닉스", "HBM 리더", "🇰🇷", "#FFD700",
         "엔비디아 H200/B200 AI GPU 내부의 HBM3E 공급. "
         "SK하이닉스 HBM 없이는 엔비디아 AI 데이터센터 칩 생산 불가. "
         "2025~2026년까지 생산 용량 제한 상태."),
    ]
    chokepoints = choke_ko if lang_s == "ko" else choke_en
    cho_cols = st.columns(2)
    for ci, (cp_name, cp_role, cp_flag, cp_color, cp_desc) in enumerate(chokepoints):
        with cho_cols[ci % 2]:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1A1F35,#0F1527);
                        border:2px solid {cp_color};border-radius:12px;
                        padding:16px 18px;margin-bottom:14px;'>
                <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
                    <span style='font-size:1.4rem;'>{cp_flag}</span>
                    <span style='font-size:1.1rem;font-weight:800;color:{cp_color};'>{cp_name}</span>
                    <span style='background:{cp_color}20;color:{cp_color};border-radius:8px;
                                 padding:2px 8px;font-size:0.75rem;'>{cp_role}</span>
                </div>
                <div style='color:#B0BEC5;font-size:0.82rem;line-height:1.6;'>{cp_desc}</div>
                <div style='margin-top:10px;'>
                    {'<button onclick="void(0)" style="background:#1E2130;color:#FFFFFF;border:1px solid #3A4060;border-radius:6px;padding:4px 12px;cursor:pointer;font-size:0.78rem;">차트 보기 →</button>'
                     if lang_s=="ko" else
                     '<button style="background:#1E2130;color:#FFFFFF;border:1px solid #3A4060;border-radius:6px;padding:4px 12px;cursor:pointer;font-size:0.78rem;">View Chart →</button>'}
                </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════ SECTOR DATABASE ══════════════════
SECTOR_DB = [
    {
        "id": "ai",
        "icon": "🤖",
        "label_en": "Artificial Intelligence",
        "label_ko": "인공지능 (AI)",
        "color": "#FFA500",
        "companies": [
            ("NVDA","NVIDIA","GPU/AI Infra","🇺🇸"),
            ("MSFT","Microsoft","Copilot/Azure AI","🇺🇸"),
            ("GOOGL","Alphabet","Gemini/TPU/DeepMind","🇺🇸"),
            ("META","Meta","Llama/FAIR Research","🇺🇸"),
            ("AMZN","Amazon","Bedrock/Trainium","🇺🇸"),
            ("ORCL","Oracle","AI Cloud/GPU Infra","🇺🇸"),
            ("CRM","Salesforce","Einstein AI/Agentforce","🇺🇸"),
            ("NOW","ServiceNow","Enterprise AI Agents","🇺🇸"),
            ("PLTR","Palantir","AI/Big Data Analytics","🇺🇸"),
            ("AI","C3.ai","Enterprise AI Apps","🇺🇸"),
            ("SOUN","SoundHound AI","Voice AI","🇺🇸"),
            ("BBAI","BigBear.ai","AI Decision Intelligence","🇺🇸"),
            ("PATH","UiPath","RPA/AI Automation","🇺🇸"),
            ("SNOW","Snowflake","Data Cloud + AI","🇺🇸"),
            ("DDOG","Datadog","AI Observability","🇺🇸"),
            ("S","SentinelOne","AI Cybersecurity","🇺🇸"),
            ("CFLT","Confluent","Data Streaming AI","🇺🇸"),
            ("MDB","MongoDB","Vector DB for AI","🇺🇸"),
        ],
    },
    {
        "id": "semiconductor",
        "icon": "💾",
        "label_en": "Semiconductors",
        "label_ko": "반도체",
        "color": "#AB63FA",
        "companies": [
            ("NVDA","NVIDIA","AI GPU","🇺🇸"),
            ("TSM","TSMC","Foundry #1","🇹🇼"),
            ("AVGO","Broadcom","Networking Chips","🇺🇸"),
            ("AMD","AMD","CPU/GPU","🇺🇸"),
            ("INTC","Intel","CPU/IDM","🇺🇸"),
            ("QCOM","Qualcomm","Mobile SoC","🇺🇸"),
            ("ASML","ASML","EUV Equipment","🇳🇱"),
            ("AMAT","Applied Materials","Fab Equipment","🇺🇸"),
            ("LRCX","Lam Research","Etch Tools","🇺🇸"),
            ("KLAC","KLA Corp","Process Control","🇺🇸"),
            ("MU","Micron","DRAM/NAND/HBM","🇺🇸"),
            ("TXN","Texas Instruments","Analog/MCU","🇺🇸"),
            ("ARM","Arm Holdings","CPU IP","🇬🇧"),
            ("MRVL","Marvell","Data Center","🇺🇸"),
            ("SNPS","Synopsys","EDA Tools","🇺🇸"),
            ("CDNS","Cadence","EDA/PCB","🇺🇸"),
            ("ENTG","Entegris","Fab Materials","🇺🇸"),
            ("MPWR","Monolithic Power","Power Mgmt","🇺🇸"),
            ("005930.KS","Samsung","Memory+Foundry","🇰🇷"),
            ("000660.KS","SK Hynix","HBM/DRAM","🇰🇷"),
        ],
    },
    {
        "id": "biotech",
        "icon": "🧬",
        "label_en": "Biotech",
        "label_ko": "바이오테크",
        "color": "#00D4AA",
        "companies": [
            ("REGN","Regeneron","Eylea/Dupixent","🇺🇸"),
            ("VRTX","Vertex Pharma","CRISPR/CF Drugs","🇺🇸"),
            ("MRNA","Moderna","mRNA Vaccines","🇺🇸"),
            ("BNTX","BioNTech","mRNA Platform","🇩🇪"),
            ("GILD","Gilead","HIV/Oncology","🇺🇸"),
            ("BIIB","Biogen","Alzheimer's","🇺🇸"),
            ("ILMN","Illumina","DNA Sequencing","🇺🇸"),
            ("CRSP","CRISPR Therapeutics","Gene Editing","🇨🇭"),
            ("EDIT","Editas Medicine","CRISPR Editing","🇺🇸"),
            ("NTLA","Intellia Therapeutics","CRISPR In Vivo","🇺🇸"),
            ("BEAM","Beam Therapeutics","Base Editing","🇺🇸"),
            ("PACB","PacBio","Long-read Sequencing","🇺🇸"),
            ("RXRX","Recursion Pharma","AI Drug Discovery","🇺🇸"),
            ("EXAS","Exact Sciences","Cancer Screening","🇺🇸"),
            ("FATE","Fate Therapeutics","Cell Therapy","🇺🇸"),
            ("IONS","Ionis Pharma","RNA Therapeutics","🇺🇸"),
            ("ALNY","Alnylam","RNAi Therapy","🇺🇸"),
            ("SGEN","Seagen","Antibody-Drug Conj","🇺🇸"),
        ],
    },
    {
        "id": "healthcare",
        "icon": "🏥",
        "label_en": "Healthcare",
        "label_ko": "헬스케어",
        "color": "#64B5F6",
        "companies": [
            ("UNH","UnitedHealth","Health Insurance","🇺🇸"),
            ("JNJ","Johnson & Johnson","Pharma/MedTech","🇺🇸"),
            ("LLY","Eli Lilly","GLP-1/Diabetes","🇺🇸"),
            ("NVO","Novo Nordisk","Ozempic/Wegovy","🇩🇰"),
            ("ABBV","AbbVie","Humira/Skyrizi","🇺🇸"),
            ("MRK","Merck","Keytruda/Vaccines","🇺🇸"),
            ("PFE","Pfizer","Vaccines/Oncology","🇺🇸"),
            ("TMO","Thermo Fisher","Lab Instruments","🇺🇸"),
            ("ABT","Abbott","Diagnostics/CGM","🇺🇸"),
            ("ISRG","Intuitive Surgical","Robotic Surgery","🇺🇸"),
            ("SYK","Stryker","Orthopedics","🇺🇸"),
            ("BSX","Boston Scientific","Cardiovascular","🇺🇸"),
            ("EW","Edwards Life","Heart Valves","🇺🇸"),
            ("DXCM","Dexcom","CGM Glucose Monitor","🇺🇸"),
            ("HCA","HCA Healthcare","Hospital Network","🇺🇸"),
            ("CVS","CVS Health","Pharmacy/Insurance","🇺🇸"),
            ("CI","Cigna","Health Insurance","🇺🇸"),
            ("HUM","Humana","Medicare Advantage","🇺🇸"),
        ],
    },
    {
        "id": "quantum",
        "icon": "⚛️",
        "label_en": "Quantum Computing",
        "label_ko": "양자컴퓨터",
        "color": "#CE93D8",
        "companies": [
            ("IONQ","IonQ","Trapped Ion QC","🇺🇸"),
            ("RGTI","Rigetti","Superconducting QC","🇺🇸"),
            ("QBTS","D-Wave Quantum","Quantum Annealing","🇨🇦"),
            ("QUBT","Quantum Computing Inc","Photonic QC","🇺🇸"),
            ("IBM","IBM","IBM Quantum/1000+ qubits","🇺🇸"),
            ("GOOGL","Alphabet","Willow Quantum Chip","🇺🇸"),
            ("MSFT","Microsoft","Topological Qubit","🇺🇸"),
            ("AMZN","Amazon","Braket Quantum Cloud","🇺🇸"),
            ("HON","Honeywell","Quantinuum (spin-off)","🇺🇸"),
            ("ARQQ","Arqit Quantum","QKD Encryption","🇬🇧"),
            ("BFLY","Butterfly Network","Quantum Sensing","🇺🇸"),
        ],
    },
    {
        "id": "oil",
        "icon": "🛢️",
        "label_en": "Oil & Petroleum",
        "label_ko": "석유",
        "color": "#8B4513",
        "companies": [
            ("XOM","ExxonMobil","Integrated Oil Major","🇺🇸"),
            ("CVX","Chevron","Integrated Oil Major","🇺🇸"),
            ("COP","ConocoPhillips","E&P Focus","🇺🇸"),
            ("EOG","EOG Resources","Shale/Permian E&P","🇺🇸"),
            ("PXD","Pioneer Natural","Permian Basin","🇺🇸"),
            ("OXY","Occidental","Enhanced Oil Recovery","🇺🇸"),
            ("MPC","Marathon Petroleum","Refining","🇺🇸"),
            ("PSX","Phillips 66","Refining/Midstream","🇺🇸"),
            ("VLO","Valero Energy","Largest US Refiner","🇺🇸"),
            ("HAL","Halliburton","Oilfield Services","🇺🇸"),
            ("SLB","SLB (Schlumberger)","Oilfield Services #1","🇺🇸"),
            ("BKR","Baker Hughes","Oilfield Services","🇺🇸"),
            ("SHEL","Shell","European Oil Major","🇬🇧"),
            ("BP","BP","European Oil Major","🇬🇧"),
            ("TTE","TotalEnergies","European Oil Major","🇫🇷"),
            ("E","Eni","Italian Oil Major","🇮🇹"),
            ("CL=F","Crude Oil WTI","Commodity","🌐"),
        ],
    },
    {
        "id": "natgas",
        "icon": "🔥",
        "label_en": "Natural Gas & Shale",
        "label_ko": "천연가스·셰일가스",
        "color": "#FF7043",
        "companies": [
            ("LNG","Cheniere Energy","LNG Export #1 US","🇺🇸"),
            ("AR","Antero Resources","Appalachian Natgas","🇺🇸"),
            ("EQT","EQT Corp","Largest US Natgas E&P","🇺🇸"),
            ("RRC","Range Resources","Marcellus Shale","🇺🇸"),
            ("SWN","Southwestern Energy","Appalachian Shale","🇺🇸"),
            ("CTRA","Coterra Energy","Permian+Marcellus","🇺🇸"),
            ("CNX","CNX Resources","Appalachian Gas","🇺🇸"),
            ("KMI","Kinder Morgan","Gas Pipeline #1","🇺🇸"),
            ("WMB","Williams Companies","Gas Midstream","🇺🇸"),
            ("OKE","ONEOK","NGL Midstream","🇺🇸"),
            ("ET","Energy Transfer","Gas Pipelines","🇺🇸"),
            ("NG=F","Natural Gas Futures","Commodity","🌐"),
        ],
    },
    {
        "id": "solar",
        "icon": "☀️",
        "label_en": "Solar Energy",
        "label_ko": "태양광·태양열",
        "color": "#FFD700",
        "companies": [
            ("ENPH","Enphase Energy","Microinverters","🇺🇸"),
            ("SEDG","SolarEdge","String Inverters","🇮🇱"),
            ("FSLR","First Solar","Thin-Film Panels","🇺🇸"),
            ("SPWR","SunPower","Residential Solar","🇺🇸"),
            ("RUN","Sunrun","Rooftop Solar Lease","🇺🇸"),
            ("ARRY","Array Technologies","Solar Trackers","🇺🇸"),
            ("NOVA","Sunnova Energy","Solar+Storage","🇺🇸"),
            ("SHLS","Shoals Technologies","BOS Components","🇺🇸"),
            ("CSIQ","Canadian Solar","Global Panel Maker","🇨🇦"),
            ("JKS","JinkoSolar","China #1 Panel Maker","🇨🇳"),
            ("DQ","Daqo New Energy","Polysilicon","🇨🇳"),
            ("NEE","NextEra Energy","Solar+Wind Utility","🇺🇸"),
            ("AES","AES Corp","Renewable Utility","🇺🇸"),
            ("CWEN","Clearway Energy","Solar/Wind YieldCo","🇺🇸"),
        ],
    },
    {
        "id": "energy_general",
        "icon": "⚡",
        "label_en": "Energy & Utilities",
        "label_ko": "에너지·전력·유틸리티",
        "color": "#FFCA28",
        "companies": [
            ("NEE","NextEra Energy","Largest US Utility/Solar","🇺🇸"),
            ("DUK","Duke Energy","Nuclear+Coal Utility","🇺🇸"),
            ("SO","Southern Company","Nuclear+Gas Utility","🇺🇸"),
            ("D","Dominion Energy","Mid-Atlantic Utility","🇺🇸"),
            ("EXC","Exelon","Nuclear Power Largest","🇺🇸"),
            ("CEG","Constellation Energy","Nuclear Clean Power","🇺🇸"),
            ("VST","Vistra","Nuclear+Gas Power","🇺🇸"),
            ("NRG","NRG Energy","Competitive Power","🇺🇸"),
            ("PCG","PG&E","California Utility","🇺🇸"),
            ("ED","Consolidated Edison","NYC Utility","🇺🇸"),
            ("AEP","American Electric","Transmission Grid","🇺🇸"),
            ("ETR","Entergy","Nuclear South","🇺🇸"),
            ("AWK","American Water","Water Utility","🇺🇸"),
            ("WEC","WEC Energy","Midwest Utility","🇺🇸"),
            ("ES","Eversource","New England Utility","🇺🇸"),
        ],
    },
    {
        "id": "defense",
        "icon": "🛡️",
        "label_en": "Defense & Aerospace",
        "label_ko": "방산·항공우주",
        "color": "#607D8B",
        "companies": [
            ("LMT","Lockheed Martin","F-35/Missiles/Space","🇺🇸"),
            ("RTX","RTX Corp","Missiles/Jet Engines","🇺🇸"),
            ("NOC","Northrop Grumman","B-21/Cyber/Space","🇺🇸"),
            ("GD","General Dynamics","Submarines/Gulfstream","🇺🇸"),
            ("BA","Boeing","Aircraft/Defense","🇺🇸"),
            ("HII","Huntington Ingalls","Naval Ships","🇺🇸"),
            ("L3","L3Harris","Electronics/Sensors","🇺🇸"),
            ("LDOS","Leidos","IT/Defense Services","🇺🇸"),
            ("SAIC","SAIC","Defense IT","🇺🇸"),
            ("KTOS","Kratos Defense","Drone/Hypersonic","🇺🇸"),
            ("PLTR","Palantir","AI/Data for Military","🇺🇸"),
            ("AVAV","AeroVironment","Tactical Drones","🇺🇸"),
            ("ACHR","Archer Aviation","eVTOL/Urban Air","🇺🇸"),
            ("JOBY","Joby Aviation","Air Taxi","🇺🇸"),
            ("AIR","AAR Corp","MRO/Aviation Support","🇺🇸"),
            ("HEI","HEICO","Aerospace Parts","🇺🇸"),
            ("TDG","TransDigm","Aerospace Components","🇺🇸"),
        ],
    },
    {
        "id": "battery",
        "icon": "🔋",
        "label_en": "Battery & Energy Storage",
        "label_ko": "배터리·에너지저장",
        "color": "#69F0AE",
        "companies": [
            ("TSLA","Tesla","Megapack/4680","🇺🇸"),
            ("ENVX","Enovix","Silicon Anode Battery","🇺🇸"),
            ("QS","QuantumScape","Solid-State Battery","🇺🇸"),
            ("FREYR","FREYR Battery","European Gigafactory","🇳🇴"),
            ("AMPS","Altus Power","Solar+Storage","🇺🇸"),
            ("STEM","Stem Inc","AI Battery Storage","🇺🇸"),
            ("FLUX","Flux Power","Lithium Forklift","🇺🇸"),
            ("NKLA","Nikola","Hydrogen/BEV Trucks","🇺🇸"),
            ("006400.KS","Samsung SDI","EV+ESS Batteries","🇰🇷"),
            ("051910.KS","LG Chem","Battery Materials","🇰🇷"),
            ("373220.KS","LG Energy Solution","EV Batteries","🇰🇷"),
            ("096770.KS","SK Innovation","EV Batteries","🇰🇷"),
            ("CATL","CATL","China #1 Battery","🇨🇳"),
            ("ALB","Albemarle","Lithium Mining","🇺🇸"),
            ("SQM","SQM","Lithium (Chile)","🇨🇱"),
            ("PLL","Piedmont Lithium","US Lithium Mining","🇺🇸"),
            ("LAC","Lithium Americas","Nevada Lithium","🇨🇦"),
        ],
    },
    {
        "id": "automotive",
        "icon": "🚗",
        "label_en": "Automotive & EV",
        "label_ko": "자동차·전기차",
        "color": "#4FC3F7",
        "companies": [
            ("TSLA","Tesla","EV Leader","🇺🇸"),
            ("TM","Toyota","Hybrid/Fuel Cell","🇯🇵"),
            ("GM","General Motors","EV Transition","🇺🇸"),
            ("F","Ford","F-150 Lightning","🇺🇸"),
            ("STLA","Stellantis","Jeep/RAM/Fiat","🇮🇹"),
            ("RIVN","Rivian","EV Trucks/Amazon Van","🇺🇸"),
            ("LCID","Lucid Motors","Luxury Long-Range EV","🇺🇸"),
            ("NIO","NIO","China EV Premium","🇨🇳"),
            ("LI","Li Auto","China EREV","🇨🇳"),
            ("XPEV","XPeng","China EV+ADAS","🇨🇳"),
            ("005380.KS","Hyundai","Ioniq/EV6","🇰🇷"),
            ("000270.KS","Kia","EV6/EV9","🇰🇷"),
            ("VOW","Volkswagen","ID Series EV","🇩🇪"),
            ("BMW","BMW","iX/i4 EV","🇩🇪"),
            ("MBGAF","Mercedes-Benz","EQS EV","🇩🇪"),
            ("MBIN","Merchants Fleet","Fleet Management","🇺🇸"),
            ("GOEV","Canoo","EV Van/Truck","🇺🇸"),
            ("CARZ","Carvana (related ETF)","Used EV Market","🇺🇸"),
        ],
    },
    {
        "id": "tech",
        "icon": "💻",
        "label_en": "Big Tech & Software",
        "label_ko": "빅테크·소프트웨어",
        "color": "#29B6F6",
        "companies": [
            ("AAPL","Apple","iPhone/Mac/Services","🇺🇸"),
            ("MSFT","Microsoft","Windows/Azure/AI","🇺🇸"),
            ("GOOGL","Alphabet","Search/Cloud/AI","🇺🇸"),
            ("META","Meta","Social/AR/VR","🇺🇸"),
            ("AMZN","Amazon","eCommerce/AWS","🇺🇸"),
            ("NFLX","Netflix","Streaming","🇺🇸"),
            ("ADBE","Adobe","Creative Cloud/AI","🇺🇸"),
            ("CRM","Salesforce","CRM/AI","🇺🇸"),
            ("NOW","ServiceNow","Enterprise Automation","🇺🇸"),
            ("SHOP","Shopify","eCommerce Platform","🇨🇦"),
            ("UBER","Uber","Rideshare/Delivery","🇺🇸"),
            ("LYFT","Lyft","Rideshare","🇺🇸"),
            ("ABNB","Airbnb","Travel Platform","🇺🇸"),
            ("DASH","DoorDash","Food Delivery","🇺🇸"),
            ("SPOT","Spotify","Music Streaming","🇸🇪"),
            ("SNAP","Snap","Social/AR","🇺🇸"),
            ("X","X (Twitter)","Social Media","🇺🇸"),
            ("PINS","Pinterest","Visual Search","🇺🇸"),
            ("TWLO","Twilio","Communication API","🇺🇸"),
            ("ZM","Zoom","Video Conferencing","🇺🇸"),
        ],
    },
    {
        "id": "electronics",
        "icon": "📱",
        "label_en": "Electronics & Consumer Tech",
        "label_ko": "전자제품·소비가전",
        "color": "#FF8A65",
        "companies": [
            ("AAPL","Apple","Consumer Electronics","🇺🇸"),
            ("005930.KS","Samsung","TVs/Appliances/Phones","🇰🇷"),
            ("SONY","Sony","PS5/TV/Camera","🇯🇵"),
            ("LG","LG Electronics","OLED TV/Appliances","🇰🇷"),
            ("PANAY","Panasonic","Batteries/B2B","🇯🇵"),
            ("MSI","MSI/Micro-Star","Gaming PCs/GPUs","🇹🇼"),
            ("HPQ","HP Inc","PC/Printer","🇺🇸"),
            ("DELL","Dell Tech","PC/Server","🇺🇸"),
            ("LNVGY","Lenovo","PC #1 Global","🇨🇳"),
            ("BBY","Best Buy","Electronics Retail","🇺🇸"),
            ("GME","GameStop","Gaming Retail","🇺🇸"),
            ("AMZN","Amazon","Electronics eRetail","🇺🇸"),
            ("WMT","Walmart","Electronics Retail","🇺🇸"),
            ("NFLX","Netflix","Streaming Device","🇺🇸"),
            ("ROKU","Roku","Streaming Platform","🇺🇸"),
            ("VZIO","Vizio","Smart TVs","🇺🇸"),
            ("HEAR","Turtle Beach","Gaming Headsets","🇺🇸"),
        ],
    },
    {
        "id": "finance",
        "icon": "🏦",
        "label_en": "Finance & Banking",
        "label_ko": "금융·은행",
        "color": "#42A5F5",
        "companies": [
            ("JPM","JPMorgan Chase","Largest US Bank","🇺🇸"),
            ("BAC","Bank of America","Retail Banking","🇺🇸"),
            ("WFC","Wells Fargo","Consumer Banking","🇺🇸"),
            ("GS","Goldman Sachs","Investment Bank","🇺🇸"),
            ("MS","Morgan Stanley","Wealth Mgmt","🇺🇸"),
            ("BRK-B","Berkshire Hathaway","Conglomerate/Insurance","🇺🇸"),
            ("V","Visa","Payment Network","🇺🇸"),
            ("MA","Mastercard","Payment Network","🇺🇸"),
            ("PYPL","PayPal","Digital Payments","🇺🇸"),
            ("SQ","Block (Square)","Fintech","🇺🇸"),
            ("COIN","Coinbase","Crypto Exchange","🇺🇸"),
            ("BX","Blackstone","Private Equity","🇺🇸"),
            ("KKR","KKR","Private Equity","🇺🇸"),
            ("APO","Apollo Global","Alternative Assets","🇺🇸"),
            ("SCHW","Charles Schwab","Brokerage","🇺🇸"),
            ("ICE","ICE","Exchange Operator","🇺🇸"),
            ("CME","CME Group","Derivatives Exchange","🇺🇸"),
        ],
    },
    {
        "id": "crypto",
        "icon": "₿",
        "label_en": "Crypto & Digital Assets",
        "label_ko": "암호화폐·디지털자산",
        "color": "#F7931A",
        "companies": [
            ("BTC-USD","Bitcoin","Store of Value","🌐"),
            ("ETH-USD","Ethereum","Smart Contract L1","🌐"),
            ("SOL-USD","Solana","High-Speed L1","🌐"),
            ("BNB-USD","BNB","Binance Chain","🌐"),
            ("COIN","Coinbase","Crypto Exchange","🇺🇸"),
            ("MSTR","MicroStrategy","Bitcoin Treasury","🇺🇸"),
            ("MARA","Marathon Digital","Bitcoin Mining","🇺🇸"),
            ("RIOT","Riot Platforms","Bitcoin Mining","🇺🇸"),
            ("HUT","Hut 8","Bitcoin Mining","🇨🇦"),
            ("CLSK","CleanSpark","Green BTC Mining","🇺🇸"),
            ("CIFR","Cipher Mining","Bitcoin Mining","🇺🇸"),
            ("CRCL","Circle (private)","USDC Stablecoin","🇺🇸"),
            ("GBTC","Grayscale BTC Trust","BTC ETF","🇺🇸"),
            ("IBIT","iShares BTC ETF","BTC ETF (BlackRock)","🇺🇸"),
        ],
    },
    {
        "id": "space",
        "icon": "🚀",
        "label_en": "Space & Satellites",
        "label_ko": "우주·위성",
        "color": "#7C4DFF",
        "companies": [
            ("SPCE","Virgin Galactic","Space Tourism","🇺🇸"),
            ("RKLB","Rocket Lab","Small Satellite Launch","🇺🇸"),
            ("ASTS","AST SpaceMobile","Space Cellular","🇺🇸"),
            ("LUNR","Intuitive Machines","Lunar Landing","🇺🇸"),
            ("PL","Planet Labs","Earth Observation","🇺🇸"),
            ("SATL","Satellogic","Earth Imaging","🇺🇸"),
            ("MAXR","Maxar Technologies","Satellite Imagery","🇺🇸"),
            ("IRDM","Iridium","Satellite IoT/Phone","🇺🇸"),
            ("VSAT","Viasat","Satellite Internet","🇺🇸"),
            ("TSAT","Telesat","Satellite Broadband","🇨🇦"),
            ("SRAC","Momentus","In-Space Transport","🇺🇸"),
            ("LMT","Lockheed","Orion/Defense Space","🇺🇸"),
            ("NOC","Northrop","James Webb/B-21","🇺🇸"),
            ("BA","Boeing","Starliner/SLS","🇺🇸"),
        ],
    },
    {
        "id": "industrials",
        "icon": "🏗️",
        "label_en": "Industrials & Infrastructure",
        "label_ko": "산업재·인프라",
        "color": "#90A4AE",
        "companies": [
            ("CAT","Caterpillar","Heavy Equipment","🇺🇸"),
            ("DE","John Deere","Ag/Mining Equipment","🇺🇸"),
            ("HON","Honeywell","Industrial Automation","🇺🇸"),
            ("MMM","3M","Diversified Industrial","🇺🇸"),
            ("GE","GE Aerospace","Jet Engines","🇺🇸"),
            ("ETN","Eaton","Power Management","🇮🇪"),
            ("EMR","Emerson Electric","Automation","🇺🇸"),
            ("ROK","Rockwell Auto","Factory Automation","🇺🇸"),
            ("PWR","Quanta Services","Grid/EV Infra","🇺🇸"),
            ("PRIM","Primoris","Engineering/Infra","🇺🇸"),
            ("URI","United Rentals","Equipment Rental","🇺🇸"),
            ("AME","AMETEK","Electronic Instruments","🇺🇸"),
            ("FTV","Fortive","Industrial Tech","🇺🇸"),
            ("GNRC","Generac","Backup Power","🇺🇸"),
            ("CARR","Carrier Global","HVAC/Refrigeration","🇺🇸"),
            ("TT","Trane Technologies","HVAC Systems","🇮🇪"),
        ],
    },
    {
        "id": "realestate",
        "icon": "🏢",
        "label_en": "Real Estate & REITs",
        "label_ko": "부동산·리츠",
        "color": "#A1887F",
        "companies": [
            ("AMT","American Tower","Cell Tower REIT","🇺🇸"),
            ("PLD","Prologis","Industrial REIT","🇺🇸"),
            ("EQIX","Equinix","Data Center REIT","🇺🇸"),
            ("DLR","Digital Realty","Data Center REIT","🇺🇸"),
            ("CCI","Crown Castle","Cell Tower REIT","🇺🇸"),
            ("SPG","Simon Property","Mall REIT","🇺🇸"),
            ("O","Realty Income","Net Lease REIT","🇺🇸"),
            ("WELL","Welltower","Senior Housing REIT","🇺🇸"),
            ("AVB","AvalonBay","Apartment REIT","🇺🇸"),
            ("EQR","Equity Residential","Apartment REIT","🇺🇸"),
            ("PSA","Public Storage","Self-Storage REIT","🇺🇸"),
            ("VICI","VICI Properties","Gaming REIT","🇺🇸"),
            ("GLPI","Gaming & Leisure","Casino REIT","🇺🇸"),
            ("IRM","Iron Mountain","Document Storage REIT","🇺🇸"),
            ("SBAC","SBA Communications","Tower REIT","🇺🇸"),
        ],
    },
    {
        "id": "consumer",
        "icon": "🛒",
        "label_en": "Consumer & Retail",
        "label_ko": "소비재·유통",
        "color": "#EF9A9A",
        "companies": [
            ("AMZN","Amazon","eCommerce/AWS","🇺🇸"),
            ("WMT","Walmart","Big-Box Retail","🇺🇸"),
            ("COST","Costco","Membership Retail","🇺🇸"),
            ("TGT","Target","Discount Retail","🇺🇸"),
            ("HD","Home Depot","Home Improvement","🇺🇸"),
            ("LOW","Lowe's","Home Improvement","🇺🇸"),
            ("NKE","Nike","Sportswear","🇺🇸"),
            ("LULU","Lululemon","Athletic Apparel","🇨🇦"),
            ("MCD","McDonald's","Fast Food","🇺🇸"),
            ("SBUX","Starbucks","Coffee Chain","🇺🇸"),
            ("CMG","Chipotle","Fast Casual","🇺🇸"),
            ("PG","Procter & Gamble","Consumer Staples","🇺🇸"),
            ("KO","Coca-Cola","Beverages","🇺🇸"),
            ("PEP","PepsiCo","Beverages/Snacks","🇺🇸"),
            ("MDLZ","Mondelez","Snacks/Oreo/Cadbury","🇺🇸"),
            ("HSY","Hershey","Chocolate/Candy","🇺🇸"),
            ("TSLA","Tesla","EV Consumer","🇺🇸"),
        ],
    },
    {
        "id": "nuclear",
        "icon": "☢️",
        "label_en": "Nuclear Energy",
        "label_ko": "원자력 에너지",
        "color": "#FF6E40",
        "companies": [
            ("CEG","Constellation Energy","US Nuclear #1","🇺🇸"),
            ("VST","Vistra","Nuclear+Gas","🇺🇸"),
            ("ETR","Entergy","Nuclear South US","🇺🇸"),
            ("EXC","Exelon","Nuclear Largest Fleet","🇺🇸"),
            ("CCJ","Cameco","Uranium Mining","🇨🇦"),
            ("NXE","NexGen Energy","Uranium Athabasca","🇨🇦"),
            ("DNN","Denison Mines","Uranium","🇨🇦"),
            ("URA","Global X Uranium ETF","Uranium Basket","🇺🇸"),
            ("UUUU","Energy Fuels","US Uranium/Vanadium","🇺🇸"),
            ("UEC","Uranium Energy","US ISR Uranium","🇺🇸"),
            ("OKLO","Oklo","Micro-reactor SMR","🇺🇸"),
            ("SMR","NuScale Power","SMR Reactor Design","🇺🇸"),
            ("BWXT","BWX Technologies","Nuclear Components","🇺🇸"),
            ("GEV","GE Vernova","Nuclear+Grid Tech","🇺🇸"),
        ],
    },
]

# session state for selected sector
if "selected_sector" not in st.session_state:
    st.session_state.selected_sector = None

@st.cache_data(ttl=300)
def get_batch_prices(tickers: tuple) -> dict:
    result = {}
    for t in tickers:
        try:
            fi = yf.Ticker(t).fast_info
            price = getattr(fi, "last_price", 0) or 0
            prev  = getattr(fi, "previous_close", price) or price
            chg   = (price - prev) / prev * 100 if prev else 0
            result[t] = {"price": price, "chg": chg}
        except Exception:
            result[t] = {"price": 0, "chg": 0}
    return result

# ══════════════════ SECTOR EXPLORER (sidebar nav) ══════════════════
if st.session_state.sidebar_view == "sectors":
    lang_se = lang

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0D1B2A,#1A2744);border-radius:14px;
                padding:16px 24px;margin-bottom:20px;border:1px solid #2E3250;'>
        <div style='font-size:1.3rem;font-weight:800;color:#FFA500;'>
            {'🗂️ 전 산업 섹터 탐색기' if lang_se=='ko' else '🗂️ Global Sector Explorer'}
        </div>
        <div style='color:#8B9DB0;font-size:0.82rem;margin-top:4px;'>
            {'섹터 클릭 → 기업 목록 보기 → 기업 클릭 → 바로 차트·예측으로 이동'
             if lang_se=='ko' else
             'Click sector → view companies → click company → instant chart & analysis'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sector grid ──
    st.markdown(f"<div class='section-header'>{'섹터 선택' if lang_se=='ko' else 'Select Sector'}</div>",
                unsafe_allow_html=True)

    GRID_COLS = 5
    sector_rows = [SECTOR_DB[i:i+GRID_COLS] for i in range(0, len(SECTOR_DB), GRID_COLS)]
    for sector_row in sector_rows:
        row_cols = st.columns(GRID_COLS)
        for col_se, sector in zip(row_cols, sector_row):
            label = sector["label_ko"] if lang_se == "ko" else sector["label_en"]
            is_active = st.session_state.selected_sector == sector["id"]
            border = f"2px solid {sector['color']}" if is_active else f"1px solid {sector['color']}40"
            bg = f"{sector['color']}25" if is_active else "#1A1F35"
            with col_se:
                if st.button(
                    f"{sector['icon']} {label}",
                    key=f"sector_btn_{sector['id']}",
                    use_container_width=True,
                ):
                    if st.session_state.selected_sector == sector["id"]:
                        st.session_state.selected_sector = None
                    else:
                        st.session_state.selected_sector = sector["id"]
                    st.rerun()
                n_cos = len(sector["companies"])
                st.markdown(
                    f"<div style='text-align:center;color:{sector['color']};font-size:0.7rem;"
                    f"margin-top:-8px;margin-bottom:6px;'>{n_cos} {'기업' if lang_se=='ko' else 'companies'}</div>",
                    unsafe_allow_html=True,
                )

    # ── Company list for selected sector ──
    if st.session_state.selected_sector:
        sector_data = next((s for s in SECTOR_DB if s["id"] == st.session_state.selected_sector), None)
        if sector_data:
            st.markdown("<br>", unsafe_allow_html=True)
            label_s = sector_data["label_ko"] if lang_se == "ko" else sector_data["label_en"]
            color_s = sector_data["color"]
            st.markdown(f"""
            <div style='border-left:4px solid {color_s};padding:10px 18px;
                        background:{color_s}15;border-radius:0 10px 10px 0;margin-bottom:18px;'>
                <span style='font-size:1.1rem;font-weight:700;color:{color_s};'>
                    {sector_data['icon']} {label_s}
                </span>
                <span style='color:#8B9DB0;font-size:0.82rem;margin-left:12px;'>
                    {len(sector_data['companies'])} {'기업' if lang_se=='ko' else 'companies'} — {'클릭하면 분석 화면으로 이동' if lang_se=='ko' else 'Click to analyze'}
                </span>
            </div>
            """, unsafe_allow_html=True)

            # Fetch live prices
            all_tickers = tuple(c[0] for c in sector_data["companies"]
                                if not c[0].endswith("=F") and "." not in c[0]
                                and c[0] not in ("CATL","X","VOW","BMW","LG","CRCL","MDIA","BRCM"))
            with st.spinner("Loading prices..." if lang_se == "en" else "시세 로딩 중..."):
                prices_se = get_batch_prices(all_tickers)

            # Company cards
            CARD_COLS = 4
            companies = sector_data["companies"]
            for row_s in range(0, len(companies), CARD_COLS):
                row_cos = companies[row_s:row_s+CARD_COLS]
                card_cols = st.columns(CARD_COLS)
                for card_col, (sym, name, role, flag) in zip(card_cols, row_cos):
                    pd_s    = prices_se.get(sym, {})
                    price_s = pd_s.get("price", 0)
                    chg_s   = pd_s.get("chg", 0)
                    chg_c   = "#FF4040" if chg_s >= 0 else "#4488FF"
                    arrow_s = "▲" if chg_s >= 0 else "▼"
                    price_display = f"${price_s:,.2f}" if price_s else "—"

                    with card_col:
                        if st.button(
                            f"📈 {name}  {price_display}  {arrow_s}{abs(chg_s):.1f}%",
                            key=f"co_btn_{sector_data['id']}_{sym}",
                            use_container_width=True,
                            type="primary",
                            help=f"{sym} | {role} | {flag} | 클릭하면 분석으로 이동",
                        ):
                            st.session_state.ticker = sym
                            st.session_state.sidebar_view = None
                            st.rerun()
                        st.markdown(f"""
                        <div style='background:#0F1527;border:1px solid {color_s}50;
                                    border-radius:0 0 10px 10px;padding:6px 12px 10px 12px;margin:-8px 0 12px 0;'>
                            <div style='display:flex;justify-content:space-between;'>
                                <span style='color:{color_s};font-size:0.76rem;font-weight:700;'>{sym}</span>
                                <span style='font-size:0.78rem;'>{flag}</span>
                            </div>
                            <div style='color:#8B9DB0;font-size:0.7rem;margin-top:2px;'>{role}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Mini performance chart for top stocks in sector
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='section-header'>{'섹터 주요 종목 1년 수익률 비교' if lang_se=='ko' else 'Sector Top Stocks — 1Y Return Comparison'}</div>",
                        unsafe_allow_html=True)
            top_tickers = [c[0] for c in companies[:8]
                           if c[0] not in ("CL=F","NG=F","BTC-USD","ETH-USD","SOL-USD","BNB-USD","CATL","VOW","BMW","LG")]
            if top_tickers:
                with st.spinner("Loading comparison..." if lang_se=="en" else "비교 데이터 로딩 중..."):
                    returns_data = {}
                    for t_cmp in top_tickers[:8]:
                        try:
                            df_cmp = yf.download(t_cmp, period="1y", progress=False, auto_adjust=True)
                            if not df_cmp.empty:
                                cl = df_cmp["Close"]
                                if cl.ndim == 2: cl = cl.iloc[:,0]
                                cl = cl.astype(float).dropna()
                                if len(cl) > 5:
                                    ret = (cl.iloc[-1] / cl.iloc[0] - 1) * 100
                                    returns_data[t_cmp] = round(float(ret), 2)
                        except Exception:
                            pass

                if returns_data:
                    sorted_r = sorted(returns_data.items(), key=lambda x: x[1], reverse=True)
                    names_r  = [x[0] for x in sorted_r]
                    vals_r   = [x[1] for x in sorted_r]
                    colors_r = [color_s if v >= 0 else "#4488FF" for v in vals_r]

                    fig_sr = go.Figure(go.Bar(
                        x=names_r, y=vals_r,
                        marker_color=colors_r,
                        text=[f"{v:+.1f}%" for v in vals_r],
                        textposition="outside",
                    ))
                    fig_sr.add_hline(y=0, line_color="#555", line_width=1)
                    fig_sr.update_layout(
                        template="plotly_dark", height=320,
                        title=f"{'1년 수익률 비교 (%)' if lang_se=='ko' else '1-Year Return Comparison (%)'}",
                        margin=dict(l=0,r=0,t=40,b=0),
                        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                        yaxis_title="Return %" if lang_se=="en" else "수익률 %",
                    )
                    st.plotly_chart(fig_sr, use_container_width=True)
    else:
        st.markdown(f"""
        <div style='text-align:center;padding:40px 20px;color:#4A5568;'>
            <div style='font-size:2.5rem;margin-bottom:12px;'>👆</div>
            <div style='font-size:1rem;'>
                {'위의 섹터 버튼을 클릭하면 해당 섹터 기업 목록이 표시됩니다.'
                 if lang_se=='ko' else
                 'Click any sector button above to see the companies in that sector.'}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════ TAB 8: INVESTMENT ANALYSIS ══════════════════
if _show_tabs:
    with tabs[7]:
        lang_inv = st.session_state.lang
        st.markdown(f"<div class='section-header'>{'📈 투자 분석 대시보드' if lang_inv=='ko' else '📈 Investment Analysis Dashboard'}</div>", unsafe_allow_html=True)
        st.markdown(f"<small style='color:#8B9DB0;'>{'실시간 yFinance 데이터 기반 | 투자 결정은 전문가와 상담하세요' if lang_inv=='ko' else 'Real-time yFinance data | Consult a professional before investing'}</small>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Fetch fresh financials
        inv_info = info  # reuse already loaded info dict

        def safe(val, fmt=None, suffix=""):
            if val is None or val == "N/A" or (isinstance(val, float) and (val != val)):
                return "N/A"
            try:
                if fmt == "pct":
                    return f"{float(val)*100:.1f}%"
                elif fmt == "x":
                    return f"{float(val):.2f}x"
                elif fmt == "f2":
                    return f"{float(val):.2f}"
                elif fmt == "big":
                    v = float(val)
                    if abs(v) >= 1e12: return f"${v/1e12:.2f}T"
                    if abs(v) >= 1e9:  return f"${v/1e9:.1f}B"
                    if abs(v) >= 1e6:  return f"${v/1e6:.0f}M"
                    return f"${v:,.0f}"
                else:
                    return str(val)
            except Exception:
                return "N/A"

        # ── Section 1: Valuation ──
        st.markdown(f"### {gl('PER', '💰 가치평가 (Valuation)' if lang_inv=='ko' else '💰 Valuation')}", unsafe_allow_html=True)

        per  = inv_info.get("trailingPE") or inv_info.get("forwardPE")
        pbr  = inv_info.get("priceToBook")
        peg  = inv_info.get("pegRatio")
        ps   = inv_info.get("priceToSalesTrailing12Months")
        ev_ebitda = inv_info.get("enterpriseToEbitda")

        val_metrics = [
            ("PER", safe(per, "f2"), "PER", "낮을수록 저평가" if lang_inv=="ko" else "Lower = undervalued"),
            ("PBR", safe(pbr, "f2"), "PBR", "1 미만 = 자산 대비 저평가" if lang_inv=="ko" else "< 1 = below book value"),
            ("PEG", safe(peg, "f2"), "PEG", "1 미만 = 성장 대비 저평가" if lang_inv=="ko" else "< 1 = undervalued vs growth"),
            ("P/S", safe(ps, "f2"), "P/S Ratio", "낮을수록 매출 대비 저평가" if lang_inv=="ko" else "Lower = cheaper vs sales"),
            ("EV/EBITDA", safe(ev_ebitda, "f2"), "EV/EBITDA", "10 미만 = 저평가 기준" if lang_inv=="ko" else "Below 10 = generally cheap"),
        ]

        v_cols = st.columns(5)
        for col, (name, val, gl_key, hint) in zip(v_cols, val_metrics):
            try:
                fval = float(val.replace("x","").replace("%","")) if val != "N/A" else None
            except Exception:
                fval = None
            # Color coding
            color = "#8B9DB0"
            if name == "PER" and fval is not None:
                color = "#00D4AA" if fval < 15 else "#FFA500" if fval < 30 else "#FF4B4B"
            elif name == "PBR" and fval is not None:
                color = "#00D4AA" if fval < 1 else "#FFA500" if fval < 3 else "#FF4B4B"
            elif name == "PEG" and fval is not None:
                color = "#00D4AA" if fval < 1 else "#FFA500" if fval < 2 else "#FF4B4B"
            elif name == "EV/EBITDA" and fval is not None:
                color = "#00D4AA" if fval < 10 else "#FFA500" if fval < 20 else "#FF4B4B"
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{gl(gl_key, name)}</div>
                <div style='font-size:1.4rem;font-weight:700;color:{color};'>{val}</div>
                <div style='font-size:0.72rem;color:#6B7A8D;margin-top:4px;'>{hint}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Section 2: Profitability ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### {'🏆 수익성 (Profitability)' if lang_inv=='ko' else '🏆 Profitability'}", unsafe_allow_html=True)

        roe  = inv_info.get("returnOnEquity")
        roa  = inv_info.get("returnOnAssets")
        op_margin  = inv_info.get("operatingMargins")
        net_margin = inv_info.get("profitMargins")
        gross_margin = inv_info.get("grossMargins")

        # ROIC approximation: Net Income / (Total Assets - Current Liabilities)
        net_income = inv_info.get("netIncomeToCommon", 0) or 0
        total_assets = inv_info.get("totalAssets", 0) or 0
        curr_liab = inv_info.get("totalCurrentLiabilities", 0) or 0
        roic_val = (net_income / (total_assets - curr_liab)) if (total_assets - curr_liab) > 0 else None

        prof_metrics = [
            ("ROE", safe(roe, "pct"), "ROE", "15%↑ 우수" if lang_inv=="ko" else "15%+ excellent"),
            ("ROA", safe(roa, "pct"), "ROA", "5%↑ 양호" if lang_inv=="ko" else "5%+ good"),
            ("ROIC", safe(roic_val, "pct") if roic_val else "N/A", "ROIC", "WACC 초과 시 가치창출" if lang_inv=="ko" else "Above WACC = value creation"),
            ("영업이익률" if lang_inv=="ko" else "Op. Margin", safe(op_margin, "pct"), "Operating Margin", "높을수록 경쟁우위" if lang_inv=="ko" else "Higher = stronger moat"),
            ("순이익률" if lang_inv=="ko" else "Net Margin", safe(net_margin, "pct"), "Net Margin", "순수 수익성" if lang_inv=="ko" else "Final profitability"),
        ]

        p_cols = st.columns(5)
        for col, (name, val, gl_key, hint) in zip(p_cols, prof_metrics):
            try:
                fval = float(val.replace("%","")) if val != "N/A" else None
            except Exception:
                fval = None
            color = "#8B9DB0"
            if fval is not None:
                color = "#00D4AA" if fval >= 15 else "#FFA500" if fval >= 5 else "#FF4B4B"
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{gl(gl_key, name)}</div>
                <div style='font-size:1.4rem;font-weight:700;color:{color};'>{val}</div>
                <div style='font-size:0.72rem;color:#6B7A8D;margin-top:4px;'>{hint}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Section 3: Growth ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### {'📈 성장성 (Growth)' if lang_inv=='ko' else '📈 Growth'}", unsafe_allow_html=True)

        rev_growth   = inv_info.get("revenueGrowth")
        earn_growth  = inv_info.get("earningsGrowth")
        eps_trail    = inv_info.get("trailingEps")
        eps_fwd      = inv_info.get("forwardEps")
        eps_growth   = ((eps_fwd - eps_trail) / abs(eps_trail)) if eps_trail and eps_fwd and eps_trail != 0 else None
        revenue_ttm  = inv_info.get("totalRevenue")
        analyst_tgt  = inv_info.get("targetMeanPrice")
        curr_pr      = inv_info.get("currentPrice") or inv_info.get("regularMarketPrice") or current_price
        upside       = ((analyst_tgt - curr_pr) / curr_pr) if analyst_tgt and curr_pr else None

        growth_metrics = [
            ("매출 성장률" if lang_inv=="ko" else "Revenue Growth", safe(rev_growth, "pct"), "Revenue Growth", "YoY 성장" if lang_inv=="ko" else "YoY growth"),
            ("순이익 성장률" if lang_inv=="ko" else "Earnings Growth", safe(earn_growth, "pct"), "EPS Growth", "YoY 이익 성장" if lang_inv=="ko" else "YoY earnings growth"),
            ("EPS (TTM)", safe(eps_trail, "f2"), "EPS", "주당순이익" if lang_inv=="ko" else "Trailing 12M EPS"),
            ("EPS (선행)" if lang_inv=="ko" else "EPS (Fwd)", safe(eps_fwd, "f2"), "EPS Growth", "예상 주당순이익" if lang_inv=="ko" else "Forward EPS estimate"),
            ("애널리스트 목표가" if lang_inv=="ko" else "Analyst Target", f"${analyst_tgt:,.2f}" if analyst_tgt else "N/A", "EPS Growth",
             f"상승 여력 {upside*100:.1f}%" if upside and upside>=0 and lang_inv=="ko"
             else f"Upside {upside*100:.1f}%" if upside and upside>=0
             else f"하락 여지 {abs(upside)*100:.1f}%" if upside and lang_inv=="ko"
             else f"Downside {abs(upside)*100:.1f}%" if upside else "N/A"),
        ]

        g_cols = st.columns(5)
        for col, (name, val, gl_key, hint) in zip(g_cols, growth_metrics):
            try:
                fval = float(val.replace("%","").replace("$","").replace(",","")) if val not in ("N/A","") else None
            except Exception:
                fval = None
            color = "#8B9DB0"
            if "성장" in name or "Growth" in name:
                if fval is not None:
                    color = "#00D4AA" if fval >= 10 else "#FFA500" if fval >= 0 else "#FF4B4B"
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{gl(gl_key, name)}</div>
                <div style='font-size:1.3rem;font-weight:700;color:{color};'>{val}</div>
                <div style='font-size:0.72rem;color:#6B7A8D;margin-top:4px;'>{hint}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Section 4: Stability ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### {'🛡️ 안정성 (Stability)' if lang_inv=='ko' else '🛡️ Stability'}", unsafe_allow_html=True)

        total_debt  = inv_info.get("totalDebt", 0) or 0
        eq          = inv_info.get("totalStockholderEquity") or inv_info.get("bookValue", 0) or 0
        de_ratio    = (total_debt / (eq * inv_info.get("sharesOutstanding", 1))) if eq and inv_info.get("sharesOutstanding") else inv_info.get("debtToEquity")
        curr_ratio  = inv_info.get("currentRatio")
        quick_ratio = inv_info.get("quickRatio")
        ebit        = inv_info.get("ebit", 0) or 0
        int_exp     = inv_info.get("interestExpense", 0) or 0
        int_cov     = abs(ebit / int_exp) if int_exp and int_exp != 0 and ebit else None
        beta_val    = inv_info.get("beta")

        stab_metrics = [
            ("부채비율" if lang_inv=="ko" else "Debt/Equity", safe(de_ratio, "f2"), "Debt/Equity", "낮을수록 안전" if lang_inv=="ko" else "Lower = safer"),
            ("유동비율" if lang_inv=="ko" else "Current Ratio", safe(curr_ratio, "f2"), "Current Ratio", "1.5↑ 안전" if lang_inv=="ko" else "1.5+ healthy"),
            ("당좌비율" if lang_inv=="ko" else "Quick Ratio", safe(quick_ratio, "f2"), "Current Ratio", "1.0↑ 양호" if lang_inv=="ko" else "1.0+ good"),
            ("이자보상배율" if lang_inv=="ko" else "Interest Coverage", safe(int_cov, "f2") if int_cov else "N/A", "Interest Coverage", "3↑ 안전" if lang_inv=="ko" else "3+ safe"),
            ("베타" if lang_inv=="ko" else "Beta", safe(beta_val, "f2"), "Beta vs S&P 500", "1 초과=고변동성" if lang_inv=="ko" else ">1 = more volatile"),
        ]

        s_cols = st.columns(5)
        for col, (name, val, gl_key, hint) in zip(s_cols, stab_metrics):
            try:
                fval = float(val.replace("%","")) if val != "N/A" else None
            except Exception:
                fval = None
            color = "#8B9DB0"
            if "부채" in name or "Debt" in name:
                if fval is not None:
                    color = "#00D4AA" if fval < 1 else "#FFA500" if fval < 2 else "#FF4B4B"
            elif "유동" in name or "Current" in name or "Quick" in name:
                if fval is not None:
                    color = "#00D4AA" if fval >= 1.5 else "#FFA500" if fval >= 1 else "#FF4B4B"
            elif "이자" in name or "Interest" in name:
                if fval is not None:
                    color = "#00D4AA" if fval >= 3 else "#FFA500" if fval >= 1.5 else "#FF4B4B"
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{gl(gl_key, name)}</div>
                <div style='font-size:1.4rem;font-weight:700;color:{color};'>{val}</div>
                <div style='font-size:0.72rem;color:#6B7A8D;margin-top:4px;'>{hint}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Section 5: Cash Flow ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### {'💵 현금흐름 (Cash Flow)' if lang_inv=='ko' else '💵 Cash Flow'}", unsafe_allow_html=True)

        op_cf   = inv_info.get("operatingCashflow") or inv_info.get("totalCashFromOperatingActivities")
        capex   = inv_info.get("capitalExpenditures", 0) or 0
        fcf_val = (op_cf + capex) if op_cf else None  # capex is usually negative in yf
        mkt_cap_v = inv_info.get("marketCap", 0) or 0
        fcf_yield_v = (fcf_val / mkt_cap_v) if fcf_val and mkt_cap_v else None
        div_yield   = inv_info.get("dividendYield")
        payout_r    = inv_info.get("payoutRatio")
        free_cf     = inv_info.get("freeCashflow")
        if free_cf:
            fcf_val = free_cf  # prefer direct FCF if available

        cf_metrics = [
            ("영업현금흐름" if lang_inv=="ko" else "Operating CF", safe(op_cf, "big"), "Operating CF", "실제 현금 창출력" if lang_inv=="ko" else "Real cash generation"),
            ("FCF", safe(fcf_val, "big") if fcf_val else "N/A", "FCF", "주주 환원 여력" if lang_inv=="ko" else "Available for shareholders"),
            ("FCF 수익률" if lang_inv=="ko" else "FCF Yield", safe(fcf_yield_v, "pct") if fcf_yield_v else "N/A", "FCF Yield", "높을수록 저평가" if lang_inv=="ko" else "Higher = undervalued"),
            ("배당수익률" if lang_inv=="ko" else "Div. Yield", safe(div_yield, "pct") if div_yield else "무배당" if lang_inv=="ko" else "No dividend", "FCF", "현금 배당 비율" if lang_inv=="ko" else "Cash return to shareholders"),
            ("배당성향" if lang_inv=="ko" else "Payout Ratio", safe(payout_r, "pct") if payout_r else "N/A", "FCF", "순이익 중 배당 비중" if lang_inv=="ko" else "% of earnings paid as dividend"),
        ]

        c_cols = st.columns(5)
        for col, (name, val, gl_key, hint) in zip(c_cols, cf_metrics):
            color = "#FFA500"
            if "무배당" in str(val) or "No dividend" in str(val):
                color = "#8B9DB0"
            col.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{gl(gl_key, name)}</div>
                <div style='font-size:1.2rem;font-weight:700;color:{color};'>{val}</div>
                <div style='font-size:0.72rem;color:#6B7A8D;margin-top:4px;'>{hint}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Section 6: Academic Models ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### {'🎓 학술 투자 모델 분석' if lang_inv=='ko' else '🎓 Academic Investment Models'}", unsafe_allow_html=True)

        model_col1, model_col2 = st.columns(2)

        # Graham Number
        with model_col1:
            eps_v = inv_info.get("trailingEps", 0) or 0
            bps_v = inv_info.get("bookValue", 0) or 0
            graham = None
            if eps_v > 0 and bps_v > 0:
                graham = (22.5 * eps_v * bps_v) ** 0.5
            graham_str = f"${graham:,.2f}" if graham else "N/A (음수 EPS/BPS)"
            upside_g = ((graham - curr_pr) / curr_pr * 100) if graham and curr_pr else None
            g_color = "#00D4AA" if upside_g and upside_g > 0 else "#FF4B4B" if upside_g else "#8B9DB0"
            graham_interpret = (
                f"현재가 대비 {'저평가' if upside_g and upside_g>0 else '고평가'} {abs(upside_g):.1f}%" if upside_g else
                ("EPS 또는 BPS가 음수여서 계산 불가" if lang_inv=="ko" else "Cannot compute: negative EPS or BPS")
            )
            st.markdown(f"""
            <div class='geo-card'>
                <div style='font-weight:700;color:#FFD700;font-size:1rem;'>{gl("Graham Number", "📐 그레이엄 넘버 (Graham Number)")}</div>
                <div style='font-size:0.82rem;color:#B0BEC5;margin:6px 0;'>
                    {"벤저민 그레이엄의 안전마진 계산: √(22.5 × EPS × BPS)" if lang_inv=="ko" else "Benjamin Graham's intrinsic value: √(22.5 × EPS × BPS)"}
                </div>
                <div style='font-size:1.6rem;font-weight:800;color:{g_color};'>{graham_str}</div>
                <div style='font-size:0.85rem;color:#B0BEC5;margin-top:6px;'>{graham_interpret}</div>
                <div style='font-size:0.78rem;color:#6B7A8D;margin-top:4px;'>EPS: {safe(eps_v,"f2")} | BPS: {safe(bps_v,"f2")}</div>
            </div>
            """, unsafe_allow_html=True)

        # Piotroski F-Score
        with model_col2:
            # Calculate simplified Piotroski F-Score (9 criteria)
            pio_score = 0
            pio_details = []
            roa_v = inv_info.get("returnOnAssets", 0) or 0
            op_cf_v2 = (op_cf or 0)
            # Profitability (4 signals)
            if roa_v > 0: pio_score += 1; pio_details.append(("ROA > 0", True))
            else: pio_details.append(("ROA > 0", False))
            if op_cf_v2 > 0: pio_score += 1; pio_details.append(("영업현금흐름 > 0" if lang_inv=="ko" else "Op. CF > 0", True))
            else: pio_details.append(("영업현금흐름 > 0" if lang_inv=="ko" else "Op. CF > 0", False))
            if earn_growth and earn_growth > 0: pio_score += 1; pio_details.append(("이익 증가" if lang_inv=="ko" else "Earnings↑", True))
            else: pio_details.append(("이익 증가" if lang_inv=="ko" else "Earnings↑", False))
            if op_cf_v2 > 0 and roa_v > 0 and op_cf_v2 > net_income: pio_score += 1; pio_details.append(("Accruals 건전" if lang_inv=="ko" else "Accruals OK", True))
            else: pio_details.append(("Accruals 건전" if lang_inv=="ko" else "Accruals OK", False))
            # Leverage / Liquidity (3 signals)
            de_num = inv_info.get("debtToEquity", 100) or 100
            if de_num < 100: pio_score += 1; pio_details.append(("부채비율 감소" if lang_inv=="ko" else "Leverage↓", True))
            else: pio_details.append(("부채비율 감소" if lang_inv=="ko" else "Leverage↓", False))
            cr_v = inv_info.get("currentRatio", 0) or 0
            if cr_v > 1.5: pio_score += 1; pio_details.append(("유동비율 양호" if lang_inv=="ko" else "Liquidity OK", True))
            else: pio_details.append(("유동비율 양호" if lang_inv=="ko" else "Liquidity OK", False))
            pio_details.append(("주식희석 없음" if lang_inv=="ko" else "No dilution", None))  # simplified
            # Operating Efficiency (2 signals)
            gm = inv_info.get("grossMargins", 0) or 0
            if gm > 0.3: pio_score += 1; pio_details.append(("매출총이익률 양호" if lang_inv=="ko" else "Gross Margin OK", True))
            else: pio_details.append(("매출총이익률 양호" if lang_inv=="ko" else "Gross Margin OK", False))
            at = inv_info.get("assetTurnover") or (inv_info.get("totalRevenue", 0) / total_assets if total_assets else None)
            if at and at > 0.5: pio_score += 1; pio_details.append(("자산회전율 양호" if lang_inv=="ko" else "Asset Turnover OK", True))
            else: pio_details.append(("자산회전율 양호" if lang_inv=="ko" else "Asset Turnover OK", False))

            pio_color = "#00D4AA" if pio_score >= 7 else "#FFA500" if pio_score >= 4 else "#FF4B4B"
            pio_label = ("강한 매수 신호" if pio_score >= 7 else "중립" if pio_score >= 4 else "약세 신호") if lang_inv=="ko" else ("Strong Buy Signal" if pio_score >= 7 else "Neutral" if pio_score >= 4 else "Weak Signal")
            details_html = " ".join([
                f"<span style='color:{'#00D4AA' if ok else '#FF4B4B' if ok is not None else '#8B9DB0'};font-size:0.72rem;'>{'✓' if ok else '✗' if ok is not None else '?'} {d}</span>"
                for d, ok in pio_details
            ])
            st.markdown(f"""
            <div class='geo-card'>
                <div style='font-weight:700;color:#AB63FA;font-size:1rem;'>{gl("Piotroski F-Score", "📊 피오트로스키 F-스코어")}</div>
                <div style='font-size:0.82rem;color:#B0BEC5;margin:6px 0;'>
                    {"재무 건전성 9개 항목 평가 (0~9점)" if lang_inv=="ko" else "9-point financial health scoring (0–9)"}
                </div>
                <div style='font-size:2rem;font-weight:800;color:{pio_color};'>{pio_score} <span style='font-size:1rem;'>/9</span></div>
                <div style='font-size:0.88rem;color:{pio_color};font-weight:600;'>{pio_label}</div>
                <div style='margin-top:8px;line-height:1.8;'>{details_html}</div>
            </div>
            """, unsafe_allow_html=True)

        # Row 2: Altman Z-Score + Magic Formula
        az_col, mf_col = st.columns(2)

        with az_col:
            # Altman Z-Score (simplified for large public companies)
            shares_out = inv_info.get("sharesOutstanding", 0) or 0
            try:
                mkt_cap_z = curr_pr * shares_out if curr_pr and shares_out else (mkt_cap_v or 0)
                ta = float(inv_info.get("totalAssets", 1) or 1)
                wc = float((inv_info.get("totalCurrentAssets", 0) or 0) - (inv_info.get("totalCurrentLiabilities", 0) or 0))
                re = float(inv_info.get("retainedEarnings", 0) or 0)
                ebit_z = float(inv_info.get("ebit", 0) or 0)
                td = float(inv_info.get("totalDebt", 0) or 0)
                rev_z = float(inv_info.get("totalRevenue", 0) or 0)
                if ta > 0 and td > 0:
                    X1 = wc / ta
                    X2 = re / ta
                    X3 = ebit_z / ta
                    X4 = mkt_cap_z / td
                    X5 = rev_z / ta
                    z_score = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
                    z_color = "#00D4AA" if z_score > 3 else "#FFA500" if z_score > 1.8 else "#FF4B4B"
                    z_label = ("안전 구간" if z_score > 3 else "회색 지대" if z_score > 1.8 else "위험 구간") if lang_inv=="ko" else ("Safe Zone" if z_score > 3 else "Grey Zone" if z_score > 1.8 else "Distress Zone")
                    z_str = f"{z_score:.2f}"
                else:
                    z_str, z_color, z_label = "N/A", "#8B9DB0", "데이터 부족" if lang_inv=="ko" else "Insufficient data"
            except Exception:
                z_str, z_color, z_label = "N/A", "#8B9DB0", "계산 오류" if lang_inv=="ko" else "Calc error"

            az_col.markdown(f"""
            <div class='geo-card'>
                <div style='font-weight:700;color:#FF8C00;font-size:1rem;'>{gl("Altman Z-Score", "⚠️ 알트만 Z-스코어")}</div>
                <div style='font-size:0.82rem;color:#B0BEC5;margin:6px 0;'>
                    {"부도 위험 예측 모델 | 3↑ 안전, 1.8~3 회색지대, 1.8↓ 위험" if lang_inv=="ko" else "Bankruptcy prediction model | >3 safe, 1.8-3 grey, <1.8 distress"}
                </div>
                <div style='font-size:2rem;font-weight:800;color:{z_color};'>{z_str}</div>
                <div style='font-size:0.9rem;color:{z_color};font-weight:600;'>{z_label}</div>
                <div style='font-size:0.78rem;color:#6B7A8D;margin-top:6px;'>
                    {"공식: 1.2×유동자본/자산 + 1.4×유보이익/자산 + 3.3×EBIT/자산 + 0.6×시총/부채 + 매출/자산" if lang_inv=="ko"
                     else "Formula: 1.2×WC/TA + 1.4×RE/TA + 3.3×EBIT/TA + 0.6×MktCap/Debt + Rev/TA"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with mf_col:
            # Magic Formula (Joel Greenblatt): high ROIC + low EV/EBIT
            try:
                ev = inv_info.get("enterpriseValue", 0) or 0
                ebit_mf = inv_info.get("ebit", 0) or 0
                ev_ebit = ev / ebit_mf if ebit_mf and ebit_mf > 0 and ev > 0 else None
                roic_mf = roic_val
                if ev_ebit and roic_mf:
                    # Simplified rank: lower ev_ebit + higher roic = better
                    mf_score_str = f"EV/EBIT: {ev_ebit:.1f}x | ROIC: {roic_mf*100:.1f}%"
                    mf_good = ev_ebit < 15 and roic_mf > 0.15
                    mf_ok = ev_ebit < 25 and roic_mf > 0.08
                    mf_color = "#00D4AA" if mf_good else "#FFA500" if mf_ok else "#FF4B4B"
                    mf_label = ("매력적" if mf_good else "보통" if mf_ok else "비매력적") if lang_inv=="ko" else ("Attractive" if mf_good else "Neutral" if mf_ok else "Unattractive")
                else:
                    mf_score_str = "N/A"
                    mf_color = "#8B9DB0"
                    mf_label = "데이터 부족" if lang_inv=="ko" else "Insufficient data"
            except Exception:
                mf_score_str = "N/A"
                mf_color = "#8B9DB0"
                mf_label = "계산 오류" if lang_inv=="ko" else "Calc error"

            mf_col.markdown(f"""
            <div class='geo-card'>
                <div style='font-weight:700;color:#64B5F6;font-size:1rem;'>{gl("Magic Formula", "✨ 매직 포뮬러 (그린블라트)")}</div>
                <div style='font-size:0.82rem;color:#B0BEC5;margin:6px 0;'>
                    {"높은 ROIC + 낮은 EV/EBIT = 저평가 고수익 기업 선별" if lang_inv=="ko" else "High ROIC + Low EV/EBIT = undervalued high-quality company"}
                </div>
                <div style='font-size:1.3rem;font-weight:700;color:{mf_color};'>{mf_score_str}</div>
                <div style='font-size:0.9rem;color:{mf_color};font-weight:600;margin-top:4px;'>{mf_label}</div>
                <div style='font-size:0.78rem;color:#6B7A8D;margin-top:6px;'>
                    {"EV/EBIT 15↓ + ROIC 15%↑ = 강한 매수 신호" if lang_inv=="ko" else "EV/EBIT < 15 + ROIC > 15% = strong buy signal"}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Section 7: DCF Simplified ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### {gl('DCF', '🔢 DCF 간이 내재가치 분석' if lang_inv=='ko' else '🔢 Simplified DCF Intrinsic Value')}", unsafe_allow_html=True)

        dcf_col1, dcf_col2 = st.columns([2, 1])
        with dcf_col1:
            try:
                fcf_dcf = fcf_val or (op_cf + capex if op_cf else None)
                if fcf_dcf and fcf_dcf > 0 and shares_out > 0:
                    # DCF with 3-stage growth
                    wacc = 0.09  # typical 9% WACC
                    g1 = min(max(float(rev_growth or 0.05), 0.01), 0.30)  # Stage 1: current growth (capped)
                    g2 = g1 * 0.5  # Stage 2: half of current growth
                    g3 = 0.025    # Terminal growth rate

                    pv = 0
                    cf = fcf_dcf
                    for yr in range(1, 6):   # Stage 1: 5 years
                        cf *= (1 + g1)
                        pv += cf / (1 + wacc)**yr
                    for yr in range(6, 11):  # Stage 2: 5 years
                        cf *= (1 + g2)
                        pv += cf / (1 + wacc)**yr
                    terminal = cf * (1 + g3) / (wacc - g3)
                    pv += terminal / (1 + wacc)**10

                    dcf_per_share = pv / shares_out
                    margin_of_safety = (dcf_per_share - curr_pr) / curr_pr * 100 if curr_pr else 0
                    dcf_color = "#00D4AA" if margin_of_safety > 20 else "#FFA500" if margin_of_safety > -20 else "#FF4B4B"
                    dcf_signal = ("매수 유망 (안전마진 확보)" if margin_of_safety > 20 else "적정 가격" if margin_of_safety > -20 else "고평가 주의") if lang_inv=="ko" else ("Attractive (margin of safety)" if margin_of_safety > 20 else "Fairly valued" if margin_of_safety > -20 else "Potentially overvalued")

                    st.markdown(f"""
                    <div class='prediction-card' style='text-align:left;'>
                        <div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;'>
                            <div>
                                <div style='color:#8B9DB0;font-size:0.85rem;margin-bottom:4px;'>{"DCF 내재가치 (주당)" if lang_inv=="ko" else "DCF Intrinsic Value (per share)"}</div>
                                <div style='font-size:2rem;font-weight:800;color:{dcf_color};'>${dcf_per_share:,.2f}</div>
                                <div style='font-size:0.9rem;color:{dcf_color};margin-top:4px;'>{dcf_signal}</div>
                            </div>
                            <div>
                                <div style='color:#8B9DB0;font-size:0.85rem;'>{"현재가" if lang_inv=="ko" else "Current Price"}</div>
                                <div style='font-size:1.4rem;font-weight:700;color:#FFFFFF;'>${curr_pr:,.2f}</div>
                                <div style='font-size:0.85rem;color:{dcf_color};'>{margin_of_safety:+.1f}% {"괴리" if lang_inv=="ko" else "deviation"}</div>
                            </div>
                            <div>
                                <div style='color:#8B9DB0;font-size:0.8rem;'>{"가정 (WACC / 성장률1 / 성장률2 / 영구)" if lang_inv=="ko" else "Assumptions (WACC / G1 / G2 / Terminal)"}</div>
                                <div style='font-size:0.85rem;color:#B0BEC5;'>{wacc*100:.1f}% / {g1*100:.1f}% / {g2*100:.1f}% / {g3*100:.1f}%</div>
                                <div style='font-size:0.78rem;color:#6B7A8D;margin-top:4px;'>{"※ 단순화된 추정치. 실제 투자 시 전문가 분석 필요" if lang_inv=="ko" else "⚠ Simplified estimate. Consult expert before investing"}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("FCF가 0 이하이거나 데이터 부족으로 DCF 계산이 불가합니다." if lang_inv=="ko" else "Cannot compute DCF: FCF ≤ 0 or insufficient data.")
            except Exception as e:
                st.info(f"DCF 계산 중 오류: {e}" if lang_inv=="ko" else f"DCF calculation error: {e}")

        with dcf_col2:
            st.markdown(f"""
            <div class='summary-box' style='height:100%;'>
                <div style='font-weight:700;color:#FFA500;margin-bottom:8px;'>{gl("DCF", "DCF 모델이란?")}</div>
                <div style='font-size:0.82rem;color:#B0BEC5;line-height:1.7;'>
                    {"• 미래 잉여현금흐름을 현재 가치로 할인<br>• 3단계 성장 모델 적용<br>• 1~5년: 현재 성장률 유지<br>• 6~10년: 절반으로 감속<br>• 10년 이후: 영구성장률 2.5%<br>• WACC 9% 가정 (시장 평균)<br><br><span style='color:#FF8C00;'>⚠ 단순화된 모델로 참고용만 사용" if lang_inv=="ko" else
                    "• Discounts future free cash flows<br>• 3-stage growth model<br>• Yr 1-5: Current growth rate<br>• Yr 6-10: Half of current growth<br>• Beyond 10: 2.5% terminal growth<br>• WACC assumed at 9%<br><br><span style='color:#FF8C00;'>⚠ Simplified model — reference only"}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Section 8: Multi-Model Fair Value Analysis ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### {'🎯 적정 주가 종합 분석 (멀티 밸류에이션)' if lang_inv=='ko' else '🎯 Fair Value Analysis — Multi-Model'}", unsafe_allow_html=True)
        st.markdown(f"<small style='color:#8B9DB0;'>{'6가지 밸류에이션 모델로 적정가를 산출하고 현재가와 괴리를 분석합니다' if lang_inv=='ko' else '6 valuation models to estimate fair value and explain the gap from current price'}</small>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        _curr = curr_pr or 0
        fv_methods = []  # list of (method_name, fair_value, weight, color, description)

        # 1) PER-based: industry-avg P/E × trailing EPS
        _eps_t = inv_info.get("trailingEps") or 0
        _per_fwd = inv_info.get("forwardPE")
        _per_trail = inv_info.get("trailingPE")
        # Use sector median P/E heuristic (technology ≈ 25, broad market ≈ 18)
        _sector_pe = 25.0 if "tech" in (inv_info.get("sector","") or "").lower() else 20.0
        _fv_per = _sector_pe * _eps_t if _eps_t and _eps_t > 0 else None

        # 2) PBR-based: sector-avg P/B × book value per share
        _bps = inv_info.get("bookValue") or 0
        _sector_pb = 4.0 if "tech" in (inv_info.get("sector","") or "").lower() else 2.5
        _fv_pbr = _sector_pb * _bps if _bps and _bps > 0 else None

        # 3) P/S-based: sector-avg P/S × revenue per share
        _rev = inv_info.get("totalRevenue") or 0
        _shares = inv_info.get("sharesOutstanding") or 1
        _rev_ps = _rev / _shares if _shares else 0
        _sector_ps = 8.0 if "tech" in (inv_info.get("sector","") or "").lower() else 2.0
        _fv_ps = _sector_ps * _rev_ps if _rev_ps else None

        # 4) EV/EBITDA-based
        _ebitda = inv_info.get("ebitda") or 0
        _td2 = inv_info.get("totalDebt") or 0
        _cash = inv_info.get("totalCash") or 0
        _sector_ev_ebitda = 20.0 if "tech" in (inv_info.get("sector","") or "").lower() else 12.0
        if _ebitda and _ebitda > 0 and _shares:
            _ev_fair = _sector_ev_ebitda * _ebitda
            _eq_fair = _ev_fair - _td2 + _cash
            _fv_evebitda = _eq_fair / _shares if _eq_fair > 0 else None
        else:
            _fv_evebitda = None

        # 5) Graham Number
        _fv_graham = graham  # already computed above (may be None)

        # 6) DCF
        try:
            _fv_dcf = dcf_per_share if 'dcf_per_share' in dir() else None
        except Exception:
            _fv_dcf = None

        # 7) Analyst consensus
        _fv_analyst = inv_info.get("targetMeanPrice") or None
        _fv_analyst_low  = inv_info.get("targetLowPrice") or None
        _fv_analyst_high = inv_info.get("targetHighPrice") or None

        # Build table
        _methods_raw = [
            ("PER 기반" if lang_inv=="ko" else "P/E Based",       _fv_per,      1.5, "#FFA500",
             f"섹터 평균 PER {_sector_pe:.0f}배 × EPS({_eps_t:.2f})" if lang_inv=="ko"
             else f"Sector avg P/E {_sector_pe:.0f}x × EPS({_eps_t:.2f})"),
            ("PBR 기반" if lang_inv=="ko" else "P/B Based",       _fv_pbr,      1.0, "#64B5F6",
             f"섹터 평균 PBR {_sector_pb:.1f}배 × BPS({_bps:.2f})" if lang_inv=="ko"
             else f"Sector avg P/B {_sector_pb:.1f}x × BPS({_bps:.2f})"),
            ("P/S 기반" if lang_inv=="ko" else "P/S Based",       _fv_ps,       0.8, "#AB63FA",
             f"섹터 평균 P/S {_sector_ps:.1f}배 × 주당매출({_rev_ps:.2f})" if lang_inv=="ko"
             else f"Sector avg P/S {_sector_ps:.1f}x × RevPS({_rev_ps:.2f})"),
            ("EV/EBITDA 기반" if lang_inv=="ko" else "EV/EBITDA",  _fv_evebitda, 1.2, "#FF8C00",
             f"섹터 평균 EV/EBITDA {_sector_ev_ebitda:.0f}배 적용" if lang_inv=="ko"
             else f"Sector avg EV/EBITDA {_sector_ev_ebitda:.0f}x applied"),
            ("그레이엄 넘버" if lang_inv=="ko" else "Graham Number", _fv_graham,  1.0, "#FFD700",
             "√(22.5 × EPS × BPS) — 안전마진 기준"),
            ("DCF 내재가치" if lang_inv=="ko" else "DCF Value",    _fv_dcf,      2.0, "#00D4AA",
             "3단계 성장 DCF 모델 (WACC 9%)" if lang_inv=="ko" else "3-stage DCF model (WACC 9%)"),
            ("애널리스트 목표가" if lang_inv=="ko" else "Analyst Target", _fv_analyst, 1.5, "#E91E8C",
             f"기관 애널리스트 평균 목표가 (범위: ${_fv_analyst_low or '?'}~${_fv_analyst_high or '?'})" if lang_inv=="ko"
             else f"Consensus analyst target (range: ${_fv_analyst_low or '?'}~${_fv_analyst_high or '?'})"),
        ]
        _valid = [(n, v, w, c, d) for n, v, w, c, d in _methods_raw if v and v > 0]

        if _valid and _curr > 0:
            # Weighted average fair value
            _total_w = sum(w for _, _, w, _, _ in _valid)
            _wavg_fv = sum(v * w for _, v, w, _, _ in _valid) / _total_w
            _simple_avg = sum(v for _, v, _, _, _ in _valid) / len(_valid)
            _gap_pct = (_wavg_fv - _curr) / _curr * 100
            _gap_color = "#FF4040" if _gap_pct > 10 else "#4488FF" if _gap_pct < -10 else "#FFA500"
            _gap_label = (
                ("🟢 저평가 — 매수 고려 구간" if _gap_pct > 20
                 else "🟡 약간 저평가" if _gap_pct > 10
                 else "🟡 적정 가격 근접" if _gap_pct > -10
                 else "🟠 약간 고평가" if _gap_pct > -20
                 else "🔴 고평가 — 주의 구간")
                if lang_inv == "ko" else
                ("🟢 Undervalued — Consider buying" if _gap_pct > 20
                 else "🟡 Slightly undervalued" if _gap_pct > 10
                 else "🟡 Near fair value" if _gap_pct > -10
                 else "🟠 Slightly overvalued" if _gap_pct > -20
                 else "🔴 Overvalued — Caution")
            )

            # ── Summary banner ──
            _min_fv = min(v for _, v, _, _, _ in _valid)
            _max_fv = max(v for _, v, _, _, _ in _valid)
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#1A1F35,#0F1527);border:2px solid {_gap_color};
                        border-radius:14px;padding:20px 28px;margin-bottom:20px;'>
                <div style='display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;'>
                    <div>
                        <div style='font-size:0.82rem;color:#8B9DB0;margin-bottom:4px;'>
                            {"가중평균 적정주가 (" + str(len(_valid)) + "개 모델)" if lang_inv=="ko"
                             else "Weighted Avg Fair Value (" + str(len(_valid)) + " models)"}
                        </div>
                        <div style='font-size:2.4rem;font-weight:900;color:{_gap_color};'>${_wavg_fv:,.2f}</div>
                        <div style='font-size:1rem;color:{_gap_color};font-weight:700;margin-top:4px;'>{_gap_label}</div>
                    </div>
                    <div>
                        <div style='font-size:0.82rem;color:#8B9DB0;'>{"현재가" if lang_inv=="ko" else "Current Price"}</div>
                        <div style='font-size:1.8rem;font-weight:800;color:#FFFFFF;'>${_curr:,.2f}</div>
                        <div style='font-size:1.1rem;font-weight:700;color:{_gap_color};margin-top:4px;'>{_gap_pct:+.1f}% {"괴리율" if lang_inv=="ko" else "gap"}</div>
                    </div>
                    <div>
                        <div style='font-size:0.82rem;color:#8B9DB0;'>{"적정가 범위" if lang_inv=="ko" else "Fair Value Range"}</div>
                        <div style='font-size:1rem;color:#B0BEC5;margin-top:4px;'>${_min_fv:,.2f} ~ ${_max_fv:,.2f}</div>
                        <div style='font-size:0.82rem;color:#8B9DB0;margin-top:4px;'>{"단순 평균" if lang_inv=="ko" else "Simple avg"}: ${_simple_avg:,.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Method-by-method table + bar chart ──
            fv_chart_col, fv_table_col = st.columns([3, 2])

            with fv_chart_col:
                fig_fv = go.Figure()
                names_fv = [n for n, _, _, _, _ in _valid]
                vals_fv  = [v for _, v, _, _, _ in _valid]
                colors_fv = [c for _, _, _, c, _ in _valid]
                gaps_fv   = [(v - _curr) / _curr * 100 for v in vals_fv]

                fig_fv.add_trace(go.Bar(
                    x=names_fv, y=vals_fv,
                    marker_color=colors_fv,
                    text=[f"${v:,.0f}<br>{g:+.1f}%" for v, g in zip(vals_fv, gaps_fv)],
                    textposition="outside",
                    hovertemplate="<b>%{x}</b><br>적정가: $%{y:,.2f}<br>괴리: %{text}<extra></extra>",
                ))
                # Current price line
                fig_fv.add_hline(
                    y=_curr, line_dash="dash", line_color="#FFFFFF", line_width=2,
                    annotation_text=f"  현재가 ${_curr:,.2f}" if lang_inv=="ko" else f"  Current ${_curr:,.2f}",
                    annotation_font_color="#FFFFFF",
                )
                # Weighted avg line
                fig_fv.add_hline(
                    y=_wavg_fv, line_dash="dot", line_color=_gap_color, line_width=2,
                    annotation_text=f"  적정가 ${_wavg_fv:,.2f}" if lang_inv=="ko" else f"  Fair Value ${_wavg_fv:,.2f}",
                    annotation_font_color=_gap_color,
                )
                fig_fv.update_layout(
                    template="plotly_dark",
                    height=380,
                    title=f"{'모델별 적정가 vs 현재가' if lang_inv=='ko' else 'Fair Value by Model vs Current Price'}",
                    margin=dict(l=0, r=0, t=50, b=0),
                    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                    yaxis_title="Price (USD)" if lang_inv=="en" else "주가 (USD)",
                    showlegend=False,
                )
                st.plotly_chart(fig_fv, use_container_width=True)

            with fv_table_col:
                st.markdown(f"<div style='font-weight:700;color:#FFA500;margin-bottom:10px;'>{'모델별 상세' if lang_inv=='ko' else 'Model Detail'}</div>", unsafe_allow_html=True)
                for n, v, w, c, desc in _valid:
                    _g = (v - _curr) / _curr * 100
                    _g_c = "#FF4040" if _g > 0 else "#4488FF"
                    st.markdown(f"""
                    <div style='background:#1A1F35;border-left:3px solid {c};border-radius:0 8px 8px 0;
                                padding:10px 14px;margin-bottom:8px;'>
                        <div style='display:flex;justify-content:space-between;align-items:center;'>
                            <span style='font-weight:700;color:{c};font-size:0.85rem;'>{n}</span>
                            <span style='font-size:1rem;font-weight:800;color:#FFFFFF;'>${v:,.2f}</span>
                        </div>
                        <div style='display:flex;justify-content:space-between;margin-top:4px;'>
                            <span style='color:#8B9DB0;font-size:0.72rem;'>{desc[:40]}{"..." if len(desc)>40 else ""}</span>
                            <span style='color:{_g_c};font-size:0.82rem;font-weight:700;'>{_g:+.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Gap analysis explanation ──
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='section-header'>{'🔍 괴리율 원인 분석' if lang_inv=='ko' else '🔍 Gap Analysis'}</div>", unsafe_allow_html=True)

            def _gap_explanation(gap_pct, inv_info, lang_inv, per, pbr, roe, rev_growth, fcf_val):
                reasons = []
                if gap_pct > 20:
                    reasons.append("📉 " + ("현재 주가가 여러 밸류에이션 모델 대비 크게 저평가되어 있습니다. 시장이 단기 악재를 과도하게 반영했거나, 아직 성장 잠재력이 충분히 인정받지 못했을 가능성이 있습니다." if lang_inv=="ko" else "The stock appears significantly undervalued vs. multiple models. The market may be over-pricing short-term risks or the growth potential may not yet be fully recognized."))
                elif gap_pct > 5:
                    reasons.append("📊 " + ("현재가가 적정가보다 다소 낮습니다. 단기 모멘텀 부재나 섹터 전반의 약세가 원인일 수 있습니다." if lang_inv=="ko" else "Price is slightly below fair value. Short-term momentum weakness or sector-wide selling pressure may be the cause."))
                elif gap_pct > -5:
                    reasons.append("⚖️ " + ("현재 주가가 여러 모델의 적정가와 거의 일치합니다. 시장이 적절히 가격을 반영한 상태입니다." if lang_inv=="ko" else "Current price is well-aligned with multi-model fair values. The market appears to be pricing the stock fairly."))
                elif gap_pct > -20:
                    reasons.append("📈 " + ("현재가가 적정가보다 높습니다. 성장 프리미엄·브랜드 가치 등이 반영됐거나 시장 과열 신호일 수 있습니다." if lang_inv=="ko" else "Price is above fair value. Growth premium, brand value, or market exuberance may be reflected."))
                else:
                    reasons.append("🚨 " + ("현재 주가가 대부분의 밸류에이션 모델 대비 크게 고평가되어 있습니다. 투자 시 주의가 필요합니다." if lang_inv=="ko" else "The stock appears significantly overvalued vs. most models. Caution is advised."))

                # Specific factor analysis
                if per and float(per) > 40:
                    reasons.append("🔺 " + (f"PER {float(per):.1f}배로 업종 평균 대비 높아 성장 기대감이 주가에 선반영된 상태입니다." if lang_inv=="ko"
                                   else f"P/E of {float(per):.1f}x is above sector avg — high growth expectations are priced in."))
                if pbr and float(pbr) > 5:
                    reasons.append("🔺 " + (f"PBR {float(pbr):.1f}배로 강력한 무형자산(브랜드·기술·특허) 가치가 반영된 것으로 해석됩니다." if lang_inv=="ko"
                                   else f"P/B of {float(pbr):.1f}x suggests strong intangible assets (brand/tech/IP) are priced in."))
                if roe and float(roe) * 100 > 20:
                    reasons.append("✅ " + (f"ROE {float(roe)*100:.1f}%의 높은 수익성이 프리미엄 밸류에이션을 정당화합니다." if lang_inv=="ko"
                                   else f"ROE of {float(roe)*100:.1f}% justifies premium valuation."))
                if rev_growth and float(rev_growth) * 100 > 20:
                    reasons.append("✅ " + (f"매출 성장률 {float(rev_growth)*100:.1f}%의 고성장이 현재 주가를 지지합니다." if lang_inv=="ko"
                                   else f"{float(rev_growth)*100:.1f}% revenue growth supports the current price level."))
                if fcf_val and fcf_val < 0:
                    reasons.append("⚠️ " + ("FCF가 마이너스로 성장 투자 단계의 기업입니다. 미래 수익성에 대한 신뢰가 가격 결정의 핵심입니다." if lang_inv=="ko"
                                   else "Negative FCF indicates a growth-stage company. Future profitability expectations drive the price."))

                analyst_tgt = inv_info.get("targetMeanPrice")
                n_analysts  = inv_info.get("numberOfAnalystOpinions") or 0
                if analyst_tgt and n_analysts:
                    reasons.append("📋 " + (f"총 {n_analysts}명의 애널리스트 평균 목표가는 ${analyst_tgt:,.2f}입니다." if lang_inv=="ko"
                                   else f"{n_analysts} analysts have an average target of ${analyst_tgt:,.2f}."))
                return reasons

            _reasons = _gap_explanation(_gap_pct, inv_info, lang_inv, per, pbr, roe, rev_growth, fcf_val)
            for r in _reasons:
                st.markdown(f"""
                <div style='background:#1A1F35;border-left:3px solid {_gap_color};border-radius:0 8px 8px 0;
                            padding:10px 16px;margin-bottom:8px;font-size:0.88rem;color:#E0E0E0;line-height:1.6;'>
                    {r}
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style='background:#0D1120;border:1px solid #2E3250;border-radius:10px;padding:12px 16px;margin-top:8px;'>
                <div style='font-size:0.78rem;color:#6B7A8D;'>
                    {"※ 적정가는 모델·가정에 따라 크게 달라집니다. 섹터 평균 배수는 시장 상황에 따라 변동되며, 본 분석은 참고용입니다." if lang_inv=="ko"
                     else "⚠ Fair values vary significantly by model and assumptions. Sector multiples shift with market conditions. For reference only."}
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.info("현재가 또는 재무 데이터 부족으로 적정가 분석이 어렵습니다." if lang_inv=="ko"
                    else "Insufficient financial data to perform fair value analysis.")

        # ── Section 9: AI One-line Summary ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"### {'🤖 AI 종합 투자 분석 요약' if lang_inv=='ko' else '🤖 AI Investment Summary'}", unsafe_allow_html=True)

        # Generate rule-based AI summary
        def generate_invest_summary(inv_info, lang_inv, per, pbr, roe, op_margin, rev_growth, fcf_val, pio_score, z_str):
            signals = []
            concerns = []
            # Valuation
            if per and float(per) < 15: signals.append("저PER 저평가" if lang_inv=="ko" else "low P/E undervaluation")
            elif per and float(per) > 35: concerns.append("고PER 고평가 우려" if lang_inv=="ko" else "high P/E overvaluation risk")
            if pbr and float(pbr) < 1: signals.append("PBR 1배 미만 자산 저평가" if lang_inv=="ko" else "trading below book value")
            # Profitability
            if roe and float(roe)*100 > 15: signals.append(f"ROE {float(roe)*100:.0f}% 우수 수익성" if lang_inv=="ko" else f"excellent ROE {float(roe)*100:.0f}%")
            if op_margin and float(op_margin)*100 > 20: signals.append("높은 영업이익률로 경쟁우위 확보" if lang_inv=="ko" else "high operating margin competitive moat")
            # Growth
            if rev_growth and float(rev_growth)*100 > 15: signals.append(f"매출 {float(rev_growth)*100:.0f}% 고성장" if lang_inv=="ko" else f"{float(rev_growth)*100:.0f}% revenue growth")
            elif rev_growth and float(rev_growth)*100 < 0: concerns.append("매출 역성장" if lang_inv=="ko" else "revenue decline")
            # Cash flow
            if fcf_val and fcf_val > 0: signals.append("양호한 잉여현금흐름" if lang_inv=="ko" else "positive free cash flow")
            else: concerns.append("FCF 마이너스" if lang_inv=="ko" else "negative FCF")
            # Piotroski
            if pio_score >= 7: signals.append(f"F-스코어 {pio_score}/9 재무 건전" if lang_inv=="ko" else f"F-Score {pio_score}/9 financially strong")
            elif pio_score <= 2: concerns.append(f"F-스코어 {pio_score}/9 재무 부실" if lang_inv=="ko" else f"F-Score {pio_score}/9 financially weak")
            # Z-Score
            try:
                z_v = float(z_str)
                if z_v > 3: signals.append("부도 위험 낮음" if lang_inv=="ko" else "low bankruptcy risk")
                elif z_v < 1.8: concerns.append("부도 위험 경고" if lang_inv=="ko" else "bankruptcy risk warning")
            except Exception:
                pass

            if not signals and not concerns:
                return ("데이터 부족으로 자동 분석이 어렵습니다. 직접 재무제표를 확인하세요." if lang_inv=="ko"
                        else "Insufficient data for automated analysis. Please review financial statements directly.")

            summary_parts = []
            if signals:
                summary_parts.append(("✅ 긍정 신호: " if lang_inv=="ko" else "✅ Positives: ") + " · ".join(signals[:3]))
            if concerns:
                summary_parts.append(("⚠️ 주의 사항: " if lang_inv=="ko" else "⚠️ Concerns: ") + " · ".join(concerns[:3]))
            return " | ".join(summary_parts)

        ai_summary = generate_invest_summary(inv_info, lang_inv, per, pbr, roe, op_margin, rev_growth, fcf_val, pio_score, z_str)
        st.markdown(f"""
        <div class='summary-box' style='border-color:#FFA500;'>
            <div style='font-size:0.78rem;color:#8B9DB0;margin-bottom:6px;'>🤖 {"규칙 기반 자동 분석" if lang_inv=="ko" else "Rule-based Auto Analysis"} · {company_name} ({ticker})</div>
            <div style='font-size:0.92rem;color:#EAEAEA;line-height:1.8;'>{ai_summary}</div>
        </div>
        """, unsafe_allow_html=True)

        # Disclaimer
        st.markdown(f"<small style='color:#4A5568;'>{'⚠️ 모든 지표는 참고용입니다. yFinance 실시간 데이터 기반. 투자는 본인 책임입니다.' if lang_inv=='ko' else '⚠️ All metrics for reference only. Based on yFinance real-time data. Invest at your own risk.'}</small>", unsafe_allow_html=True)

    # ─── FOOTER ───────────────────────────────────────────────────────────────────
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align:center;color:#4A5568;font-size:0.8rem;padding:16px;border-top:1px solid #2E3250;'>
        {'⚠️ 이 프로그램은 교육 및 분석 목적으로만 제공됩니다. 투자 결정은 전문 금융 어드바이저와 상담하세요.' if lang == 'ko'
         else '⚠️ This system is for educational and analytical purposes only. Consult a qualified financial advisor before making investment decisions.'}<br>
        Data: Yahoo Finance | News: Reuters, CNBC, MarketWatch | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC
    </div>
    """, unsafe_allow_html=True)
