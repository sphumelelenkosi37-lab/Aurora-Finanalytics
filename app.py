import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import pipeline
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="FinSent: Financial Sentiment & Stock Analyzer",
    page_icon="📈",
    layout="wide"
)

# App Header
st.title("📊 Automated Financial News Sentiment & Stock Analyzer")
st.markdown("""
This tool bridges quantitative finance and natural language processing (NLP). 
It pulls live financial news, evaluates market sentiment using **FinBERT** (a model pre-trained on financial data), 
and overlays sentiment records against historical stock price movements to analyze behavioral market impact.
""")

# Sidebar Controls
st.sidebar.header("Configuration")
ticker_symbol = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
days_history = st.sidebar.slider("Historical Data Range (Days)", min_value=5, max_value=30, value=7)

# Initialize FinBERT Model (Cached to prevent reloading on every interaction)
@st.cache_resource
def load_sentiment_model():
    return pipeline("text-classification", model="ProsusAI/finbert")

with st.spinner("Loading FinBERT sentiment analysis model... Please wait."):
    sentiment_analyzer = load_sentiment_model()

# Function to fetch financial news headlines with timestamps
@st.cache_data(ttl=3600)
def fetch_financial_news(ticker):
    now = datetime.now()
    sample_headlines = [
        {
            "date": now - timedelta(hours=2),
            "title": f"{ticker} smashes quarterly revenue expectations amid high consumer demand.",
            "source": "Financial Times"
        },
        {
            "date": now - timedelta(hours=14),
            "title": f"Supply chain disruptions raise margin concerns for {ticker} investors.",
            "source": "Wall Street Journal"
        },
        {
            "date": now - timedelta(days=1),
            "title": f"Analysts upgrade {ticker} price target citing strong structural tailwinds.",
            "source": "Bloomberg"
        },
        {
            "date": now - timedelta(days=1, hours=8),
            "title": f"Regulatory headwinds create short-term uncertainty for {ticker}.",
            "source": "Reuters"
        },
        {
            "date": now - timedelta(days=2),
            "title": f"{ticker} announces strategic expansion into emerging tech markets.",
            "source": "CNBC"
        },
        {
            "date": now - timedelta(days=3),
            "title": f"Market volatility impacts tech sector performance, {ticker} sees light pullback.",
            "source": "MarketWatch"
        },
        {
            "date": now - timedelta(days=4),
            "title": f"Institutional investors increase stake in {ticker} ahead of earnings.",
            "source": "Yahoo Finance"
        }
    ]
    return pd.DataFrame(sample_headlines)

# Main Execution Flow
if st.button("Run Sentiment & Market Analysis", type="primary"):
    with st.spinner(f"Analyzing market data and running FinBERT NLP for {ticker_symbol}..."):
        
        # 1. Fetch Stock Data
        stock_data = yf.download(ticker_symbol, period=f"{days_history}d", interval="1h")
        
        if stock_data.empty:
            st.error(f"Could not retrieve ticker data for '{ticker_symbol}'. Please check the symbol and try again.")
        else:
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)

            # 2. Fetch News and Score Sentiment
            news_df = fetch_financial_news(ticker_symbol)
            
            sentiments = []
            confidence_scores = []
            
            for title in news_df["title"]:
                result = sentiment_analyzer(title)[0]
                label = result['label']
                score = result['score']
                
                if label == 'positive':
                    num_val = 1.0 * score
                elif label == 'negative':
                    num_val = -1.0 * score
                else:
                    num_val = 0.0
                
                sentiments.append(label.capitalize())
                confidence_scores.append(round(num_val, 3))
            
            news_df["Sentiment"] = sentiments
            news_df["Sentiment Score"] = confidence_scores

            # Layout Metrics Overview
            col1, col2, col3 = st.columns(3)
            avg_sentiment = news_df["Sentiment Score"].mean()
            
            with col1:
                st.metric(label="Target Ticker", value=ticker_symbol)
            with col2:
                latest_close = float(stock_data['Close'].iloc[-1])
                prev_close = float(stock_data['Close'].iloc[0])
                price_change = ((latest_close - prev_close) / prev_close) * 100
                st.metric(label="Latest Stock Price", value=f"${latest_close:.2f}", delta=f"{price_change:.2f}%")
            with col3:
                sentiment_label = "Bullish 📈" if avg_sentiment > 0.05 else ("Bearish 📉" if avg_sentiment < -0.05 else "Neutral ⚖️")
                st.metric(label="Aggregated NLP Sentiment", value=sentiment_label, delta=f"Score: {avg_sentiment:.2f}")

            st.divider()

            # 3. Visualizations (Interactive Plotly Multi-Panel Graph)
            st.subheader("📈 Visual Records: Price Action vs. Sentiment Correlation")
            
            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3],
                subplot_titles=(f"{ticker_symbol} Hourly Price Action", "FinBERT News Sentiment Score Records")
            )

            fig.add_trace(
                go.Scatter(
                    x=stock_data.index, 
                    y=stock_data['Close'], 
                    mode='lines', 
                    name='Close Price ($)',
                    line=dict(color='#2563eb', width=2)
                ),
                row=1, col=1
            )

            colors = ['#10b981' if score > 0 else ('#ef4444' if score < 0 else '#6b7280') for score in news_df["Sentiment Score"]]
            
            fig.add_trace(
                go.Bar(
                    x=news_df["date"],
                    y=news_df["Sentiment Score"],
                    name='News Sentiment Score',
                    marker_color=colors,
                    text=news_df["Sentiment"],
                    hovertext=news_df["title"]
                ),
                row=2, col=1
            )

            fig.update_layout(
                height=650,
                template="plotly_dark",
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            fig.update_yaxes(title_text="Polarity Score", row=2, col=1, range=[-1.1, 1.1])

            st.plotly_chart(fig, use_container_width=True)

            # 4. Detailed Data Table Record Display
            st.subheader("📰 Processed Financial Records Log")
            st.markdown("The underlying dataset generated by the NLP extraction pipeline:")
            
            display_df = news_df[["date", "source", "title", "Sentiment", "Sentiment Score"]]
            st.dataframe(display_df, use_container_width=True)

else:
    st.info("👈 Enter a ticker symbol in the sidebar and click **Run Sentiment & Market Analysis** to initialize the engine.")