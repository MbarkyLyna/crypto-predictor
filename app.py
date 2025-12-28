import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from utils.data_fetcher import CryptoDataFetcher
from utils.predictor import CryptoPredictor

# Page config
st.set_page_config(
    page_title="Crypto Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Initialize
@st.cache_resource
def init_components():
    fetcher = CryptoDataFetcher()
    predictor = CryptoPredictor()
    return fetcher, predictor

fetcher, predictor = init_components()

# Sidebar
st.sidebar.title("⚙️ Settings")
selected_coin = st.sidebar.selectbox(
    "Select Cryptocurrency",
    ['bitcoin', 'ethereum', 'cardano', 'solana', 'ripple'],
    index=0
)

historical_days = st.sidebar.slider(
    "Historical Data (days)",
    min_value=30,
    max_value=365,
    value=90
)

prediction_hours = st.sidebar.slider(
    "Prediction Horizon (hours)",
    min_value=6,
    max_value=72,
    value=24
)

auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)

# Main title
st.title("📈 Real-Time Crypto Price Predictor")
st.markdown(f"**Live predictions powered by Machine Learning** | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load data
with st.spinner(f"Loading {selected_coin.upper()} data..."):
    # Current price
    current_data = fetcher.get_current_price(selected_coin)
    
    # Historical data
    historical_data = fetcher.get_historical_data(selected_coin, days=historical_days)
    
    if historical_data is None or current_data is None:
        st.error("Failed to fetch data. Please try again.")
        st.stop()

# Train model
with st.spinner("Training ML model..."):
    mape = predictor.train(historical_data)

# Make predictions
with st.spinner("Generating predictions..."):
    predictions = predictor.predict_next_24h(historical_data, hours=prediction_hours)
    signal = predictor.generate_signal(current_data['price'], predictions)

# Display metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Price",
        f"${current_data['price']:,.2f}",
        f"{current_data['change_24h']:.2f}%"
    )

with col2:
    st.metric(
        "24h Volume",
        f"${current_data['volume_24h']/1e9:.2f}B"
    )

with col3:
    st.metric(
        "Predicted Avg (24h)",
        f"${signal['predicted_avg']:,.2f}",
        f"{signal['expected_change']:.2f}%"
    )

with col4:
    st.metric(
        "Model Accuracy",
        f"{100-mape:.1f}%",
        "MAPE"
    )

# Trading Signal
st.markdown("---")
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### Trading Signal")
    signal_html = f"""
    <div style="padding: 20px; border-radius: 10px; background-color: {'#d4edda' if 'BUY' in signal['signal'] else '#f8d7da' if 'SELL' in signal['signal'] else '#fff3cd'}; text-align: center;">
        <h1 style="margin: 0; font-size: 48px;">{signal['signal']}</h1>
        <p style="font-size: 20px; margin: 10px 0;">Confidence: {signal['confidence']:.0f}%</p>
        <p style="font-size: 16px;">Expected Change: {signal['expected_change']:.2f}%</p>
    </div>
    """
    st.markdown(signal_html, unsafe_allow_html=True)

with col2:
    st.markdown("### Signal Explanation")
    if 'BUY' in signal['signal']:
        st.success(f"""
        **Buy Signal Detected!** 
        - Model predicts price will increase by {signal['expected_change']:.2f}% in next {prediction_hours} hours
        - Current: ${signal['current_price']:,.2f}
        - Predicted Avg: ${signal['predicted_avg']:,.2f}
        - Consider entering a long position
        """)
    elif 'SELL' in signal['signal']:
        st.error(f"""
        **Sell Signal Detected!**
        - Model predicts price will decrease by {signal['expected_change']:.2f}% in next {prediction_hours} hours
        - Current: ${signal['current_price']:,.2f}
        - Predicted Avg: ${signal['predicted_avg']:,.2f}
        - Consider exiting positions or shorting
        """)
    else:
        st.warning(f"""
        **Hold Position**
        - Model predicts minimal price movement ({signal['expected_change']:.2f}%)
        - Current: ${signal['current_price']:,.2f}
        - Predicted Avg: ${signal['predicted_avg']:,.2f}
        - Wait for clearer signals
        """)

# Charts
st.markdown("---")
st.markdown("### 📊 Price Analysis")

tab1, tab2, tab3 = st.tabs(["Historical + Predictions", "Technical Indicators", "Volume Analysis"])

with tab1:
    # Historical + Prediction chart
    fig = go.Figure()
    
    # Historical prices
    fig.add_trace(go.Scatter(
        x=historical_data['timestamp'],
        y=historical_data['price'],
        mode='lines',
        name='Historical Price',
        line=dict(color='blue', width=2)
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=predictions['timestamp'],
        y=predictions['predicted_price'],
        mode='lines',
        name='Predicted Price',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Current price marker
    fig.add_trace(go.Scatter(
        x=[datetime.now()],
        y=[current_data['price']],
        mode='markers',
        name='Current Price',
        marker=dict(size=15, color='green')
    ))
    
    fig.update_layout(
        title=f"{selected_coin.upper()} Price - Historical & Predicted",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Calculate moving averages
    historical_data['MA7'] = historical_data['price'].rolling(window=7).mean()
    historical_data['MA30'] = historical_data['price'].rolling(window=30).mean()
    
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=historical_data['timestamp'],
        y=historical_data['price'],
        name='Price',
        line=dict(color='blue')
    ))
    
    fig2.add_trace(go.Scatter(
        x=historical_data['timestamp'],
        y=historical_data['MA7'],
        name='7-Day MA',
        line=dict(color='orange', dash='dot')
    ))
    
    fig2.add_trace(go.Scatter(
        x=historical_data['timestamp'],
        y=historical_data['MA30'],
        name='30-Day MA',
        line=dict(color='green', dash='dot')
    ))
    
    fig2.update_layout(
        title="Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    # Volume chart
    if 'volume' in historical_data.columns:
        fig3 = go.Figure()
        
        fig3.add_trace(go.Bar(
            x=historical_data['timestamp'],
            y=historical_data['volume'],
            name='Volume',
            marker_color='lightblue'
        ))
        
        fig3.update_layout(
            title="Trading Volume",
            xaxis_title="Date",
            yaxis_title="Volume (USD)",
            height=500
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Volume data not available for this timeframe")

# Prediction table
st.markdown("---")
st.markdown("### 📋 Detailed Predictions")

col1, col2 = st.columns([2, 1])

with col1:
    # Show predictions table
    pred_display = predictions.copy()
    pred_display['predicted_price'] = pred_display['predicted_price'].apply(lambda x: f"${x:,.2f}")
    pred_display['timestamp'] = pred_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(
        pred_display,
        use_container_width=True,
        height=400
    )

with col2:
    st.markdown("#### Statistics")
    st.metric("Min Predicted", f"${predictions['predicted_price'].min():,.2f}")
    st.metric("Max Predicted", f"${predictions['predicted_price'].max():,.2f}")
    st.metric("Avg Predicted", f"${predictions['predicted_price'].mean():,.2f}")
    st.metric("Price Range", f"${predictions['predicted_price'].max() - predictions['predicted_price'].min():,.2f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    <p>⚠️ <strong>Disclaimer:</strong> This is for educational purposes only. Not financial advice. Always do your own research.</p>
    <p>Data source: CoinGecko API | Model: Random Forest Regressor</p>
</div>
""", unsafe_allow_html=True)

# Auto-refresh
if auto_refresh:
    time.sleep(60)
    st.rerun()