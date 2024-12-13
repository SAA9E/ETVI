import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# 데이터
ETVI = pd.read_csv("./ETVI_final_df.csv")

# 페이지 레이아웃
st.set_page_config(layout="wide", page_title="지수 대시보드", initial_sidebar_state="expanded")

import streamlit as st

# 사이드바 디자인
with st.sidebar:
    st.title("🔍 ETVI의 구성은?")

    with st.expander("🏭 **미국 core PPI**"):
        st.info("""
        core PPI는 근원 생산자물가지수로 식품과 에너지를 제외한 상품 및 서비스의 가격 변동 지수입니다. 
        
         소비자 인플레이션의 선행지표로 간주되는데, 이는 생산자 상품과 서비스에 더 많은 비용을 지출할수록 
        소비자에게 더 많은 비용을 전가할 가능성이 높다는 상관성이 있기 때문입니다.
        
        PPI는 물가 상승 압박이 클 경우, 경제 불확실성이 증가하고 시장의 변동성이 커질 가능성을 시사합니다.
        특히, 생산자물가 상승은 원자재 가격 상승과 같은 비용 증가를 유발하여 투자시장에 부정적인 영향을 미칠 수 있습니다.        
        """)

    with st.expander("🧑‍💼 **Real-time Sahm**"):
        st.info("""
        
        Sham은  시장의 변동성과 불확실성을 실시간으로 측정하는 지표입니다. 

        주로 경제적 위기나 급격한 변동성을 평가하는 데 사용되며,
        GDP, 소비자 신뢰지수, 산업생산 등  다양한 경제지표로 계산되는 대표적인 실업률지표입니다.
                
        미국경제가 침체에 접어들경우 투자자는 불안정성을 느끼고 매도압력이 강해져 전체 시장의 변동성이 강해질 수 있음을 시사합니다.
        """)

    with st.expander(" **💫 미국 MSCI**"):
        st.info("""
        MSCI(Morgan Stanley Capital International)이 발표하는 세계주가지수입니다.
        
        2020년 말 기준 미국계 펀드 95%는 MSCI를 참조하여 리스크를 관리 및 운용합니다.
        지수 구성종목은 1년에 4번 리밸런싱되는데 이때 세계 투자자금이 함께 활발히 움직이는 경향이 있습니다.
        
        이는 미국 시장의 변동성이 클수록 한국 ETF 시장의 변동성도 상승한다는 것을 의미합니다.
        즉, 글로벌 시장의 불확실성이 한국 시장에 영향을 미치며, 미국 MSCI 지수의 변화가 한국 ETF 시장의 리스크를 예측하는 데 중요한 역할을 합니다.
        """)

    
    with st.expander("🪙 **금(표준편차)**"):
        st.info("""
        금은 한국 ETF시장의 변동성과 큰 관련성을 보입니다. 
                
        이는 안전자산인 금은 다른 변수들에 비해 변동이 작기에, 1단위 값의 상승이
        전체시장의 큰 불확실성과  변동을 반영하는 것으로  해석할 수 있습니다.
       """)
                
    with st.expander("🌐 **한국 무역수지(표준편차)**"):
        st.info("""
        한국 무역수지와 ETF 시장 변동성은 음의 관계를 보입니다.
        이는 한국의 무역이 악화될 경우, 외환시장 불안정성이나 경제적 불확실성이 증가하며, 
        그로 인해 ETF 시장의 변동성도 상승할 수 있음을 시사합니다.
        반대로, 무역수지가 개선되면 시장 불확실성이 줄어들어 
        ETF 시장의 변동성도 낮아질 가능성이 있습니다.
        """)

    with st.expander("🚩 **MACD (Moving Average Convergence Divergence)**"):
        st.info("""
        MACD는 MACD(Moving Average Convergence Divergence)는 금융 시장에서 주식, ETF, 지수 등의 추세 강도와 변화 방향을 분석하기 위해 사용되는 기술적 지표입니다.
        
        단기 지수이동평균(EMA) - 장기 지수이동평균의 차이(12일 EMA와 26일 EMA)로 계산됩니다.
        MACD가 신호선을 상향 돌파하면 매수 신호, 하향 돌파하면 매도 신호로 해석됩니다. 이는 현재 시장의 강세 혹은 약세를 반영합니다.
       
        이는 상승 추세에서 투자자들이 더 많은 거래를 통해 기회를 찾고 있음을 시사합니다. 
        강세장에서 나타나는 일반적인 패턴입니다.
        """)

    with st.expander("💼 **주식시장 모멘텀 (Market Momentum)**"):
        st.info("""
        CNN 공포탐욕지수를 벤치마킹하여 한국시장에 적용한 지표입니다.
        S&P500을 KOSPI로 치환하여, KOSPI 지수가 125일 이동평균을 초과하거나 하회하는 정도를 측정하는 시장의 추세 지표입니다.
                
        KOSPI가 이평보다 높을수록 탐욕이 상승하고 긍정적 모멘텀으로 인식됩니다. 반대의 경우 투자자들의 불안이 올라가는 상황으로 해석될 수 있습니다.
                
        이는 시장이 상승하거나 하락하는 추세가 강할 때,
        투자자 심리와 시장 반응이 더 극적으로 나타나 ETF 수익률의 변동성 증가에 영향을 줄 수 있음을 의미합니다.
        """)

    with st.expander("🚨 **볼린저밴드 GAP (Bollinger Band GAP)**"):
        st.info("""
        이동평균의 표준편차를 기반으로 한 대표적인 변동성 지표로, 상단과 하단 밴드 간의 간격(GAP)을 분석합니다.
        시장의 현재 변동성수준을 직관적으로 확인하는 대표적인 방법입니다.
        GAP이 클수록 시장 변동성이 크며 투자자들의 불안 심리가 반영됩니다.
        높은 GAP은 현재 투자자들이 시장을 불안정하게 인식하고 있음을 반영합니다.
        """)


    with st.expander("💰 **P/C ratio**"):
        st.info("""
        P/C ratio는 CNN의 공포탐욕지수에서 벤치마킹되어 한국시장에 적용된 지수입니다.
        풋옵션 거래량/ 콜옵션 거래량으로 계산되며, 투자시장의 하락배팅과 상승배팅의 비율로 시장의 변동성을 나타내는 지표입니다.
        즉, 1을 초과하면 풋옵션의 거래가 많음 = 매도세가 강한 지표가 되고, 0.7 미만~ 0.5에 근접한다면 매수세가 강한 시장임을 시사합니다.
        
        P/C ratio는 시장의 감정과 심리를 반영합니다.
        즉 높은 P/C는 투자자들이 하락에 대비하며 공포가 팽배한 시장임을 시사하며,
        반대로 낮은 P/C는 과신이나 과열상태를 시사합니다.
        또한 극단적인 비율은 시장반전가능성 등 시장의 변동성에 영향을 줄 수 있습니다.      
                """)


    
# 배경 및 스타일
st.markdown("""
    <style>
        body {
            background-color: #f0f4f7;
        }
        .reportview-container .main .block-container {
            padding: 2rem;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.1);
        }
        h1, h2 {
            color: #333;
        }
        .stButton>button {
            background-color: #0073e6;
            color: white;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #005bb5;
        }
        .stMarkdown {
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# 제목 및 설명
st.title("📈 ETVI : ETF Variance Index ")


# HTML을 사용하여 기본적으로 열려있는 Expander처럼 보이게 만들기
st.markdown("""
    <details open>
        <summary>🔍 ETVI란?</summary>
        개인 투자자의 빠르고 효율적인 투자결정을 돕는 지표입니다. ETF 매수시점의 ETVI와 현재의 ETVI를 손쉽게 비교해서, 투자기간 동안의 경제 변화 및 시장 변동 정도를 집약적으로 파악할 수 있습니다.
    </details>
""", unsafe_allow_html=True)






# 'Date' 열을 datetime 형식으로 변환
ETVI['Date'] = pd.to_datetime(ETVI['Date'])

# start_date와 end_date에 해당하는 ETVI 등급을 가져오는 함수
def get_grade_for_date(date, df):
    return df[df['Date'] == date]['ETVI 등급'].values[0] if not df[df['Date'] == date].empty else None

# get_grade_value 함수를 작성하여 ETVI 점수에 맞는 값 설정
def get_grade_value(date, df):
    etvi_score = df[df['Date'] == date]['ETVI 점수'].values[0] if not df[df['Date'] == date].empty else 0
    return etvi_score  # 'ETVI 점수' 값 반환

# 게이지 차트를 생성하는 함수
def create_gauge_chart(etvi_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=etvi_score,
        title={'text': "ETVI 점수"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "lightblue"},
            'steps': [
                {'range': [0, 20], 'color': "green"},
                {'range': [20, 40], 'color': "lightgreen"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"},
            ]
        }
    ))
    return fig

# 날짜 범위 선택 (기본값 설정)
st.title("")
st.markdown("##### 투자한 날짜와 매도 시점을 선택하세요! 😊")

start_date = st.date_input(
    "시작 날짜", 
    value=pd.to_datetime('2020-01-02').date(),  # 기본값: 2020-01-01
    min_value=ETVI['Date'].min().date(),  
    max_value=ETVI['Date'].max().date()
)
end_date = st.date_input(
    "끝 날짜", 
    value=pd.to_datetime('2024-07-01').date(),  # 기본값: 2024-07-01
    min_value=ETVI['Date'].min().date(),  
    max_value=ETVI['Date'].max().date()
)

# 날짜 형식 변환
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# start_date와 end_date에 해당하는 ETVI 점수 구하기
start_etvi_score = get_grade_value(start_date, ETVI)
end_etvi_score = get_grade_value(end_date, ETVI)

# 타임시리즈 차트
st.markdown("## 📅 ETVI Interactive Time Series")

# 필터링된 데이터 확인
filtered_data = ETVI[(ETVI["Date"] >= start_date) & (ETVI["Date"] <= end_date)]

# 첫 번째 열 (타임 시리즈 그래프)
fig = go.Figure()

# ETVI 점수 시각화
fig.add_trace(go.Scatter(
    x=filtered_data['Date'], 
    y=filtered_data['ETVI 점수'], 
    mode='lines', 
    name='ETVI 점수', 
    line=dict(color='skyblue', width=2)
))

# ETF시장의 변동성 시각화
fig.add_trace(go.Scatter(
    x=filtered_data['Date'], 
    y=filtered_data['ETF시장의 변동성'], 
    mode='lines', 
    name='ETF시장의 변동성', 
    line=dict(color='blue', width=3)
))

# 그래프 레이아웃 설정
fig.update_layout(
    xaxis_title='Date',
    yaxis_title='값',
    template='plotly',
    plot_bgcolor='rgba(233, 246, 252, 0.8)',  # 플롯 영역 하늘색 배경
    font=dict(family="Arial, sans-serif", size=12, color="black"),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)

# 타임 시리즈 그래프 표시
st.plotly_chart(fig, use_container_width=True)

# Gauge 차트 아래 나란히 배치 (양옆에)
st.markdown("## 🚦 ETVI 등급 Gauge Charts")

# Gauge 차트 배치
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"##### **{start_date.date()}의 ETVI 점수**")
    start_gauge_chart = create_gauge_chart(start_etvi_score)
    st.plotly_chart(start_gauge_chart)

with col2:
    st.markdown(f"##### **{end_date.date()}의 ETVI 점수**")
    end_gauge_chart = create_gauge_chart(end_etvi_score)
    st.plotly_chart(end_gauge_chart)

# 필터링된 데이터 확인 (디버깅용)
st.markdown("##### 🔍 더 자세한 정보")
st.write(filtered_data)

st.write("")

st.markdown("""
    <div style="background-color: #E9F6FC; padding: 10px; border-radius: 5px;">
        💡 ETVI 지표 값은 과거 일별 데이터를 기반으로 계산되었습니다.  
        본 지표는 ETF의 변동 위험을 설명합니다.
    </div>
""", unsafe_allow_html=True)
