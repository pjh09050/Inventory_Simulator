import streamlit as st
import yaml
import pandas as pd
from datetime import datetime
# from function import *
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import platform
import os
if platform.system() == 'Windows':
    rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
else: # linux
    rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')

st.markdown("""
    <style>
    .block-container {
        padding-top: 2.5rem;
        max-width: 80%;  /* Increase the width to 70% of the page */
    }
    /* Increase font size for all elements */
    html, body {
        font-size: 1.3rem; /* Global font size adjustment */
    }
    h1 {
        font-size: 1.5rem; /* Increase title size */
    }
    h2, h3 {
        font-size: 1.5rem; /* Increase subheader size */
    }
    h4 {
        font-size: 1.3rem; /* Increase subheader size */
    }
    .stButton button {
        font-size: 1.2rem; /* Button font size */
    }
    </style>
    """, unsafe_allow_html=True)

# Title and header
st.markdown('# Inventory Simulator')

st.markdown('### 재고관리 시뮬레이션 및 최적화 ')

col0, col1, col2 = st.columns([1,1,1])

with col0:
    st.markdown('재고 비용')
    st.markdown('익일 재고 수준에 기반하여 재고 비용을 계산')

with col1:
    st.markdown('백로그 비용')
    st.markdown('주문이 들어온 상황에서 제공하지 못하는 상황')
    st.markdown('재고가 음수로 내려가는 상황이 지속된 경우 지수적으로 비용이 증가')
    
with col2:
    st.markdown('주문비용')
    st.markdown('주문이 발생할 때 발생하는 고정 비용과 주문량에 비례해서 선형적으로 증가하는 변동 비용')
    st.markdown('')

with st.expander('시뮬레이터 파라미터 설정 방법'):
    st.markdown('-'*50)
    st.markdown('### 시뮬레이터를 위한 분포 추론')
    st.markdown('정규분포를 활용한 주문량 생성')
    st.markdown('포아송분포를 활용한 주문간격 생성')

with st.expander('### 재고 최소화를 위한 목적함수'):    
    st.markdown("### Objective Fuction :")
    st.latex(r"Q(SS, EOQ | T) = \sum_{t=0}^{T} (\text{Inventory cost}_t + \text{backlog cost}_t + \text{ordering cost}_t)")

    st.markdown("### Where :")

    st.markdown("1. **Inventory Cost:**")
    st.latex(r"\text{Inventory cost}_t = CS_t \times \alpha \%")

    st.markdown("2. **Backlog Cost:**")
    st.latex(r"\text{backlog cost}_t = \max(0, CS_t) \times \beta \times \exp(\lambda \times \tau_t)")

    st.markdown("3. **Ordering Cost:**")
    st.latex(r"\text{ordering cost}_t = EOQ \times \text{cost} \times \gamma + \delta")

    st.markdown("4. **Inventory Balance Equation:**")
    st.latex(r"CS_t = CS_{t-1} + S_t - D_t, \quad P_t \sim \text{Poisson}(\lambda)")

    st.markdown("5. **Backlog Time $\\tau_t$ :**")
    st.latex(r"""
    \tau_t = 
    \begin{cases}
    \tau_{t-1} + 1, & \text{if } D_t > (CS_t + S_t) \\
    0, & \text{otherwise}
    \end{cases}
    """)

    st.markdown("6. **Reorder Point (ROP):**")
    st.latex(r"ROP = SS + \sum_{k=t}^{t+L} D_k")

    st.markdown("7. **Demand ($D_t$)**")
    st.latex(r"""
    D_t = 
    \begin{cases}
    N(\bar{D}, \sigma_D), & \text{if } P_t = 1 \\
    0, & \text{otherwise}
    \end{cases}
    """)

    st.markdown("8. **Order Quantity Decision ($S_{t+1}$))**")
    st.latex(r"""
    S_{t+1} = 
    \begin{cases}
    EOQ, & \text{if } CS_t < ROP \\
    0, & \text{otherwise}
    \end{cases}, \quad I_t \sim N(\bar{I}, \sigma_I)
    """)

    st.markdown("### Parameter Definitions:")
    st.markdown("- $ CS_t $: On-hand inventory at time \(t\)")
    st.markdown("- $ D_t $: Demand at time \(t\)")
    st.markdown("- $ S_t $: Order quantity at time \(t\)")
    st.markdown("- $ t $: Time period")
    st.markdown("- $ \lambda $: Demand arrival rate (Poisson parameter)")
    st.markdown("- $ \\alpha,\\beta, \gamma, \delta $: Parameters:")
    st.markdown("  - $\\alpha $: Inventory holding cost rate (%)")
    st.markdown("  - $\\beta $: Backlog cost factor (%)")
    st.markdown("  - $\gamma $: Fixed order cost per EOQ")
    st.markdown("  - $\delta $: Constant ordering cost per period")
    st.markdown("-  $\\bar{D}, \sigma_D $): Mean and standard deviation of demand")
    st.markdown("- $\\bar{I}, \sigma_I $: Mean and standard deviation of lead time demand")
    st.markdown("- $L$: Lead time")
    st.markdown("- $ ROP $: Reorder point")
    st.markdown("- $ EOQ $: Economic order quantity")
    st.markdown("- $ SS $: Safety stock level")
    st.markdown("- $ P_t $: Binary variable indicating demand occurrence at time $t$")
