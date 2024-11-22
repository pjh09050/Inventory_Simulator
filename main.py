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

st.markdown(' ')

col0, col1 = st.columns([8,2])
with col0:
    st.markdown('# Inventory Simulator')
    st.markdown('## 재고관리 시뮬레이션 및 최적화 플랫폼')

with col1:
    st.markdown(' ')
    st.image('./images/gs칼텍스 로고.png')

st.markdown('-'*50)


col0, col1, col2 = st.columns([1,1,1])

with st.expander(rf'''###### :pencil: **재고 산정 방법**''', expanded=False):
    
    st.markdown('### 재고 비용')
    st.markdown('- 재고 비용은 익일 재고 수준에 기반하여 계산됩니다.')
    st.markdown('- 재고량(CS)에 재고 보유 비율(α%)을 곱하여 계산됩니다.')
    st.markdown(r'''$$Inventory \, Cost_t = CS_t \times \alpha$$''')
    st.markdown('-'*50)

    st.markdown('### 백로그 비용')
    st.markdown('- 백로그 비용은 주문을 제공하지 못하는 상황에서 발생합니다.')
    st.markdown('- 재고가 음수로 내려가는 상황이 지속되면, 지수적으로 비용이 증가합니다.')
    st.markdown(r'''$$Backlog \, Cost_t = max(0, D_t - (CS_t + S_t)) \times \beta \times exp(\lambda \times \tau_t)$$''')
    st.markdown('-'*50)

    st.markdown('### 주문비용')
    st.markdown('- 주문 비용은 새로운 재고를 주문하거나 공급받는 데 발생하는 비용입니다.')
    st.markdown('- Economic Order Quantity(EOQ)를 기반으로 계산되며, 고정 및 가변 비용을 포함합니다.')
    st.markdown(r'''$$Ordering \, Cost_t = EOQ \times \gamma + \delta$$''')
    st.markdown('-'*50)

    st.markdown('### 재고 파라미터 설정 예시')
    alpha = st.slider("재고 비용 계수 (α)", 0.01, 0.5, 0.1, 0.01)
    beta = st.slider("백로그 비용 계수 (β)", 0.1, 5.0, 1.0, 0.1)
    lambda_param = st.slider("백로그 지연 민감도 (λ)", 0.1, 2.0, 0.5, 0.1)
    gamma = st.slider("주문 비용 계수 (γ)", 1.0, 20.0, 5.0, 1.0)
    delta = st.slider("주문 고정 비용 (δ)", 50, 500, 100, 10)

    # 데이터 생성
    x = np.linspace(0, 100, 100)  # 재고량, 수요량, 주문량 등

    # 비용 계산
    inventory_cost = x * alpha  # 재고 비용 (선형)
    backlog_cost = np.maximum(0, x - 50) * beta * np.exp(lambda_param * (x / 50))  # 백로그 비용 (지수)
    ordering_cost = (x / 10) * gamma + delta  # 주문 비용 (선형 증가 + 고정 비용)

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, inventory_cost, label=f"재고 비용 (선형)", color='blue')
    ax.plot(x, backlog_cost, label=f"백로그 비용 (지수 증가)", color='red')
    ax.plot(x, ordering_cost, label=f"주문 비용 (선형 증가)", color='green')
    ax.set_xlabel("재고량 또는 수요량", fontsize=12)
    ax.set_ylabel("비용", fontsize=12)
    ax.set_title("비용 구성 요소 시각화", fontsize=14)
    ax.legend()
    ax.grid()

    # Streamlit에 그래프 표시
    st.pyplot(fig)

    # 동적 설명
    st.write(f"### 파라미터 설정")
    st.write(f"- **재고 비용 계수 (α)**: {alpha}")
    st.write(f"- **백로그 비용 계수 (β)**: {beta}, **백로그 민감도 (λ)**: {lambda_param}")
    st.write(f"- **주문 비용 계수 (γ)**: {gamma}, **고정 주문 비용 (δ)**: {delta}")
    st.write("---")

with st.expander('###### :bookmark_tabs: **시뮬레이터 파라미터 설정 방법** '):
    st.markdown('-'*50)
    st.markdown('### 시뮬레이터를 위한 분포 추론')
    st.markdown('- **정규분포**: 주문량 생성에 활용됩니다.')
    st.markdown(r'''$$D_t \sim N(\mu_D, \sigma_D)$$''')
    st.markdown('- **포아송분포**: 주문 간격 생성에 활용됩니다.')
    st.markdown(r'''$$P_t \sim Poisson(\lambda)$$''')
    st.markdown('-'*50)

    st.markdown("분포 모수 설정")
    mu_D = st.slider("정규분포 평균 (μ_D)", 50, 200, 100, 10)
    sigma_D = st.slider("정규분포 표준편차 (σ_D)", 5, 50, 10, 5)
    lambda_param = st.slider("포아송분포 평균 발생률 (λ)", 0.1, 2.0, 1.0, 0.1)

    # 데이터 생성
    np.random.seed(42)  # 결과 재현성을 위해 고정
    num_samples = 100  # 시뮬레이션 데이터 포인트 개수

    # 정규분포에 기반한 주문량 데이터 생성
    demand = np.random.normal(loc=mu_D, scale=sigma_D, size=num_samples)
    demand = np.maximum(demand, 0)  # 음수 값 제거 (재고량은 0 이상이어야 함)

    # 포아송분포에 기반한 주문 간격 데이터 생성
    order_occurrence = np.random.poisson(lam=lambda_param, size=100)
    order_occurrence = np.minimum(order_occurrence, 1)  # 포아송 결과를 0 또는 1로 변환


    # 시뮬레이션 결과 시각화
    fig, ax = plt.subplots(2, 1, figsize=(20, 12))

    # 주문량 시뮬레이션 결과
    ax[0].plot(range(num_samples), demand, marker='o', label="주문량 (정규분포 기반)")
    ax[0].set_title("주문량 시뮬레이션 결과", fontsize=14)
    ax[0].set_xlabel("일", fontsize=12)
    ax[0].set_ylabel("일(Day)", fontsize=12)
    ax[0].legend()
    ax[0].grid()

    # 주문 간격 시뮬레이션 결과
    ax[1].bar(range(100), order_occurrence, label="주문 발생 여부 (0: 없음, 1: 발생)", color="orange")
    ax[1].set_title("주문 발생 여부 (100일)", fontsize=14)
    ax[1].set_xlabel("날짜", fontsize=12)
    ax[1].set_ylabel("발생 여부", fontsize=12)
    ax[1].legend()
    ax[1].grid()

    # Streamlit에서 그래프 출력
    st.pyplot(fig)


with st.expander(rf''' ###### :dart: **재고 최소화를 위한 목적함수**'''):    
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

    
