import streamlit as st
import yaml
import pandas as pd
from datetime import datetime
from function import *
import time
import threading
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
        font-size: 1.5rem; /* Global font size adjustment */
    }
    h1 {
        font-size: 2rem; /* Increase title size */
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
st.subheader('Parameter settings')

# 초기 세팅
if 'parameters_loaded' not in st.session_state:
    st.session_state['parameters_loaded'] = False

# yaml 파일 로드
if st.checkbox('Load parameters'):
    try:
        yaml_files = [f for f in os.listdir() if f.endswith('.yaml') and f != 'meta_info.yaml']
        if yaml_files:
            st.session_state['load_filename'] = st.selectbox('Select a YAML file to load', yaml_files)
            if st.button('Load parameters(yaml 파일)'):
                with open(st.session_state['load_filename'], 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                st.session_state['config'] = config
                st.success(f'{st.session_state["load_filename"]} 파라미터 로드 완료')
                st.session_state['parameters_loaded'] = True
        else:
            st.write("No YAML files found.")
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
    if 'config' in st.session_state:
        if st.checkbox('Show parameters', key='show_param', value=False):
            st.write(st.session_state['config'])

if st.session_state['parameters_loaded'] and 'config' in st.session_state:
    if 'config' in st.session_state:
        config = st.session_state['config']
        st.session_state['data_dict'] = eda(config['data_path'])
        st.session_state['target'] = config['target']
        st.session_state['start_date'] = find_closest_date(pd.to_datetime(config['start_date']), st.session_state['data_dict'][st.session_state['target']], '날짜')
        st.session_state['end_date'] = pd.to_datetime(config['end_date'])
        st.session_state['run_start_date'] = pd.to_datetime(config['run_start_date'])
        st.session_state['run_end_date'] = pd.to_datetime(config['run_end_date'])
        st.session_state['safety_stock'] = config['safety_stock']
        st.session_state['data_dict'][st.session_state['target']]['날짜'] = pd.to_datetime(st.session_state['data_dict'][st.session_state['target']]['날짜'])
        st.session_state['initial_stock'] = st.session_state['data_dict'][st.session_state['target']].groupby(['날짜']).sum(numeric_only=True)['수량'].loc[st.session_state['start_date']]
        st.session_state['lead_time_mu'] = config['납기(일)-Average Lead Time (Days) /Max/Min.1']
        st.session_state['lead_time_std'] = config['Lead Time Standard Deviation (Days)']
        st.session_state['maintenance_mu'] = config['maintenance_mu']
        st.session_state['maintenance_std'] = config['maintenance_std']
        st.session_state["item_name"] = config['품명']

st.subheader('Cost Optimization')

if not st.session_state.get('parameters_loaded', False):
    st.warning("⚠️ 파라미터가 로드되지 않았습니다. 파라미터를 로드해 주세요.")
else:
    with st.expander("변수 설명 및 현재 값", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            - **target**: 시뮬레이션할 재고의 대상 ID
                - {st.session_state['target']}
            - **name**: 시뮬레이션할 재고 이름
                - {st.session_state['item_name']}            
            - **start_date**: 분포 참조 기간의 시작 날짜
                - {st.session_state['start_date'].date()}
            - **end_date**: 분포 참조 기간의 종료 날짜
                - {st.session_state['end_date'].date()}
            - **run_start_date**: 실제 시뮬레이션 실행 시작 날짜
                - {st.session_state['run_start_date'].date()}
            - **run_end_date**: 실제 시뮬레이션 실행 종료 날짜
                - {st.session_state['run_end_date'].date()}
            """)
        with col2:
            st.markdown(f"""
            - **initial_stock**: 초기 재고 수량
                - {st.session_state['initial_stock']}
            - **safety_stock**: 안전 재고량
                - {st.session_state['safety_stock']}
            - **lead_time_mu**: 리드 타임의 평균
                - {st.session_state['lead_time_mu']}
            - **lead_time_std**: 리드 타임의 표준 편차
                - {st.session_state['lead_time_std']}
            - **maintenance_mu**: 유지보수 주기의 평균값
                - {st.session_state['maintenance_mu']}
            - **maintenance_std**: 유지보수 주기의 표준 편차
                - {st.session_state['maintenance_std']}
            """)

    if st.checkbox('GA 파라미터 세팅'):
        st.markdown("##### EOQ와 SS 범위 설정")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state["EOQ_LOW"] = st.slider("EOQ_LOW", min_value=0, max_value=100, value=10, step=1)
            st.session_state["EOQ_HIGH"] = st.slider("EOQ_HIGH", min_value=0, max_value=200, value=50, step=1)
        with col2:
            st.session_state["SS_LOW"] = st.slider("SS_LOW", min_value=0, max_value=100, value=10, step=1)
            st.session_state["SS_HIGH"] = st.slider("SS_HIGH", min_value=0, max_value=100, value=50, step=1)

        st.markdown("##### 비용 파라미터 설정")
        col3, col4, col5, col6, col7 = st.columns(5)
        with col3:
            st.session_state["alpha"] = st.number_input("alpha", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        with col4:
            st.session_state["beta"] = st.number_input("beta", min_value=0, max_value=100000, value=50000, step=1000)
        with col5:
            st.session_state["gamma"] = st.number_input("gamma", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
        with col6:
            st.session_state["delta"] = st.number_input("delta", min_value=0, max_value=1000000, value=700000, step=10000)
        with col7:
            st.session_state["lambda_param"] = st.number_input("lambda_param", min_value=0, max_value=10, value=3, step=1)

        st.markdown("##### Genetic Algorithm 파라미터 설정")
        col8, col9 = st.columns(2)
        with col8:
            st.session_state["population_size"] = st.number_input("초기 개체 수 (population size)", min_value=1, max_value=100, value=15, step=1)
            st.session_state["NGEN"] = st.number_input("세대 수 (NGEN)", min_value=1, max_value=1000, value=50, step=1)
        with col9:
            st.session_state["CXPB"] = st.slider("교차 확률 (CXPB)", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
            st.session_state["MUTPB"] = st.slider("변이 확률 (MUTPB)", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        
        if 'meta_dict' in st.session_state:
            meta_dict = st.session_state['meta_dict']
        else:
            st.error("Meta dict 변수가 선언되어있지 않습니다. 경로를 다시 정의해주세요")
            meta_path = st.text_input("Meta Path", './data/project_zwms03s_df.pickle')
            
            if st.button("Load Meta Dict"):
                st.session_state['meta_dict'] = load_meta_info(meta_path)
                meta_dict = st.session_state['meta_dict']
                st.success("Meta dict loaded successfully!")

#############################################################################################################
    if 'optimize_checkbox' not in st.session_state:
        st.session_state['optimize_checkbox'] = False
    if 'optimization_complete' not in st.session_state:
        st.session_state['optimization_complete'] = False

    optimize_checkbox = st.checkbox("최적화 실행", key='optimize_checkbox')
    if optimize_checkbox and not st.session_state['optimization_complete']:
        time.sleep(0.2)
        st.session_state['type'] = "optimal"
        if 'meta_dict' in st.session_state:
            status_text = st.empty()
            status_text.markdown("<h3 style='color: red;'>최적화 진행 중...</h3>", unsafe_allow_html=True)

            warmup_stock_levels_df, warmup_pending_orders_df, warmup_order_dates, warmup_arrival_dates, warmup_rop_values, warmup_dates = warmup_simulator(
                None, st.session_state['safety_stock'], st.session_state['data_dict'], st.session_state['target'], st.session_state['initial_stock'],
                st.session_state['lead_time_mu'], st.session_state['lead_time_std'], st.session_state['start_date'], st.session_state['end_date'], 
                st.session_state['run_start_date'], type="optimal"
            )

            if not warmup_pending_orders_df.empty and 'arrival_date' in warmup_pending_orders_df.columns:
                st.session_state['order_dates'] = warmup_pending_orders_df.groupby(['arrival_date']).sum().reset_index()['arrival_date'].tolist()
                st.session_state['order_values'] = warmup_pending_orders_df.groupby(['arrival_date']).sum().reset_index()['quantity'].tolist()
                st.session_state['arrival_dates'] = warmup_pending_orders_df.groupby(['arrival_date']).sum().reset_index()['arrival_date'].tolist()
                st.session_state['pending_orders'] = warmup_pending_orders_df.groupby(['arrival_date']).sum().reset_index()
            else:
                st.session_state['order_dates'] = []
                st.session_state['order_values'] = []
                st.session_state['arrival_dates'] = []
                st.session_state['pending_orders'] = pd.DataFrame() 

            st.session_state["ga_result"], st.session_state['minus_value'] = run_genetic_algorithm(
                data_dict=st.session_state['data_dict'],
                meta_dict=st.session_state['meta_dict'],
                target=st.session_state['target'],
                initial_stock=st.session_state['initial_stock'],
                start_date=st.session_state['start_date'],
                end_date=st.session_state['end_date'],
                run_start_date=st.session_state['run_start_date'],
                run_end_date=st.session_state['run_end_date'],
                type=st.session_state['type'],
                lead_time_mu=st.session_state['lead_time_mu'],
                lead_time_std=st.session_state['lead_time_std'],
                order_dates=st.session_state['order_dates'],
                order_values=st.session_state['order_values'],
                arrival_dates=st.session_state['arrival_dates'],
                pending_orders=st.session_state['pending_orders'],
                EOQ_LOW=st.session_state['EOQ_LOW'],
                EOQ_HIGH=st.session_state['EOQ_HIGH'],
                SS_LOW=st.session_state['SS_LOW'],
                SS_HIGH=st.session_state['SS_HIGH'],
                alpha=st.session_state['alpha'],
                beta=st.session_state['beta'],
                gamma=st.session_state['gamma'],
                delta=st.session_state['delta'],
                lambda_param=st.session_state['lambda_param'],
                population_size=st.session_state['population_size'],
                ngen=st.session_state['NGEN'],
                cxpb=st.session_state['CXPB'],
                mutpb=st.session_state['MUTPB'],
                elitism_percent=st.session_state.get('elitism_percent', 0.02)
            )
            st.session_state["optimization_complete"] = True
            status_text.markdown("<h3 style='color: green;'>최적화 완료</h3>", unsafe_allow_html=True)

    if st.session_state.get('optimization_complete', False) and optimize_checkbox:
        if 'ga_result' in st.session_state:
            st.write("다음은 모든 EOQ와 SS 조합입니다:")
            options = [f"EOQ = {eoq}, SS = {ss}" for eoq, ss in st.session_state['ga_result']]
            selected_option = st.selectbox("최적의 조합을 선택하세요:", options)

            selected_index = options.index(selected_option)
            st.session_state["selected_eoq"], st.session_state["selected_ss"] = st.session_state['ga_result'][selected_index]
            st.markdown(f"""
                <div style="text-align: center; font-size: 24px; color: #333;">
                    <span style="font-weight: bold; color: #007BFF;">선택한 EOQ:</span> {st.session_state["selected_eoq"]}<br>
                    <span style="font-weight: bold; color: #FF6347;">선택한 SS:</span> {st.session_state["selected_ss"]}
                </div>
            """, unsafe_allow_html=True)

        if st.checkbox("그래프 보기") and st.session_state.get("optimization_complete", False):
            st.session_state['type'] = "optimal"
            stock_levels_df_result, pending_orders_result, orders_df_result, rop_values_result, dates, st.session_state['total_cost_value'] = total_cost_result(
                st.session_state["selected_eoq"], 
                st.session_state["selected_ss"], 
                st.session_state['data_dict'], 
                st.session_state['meta_dict'],
                st.session_state['target'], 
                st.session_state['initial_stock'], 
                st.session_state['start_date'], 
                st.session_state['end_date'], 
                st.session_state['run_start_date'], 
                st.session_state['run_end_date'], 
                st.session_state['type'], 
                st.session_state['lead_time_mu'], 
                st.session_state['lead_time_std'], 
                st.session_state['order_dates'], 
                st.session_state['order_values'], 
                st.session_state['arrival_dates'], 
                st.session_state['pending_orders'], 
                st.session_state['alpha'], 
                st.session_state['beta'], 
                st.session_state['gamma'], 
                st.session_state['delta'], 
                st.session_state['lambda_param'],
                st.session_state['minus_value']
            )
            st.session_state['stock_levels_df_result'] = stock_levels_df_result
            st.session_state['pending_orders_result'] = pending_orders_result
            st.session_state['orders_df_result'] = orders_df_result
            st.session_state['rop_values_result'] = rop_values_result
            st.session_state['dates'] = dates
            time.sleep(0.5)
            st.markdown(f"<h4 style='color: black;'>자재명: {st.session_state['item_name']}</h3>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style="display: flex; justify-content: flex-start; align-items: center;">
                    <h4 style="color: blue;">EOQ: {st.session_state['selected_eoq']}, Safety Stock: {st.session_state['selected_ss']}</h4>
                    <h4 style="color: green; margin-right: 10px;">총 비용: {st.session_state['total_cost_value']:,}</h4>
                </div>
                """, 
                unsafe_allow_html=True
            )
            fig = plot_inventory_simulation(st.session_state['dates'], st.session_state['selected_ss'], st.session_state['rop_values_result'], st.session_state['stock_levels_df_result'], 
                                st.session_state['orders_df_result'], st.session_state["target"], st.session_state["initial_stock"])
            st.plotly_chart(fig)
            minus_df_display = st.session_state['minus_value'].copy().reset_index()
            minus_df_display['날짜'] = minus_df_display['날짜'].dt.strftime('%Y-%m-%d')
            with st.expander("출고 정보 보기", expanded=False):
                st.markdown(
                    f"<div style='display: flex; font-size: 20px; font-weight: bold; border-bottom: 2px solid #e0e0e0; padding-bottom: 1px; margin-bottom: 5px;'>"
                    f"<div style='flex: 1; padding: 1px;'>날짜</div>"
                    f"<div style='flex: 10; padding: 1px;'>수량</div>"
                    f"</div>", unsafe_allow_html=True
                )
                for idx, row in minus_df_display.iterrows():
                    st.markdown(
                        f"<div style='display: flex; font-size: 18px; border-bottom: 0.5px solid #e0e0e0; padding: 5px;'>"
                        f"<div style='flex: 1; padding: 1px;'>{row['날짜']}</div>"
                        f"<div style='flex: 10; padding: 1px;'>{row['수량']}</div>"
                        f"</div>", unsafe_allow_html=True
                )
            new_filename = st.text_input("결과를 저장할 파일 이름:", f"{st.session_state['load_filename'].split('.')[0]}_{st.session_state['selected_eoq']}_{st.session_state['selected_ss']}.yaml")
            if st.button("결과 저장"):
                try:
                    with open(st.session_state['load_filename'], 'r', encoding='utf-8') as file:
                        original_data = ordered_load(file)
                    new_results = OrderedDict({
                        "EOQ": int(st.session_state.get("selected_eoq")),
                        "Optimal_Safety_Stock": int(st.session_state.get("selected_ss")),
                        "Cost": float(st.session_state.get("total_cost_value"))
                    })
                    combined_data = OrderedDict(list(original_data.items()) + list(new_results.items()))
                    with open(new_filename, "w", encoding="utf-8") as file:
                        ordered_dump(combined_data, file, allow_unicode=True, default_flow_style=False)
                    st.success(f"Results saved to {new_filename}")
                except Exception as e:
                    st.error(f"Error saving results: {e}")
    else:
        st.warning("최적화가 실행되지 않았습니다. '최적화 실행'을 먼저 실행시켜주세요.")
        
    # 체크박스를 끄면 최적화가 다시 필요하게 설정
    if not optimize_checkbox:
        st.session_state['optimization_complete'] = False