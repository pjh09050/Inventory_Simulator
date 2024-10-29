import streamlit as st
import yaml
import pandas as pd
from datetime import datetime
from function import *
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
        font-size: 1.5rem; /* Global font size adjustment */
    }
    h1 {
        font-size: 2rem; /* Increase title size */
    }
    h2, h3 {
        font-size: 1.5rem; /* Increase subheader size */
    }
    .stButton button {
        font-size: 1.2rem; /* Button font size */
    }
    </style>
    """, unsafe_allow_html=True)

# Title and header
st.subheader('Parameter settings')

# 체크박스를 사용하여 파라미터 설정
set_param = st.checkbox('Set & Save parameters')

if set_param:
    data_path = st.text_input("Data Path", './data/project_mb51_df.pickle')
    meta_path = st.text_input("Meta Path", './data/project_zwms03s_df.pickle')
    target = st.text_input("Target", '61-200161000000')

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_date = st.date_input("Start Date", datetime(2022, 12, 30))
        initial_stock = st.number_input("Initial Stock", value=34.0, step=0.01, format="%.2f")
    with col2:
        end_date = st.date_input("End Date", datetime(2023, 12, 30))
        safety_stock = st.number_input("Safety Stock", value=17.096, step=0.01, format="%.2f")
    with col3:
        run_start_date = st.date_input("Run Start Date", datetime(2024, 1, 1))
        maintenance_mu = st.number_input("Maintenance Mean (mu)", value=70.0, step=0.01, format="%.2f")
        # reorder_point = st.number_input("Reorder Point", value=32.572, step=0.01, format="%.2f")
    with col4:
        run_end_date = st.date_input("Run End Date", datetime(2024, 1, 30))
        maintenance_std = st.number_input("Maintenance Std Dev (std)", value=13.0, step=0.01, format="%.2f")

    save_config = {
        'data_path': data_path,
        'meta_path': os.path.abspath(meta_path),
        'target': target,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'run_start_date': run_start_date.strftime('%Y-%m-%d'),
        'run_end_date': run_end_date.strftime('%Y-%m-%d'),
        'safety_stock': safety_stock,
        'initial_stock': initial_stock,
        # 'reorder_point': reorder_point,
        'maintenance_mu': maintenance_mu,
        'maintenance_std': maintenance_std
    }

    save_filename = st.text_input("Save File Name", "param_set.yaml")
    if st.button('Save parameters'):
        try:
            if not save_filename.endswith('.yaml'):
                save_filename += '.yaml'

            save_config.update({
                'data_path': data_path,
                'meta_path': meta_path,
                'target': target,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'run_start_date': run_start_date.strftime('%Y-%m-%d'),
                'run_end_date': run_end_date.strftime('%Y-%m-%d'),
                'safety_stock': safety_stock,
                'initial_stock': initial_stock,
                # 'reorder_point': reorder_point,
                'maintenance_mu': maintenance_mu,
                'maintenance_std': maintenance_std
            })
            st.session_state['meta_dict'] = load_meta_info(meta_path)
            save_config.update(st.session_state['meta_dict'][target])
            with open(save_filename, 'w', encoding='utf-8') as file:
                yaml.dump(save_config, file, default_flow_style=False, sort_keys=False, allow_unicode=True)
            st.success(f"Configuration saved to '{save_filename}'")
        except Exception as e:
            st.error(f"Error saving configuration: {e}")

################################################################################################
st.subheader('Run simulation')
# st.markdown("---")
# 초기 상태 설정
if 'run_simulation_started' not in st.session_state:
    st.session_state['run_simulation_started'] = False
if 'parameters_loaded' not in st.session_state:
    st.session_state['parameters_loaded'] = False
if 'EOQ_simulation_started' not in st.session_state:
    st.session_state['EOQ_simulation_started'] = False

# yaml 파일 로드
if st.checkbox('Load parameters'):
    try:
        yaml_files = [f for f in os.listdir() if f.endswith('.yaml') and f != 'meta_info.yaml']
        if yaml_files:
            load_filename = st.selectbox('Select a YAML file to load', yaml_files)
            if st.button('Load parameters(yaml 파일)'):
                with open(load_filename, 'r', encoding='utf-8') as file:
                    config = yaml.safe_load(file)
                st.session_state['config'] = config
                st.markdown(f'{load_filename} 파라미터 로드 완료')
                st.session_state['parameters_loaded'] = True
        else:
            st.write("No YAML files found.")
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
    if 'config' in st.session_state:
        if st.checkbox('Show parameters', key='show_param', value=False):
            st.write(st.session_state['config'])
else:
    st.session_state['parameters_loaded'] = False

################################################################################################
if st.session_state['parameters_loaded']:
    if st.checkbox('Run simulation'):
        st.session_state['run_simulation_started'] = True
        if 'config' in st.session_state:
            config = st.session_state['config']
            data_dict = eda(config['data_path'])
            target = config['target']
            start_date = find_closest_date(pd.to_datetime(config['start_date']), data_dict[target], '날짜')
            end_date = pd.to_datetime(config['end_date'])
            run_start_date = pd.to_datetime(config['run_start_date'])
            run_end_date = pd.to_datetime(config['run_end_date'])
            safety_stock = config['safety_stock']
            data_dict[target]['날짜'] = pd.to_datetime(data_dict[target]['날짜'])
            initial_stock = data_dict[target].groupby(['날짜']).sum(numeric_only=True)['수량'].loc[start_date]
            lead_time_mu = config['납기(일)-Average Lead Time (Days) /Max/Min.1']
            lead_time_std = config['Lead Time Standard Deviation (Days)']
            maintenance_mu = config['maintenance_mu']
            maintenance_std = config['maintenance_std']
            item_name = config['품명']
            # 데이터를 session_state에 저장
            st.session_state['data_dict'] = data_dict
            st.session_state['target'] = target
            st.session_state['start_date'] = start_date
            st.session_state['end_date'] = end_date
            st.session_state['run_start_date'] = run_start_date
            st.session_state['run_end_date'] = run_end_date
            st.session_state['safety_stock'] = safety_stock
            st.session_state['initial_stock'] = initial_stock
            st.session_state['lead_time_mu'] = lead_time_mu
            st.session_state['lead_time_std'] = lead_time_std
            st.session_state['maintenance_mu'] = maintenance_mu
            st.session_state['maintenance_std'] = maintenance_std
            st.session_state["item_name"] = item_name

            # 시뮬레이션이 시작된 후에만 라디오 버튼이 표시되도록 조건 추가
            col10, col11, col12, col13 = st.columns(4)
            with col10:
                simulation_type = st.radio("Choose Simulation Type", ('EOQ 시뮬레이션', '분포 시뮬레이션'))

            if simulation_type == 'EOQ 시뮬레이션':
                simulation_type = 'optimal'
                with col11:
                    EOQ = st.number_input("EOQ", value=0, step=1, format="%d")
                with col12:
                    SS = st.number_input("Safety Stock", value=0, step=1, format="%d")
                if st.checkbox('EOQ 시뮬레이션 시작'):
                    st.session_state['EOQ_simulation_started'] = True
                    status_text = st.empty()
                    status_text.write("EOQ 시뮬레이션 실행 중...")
                    warmup_stock_levels_df, warmup_pending_orders_df, warmup_order_dates, warmup_arrival_dates, warmup_rop_values, warmup_dates = warmup_simulator(
                        EOQ, SS, st.session_state['data_dict'], st.session_state['target'], st.session_state['initial_stock'],
                        st.session_state['lead_time_mu'], st.session_state['lead_time_std'], st.session_state['start_date'], st.session_state['end_date'], 
                        st.session_state['run_start_date'], simulation_type
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
                    stock_levels_df_result, pending_orders_result, orders_df_result, rop_values_result, dates, simulation_info = run_simulation(
                        EOQ, SS, st.session_state['data_dict'], st.session_state['target'], st.session_state['initial_stock'], st.session_state['start_date'], st.session_state['end_date'], 
                        st.session_state['run_start_date'], st.session_state['run_end_date'], st.session_state['lead_time_mu'], st.session_state['lead_time_std'], st.session_state['order_dates'], 
                        st.session_state['order_values'], st.session_state['arrival_dates'], st.session_state['pending_orders'], simulation_type
                    )
                    st.session_state['stock_levels_df_result'] = stock_levels_df_result
                    st.session_state['pending_orders_result'] = pending_orders_result
                    st.session_state['orders_df_result'] = orders_df_result
                    st.session_state['rop_values_result'] = rop_values_result
                    st.session_state['dates'] = dates
                    time.sleep(0.5)
                    fig = plot_inventory_simulation(st.session_state['dates'], SS, st.session_state['rop_values_result'], st.session_state['stock_levels_df_result'], 
                                    st.session_state['orders_df_result'], st.session_state["target"], st.session_state["initial_stock"])
                    st.plotly_chart(fig)
                    status_text.write("EOQ 시뮬레이션 실행 완료")
                    if st.button("End simulation"):
                        keys_to_preserve = ['parameters_loaded']
                        for key in list(st.session_state.keys()):
                            if key not in keys_to_preserve:
                                del st.session_state[key]
                        st.session_state['run_simulation_started'] = False
                        st.session_state['parameters_loaded'] = False
                        st.session_state['EOQ_simulation_started'] = False
                        st.success("Simulation reset! Please reload parameters to start again!")
                        time.sleep(1.5)
                        st.rerun()
                    if st.checkbox('Simulation Information'):
                        st.markdown(f"{simulation_info}")

            elif simulation_type == '분포 시뮬레이션':
                status_text = st.empty()
                
                simulation_type = 'distribution'
                warmup_stock_levels_df, warmup_pending_orders_df, warmup_order_dates, warmup_arrival_dates, warmup_rop_values, warmup_dates = warmup_simulator(
                    None, st.session_state['safety_stock'], st.session_state['data_dict'], st.session_state['target'], st.session_state['initial_stock'],
                    st.session_state['lead_time_mu'], st.session_state['lead_time_std'], st.session_state['start_date'], st.session_state['end_date'], 
                    st.session_state['run_start_date'], simulation_type
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
                stock_levels_df_result, pending_orders_result, orders_df_result, rop_values_result, dates, simulation_info = run_simulation(
                    None, st.session_state['safety_stock'], st.session_state['data_dict'], st.session_state['target'], st.session_state['initial_stock'], st.session_state['start_date'], st.session_state['end_date'], 
                    st.session_state['run_start_date'], st.session_state['run_end_date'], st.session_state['lead_time_mu'], st.session_state['lead_time_std'], st.session_state['order_dates'], 
                    st.session_state['order_values'], st.session_state['arrival_dates'], st.session_state['pending_orders'], simulation_type
                )
                st.session_state['stock_levels_df_result'] = stock_levels_df_result
                st.session_state['pending_orders_result'] = pending_orders_result
                st.session_state['orders_df_result'] = orders_df_result
                st.session_state['rop_values_result'] = rop_values_result
                st.session_state['dates'] = dates
                if st.checkbox('분포 시뮬레이션 시작'):
                    status_text.write("분포 시뮬레이션 실행 중...")
                    time.sleep(0.5)
                    fig = plot_inventory_simulation(st.session_state['dates'], st.session_state['safety_stock'], st.session_state['rop_values_result'], st.session_state['stock_levels_df_result'], 
                                        st.session_state['orders_df_result'], st.session_state["target"], st.session_state["initial_stock"])
                    st.plotly_chart(fig)
                    status_text.write("분포 시뮬레이션 실행 완료")
                    if st.button("End simulation"):
                        for key in list(st.session_state.keys()):
                            if key != 'parameters_loaded':
                                del st.session_state[key]
                        st.session_state['run_simulation_started'] = False
                        st.session_state['parameters_loaded'] = False
                        st.success("Simulation reset! Please reload parameters to start again!")
                        time.sleep(1.5)
                        st.rerun()
                    if st.checkbox('Simulation Information'):
                        st.markdown(f"{simulation_info}")
        else:
            st.markdown('yaml 파일을 load 해주세요.')
    else:
        st.session_state['run_simulation_started'] = False

################################################################################################