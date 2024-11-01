import pandas as pd
import os
import yaml
import time
import streamlit as st
import scipy.stats as stats
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import random
import math
from collections import OrderedDict
from deap import base, creator, tools, algorithms

def load_meta_info(meta_path):
    df_meta_info = pd.read_pickle(meta_path)
    meta_value = df_meta_info[['자재','품명', '이동평균가', 
                            '납기(일)-Average Lead Time (Days) /Max/Min.1','Lead Time Standard Deviation (Days)',
                                'Safety Stock 1', 'Reorder Point 1', 'Safety Stock 2', 'Reorder Point 2',
                                'Safety Stock 3', 'Reorder Point 3', 'Safety Stock 4',
                                'Reorder Point 4', 'Safety Stock 5', 'Reorder Point 5',
                                'Safety Stock 6', 'Reorder Point 6', 'Recommeded Safety Stock',
                                'Recommended Reorder Point']]
    meta_value.index = meta_value['자재']
    meta_value.drop(['자재'], axis=1, inplace=True)
    meta_value = meta_value.T
    meta_dict = meta_value.to_dict()
    with open('meta_info.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(meta_dict, file, default_flow_style=False, sort_keys=False, allow_unicode=True)
    with open('meta_info.yaml', 'r', encoding='utf-8') as file:
        meta_dict = yaml.safe_load(file)
    return meta_dict

def find_closest_date(input_date, df, date_column):
    input_date = pd.to_datetime(input_date)
    df[date_column] = pd.to_datetime(df[date_column])
    df['diff'] = (df[date_column] - input_date).abs()
    closest_date = df.loc[df['diff'].idxmin(), date_column]
    df.drop(columns=['diff'], inplace=True)
    return closest_date

def check_or_create_yaml(filename, default_config):
    if not os.path.exists(filename):
        # 파일이 없으면 생성
        with open(filename, 'w') as file:
            yaml.dump(default_config, file, default_flow_style=False, sort_keys=False, allow_unicode=True)
        st.write(f"'{filename}' 파일이 생성되었습니다.")
    else:
        st.write(f"'{filename}' 파일이 이미 존재합니다.")

def moving_order_history(df, start_date, end_date):
    df['날짜'] = pd.to_datetime(df['날짜'])
    # 주어진 기간 동안의 데이터를 필터링
    df1 = df[(df['날짜'] >= start_date) & (df['날짜'] < end_date)]
    minus_df = df1[df1['입력단위수량'] < 0]
    if minus_df.empty:
        print('수요가 없는 기간입니다.')
        # df1 = df[(df['날짜'] < start_date)]
        # minus_df = df1[df1['입력단위수량'] < 0]
    # 포아송 분포 파라미터 λ 추정
    start_date = minus_df['날짜'].min()
    end_date = minus_df['날짜'].max()
    poisson_data = minus_df
    # 날짜 범위 생성
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    # '입력단위수량' 값이 유효한 행만 선택
    valid_entries = poisson_data.dropna(subset=['입력단위수량'])
    # 포아송 분포 파라미터 λ 추정 (일별 사건 수)
    valid_entries['날짜'] = pd.to_datetime(valid_entries['날짜'])
    events_per_day = valid_entries.groupby(valid_entries['날짜'].dt.date).size().reindex(date_range.date, fill_value=0)
    lambda_value = events_per_day.mean()
    # '입력단위수량' 값 추출하여 정규분포 파라미터(μ, σ) 추정
    input_quantity = minus_df['수량']
    mu, std = stats.norm.fit(input_quantity)
    return lambda_value, mu, std

def new_order(lambda_value, mu, std, start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    daily_orders = np.random.poisson(lambda_value, len(date_range))
    order_dates = []
    order_values = []
    for date, num_orders in zip(date_range, daily_orders):
        for _ in range(num_orders):
            order_dates.append(date)
            order_value = int(np.random.normal(mu, std))
            while order_value >= 0:
                order_value = int(np.random.normal(mu, std))
            order_values.append(order_value)
    new_order_df = pd.DataFrame({'날짜': order_dates,'수량': order_values})
    new_order_df.set_index('날짜', inplace=True)
    return new_order_df

def moving_store_history(df, start_date, end_date):
    plus_df = df[(df['날짜'] >= start_date) & (df['날짜'] < end_date) & (df['입력단위수량'] > 0)]
    if plus_df.empty:
        plus_df = df[(df['날짜'] < start_date) & (df['입력단위수량'] > 0)]
        input_quantity = abs(plus_df['수량'])
    # '입력단위수량' 값 추출
    input_quantity = abs(plus_df['수량'])
    # 정규분포 파라미터(평균, 표준편차) 추정
    if not input_quantity.empty:
        mu, std = stats.norm.fit(input_quantity)
    else:
        print('입고가 없는 기간입니다.')
    return mu, std

def find_closest_index(df, base_idx, candidate_indices, drop_index_list):
    """base_idx와 시간 차이가 가장 적은 candidate_indices 중 하나를 찾는 함수"""
    base_time = df.loc[base_idx, 'DateTime']
    min_time_diff = None
    closest_idx = None

    for idx in candidate_indices:
        time_diff = abs(df.loc[idx, 'DateTime'] - base_time)
        if min_time_diff is None or time_diff < min_time_diff:
            if idx not in drop_index_list:
                min_time_diff = time_diff
                closest_idx = idx
    return closest_idx

def eda(df):
    mb = pd.read_pickle(df)
    mb['시간'] = mb['시간'].astype(str)
    mb['DateTime'] = pd.to_datetime(mb['증빙일'] + ' ' + mb['시간'])
    mb['날짜'] = mb['DateTime'].dt.date
    mb['수량'] = mb['수량'].astype(float)
    null_data_index = np.where(mb.isnull().sum(axis=1) > mb.columns.size/2)[0]
    
    data_name_check = mb.loc[null_data_index]
    data_name  = data_name_check['SLoc']
    data_dict = {}
    check_dd = pd.DataFrame()

    for i, k in enumerate(data_name):
        if i == len(null_data_index) - 1:
            input_data_mb = mb.loc[null_data_index[i] + 1:].sort_values('DateTime').reset_index(drop=True)

            if len(input_data_mb['BUn'].value_counts()) == 1:
                pass
            else:
                print('error')
                break

            check_index_list = input_data_mb[input_data_mb['이동유형텍스트'] == 'PO에 대한 GR 취소'].index.tolist()
            
            drop_index_list = []
            drop_index_list += check_index_list
            for idx in check_index_list:
                current_qty = abs(input_data_mb.loc[idx, '입력단위수량'])  # 현재 허위 출고의 입력단위수량 절대값
                candidate_indices = input_data_mb[
                    (input_data_mb['이동유형텍스트'] == 'GR 입고') & 
                    (input_data_mb['입력단위수량'] == current_qty)
                ].index.tolist()

                closest_idx = find_closest_index(input_data_mb, idx, candidate_indices, drop_index_list)
                if closest_idx is not None:
                    drop_index_list.append(closest_idx)
                    drop_index_list.append(idx)
            
            input_data_mb = input_data_mb[(input_data_mb['이동유형텍스트'] != 'GR->보류재고 취소') & (input_data_mb['이동유형텍스트'] != 'GR->보류재고')]

            drop_index_list = sorted(set(drop_index_list))

            data_dict[k] = input_data_mb.drop(drop_index_list).reset_index(drop=True)
            check_dd = pd.concat([check_dd, input_data_mb.loc[drop_index_list]])

        else:
            input_data_mb = mb.loc[null_data_index[i]+1:null_data_index[i+1] -1].sort_values(['DateTime']).reset_index(drop=True)

            if len(input_data_mb['BUn'].value_counts()) == 1:
                pass
            else:
                print('error')
                break

            check_index_list = input_data_mb[input_data_mb['이동유형텍스트'] == 'PO에 대한 GR 취소'].index.tolist()

            drop_index_list = []
            drop_index_list += check_index_list

            for idx in check_index_list:
                current_qty = abs(input_data_mb.loc[idx, '입력단위수량'])  # 현재 허위 출고의 입력단위수량 절대값
                candidate_indices = input_data_mb[
                    (input_data_mb['이동유형텍스트'] == 'GR 입고') & 
                    (input_data_mb['입력단위수량'] == current_qty)
                ].index.tolist()

                closest_idx = find_closest_index(input_data_mb, idx, candidate_indices,drop_index_list)
                if closest_idx is not None:
                    drop_index_list.append(closest_idx)
                    drop_index_list.append(idx)
            
            input_data_mb = input_data_mb[(input_data_mb['이동유형텍스트'] != 'GR->보류재고 취소') & (input_data_mb['이동유형텍스트'] != 'GR->보류재고')]

            drop_index_list = sorted(set(drop_index_list))
            data_dict[k] = input_data_mb.drop(drop_index_list).reset_index(drop=True)
            check_dd = pd.concat([check_dd, input_data_mb.loc[drop_index_list]])
    
    return data_dict

def warmup_simulator(EOQ, SS, data_dict, target, initial_stock, lead_time_mu, lead_time_std, start_date, end_date, run_start_time, type):
    warm_up_start_date = run_start_time - pd.Timedelta(days=lead_time_mu)

    if initial_stock < 0:
        initial_stock = 0

    # 초기 재고 설정 및 시뮬레이션 준비
    current_stock = initial_stock
    stock_levels = []
    order_dates = []
    order_values = []
    pending_orders = []
    arrival_dates = []
    lead_time_dates = []
    rop_values = []

    # # 시뮬레이션 세팅
    # np.random.seed(1)

    # 1년간의 데이터를 사용하여 분포 추정 (시뮬레이션 시작 전)
    lambda_value, order_mu, order_std = moving_order_history(data_dict[target], start_date, end_date)
    store_mu, store_std = moving_store_history(data_dict[target], start_date, end_date)
    
    minus = new_order(lambda_value, order_mu, order_std, warm_up_start_date, run_start_time)
    minus = minus.groupby(['날짜']).sum(numeric_only=True)
    minus = minus.sort_index()

    dates = pd.date_range(start=warm_up_start_date, end=run_start_time, freq='D')
    arrival_date = pd.Timestamp(warm_up_start_date)

    for date in dates:
        minus_stock = 0
        plus_stock = 0
        # lead time 및 reorder point 업데이트
        lead_time = max(1, abs(int(np.random.normal(lead_time_mu, lead_time_std))))
        minus_check = minus[minus.index <= date]
        if minus_check.empty:
            expected_demand_during_lead_time = int(abs(np.random.normal(order_mu, order_std)) * lead_time * lambda_value)
        else:
            expected_demand_during_lead_time = int(abs(minus_check.mean()) * lead_time * lambda_value)
        reorder_point = expected_demand_during_lead_time + SS
        rop_values.append(reorder_point)

        # 입고량 업데이트
        for order in list(pending_orders):
            if order['arrival_date'] == date:
                plus_stock += order['quantity']
                pending_orders.remove(order)

        # 출고량 업데이트
        if date in minus.index:
            daily_usage = minus.loc[date, '수량']
            minus_stock += daily_usage

        # 재고량 업데이트
        current_stock += (minus_stock + plus_stock)
        stock_levels.append([current_stock, date]) 

        if current_stock <= reorder_point:
            if date < arrival_date:
                if current_stock - lead_time * lambda_value * abs(np.random.normal(order_mu, order_std)) < SS:
                    if type == 'optimal' and EOQ is not None:
                        order_value = EOQ 
                    else:
                        order_value = int(abs(np.random.normal(store_mu, store_std)))
                    current_stock += order_value
                    arrival_date = date + pd.Timedelta(days=lead_time)
                    pending_orders.append({'arrival_date': arrival_date, 'quantity': order_value})
                    lead_time_dates.append(arrival_date-date)
                    arrival_dates.append(arrival_date)
                    order_dates.append(date)
                    order_values.append(order_value)
                else:
                    pass
            else:
                if type == 'optimal' and EOQ is not None:
                    order_value = EOQ 
                else:
                    order_value = int(abs(np.random.normal(store_mu, store_std)))
                current_stock += order_value
                arrival_date = date + pd.Timedelta(days=lead_time)
                pending_orders.append({'arrival_date': arrival_date, 'quantity': order_value})
                lead_time_dates.append(arrival_date-date)
                arrival_dates.append(arrival_date)
                order_dates.append(date)
                order_values.append(order_value)

    stock_levels_df = pd.DataFrame(stock_levels)
    stock_levels_df.columns = ['Stock', 'Date']
    pending_orders_df = pd.DataFrame(pending_orders)
    return stock_levels_df, pending_orders_df, order_dates, arrival_dates, rop_values, dates

def run_simulation(EOQ, SS, data_dict, target, initial_stock, start_date, end_date, run_start_date, run_end_date, lead_time_mu, lead_time_std, order_dates, order_values, arrival_dates, pending_orders, type):
    warm_up_start_date = run_start_date - pd.Timedelta(days=lead_time_mu)
    # 초기 재고 설정 및 시뮬레이션 준비
    current_stock = initial_stock
    stock_levels = []
    lead_time_dates = []
    rop_values = []
    orders_df = pd.DataFrame(columns=['Order_Date', 'Order_Value', 'Arrival_date'])

    # # 시뮬레이션 세팅
    # np.random.seed(1)

    # 1년간의 데이터를 사용하여 분포 추정 (시뮬레이션 시작 전)
    lambda_value, order_mu, order_std = moving_order_history(data_dict[target], start_date, end_date)
    store_mu, store_std = moving_store_history(data_dict[target], start_date, end_date)

    dates = pd.date_range(start=run_start_date, end=run_end_date, freq='D')
    arrival_date = pd.Timestamp(run_start_date)
    minus = new_order(lambda_value, order_mu, order_std, run_start_date, run_end_date)
    minus = minus.groupby(['날짜']).sum(numeric_only=True)
    # main = generate_maintenance_schedule(data_dict[target], maintenance_mu, maintenance_std, simulation_start_date, simulation_start_date + pd.DateOffset(months=lookback_months))
    # minus = pd.merge(main.reset_index(), minus.reset_index(), on='날짜', how='outer', suffixes=('_main', '_minus'))
    # minus['수량'] = minus['수량_main'].combine_first(minus['수량_minus'])
    # minus = minus[['날짜', '수량']].set_index('날짜')
    minus = minus.sort_index()
    if type == "distribution":
        simulation_info = f"""
        **Simulation type**: {type}   

        **분포참조기간**: {start_date.date()} ~ {end_date.date()}  
        **하루에 출고가 발생한 횟수**: {lambda_value}   
        **분포참조기간에서 도출된 출고량 평균, 표준편차**: {order_mu}, {order_std}   
        **분포참조기간에서 도출된 입고량 평균, 표준편차**: {store_mu}, {store_std}  

        **Warm-up 시뮬레이션 기간**: {warm_up_start_date.date()} ~ {run_start_date.date()}  
        **시뮬레이션 기간**: {run_start_date.date()} ~ {run_end_date.date()}
        """
    else:
        simulation_info = f"""
        **Simulation type**: {type}   

        **분포참조기간**: {start_date.date()} ~ {end_date.date()}  
        **하루에 출고가 발생한 횟수**: {lambda_value}   
        **분포참조기간에서 도출된 출고량 평균, 표준편차**: {order_mu}, {order_std}   
        **EOQ**: {EOQ}  

        **Warm-up 시뮬레이션 기간**: {warm_up_start_date.date()} ~ {run_start_date.date()}  
        **시뮬레이션 기간**: {run_start_date.date()} ~ {run_end_date.date()}
        """

    for date in dates:
        minus_stock = 0
        plus_stock = 0

        # lead time 및 reorder point 업데이트
        lead_time = max(1, abs(int(np.random.normal(lead_time_mu, lead_time_std))))
        minus_check = minus[minus.index <= date]
        if minus_check.empty:
            expected_demand_during_lead_time = int(abs(np.random.normal(order_mu, order_std)) * lead_time * lambda_value)
        else:
            expected_demand_during_lead_time = int(abs(minus_check.mean()) * lead_time * lambda_value)
        reorder_point = expected_demand_during_lead_time + SS
        rop_values.append(reorder_point)

        # 입고량 업데이트
        for index,order in pending_orders.iterrows():
            if order['arrival_date'] == date:
                plus_stock += order['quantity']
                pending_orders.drop(index, inplace=True)

        # 출고량 업데이트
        if date in minus.index:
            daily_usage = minus.loc[date, '수량']
            minus_stock += daily_usage

        # 재고량 업데이트
        current_stock += (minus_stock + plus_stock)
        stock_levels.append([current_stock, date]) 

        if current_stock <= reorder_point:
            if date < arrival_date:
                if current_stock - lead_time * lambda_value * abs(np.random.normal(order_mu, order_std)) < SS:
                    if type == 'optimal' and EOQ is not None:
                        order_value = EOQ 
                    else:
                        order_value = int(abs(np.random.normal(store_mu, store_std)))
                    arrival_date = date + pd.Timedelta(days=lead_time)
                    new_order_df = pd.DataFrame({'quantity': order_value, 'arrival_date': arrival_date}, index=[0])
                    pending_orders = pd.concat([pending_orders, new_order_df], ignore_index=True)
                    lead_time_dates.append(arrival_date-date)
                    orders_input = [date, order_value, arrival_date]
                    orders_df.loc[len(orders_df)] = orders_input
                else:
                    pass
            else:
                if type == 'optimal' and EOQ is not None:
                    order_value = EOQ 
                else:
                    order_value = int(abs(np.random.normal(store_mu, store_std)))
                arrival_date = date + pd.Timedelta(days=lead_time)
                new_order_df = pd.DataFrame({'quantity': order_value, 'arrival_date': arrival_date}, index=[0])
                pending_orders = pd.concat([pending_orders, new_order_df], ignore_index=True)
                lead_time_dates.append(arrival_date-date)
                orders_input = [date, order_value, arrival_date]
                orders_df.loc[len(orders_df)] = orders_input

    stock_levels_df = pd.DataFrame(stock_levels)
    stock_levels_df.columns = ['Stock', 'Date']
    return stock_levels_df, pending_orders, orders_df, rop_values, dates, simulation_info, minus

def total_cost_ga(EOQ, SS, data_dict, meta_dict, target, initial_stock, start_date, end_date, run_start_date, run_end_date, type, 
                  lead_time_mu, lead_time_std, order_dates, order_values, arrival_dates, pending_orders, alpha, beta, gamma, delta, lambda_param):
    
    current_stock = initial_stock
    stock_levels = []
    lead_time_dates = []
    rop_values = []
    orders_df = pd.DataFrame(columns=['Order_Date', 'Order_Value', 'Arrival_date'])

    # 시뮬레이션 세팅
    np.random.seed(42)

    lambda_value, order_mu, order_std = moving_order_history(data_dict[target], start_date, end_date)
    dates = pd.date_range(start=run_start_date, end=run_end_date, freq='D')
    arrival_date = pd.Timestamp(run_start_date)
    minus = new_order(lambda_value, order_mu, order_std, run_start_date, run_end_date)
    minus = minus.groupby(['날짜']).sum(numeric_only=True)
    # main = generate_maintenance_schedule(data_dict[target], maintenance_mu, maintenance_std, simulation_start_date, simulation_start_date + pd.DateOffset(months=lookback_months))
    # minus = pd.merge(main.reset_index(), minus.reset_index(), on='날짜', how='outer', suffixes=('_main', '_minus'))
    # minus['수량'] = minus['수량_main'].combine_first(minus['수량_minus'])
    # minus = minus[['날짜', '수량']].set_index('날짜')
    minus = minus.sort_index()

    for date in dates:
        minus_stock = 0
        plus_stock = 0

        # lead time 및 reorder point 업데이트
        lead_time = max(1, abs(int(np.random.normal(lead_time_mu, lead_time_std))))
        minus_check = minus[minus.index <= date]
        if minus_check.empty:
            expected_demand_during_lead_time = int(abs(np.random.normal(order_mu, order_std)) * lead_time * lambda_value)
        else:
            expected_demand_during_lead_time = int(abs(minus_check.mean()) * lead_time * lambda_value)
        reorder_point = expected_demand_during_lead_time + SS
        rop_values.append(reorder_point)

        # 입고량 업데이트
        for index,order in pending_orders.iterrows():
            if order['arrival_date'] == date:
                plus_stock += order['quantity']
                pending_orders.drop(index, inplace=True)

        # 출고량 업데이트
        if date in minus.index:
            daily_usage = minus.loc[date, '수량']
            minus_stock += daily_usage

        # 재고량 업데이트
        current_stock += (minus_stock + plus_stock)
        stock_levels.append([current_stock, date]) 

        if current_stock <= reorder_point:
            if date < arrival_date:
                if current_stock - lead_time * lambda_value * abs(np.random.normal(order_mu, order_std)) < SS:
                    order_value = EOQ 
                    arrival_date = date + pd.Timedelta(days=lead_time)
                    new_order_df = pd.DataFrame({'quantity': order_value, 'arrival_date': arrival_date}, index=[0])
                    pending_orders = pd.concat([pending_orders, new_order_df], ignore_index=True)
                    lead_time_dates.append(arrival_date-date)
                    orders_input = [date, order_value, arrival_date]
                    orders_df.loc[len(orders_df)] = orders_input
                else:
                    pass
            else:
                order_value = EOQ 
                arrival_date = date + pd.Timedelta(days=lead_time)
                new_order_df = pd.DataFrame({'quantity': order_value, 'arrival_date': arrival_date}, index=[0])
                pending_orders = pd.concat([pending_orders, new_order_df], ignore_index=True)
                lead_time_dates.append(arrival_date-date)
                orders_input = [date, order_value, arrival_date]
                orders_df.loc[len(orders_df)] = orders_input

    stock_levels_df = pd.DataFrame(stock_levels)
    stock_levels_df.columns = ['Stock', 'Date']

    # 비용 계산
    total_cost_df = pd.merge(orders_df, stock_levels_df, left_on='Arrival_date', right_on='Date', how='outer')
    total_cost_df['tau'] = 0
    total_cost_df['backlog_cost'] = 0
    tau_counter = 0
    for idx in range(1, len(total_cost_df)):
        current_stock = total_cost_df['Stock'].iloc[idx]

        if current_stock < 0:
            if total_cost_df['Stock'].iloc[idx-1] < 0:
                tau_counter += 1
            else:
                tau_counter = 1

            backlog_gap = max(0, total_cost_df['Stock'].iloc[idx-1])
            backlog_cost = backlog_gap * beta * np.exp(lambda_param * tau_counter)

            total_cost_df.at[idx, 'tau'] = tau_counter
            total_cost_df.at[idx, 'backlog_cost'] = backlog_cost
        else:
            tau_counter = 0

    # 재고 비용 계산
    total_cost_df['inventory_cost'] = stock_levels_df.iloc[:, 0] * meta_dict[target]['이동평균가'] * alpha

    # 주문 비용 계산
    total_cost_df['order_cost'] = np.where(total_cost_df['Order_Date'].notna(), total_cost_df['Order_Value'] * meta_dict[target]['이동평균가'] * gamma + delta, np.nan)

    # 총 비용 계산
    total_cost_value = total_cost_df[['backlog_cost', 'inventory_cost', 'order_cost']].abs().sum().sum()
    return stock_levels_df, pending_orders, orders_df, rop_values, dates, total_cost_value, minus

def run_genetic_algorithm(data_dict, meta_dict, target, initial_stock, start_date, end_date, 
                          run_start_date, run_end_date, type, lead_time_mu, lead_time_std, 
                          order_dates, order_values, arrival_dates, pending_orders,
                          EOQ_LOW=10, EOQ_HIGH=100, SS_LOW=10, SS_HIGH=50, alpha=0.1, beta=50000,
                          gamma=0.35, delta=700000, lambda_param=3, population_size=20, 
                          ngen=100, cxpb=0.6, mutpb=0.4, elitism_percent=0.02):
    
    def calculate_bits(value_range):
        """주어진 범위에 맞는 최소 비트 수 계산"""
        low, high = value_range
        return math.ceil(math.log2(high - low + 1))

    def int_to_binary(value, num_bits, low):
        """정수를 이진수로 변환"""
        return list(map(int, bin(value - low)[2:].zfill(num_bits)))

    def binary_to_int(binary, low, high):
        """이진수를 정수로 변환"""
        value = int("".join(map(str, binary)), 2)
        return max(low, min(value + low, high))
    
    # 목적 함수 (이진수 개체를 평가)
    def eval_total_cost(individual):
        EOQ_binary, SS_binary = individual[:EOQ_BITS], individual[EOQ_BITS:]
        EOQ = binary_to_int(EOQ_binary, EOQ_LOW, EOQ_HIGH)
        SS = binary_to_int(SS_binary, SS_LOW, SS_HIGH)
        _, _, _, _, _, total_cost_value, _ = total_cost_ga(EOQ, SS, data_dict, meta_dict, target, initial_stock, start_date, end_date, 
                          run_start_date, run_end_date, type, lead_time_mu, lead_time_std, 
                          order_dates, order_values, arrival_dates, pending_orders, alpha, beta, gamma, delta, lambda_param)
        
        return total_cost_value,

    # 필요한 비트 수 계산
    EOQ_BITS = calculate_bits((EOQ_LOW, EOQ_HIGH))
    SS_BITS = calculate_bits((SS_LOW, SS_HIGH))

    # Fitness 및 개체 생성
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=EOQ_BITS + SS_BITS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # 교차 및 변이 연산 등록
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.3)
    toolbox.register("select", tools.selRoulette)
    toolbox.register("evaluate", eval_total_cost)

    # 초기 개체 수 설정
    population = toolbox.population(n=population_size)
    elitism_size = max(int(len(population) * elitism_percent), 2)

    # 초기 세대 평가
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    # 진화 과정
    status_text = st.empty()
    for gen in range(ngen):
        best_fitness = tools.selBest(population, 1)[0].fitness.values[0]
        best_individuals = [ind for ind in population if ind.fitness.values[0] == best_fitness]
        elite_individuals = tools.selBest(population, elitism_size)
        elite_individuals = list(map(toolbox.clone, elite_individuals))
        
        offspring = toolbox.select(population, len(population) - len(best_individuals) - len(elite_individuals))
        offspring = list(map(toolbox.clone, offspring))
        offspring.extend(best_individuals)
        offspring.extend(elite_individuals)
        
        # 교차 연산
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 변이 연산
        mutpb = min(0.3 + gen * 0.01, mutpb)
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 적합도 평가
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 엘리트 개체를 다음 세대로 유지
        population[:] = offspring

        # 세대별 최고 적합도 및 EOQ, SS 값 출력
        best_ind = tools.selBest(population, 1)[0]
        EOQ = binary_to_int(best_ind[:EOQ_BITS], EOQ_LOW, EOQ_HIGH)
        SS = binary_to_int(best_ind[EOQ_BITS:], SS_LOW, SS_HIGH)
        status_text.write(f"세대 {gen+1}")
        # print(f"세대 {gen+1}, 최고 fitness: {best_ind.fitness.values[0]:,.0f}, EOQ: {EOQ}, SS: {SS}")

    best_fitness = tools.selBest(population, 1)[0].fitness.values[0]
    best_individuals = [ind for ind in population if ind.fitness.values[0] == best_fitness]
    
    results = set()
    for ind in best_individuals:
        EOQ = binary_to_int(ind[:EOQ_BITS], EOQ_LOW, EOQ_HIGH)
        SS = binary_to_int(ind[EOQ_BITS:], SS_LOW, SS_HIGH)
        results.add((EOQ, SS))

    results = list(results)
    _, _, _, _, _, _, minus_value = total_cost_ga(
        EOQ, SS, data_dict, meta_dict, target, initial_stock, start_date, end_date, 
        run_start_date, run_end_date, type, lead_time_mu, lead_time_std, 
        order_dates, order_values, arrival_dates, pending_orders, alpha, beta, gamma, delta, lambda_param
    )
    return results, minus_value

def total_cost_result(EOQ, SS, data_dict, meta_dict, target, initial_stock, start_date, end_date, 
               run_start_date, run_end_date, type, lead_time_mu, lead_time_std, 
               order_dates, order_values, arrival_dates, pending_orders,
               alpha, beta, gamma, delta, lambda_param, minus):
    # 초기 재고 설정 및 시뮬레이션 준비
    current_stock = initial_stock
    stock_levels = []
    lead_time_dates = []
    rop_values = []
    orders_df = pd.DataFrame(columns=['Order_Date', 'Order_Value', 'Arrival_date'])

    # 시뮬레이션 세팅
    np.random.seed(1)

    lambda_value, order_mu, order_std = moving_order_history(data_dict[target], start_date, end_date)
    dates = pd.date_range(start=run_start_date, end=run_end_date, freq='D')
    arrival_date = pd.Timestamp(run_start_date)
    # minus = new_order(lambda_value, order_mu, order_std, run_start_date, run_end_date)
    # minus = minus.groupby(['날짜']).sum(numeric_only=True)
    # main = generate_maintenance_schedule(data_dict[target], maintenance_mu, maintenance_std, simulation_start_date, simulation_start_date + pd.DateOffset(months=lookback_months))
    # minus = pd.merge(main.reset_index(), minus.reset_index(), on='날짜', how='outer', suffixes=('_main', '_minus'))
    # minus['수량'] = minus['수량_main'].combine_first(minus['수량_minus'])
    # minus = minus[['날짜', '수량']].set_index('날짜')
    # minus = minus.sort_index()

    for date in dates:
        minus_stock = 0
        plus_stock = 0

        # lead time 및 reorder point 업데이트
        lead_time = max(1, abs(int(np.random.normal(lead_time_mu, lead_time_std))))
        minus_check = minus[minus.index <= date]
        if minus_check.empty:
            expected_demand_during_lead_time = int(abs(np.random.normal(order_mu, order_std)) * lead_time * lambda_value)
        else:
            expected_demand_during_lead_time = int(abs(minus_check.mean()) * lead_time * lambda_value)
        reorder_point = expected_demand_during_lead_time + SS
        rop_values.append(reorder_point)

        # 입고량 업데이트
        for index,order in pending_orders.iterrows():
            if order['arrival_date'] == date:
                plus_stock += order['quantity']
                pending_orders.drop(index, inplace=True)

        # 출고량 업데이트
        if date in minus.index:
            daily_usage = minus.loc[date, '수량']
            minus_stock += daily_usage

        # 재고량 업데이트
        current_stock += (minus_stock + plus_stock)
        stock_levels.append([current_stock, date]) 

        if current_stock <= reorder_point:
            if date < arrival_date:
                if current_stock - lead_time * lambda_value * abs(np.random.normal(order_mu, order_std)) < SS:
                    order_value = EOQ 
                    arrival_date = date + pd.Timedelta(days=lead_time)
                    new_order_df = pd.DataFrame({'quantity': order_value, 'arrival_date': arrival_date}, index=[0])
                    pending_orders = pd.concat([pending_orders, new_order_df], ignore_index=True)
                    lead_time_dates.append(arrival_date-date)
                    orders_input = [date, order_value, arrival_date]
                    orders_df.loc[len(orders_df)] = orders_input
                else:
                    pass
            else:
                order_value = EOQ 
                arrival_date = date + pd.Timedelta(days=lead_time)
                new_order_df = pd.DataFrame({'quantity': order_value, 'arrival_date': arrival_date}, index=[0])
                pending_orders = pd.concat([pending_orders, new_order_df], ignore_index=True)
                lead_time_dates.append(arrival_date-date)
                orders_input = [date, order_value, arrival_date]
                orders_df.loc[len(orders_df)] = orders_input

    stock_levels_df = pd.DataFrame(stock_levels)
    stock_levels_df.columns = ['Stock', 'Date']

    # 비용 계산
    total_cost_df = pd.merge(orders_df, stock_levels_df, left_on='Arrival_date', right_on='Date', how='outer')
    total_cost_df['tau'] = 0
    total_cost_df['backlog_cost'] = 0
    tau_counter = 0
    for idx in range(1, len(total_cost_df)):
        current_stock = total_cost_df['Stock'].iloc[idx]

        if current_stock < 0:
            if total_cost_df['Stock'].iloc[idx-1] < 0:
                tau_counter += 1
            else:
                tau_counter = 1

            backlog_gap = max(0, total_cost_df['Stock'].iloc[idx-1])
            backlog_cost = backlog_gap * beta * np.exp(lambda_param * tau_counter)

            total_cost_df.at[idx, 'tau'] = tau_counter
            total_cost_df.at[idx, 'backlog_cost'] = backlog_cost
        else:
            tau_counter = 0

    # 재고 비용 계산
    total_cost_df['inventory_cost'] = stock_levels_df.iloc[:, 0] * meta_dict[target]['이동평균가'] * alpha

    # 주문 비용 계산
    total_cost_df['order_cost'] = np.where(total_cost_df['Order_Date'].notna(), total_cost_df['Order_Value'] * meta_dict[target]['이동평균가'] * gamma + delta, np.nan)

    # 총 비용 계산
    total_cost_value = total_cost_df[['backlog_cost', 'inventory_cost', 'order_cost']].abs().sum().sum()
    return stock_levels_df, pending_orders, orders_df, rop_values, dates, total_cost_value

def plot_inventory_simulation(dates, safety_stock, rop_values_result, stock_levels_df_result, orders_df_result, target, initial_stock):
    arrival_dates = orders_df_result['Arrival_date']
    order_dates = orders_df_result['Order_Date']

    fig = go.Figure()
    # Safety Stock 라인
    fig.add_trace(go.Scatter(
        x=dates,
        y=[safety_stock] * len(dates),
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name=f'Safety Stock ({safety_stock:.2f})'
    ))
    # Reorder Point 라인
    fig.add_trace(go.Scatter(
        x=dates,
        y=rop_values_result,
        mode='lines+markers',
        line=dict(color='green', dash='dash', width=2),
        name='Reorder Point',
        marker=dict(size=8, symbol='circle')
    ))
    # Current Stock 라인
    fig.add_trace(go.Scatter(
        x=stock_levels_df_result['Date'],
        y=stock_levels_df_result['Stock'],
        mode='lines+markers',
        line=dict(color='blue'),
        name='Current Stock',
        marker=dict(size=8, symbol='circle')
    ))

    for date in arrival_dates:
        fig.add_shape(
            type="line",
            x0=date, y0=0, x1=date, y1=stock_levels_df_result['Stock'].max() * 1.3,
            line=dict(color="orange", width=1, dash="dash"),
        )

    for date in order_dates:
        fig.add_shape(
            type="line",
            x0=date, y0=-5, x1=date, y1=stock_levels_df_result['Stock'].max() * 1.3,
            line=dict(color="grey", width=1, dash="dash"),
        )

    for date in arrival_dates:
        if date == arrival_dates[0]:
            fig.add_trace(go.Scatter(
                x=[date],
                y=[0],
                mode="lines+markers",
                marker=dict(size=8, color="orange"),
                name="Arrival Date",
                hovertemplate="<span style='color:orange'>Arrival Date</span> (%{x|%Y-%m-%d})<extra></extra>",
                showlegend=True
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[date],
                y=[0],
                mode="lines+markers",
                marker=dict(size=8, color="orange"),
                name="Arrival Date",
                hovertemplate="<span style='color:orange'>Arrival Date</span> (%{x|%Y-%m-%d})<extra></extra>",
                showlegend=False
            ))

    for date in order_dates:
        if date == order_dates[0]:
            fig.add_trace(go.Scatter(
                x=[date],
                y=[-5],
                mode="lines+markers",
                marker=dict(size=8, color="grey"),
                name="Order Date",
                hovertemplate="<span style='color:grey'>Order Date</span> (%{x|%Y-%m-%d})<extra></extra>",
                showlegend=True
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[date],
                y=[-5],
                mode="lines+markers",
                marker=dict(size=8, color="grey"),
                name="Order Date",
                hovertemplate="<span style='color:grey'>Order Date</span> (%{x|%Y-%m-%d})<extra></extra>",
                showlegend=False
            ))      

    max_y_value = max(safety_stock, max(rop_values_result), stock_levels_df_result['Stock'].max()) * 1.2
    fig.update_layout(
        autosize=True, 
        title={
            'text': f'<span style="font-size:36px;">{target}</span><br>',
            'x': 0.5, 'xanchor': 'center'
        },
        margin=dict(l=0, r=0, t=150, b=0),  # 여백 설정으로 제목이 짤리지 않도록
        xaxis_title='날짜', yaxis_title='재고량', font=dict(size=36),
        xaxis=dict(titlefont=dict(size=24), tickformat='%Y-%m-%d', tickmode='linear',
            dtick=604800000.0, tickfont=dict(size=24), range = [min(dates)-pd.Timedelta(days=0.5), max(dates)+pd.Timedelta(days=0.5)]
        ),
        yaxis=dict(titlefont=dict(size=24), showgrid=True, tickfont=dict(size=24), range=[-10, max_y_value]),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, font=dict(size=26)),
        width=2400, height=800, hoverlabel=dict(font_size=36)
    )
    return fig

def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,lambda loader, node: object_pairs_hook(loader.construct_pairs(node)))
    return yaml.load(stream, OrderedLoader)

def ordered_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass
    OrderedDumper.add_representer(OrderedDict,lambda dumper, data: dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items()))
    return yaml.dump(data, stream, OrderedDumper, **kwds)