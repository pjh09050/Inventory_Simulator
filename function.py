import pandas as pd
import os
import yaml
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

def warmup_simulator(EOQ, SS, data_dict, target, initial_stock, lead_time_mu, lead_time_std, start_date, end_date, run_start_time, type, maintenance_mu, maintenance_std):
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
    stock_levels_df = pd.DataFrame(stock_levels)
    pending_orders_df = pd.DataFrame(pending_orders)

    # # 시뮬레이션 세팅
    # np.random.seed(1)

    # 1년간의 데이터를 사용하여 분포 추정 (시뮬레이션 시작 전)
    lambda_value, order_mu, order_std = moving_order_history(data_dict[target], start_date, end_date)
    store_mu, store_std = moving_store_history(data_dict[target], start_date, end_date)

    minus = new_order(lambda_value, order_mu, order_std, warm_up_start_date, run_start_time)
    minus = minus.groupby(['날짜']).sum(numeric_only=True)
    main = generate_maintenance_schedule(maintenance_mu, maintenance_std, warm_up_start_date, run_start_time)
    if main is not None and not main.empty:
        minus = pd.merge(main.reset_index(), minus.reset_index(), on='날짜', how='outer', suffixes=('_main', '_minus'))
        minus['수량'] = minus['수량_main'].combine_first(minus['수량_minus'])
        minus = minus[['날짜', '수량']].set_index('날짜')
    minus = minus.sort_index()
    dates = pd.date_range(start=warm_up_start_date, end=run_start_time, freq='D')
    if minus.empty:
        return stock_levels_df, pending_orders_df, order_dates, arrival_dates, rop_values, dates
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

def run_simulation(EOQ, SS, data_dict, target, initial_stock, start_date, end_date, run_start_date, run_end_date, lead_time_mu, lead_time_std, 
                   order_dates, order_values, arrival_dates, pending_orders, type, maintenance_mu, maintenance_std):
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
    main = generate_maintenance_schedule(maintenance_mu, maintenance_std, run_start_date, run_end_date)
    if main is not None and not main.empty:
        minus = pd.merge(main.reset_index(), minus.reset_index(), on='날짜', how='outer', suffixes=('_main', '_minus'))
        minus['수량'] = minus['수량_main'].combine_first(minus['수량_minus'])
        minus = minus[['날짜', '수량']].set_index('날짜')
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
    return stock_levels_df, pending_orders, orders_df, rop_values, dates, simulation_info, minus, main

def total_cost_ga(EOQ, SS, data_dict, meta_dict, target, initial_stock, start_date, end_date, run_start_date, run_end_date, type, 
                  lead_time_mu, lead_time_std, maintenance_mu, maintenance_std, order_dates, order_values, arrival_dates, pending_orders, alpha, beta, gamma, delta, lambda_param):
    
    current_stock = initial_stock
    stock_levels = []
    lead_time_list = []
    rop_values = []
    expected_demand_list = []
    orders_df = pd.DataFrame(columns=['Order_Date', 'Order_Value', 'Arrival_date'])

    # 시뮬레이션 세팅
    np.random.seed(42)

    lambda_value, order_mu, order_std = moving_order_history(data_dict[target], start_date, end_date)
    dates = pd.date_range(start=run_start_date, end=run_end_date, freq='D')

    arrival_date = pd.Timestamp(run_start_date)
    minus = new_order(lambda_value, order_mu, order_std, run_start_date, run_end_date)
    minus = minus.groupby(['날짜']).sum(numeric_only=True)
    main = generate_maintenance_schedule(maintenance_mu, maintenance_std, run_start_date, run_end_date)
    if main is not None and not main.empty:
        minus = pd.merge(main.reset_index(), minus.reset_index(), on='날짜', how='outer', suffixes=('_main', '_minus'))
        minus['수량'] = minus['수량_main'].combine_first(minus['수량_minus'])
        minus = minus[['날짜', '수량']].set_index('날짜')
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
        expected_demand_list.append(expected_demand_during_lead_time)
        rop_values.append(reorder_point)
        lead_time_list.append(lead_time)

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
                if current_stock - expected_demand_during_lead_time < SS:
                    order_value = EOQ 
                    arrival_date = date + pd.Timedelta(days=lead_time)
                    new_order_df = pd.DataFrame({'quantity': order_value, 'arrival_date': arrival_date}, index=[0])
                    pending_orders = pd.concat([pending_orders, new_order_df], ignore_index=True)
                    orders_input = [date, order_value, arrival_date]
                    orders_df.loc[len(orders_df)] = orders_input
                else:
                    pass
            else:
                order_value = EOQ 
                arrival_date = date + pd.Timedelta(days=lead_time)
                new_order_df = pd.DataFrame({'quantity': order_value, 'arrival_date': arrival_date}, index=[0])
                pending_orders = pd.concat([pending_orders, new_order_df], ignore_index=True)
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
            if total_cost_df['Stock'].iloc[idx] < 0:
                tau_counter += 1
            else:
                tau_counter = 1

            backlog_gap = max(0, abs(total_cost_df['Stock'].iloc[idx]))
            backlog_cost = backlog_gap * meta_dict[target]['이동평균가'] * beta * np.exp(lambda_param * tau_counter)

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
    return stock_levels_df, pending_orders, orders_df, rop_values, dates, total_cost_value, minus, lead_time_list, expected_demand_list, main

def run_genetic_algorithm(data_dict, meta_dict, target, initial_stock, start_date, end_date, 
                          run_start_date, run_end_date, type, lead_time_mu, lead_time_std, maintenance_mu, maintenance_std,
                          order_dates, order_values, arrival_dates, pending_orders,
                          EOQ_LOW=10, EOQ_HIGH=100, SS_LOW=10, SS_HIGH=50, alpha=0.1, beta=50000,
                          gamma=0.35, delta=700000, lambda_param=3, population_size=20, 
                          ngen=100, cxpb=0.6, mutpb=0.4):
    
    def calculate_bits(value_range):
        """주어진 범위에 맞는 최소 비트 수 계산"""
        low, high = value_range
        return math.ceil(math.log2(high - low + 1))

    def binary_to_int(binary, low, high):
        """이진수를 정수로 변환"""
        value = int("".join(map(str, binary)), 2)
        return max(low, min(value + low, high))
    
    # 목적 함수 (이진수 개체를 평가)
    def eval_total_cost(individual):
        EOQ_binary, SS_binary = individual[:EOQ_BITS], individual[EOQ_BITS:]
        EOQ = binary_to_int(EOQ_binary, EOQ_LOW, EOQ_HIGH)
        SS = binary_to_int(SS_binary, SS_LOW, SS_HIGH)
        _, _, _, _, _, total_cost_value, _, _, _, _ = total_cost_ga(EOQ, SS, data_dict, meta_dict, target, initial_stock, start_date, end_date, 
                          run_start_date, run_end_date, type, lead_time_mu, lead_time_std, maintenance_mu, maintenance_std,
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
    if len(population) <= 2:
        elitism_size = 0
    else:
        elitism_size = 2

    # 초기 세대 평가
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    # 진화 과정
    status_text = st.empty()
    for gen in range(ngen):
        elite_individuals = tools.selBest(population, elitism_size)
        elite_individuals = list(map(toolbox.clone, elite_individuals))
        
        offspring = toolbox.select(population, len(population) - len(elite_individuals))
        offspring = list(map(toolbox.clone, offspring))

        
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
        population[:] = offspring + elite_individuals

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
    _, _, _, rop_list, _, _, minus_value, lead_time_list, expected_demand_list, main = total_cost_ga(
        EOQ, SS, data_dict, meta_dict, target, initial_stock, start_date, end_date, 
        run_start_date, run_end_date, type, lead_time_mu, lead_time_std, maintenance_mu, maintenance_std,
        order_dates, order_values, arrival_dates, pending_orders, alpha, beta, gamma, delta, lambda_param
    )
    return results, minus_value, rop_list, lead_time_list, expected_demand_list, main

def total_cost_result(EOQ, SS, data_dict, meta_dict, target, initial_stock, start_date, end_date, 
               run_start_date, run_end_date, order_dates, order_values, arrival_dates, pending_orders,
               alpha, beta, gamma, delta, lambda_param, minus, rop_list, lead_time_list, expected_demand_list):
    # 초기 재고 설정 및 시뮬레이션 준비
    current_stock = initial_stock
    stock_levels = []
    orders_df = pd.DataFrame(columns=['Order_Date', 'Order_Value', 'Arrival_date'])

    # 시뮬레이션 세팅
    np.random.seed(42)

    dates = pd.date_range(start=run_start_date, end=run_end_date, freq='D')
    arrival_date = pd.Timestamp(run_start_date)

    time = 0
    for date in dates:
        minus_stock = 0
        plus_stock = 0

        lead_time = lead_time_list[time]
        reorder_point = rop_list[time]
        expected_demand = expected_demand_list[time]

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
        time += 1

        if current_stock <= reorder_point:
            if date < arrival_date:
                if current_stock - expected_demand < SS:
                    order_value = EOQ 
                    arrival_date = date + pd.Timedelta(days=lead_time)
                    new_order_df = pd.DataFrame({'quantity': order_value, 'arrival_date': arrival_date}, index=[0])
                    pending_orders = pd.concat([pending_orders, new_order_df], ignore_index=True)
                    orders_input = [date, order_value, arrival_date]
                    orders_df.loc[len(orders_df)] = orders_input
                else:
                    pass
            else:
                order_value = EOQ 
                arrival_date = date + pd.Timedelta(days=lead_time)
                new_order_df = pd.DataFrame({'quantity': order_value, 'arrival_date': arrival_date}, index=[0])
                pending_orders = pd.concat([pending_orders, new_order_df], ignore_index=True)
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
            if total_cost_df['Stock'].iloc[idx] < 0:
                tau_counter += 1
            else:
                tau_counter = 1

            backlog_gap = max(0, abs(total_cost_df['Stock'].iloc[idx]))
            backlog_cost = backlog_gap * meta_dict[target]['이동평균가'] * beta * np.exp(lambda_param * tau_counter)

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
    return stock_levels_df, pending_orders, orders_df, rop_list, dates, total_cost_value

def plot_inventory_simulation(dates, safety_stock, rop_values_result, stock_levels_df_result, orders_df_result, target, initial_stock, maintenance_date):
    arrival_dates = orders_df_result['Arrival_date']
    order_dates = orders_df_result['Order_Date']
    order_values = orders_df_result['Order_Value']
    max_y_value = max(safety_stock, max(rop_values_result), stock_levels_df_result['Stock'].max()) * 1.2
    min_y_value = min(safety_stock, min(rop_values_result), stock_levels_df_result['Stock'].min()) * 1.5
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
        marker=dict(size=7, symbol='circle')
    ))
    # Current Stock 라인
    fig.add_trace(go.Scatter(
        x=stock_levels_df_result['Date'],
        y=stock_levels_df_result['Stock'],
        mode='lines+markers',
        line=dict(color='blue'),
        name='Current Stock',
        marker=dict(size=7, symbol='circle')
    ))

    for date in arrival_dates:
        fig.add_shape(
            type="line",
            x0=date, y0=0, x1=date, y1=max_y_value,
            line=dict(color="orange", width=1, dash="dash"),
        )

    # Arrival Dates
    for i, (date, value) in enumerate(zip(arrival_dates, order_values)):
        fig.add_shape(
            type="line",
            x0=date, y0=min_y_value * 0.75, x1=date, y1=max_y_value,
            line=dict(color="orange", width=1, dash="dash"),
        )
        fig.add_trace(go.Scatter(
            x=[date],
            y=[min_y_value * 0.75],
            mode="lines+markers",
            marker=dict(size=7, color="orange"),
            name="Arrival Date",
            hovertemplate=f"<span style='color:orange'>Arrival Date</span> (%{{x|%Y-%m-%d}}, {value})<extra></extra>",
            showlegend=(i == 0)
        ))
    # Order Dates
    for i, (date, value) in enumerate(zip(order_dates, order_values)):
        fig.add_shape(
            type="line",
            x0=date, y0=min_y_value * 0.85, x1=date, y1=max_y_value,
            line=dict(color="grey", width=1, dash="dash"),
        )
        fig.add_trace(go.Scatter(
            x=[date],
            y=[min_y_value * 0.85],
            mode="lines+markers",
            marker=dict(size=7, color="grey"),
            name="Order Date",
            hovertemplate=f"<span style='color:grey'>Order Date</span> (%{{x|%Y-%m-%d}}, {value})<extra></extra>",
            showlegend=(i == 0)
        ))
    # Maintenance Dates
    for date, quantity in maintenance_date.iterrows():
        fig.add_shape(
            type="line",
            x0=date, y0=min_y_value * 0.95, x1=date, y1=max_y_value,
            line=dict(color="purple", width=2, dash="dot"),
        )
        # Maintenance Point Marker
        fig.add_trace(go.Scatter(
            x=[date],
            y=[min_y_value * 0.95],
            mode="lines+markers",
            marker=dict(size=7, color="purple", line=dict(width=2, color="purple")),
            name="Maintenance Date",
            hovertemplate=f"<span style='color:purple'>Maintenance Date</span> (%{{x|%Y-%m-%d}}, {quantity['수량']})<extra></extra>",
        ))

    fig.update_layout(
        autosize=True, 
        title={
            'text': f'<span style="font-size:36px;">{target}</span><br>',
            'x': 0.5, 'xanchor': 'center'
        },
        xaxis_title='날짜', yaxis_title='재고량', font=dict(size=36),
        xaxis=dict(titlefont=dict(size=24), tickformat='%Y-%m-%d', tickmode='linear', showline=True,
            dtick=604800000.0, tickfont=dict(size=24), tickangle=45, range = [min(dates)-pd.Timedelta(days=0.5), max(dates)+pd.Timedelta(days=0.5)]
        ),
        yaxis=dict(titlefont=dict(size=24), showgrid=True, tickfont=dict(size=24), range = [min(-30, min_y_value), max_y_value], showline=True),
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

def generate_maintenance_schedule(mu, std, start_date, end_date):
    """
    유지보수 스케줄을 생성하는 함수. 각 유지보수 날짜에 대해 수량을 생성하고, 
    수량을 화요일에 할당합니다.

    Parameters:
    - data_dict: 데이터 사전 (사용하지 않음, 필요하면 추가 활용 가능)
    - mu: 수량 생성 시 사용할 평균
    - std: 수량 생성 시 사용할 표준편차
    - start_date: 시작 날짜 (datetime 객체)
    - end_date: 종료 날짜 (datetime 객체)

    Returns:
    - maintenance_dates: 유지보수 날짜와 수량이 포함된 데이터프레임, 유지보수 날짜가 인덱스로 설정됨
    """
    # 유지보수 날짜를 저장할 빈 데이터프레임 생성
    maintenance_dates = pd.DataFrame(columns=['날짜', '수량'])

    # 각 년도별 상반기와 하반기로 유지보수 날짜 및 수량 생성
    for year in range(start_date.year, end_date.year + 1):
        # 상반기: 1월 1일 ~ 6월 30일
        first_day_of_h1 = pd.Timestamp(year, 1, 1)
        last_day_of_h1 = pd.Timestamp(year, 6, 30)

        # 하반기: 7월 1일 ~ 12월 31일
        first_day_of_h2 = pd.Timestamp(year, 7, 1)
        last_day_of_h2 = pd.Timestamp(year, 12, 31)

        # 상반기의 첫 번째 화요일
        if first_day_of_h1 <= end_date and last_day_of_h1 >= start_date:
            first_tuesday_h1 = first_day_of_h1 + pd.offsets.Week(weekday=1)
            if first_tuesday_h1 >= start_date and first_tuesday_h1 <= end_date:
                maintenance_quantity_h1 = -int(np.random.normal(mu, std))
                maintenance_dates = pd.concat([
                    maintenance_dates,
                    pd.DataFrame({'날짜': [first_tuesday_h1], '수량': [maintenance_quantity_h1]})
                ], ignore_index=True)

        # 하반기의 첫 번째 화요일
        if first_day_of_h2 <= end_date and last_day_of_h2 >= start_date:
            first_tuesday_h2 = first_day_of_h2 + pd.offsets.Week(weekday=1)
            if first_tuesday_h2 >= start_date and first_tuesday_h2 <= end_date:
                maintenance_quantity_h2 = -int(np.random.normal(mu, std))
                maintenance_dates = pd.concat([
                    maintenance_dates,
                    pd.DataFrame({'날짜': [first_tuesday_h2], '수량': [maintenance_quantity_h2]})
                ], ignore_index=True)

    # '날짜' 열을 인덱스로 설정
    maintenance_dates.set_index('날짜', inplace=True)
    return maintenance_dates

def plot_inventory_analysis(data_dict, start_date=None, end_date=None, selected_material=None):
    df = data_dict[selected_material]

    # 날짜로 그룹화하여 일별 합계 및 누적 재고 계산
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.groupby('날짜')['입력단위수량'].sum().reset_index()
    df['재고누적합'] = df['입력단위수량'].cumsum()

    # 선택한 기간 필터링
    filtered_df = df[(df['날짜'] >= pd.to_datetime(start_date)) & 
                     (df['날짜'] <= pd.to_datetime(end_date))]

    # 기간 계산
    period_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1

    # 0 이하인 누적합 값 카운트
    below_zero_count = (filtered_df['재고누적합'] <= 0).sum()

    # 기초통계량 계산
    incoming_counts = filtered_df['입력단위수량'][filtered_df['입력단위수량'] > 0].count()
    outgoing_counts = filtered_df['입력단위수량'][filtered_df['입력단위수량'] < 0].count()

    total_incoming = filtered_df['입력단위수량'][filtered_df['입력단위수량'] > 0].sum()
    total_outgoing = filtered_df['입력단위수량'][filtered_df['입력단위수량'] < 0].sum() * -1  # 출고량의 절대값

    average_incoming = total_incoming / period_days
    average_outgoing = total_outgoing / period_days
    average_inventory = filtered_df['재고누적합'].mean()
    
    std_dev_incoming = filtered_df['입력단위수량'][filtered_df['입력단위수량'] > 0].std()
    std_dev_outgoing = filtered_df['입력단위수량'][filtered_df['입력단위수량'] < 0].std()
    std_dev_inventory = filtered_df['재고누적합'].std()

    # 글씨 크기 설정
    title_font_size = 36
    axis_title_font_size = 24
    tick_font_size = 24
    legend_font_size = 26
    hoverlabel_font_size = 36

    # 입고, 출고, 누적 재고 플롯 설정
    if st.checkbox('입고 그래프 보기'):
        incoming = filtered_df[filtered_df['입력단위수량'] > 0]
        fig_incoming = go.Figure()
        fig_incoming.add_trace(
            go.Bar(x=incoming['날짜'], y=incoming['입력단위수량'], name="Incoming", marker=dict(color='blue'),
                   hovertemplate='<b>날짜:</b> %{x}<br><b>수량:</b> <span style="color:blue;">%{y}</span><extra></extra>')
        )
        fig_incoming.update_layout(
            title={
                'text' : f"{selected_material} - 입고",
                'x': 0.5, 'xanchor': 'center'
            },
            xaxis_title="날짜",
            yaxis_title="입고수량",
            title_font=dict(size=title_font_size),
            xaxis=dict(title_font=dict(size=axis_title_font_size), tickfont=dict(size=tick_font_size)),
            yaxis=dict(title_font=dict(size=axis_title_font_size), tickfont=dict(size=tick_font_size)),
            legend=dict(font=dict(size=legend_font_size)),
            hoverlabel=dict(font=dict(size=hoverlabel_font_size)),
            barmode='group'
        )
        st.plotly_chart(fig_incoming)

        # 입고 통계량 표시
        st.markdown("### 입고 통계량", unsafe_allow_html=True)
        incoming_stats = pd.DataFrame({
            '기간': [f"{start_date} ~ {end_date}"],
            '일 평균 입고량': [round(average_incoming, 2)],
            '일 평균 입고 횟수': [round(incoming_counts / period_days, 2)],
            '입고량 평균': [round(total_incoming / incoming_counts, 2) if incoming_counts else 0],
            '입고량 표준편차': [round(std_dev_incoming, 2)]
        })
        
        st.table(incoming_stats.style.set_properties(**{'font-size': '20px'}).set_table_attributes('style="font-size: 20px;"'))

    if st.checkbox('출고 그래프 보기'):
        outgoing = filtered_df[filtered_df['입력단위수량'] < 0]
        fig_outgoing = go.Figure()
        fig_outgoing.add_trace(
            go.Bar(x=outgoing['날짜'], y=outgoing['입력단위수량'] * -1, name="Outgoing", marker=dict(color='red'),
                   hovertemplate='<b>날짜:</b> %{x}<br><b>수량:</b> <span style="color:red;">%{y}</span><extra></extra>')
        )
        fig_outgoing.update_layout(
            title={
                'text' : f"{selected_material} - 출고",
                'x': 0.5, 'xanchor': 'center'
            },
            xaxis_title="날짜",
            yaxis_title="출고수량",
            title_font=dict(size=title_font_size),
            xaxis=dict(title_font=dict(size=axis_title_font_size), tickfont=dict(size=tick_font_size)),
            yaxis=dict(title_font=dict(size=axis_title_font_size), tickfont=dict(size=tick_font_size)),
            legend=dict(font=dict(size=legend_font_size)),
            hoverlabel=dict(font=dict(size=hoverlabel_font_size)),
            barmode='group'
        )
        st.plotly_chart(fig_outgoing)

        # 출고 통계량 표시
        st.markdown("### 출고 통계량", unsafe_allow_html=True)
        outgoing_stats = pd.DataFrame({
            '기간': [f"{start_date} ~ {end_date}"],
            '일 평균 출고량': [round(average_outgoing, 2)],
            '일 평균 출고 횟수': [round(outgoing_counts / period_days, 2)],
            '출고량 평균': [round(total_outgoing / outgoing_counts, 2) if outgoing_counts else 0],
            '출고량 표준편차': [round(std_dev_outgoing, 2)]
        })
        st.table(outgoing_stats.style.set_properties(**{'font-size': '20px'}).set_table_attributes('style="font-size: 20px;"'))

    if st.checkbox('입고&출고 누적합 그래프 보기'):
        fig_cumulative = go.Figure()

        above_zero = filtered_df[filtered_df['재고누적합'] > 0]
        fig_cumulative.add_trace(
            go.Scatter(
                x=above_zero['날짜'],
                y=above_zero['재고누적합'],
                mode='lines',  
                name="누적합",
                line=dict(color='green')  
            )
        )

        below_zero = filtered_df[filtered_df['재고누적합'] <= 0]
        if not below_zero.empty:
            fig_cumulative.add_trace(
                go.Scatter(
                    x=below_zero['날짜'],
                    y=below_zero['재고누적합'],
                    mode='markers',
                    name="0 이하 시점",
                    marker=dict(color='red', size=5)
                )
            )

        fig_cumulative.update_layout(
            title={
                'text' : f"{selected_material} - 누적 재고 (0 이하 값: {below_zero_count})",
                'x': 0.5, 'xanchor': 'center'
            },
            xaxis_title="날짜",
            yaxis_title="재고누적합",
            title_font=dict(size=title_font_size),
            xaxis=dict(title_font=dict(size=axis_title_font_size), tickfont=dict(size=tick_font_size)),
            yaxis=dict(title_font=dict(size=axis_title_font_size), tickfont=dict(size=tick_font_size)),
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, font=dict(size=legend_font_size)),
            hoverlabel=dict(font=dict(size=hoverlabel_font_size))
        )
        st.plotly_chart(fig_cumulative)

        # 누적 재고 통계량 표시
        st.markdown("### 누적 재고 통계량", unsafe_allow_html=True)
        inventory_stats = pd.DataFrame({
            '기간': [f"{start_date} ~ {end_date}"],
            '일 평균 재고 수준': [round(average_inventory, 2)],
            '일 평균 변동 횟수': [round(len(filtered_df) / period_days, 2)],
            '일 평균 재고 표준편차': [round(std_dev_inventory, 2)]
        })
        st.table(inventory_stats.style.set_properties(**{'font-size': '20px'}).set_table_attributes('style="font-size: 20px;"'))