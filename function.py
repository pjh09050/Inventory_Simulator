import pandas as pd
import os
import yaml
import streamlit as st
import scipy.stats as stats
import pandas as pd
import numpy as np

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

def warmup_simulator(EOQ, SS, data_dict, initial_stock, lead_time_mu, lead_time_std, target, start_date, end_date, type):
    warm_up_start_date = start_date - pd.Timedelta(days=lead_time_mu)

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

    # 시뮬레이션 세팅
    np.random.seed(1)

    # 1년간의 데이터를 사용하여 분포 추정 (시뮬레이션 시작 전)
    lambda_value, order_mu, order_std = moving_order_history(data_dict[target], start_date, end_date)
    store_mu, store_std = moving_store_history(data_dict[target], start_date, end_date)
    
    minus = new_order(lambda_value, order_mu, order_std, start_date, end_date)
    minus = minus.groupby(['날짜']).sum(numeric_only=True)
    minus = minus.sort_index()

    dates = pd.date_range(start=warm_up_start_date, end=start_date, freq='D')
    arrival_date = pd.Timestamp(warm_up_start_date)
    print(f"""
                simulation type: {type}
                재고 도착! warm_up 시뮬레이션 시작! 도착날짜: {arrival_date},
                분포참조기간: {start_date.date()} ~ {end_date.date()},
                분포참조기간에서 도출된 출고 람다, 평균, 표준편차: {lambda_value}, {order_mu}, {order_std},
                분포참조기간에서 도출된 입고 평균, 표준편차: {store_mu}, {store_std},
                초기재고: {initial_stock},
                warm-up 시뮬러닝기간: {warm_up_start_date.date()} ~ {start_date.date()}
          """)

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

def run_simulation(EOQ, SS, data_dict, target, start_date, end_date, run_start_date, run_end_date, type, lead_time_mu, lead_time_std, initial_stock, order_dates, order_values, arrival_dates, pending_orders):
    # 초기 재고 설정 및 시뮬레이션 준비
    current_stock = initial_stock
    stock_levels = []
    lead_time_dates = []
    rop_values = []
    orders_df = pd.DataFrame(columns=['Order_Date', 'Order_Value', 'Arrival_date'])

    # 시뮬레이션 세팅
    np.random.seed(1)

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
    print(f"""
                초기재고: {initial_stock},
                시뮬러닝기간: {run_start_date.date()} ~ {run_end_date.date()}
          """)
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
    
    return stock_levels_df, pending_orders, orders_df, rop_values, dates