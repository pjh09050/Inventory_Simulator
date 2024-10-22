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