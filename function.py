import pandas as pd
import os
import yaml
import streamlit as st



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