import streamlit as st
import yaml
import pandas as pd
import joblib
from datetime import datetime
from function import *
import pandas as pd
print(pd.__version__)

def load_meta_info(meta_path):
    df_meta_info = pd.read_pickle('../data/project_zwms03s_df.pickle')
    joblib.dump(df_meta_info, './data/project_zwms03s_df.joblib')
    df_meta_info = joblib.load('./data/project_zwms03s_df.joblib')
    print(df_meta_info.head())
    meta_value = df_meta_info[['자재', '품명', '이동평균값', '납기(일)-Average Lead Time (Days)',
                               'Max/Min.1', 'Lead Time Standard Deviation (Days)',
                               'Safety Stock 1', 'Reorder Point 1', 'Safety Stock 2', 'Reorder Point 2',
                               'Safety Stock 3', 'Reorder Point 3', 'Safety Stock 4', 'Reorder Point 4',
                               'Safety Stock 5', 'Reorder Point 5', 'Safety Stock 6', 'Reorder Point 6',
                               'Recommended Safety Stock', 'Recommended Reorder Point']]
    meta_value.index = meta_value['자재']
    meta_value.drop(['자재'], axis=1, inplace=True)
    meta_value = meta_value.T
    meta_dict = meta_value.to_dict()
    with open('meta_info.yaml', 'w') as file:
        yaml.dump(meta_dict, file, default_flow_style=False, sort_keys=False, allow_unicode=True)
    with open('meta_info.yaml', 'r', encoding='utf-8') as file:
        meta_dict = yaml.safe_load(file)
    return meta_dict

st.title('Simulation')
st.header('Parameter Settings')

data_path = st.text_input("Data Path", './data/project_mb51_df.pickle')
meta_path = st.text_input("Meta Path", './data/project_zwms03s_df.pickle')
target = st.text_input("Target", '61-2081610000000')

start_date = st.date_input("Start Date", datetime(2022, 12, 30))
end_date = st.date_input("End Date", datetime(2024, 1, 19))
run_start_date = st.date_input("Run Start Date", datetime(2024, 2, 1))
run_end_date = st.date_input("Run End Date", datetime(2024, 2, 28))

safety_stock = st.number_input("Safety Stock", value=17.096)
initial_stock = st.number_input("Initial Stock", value=34)
reorder_point = st.number_input("Reorder Point", value=32.572)
maintenance_mu = st.number_input("Maintenance Mean (mu)", value=70)
maintenance_std = st.number_input("Maintenance Std Dev", value=13)

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
    'reorder_point': reorder_point,
    'maintenance_mu': maintenance_mu,
    'maintenance_std': maintenance_std
}
check_or_create_yaml('param_set.yaml', save_config)

if st.button('파라미터 저장'):
    try:
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
            'reorder_point': reorder_point,
            'maintenance_mu': maintenance_mu,
            'maintenance_std': maintenance_std
        })
        meta_dict = load_meta_info(meta_path)
        save_config.update(meta_dict[target])
        with open('param_set.yaml', 'w') as file:
            yaml.dump(save_config, file, default_flow_style=False, sort_keys=False, allow_unicode=True)
        st.write("Configuration saved to 'param_set.yaml'")
    except Exception as e:
        st.error(f"Error saving configuration: {e}")

if st.button('파라미터 불러오기'):
    try:
        with open('param_set.yaml', 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        st.write(config)
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
