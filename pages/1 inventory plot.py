import streamlit as st
import yaml
import pandas as pd
from datetime import datetime
from function import *
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
        padding-top: 1rem;
        max-width: 80%;  /* Increase the width to 70% of the page */
    }
    /* Increase font size for all elements */
    html, body {
        font-size: 1.5rem; /* Global font size adjustment */
    }
    </style>
    """, unsafe_allow_html=True)

# Title and header
st.title('Simulation')
st.subheader('Parameter Settings')

data_path = st.text_input("Data Path", './data/project_mb51_df.pickle')
meta_path = st.text_input("Meta Path", './data/project_zwms03s_df.pickle')

col1, col2 = st.columns(2)
with col1:
    target = st.text_input("Target", '61-200161000000')
    start_date = st.date_input("Start Date", datetime(2022, 12, 30))
    end_date = st.date_input("End Date", datetime(2024, 1, 19))
    run_start_date = st.date_input("Run Start Date", datetime(2024, 2, 1))
    run_end_date = st.date_input("Run End Date", datetime(2024, 2, 28))
with col2:
    initial_stock = st.number_input("Initial Stock", value=34.0, step=0.01, format="%.2f")
    safety_stock = st.number_input("Safety Stock", value=17.096, step=0.01, format="%.2f")
    reorder_point = st.number_input("Reorder Point", value=32.572, step=0.01, format="%.2f")
    maintenance_mu = st.number_input("Maintenance Mean (mu)", value=70.0, step=0.01, format="%.2f")
    maintenance_std = st.number_input("Maintenance Std Dev (std)", value=13.0, step=0.01, format="%.2f")

# Configuration dictionary
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

col1, col2 = st.columns([1, 1], gap="small")  # Adjust gap as needed
with col1:
    if st.button('Save Parameters'):
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
            with open('param_set.yaml', 'w', encoding='utf-8') as file:
                yaml.dump(save_config, file, default_flow_style=False, sort_keys=False, allow_unicode=True)
            st.success("Configuration saved to 'param_set.yaml'")
        except Exception as e:
            st.error(f"Error saving configuration: {e}")

with col2:
    if st.button('Load Parameters'):
        try:
            with open('param_set.yaml', 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            st.write(config)
        except Exception as e:
            st.error(f"Error loading configuration: {e}")