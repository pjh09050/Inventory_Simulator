import streamlit as st
from function import *
import matplotlib.pyplot as plt
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
st.subheader("Inventory Plot")
data_path = st.file_uploader("Choose Data File", type=["pickle", "csv", "xlsx", "json"])
data_dict = None  

if data_path is not None:
    st.session_state['data_dict'] = eda(data_path)
    col1, col2, col3 = st.columns(3)
    st.session_state['material_numbers'] = list(st.session_state['data_dict'].keys())
    with col1:
        st.session_state['selected_material'] = st.selectbox("자재번호 선택:", st.session_state['material_numbers'])

    if st.session_state['selected_material']:
        st.session_state['df'] = st.session_state['data_dict'][st.session_state['selected_material']]
        with col2:
            st.session_state['start_date'] = st.date_input("시작 날짜:", st.session_state['df']['날짜'].min())
        with col3:
            st.session_state['end_date'] = st.date_input("종료 날짜:", st.session_state['df']['날짜'].max())
        # plot 그리기
        plot_inventory_analysis(st.session_state['data_dict'], st.session_state['start_date'], st.session_state['end_date'], st.session_state['selected_material'])
else:
    st.write("Data(mb51) 파일을 업로드하세요")