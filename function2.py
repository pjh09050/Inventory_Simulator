import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def plot_inventory_analysis(data_dict, start_date=None, end_date=None, selected_material=None):
    df = data_dict[selected_material]

    # 날짜로 그룹화하여 일별 합계 및 누적 재고 계산
    df['날짜'] = pd.to_datetime(df['날짜'])
    df = df.groupby('날짜')['입력단위수량'].sum().reset_index()
    df['재고누적'] = df['입력단위수량'].cumsum()

    # 선택한 기간 필터링
    filtered_df = df[(df['날짜'] >= pd.to_datetime(start_date)) & 
                     (df['날짜'] <= pd.to_datetime(end_date))]

    # 기간 계산
    period_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1

    # 0 이하인 누적합 값 카운트
    below_zero_count = (filtered_df['재고누적'] <= 0).sum()

    # 기초통계량 계산
    incoming_counts = filtered_df['입력단위수량'][filtered_df['입력단위수량'] > 0].count()
    outgoing_counts = filtered_df['입력단위수량'][filtered_df['입력단위수량'] < 0].count()

    total_incoming = filtered_df['입력단위수량'][filtered_df['입력단위수량'] > 0].sum()
    total_outgoing = filtered_df['입력단위수량'][filtered_df['입력단위수량'] < 0].sum() * -1  # 출고량의 절대값

    average_incoming = total_incoming / period_days
    average_outgoing = total_outgoing / period_days
    average_inventory = filtered_df['재고누적'].mean()
    
    std_dev_incoming = filtered_df['입력단위수량'][filtered_df['입력단위수량'] > 0].std()
    std_dev_outgoing = filtered_df['입력단위수량'][filtered_df['입력단위수량'] < 0].std()
    std_dev_inventory = filtered_df['재고누적'].std()

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

        above_zero = filtered_df[filtered_df['재고누적'] > 0]
        fig_cumulative.add_trace(
            go.Scatter(
                x=above_zero['날짜'],
                y=above_zero['재고누적'],
                mode='lines',  
                name="Inventory plot",
                line=dict(color='green')  
            )
        )

        below_zero = filtered_df[filtered_df['재고누적'] <= 0]
        if not below_zero.empty:
            fig_cumulative.add_trace(
                go.Scatter(
                    x=below_zero['날짜'],
                    y=below_zero['재고누적'],
                    mode='markers',
                    name="Below Zero",
                    marker=dict(color='red', size=5)
                )
            )

        fig_cumulative.update_layout(
            title={
                'text' : f"{selected_material} - 누적 재고 (0 이하 값: {below_zero_count})",
                'x': 0.5, 'xanchor': 'center'
            },
            xaxis_title="날짜",
            yaxis_title="재고 누적",
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
