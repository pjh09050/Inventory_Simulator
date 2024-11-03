import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def plot_inventory_analysis(data_dict, start_date=None, end_date=None, selected_material=None):
    df = data_dict[selected_material]  

    df['재고누적'] = df['입력단위수량'].cumsum()
    # 선택한 기간 필터링
    filtered_df = df[(pd.to_datetime(df['날짜']) >= pd.to_datetime(start_date)) & 
                     (pd.to_datetime(df['날짜']) <= pd.to_datetime(end_date))]

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

    incoming_plot_placeholder = st.empty()
    outgoing_plot_placeholder = st.empty()
    cumulative_plot_placeholder = st.empty()

    # 입고 데이터 플롯
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
            title_font=dict(size=36),
            xaxis_title="날짜",
            yaxis_title="입고수량",
            xaxis=dict(titlefont=dict(size=24), showgrid=True, tickfont=dict(size=24)),
            yaxis=dict(titlefont=dict(size=24), showgrid=True, tickfont=dict(size=24)),
            legend=dict(font=dict(size=18)),
            hoverlabel=dict(font_size=36), 
            barmode='group'
        )
        # 입고 플롯 
        st.plotly_chart(fig_incoming)

        # 입고 통계량 표시
        incoming_stats = pd.DataFrame({
            '항목': ['기간', '일 평균 입고량', '일 평균 입고 횟수', '입고량 평균', '입고량 표준편차'],
            '값': [f"{start_date} ~ {end_date}", round(average_incoming, 2), incoming_counts / period_days, round(total_incoming / incoming_counts, 2), round(std_dev_incoming, 2)]
        })

        # 항목 열 볼드체로 설정
        incoming_stats = incoming_stats.style.applymap(lambda x: 'font-weight: bold', subset=['항목'])
        st.write(incoming_stats)
    else:
        incoming_plot_placeholder.empty()


    if st.checkbox('출고 그래프 보기'):
        # 출고 데이터 플롯
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
            title_font=dict(size=36),
            xaxis_title="날짜",
            yaxis_title="출고수량",
            xaxis=dict(titlefont=dict(size=24), showgrid=True, tickfont=dict(size=24)),
            yaxis=dict(titlefont=dict(size=24), showgrid=True, tickfont=dict(size=24)),
            legend=dict(font=dict(size=18)),
            hoverlabel=dict(font_size=36),
            barmode='group'
        )
        # 출고 플롯 
        st.plotly_chart(fig_outgoing)

        # 출고 통계량 표시
        outgoing_stats = pd.DataFrame({
            '항목': ['기간', '일 평균 출고량', '일 평균 출고 횟수', '출고량 평균', '출고량 표준편차'],
            '값': [f"{start_date} ~ {end_date}", round(average_outgoing, 2), outgoing_counts / period_days, round(total_outgoing / outgoing_counts, 2), round(std_dev_outgoing, 2)]
        })

        # 항목 열 볼드체로 설정
        outgoing_stats = outgoing_stats.style.applymap(lambda x: 'font-weight: bold', subset=['항목'])
        st.write(outgoing_stats)
    else:
        outgoing_plot_placeholder.empty()

    if st.checkbox('입고&출고 누적합 그래프 보기'):
        # 누적 재고 플롯
        fig_cumulative = go.Figure()

        # 기본 누적 재고를 초록색으로 표시 (0보다 큰 값)
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

        # 0 이하 부분 표시를 위한 빨간색 점 추가
        below_zero = filtered_df[filtered_df['재고누적'] <= 0]
        if not below_zero.empty:
            fig_cumulative.add_trace(
                go.Scatter(
                    x=below_zero['날짜'],
                    y=below_zero['재고누적'],
                    mode='markers',  # 선과 점으로 표시
                    name="Inventory plot(Below Zero)",
                    marker=dict(color='red', size=3)  # 0 이하 부분을 빨간색으로
                )
            )

        fig_cumulative.update_layout(
            title={
                'text' : f"{selected_material} - 누적 재고 (0 이하 값: {below_zero_count})",
                'x': 0.5, 'xanchor': 'center'
            },
            title_font=dict(size=36),
            xaxis_title="날짜",
            yaxis_title="재고 누적",
            xaxis_title_font_size=24,
            yaxis_title_font_size=24,
            legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, font=dict(size=26)), hoverlabel=dict(font_size=36)
        
        )

        # 누적 재고 플롯 출력
        st.plotly_chart(fig_cumulative)

        # 누적 재고 통계량 표시
        inventory_stats = pd.DataFrame({
            '항목': ['기간', '일 평균 재고 수준', '일 평균 변동 횟수', '일 평균 재고 표준편차'],
            '값': [f"{start_date} ~ {end_date}", round(average_inventory, 2), len(filtered_df) / period_days, round(std_dev_inventory, 2)]
        })

        # 항목 열 볼드체로 설정
        inventory_stats = inventory_stats.style.applymap(lambda x: 'font-weight: bold', subset=['항목'])
        st.write(inventory_stats)
    else:
        cumulative_plot_placeholder.empty()  