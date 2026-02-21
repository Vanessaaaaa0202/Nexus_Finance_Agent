import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
import plotly.graph_objects as go
import time
from openai import OpenAI 

st.set_page_config(page_title="Nexus Finance Agent", layout="wide")
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Caveat:wght@400..700&family=Inter:wght@300;500;800&display=swap" rel="stylesheet">
    
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff;
    }
    
    /* 1. 主标题：完美融合 Google Fonts 官方规范 */
    .main-title {
        font-family:'Caveat', cursive !important;
        font-optical-sizing: auto;
        font-weight: 600 !important;
        font-style: normal;
        font-size: 4.2rem !important;
        color: #1a1a1a;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    
    .sub-title {
        font-family:'Caveat', cursive !important;
        color: #4b5563;
        font-weight: 500;
        font-size: 1.5rem !important;
        margin-bottom: 3rem;
    }
    .section-title {
        font-weight: 800;
        letter-spacing: -0.02em;
        font-size: 1.5rem;
        color: #1a1a1a;
        margin-bottom: 10px;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 24px !important;
        border-radius: 20px !important;
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #000000;
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. 侧边栏：上传数据与静默 API (无密码框)
st.sidebar.header("Data Center")
uploaded_file = st.sidebar.file_uploader("Upload your sales data(csv file)", type="csv")

# 静默读取系统级的 API Key (需要在 .streamlit/secrets.toml 里配置)
try:
    api_key = st.secrets["OPENAI_API_KEY"] 
    client = OpenAI(api_key=api_key)
except Exception:
    client = None
    st.sidebar.error("⚠️ The system is not configured with an API Key. Please contact the administrator.")

def clean_data(df):
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.drop_duplicates()
    df['Category'] = df['Category'].replace('Invetory', 'Inventory')
    df['Amount'] = df['Amount'].fillna(0)
    df['Status'] = df['Status'].fillna('Unknown')
    return df


# 3. 数据预处理 
df = None
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    df = clean_data(raw_df)

    revenue_categories = ['Agency Fee 1', 'Agency Fee 2', 'Holding Fee', 'Commission Fee']
    df_paid = df[df['Status'] == 'Paid']

    total_revenue = df_paid[df_paid['Category'].isin(revenue_categories)]['Amount'].sum()
    total_expense = df_paid[~df_paid['Category'].isin(revenue_categories)]['Amount'].sum()
    net_cash_flow = total_revenue - total_expense


# 4. 顶部 Header
header_col1, header_col2 = st.columns([4, 1])

with header_col1:
    st.markdown('<h1 class="main-title">Nexus Finance Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">A curated financial lens for early-stage founders.</p>', unsafe_allow_html=True)

with header_col2:
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    
    if df is not None and client:
        with st.popover("✨ Ask AI Copilot", use_container_width=True):
            st.markdown("**Nexus AI Agent**")
            st.caption("E.g., 'Predict the cash flow for next month'")
            
            user_question = st.chat_input("Type your question...")
            
            if user_question:
                columns = df.columns.tolist()
                prompt = f"""
                你是一个经验丰富、极具同理心的创业公司财务合伙人 (Fractional CFO)。现在你需要帮老板分析数据。
                变量：'df' (pandas DataFrame), 'px' (plotly.express)。列名：{columns}
                问题："{user_question}"
                请生成 Python 代码执行以下步骤：
                1. 数据清洗：Date 为 datetime。
                2. 预测逻辑：按月汇总，计算历史均值乘1.1作为预测。包含 History 和 Prediction。
                3. 🚨致命要求🚨：在画图前，必须将表示 X 轴时间的那一列转换为纯文本字符串类型！例如：df['Month'] = df['Month'].astype(str)。绝对不能把 Pandas Period 对象直接放进图表！
                4. 画图：只能用 px.bar()，x为刚才转换好的文本列，y为 Amount，color='Type'。赋值给 'fig'。
                5. 人设回复：将一句温暖、专业的分析结论赋值给变量 'answer'。用英语回复，带 1-2 个 emoji，像真人在聊天！
                仅返回 Python 代码。
                """
                
                with st.chat_message("user"):
                    st.write(user_question)
                    
                with st.chat_message("assistant"):
                    with st.spinner("✨ Nexus AI is deep diving into your ledger..."):
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[{"role": "system", "content": "只输出 Python 代码。"}, {"role": "user", "content": prompt}],
                                temperature=0.2
                            )
                            code_to_run = response.choices[0].message.content.replace('```python', '').replace('```', '').strip()
                            local_vars = {'df': df, 'pd': pd, 'np': np, 'px': px}
                            exec(code_to_run, {}, local_vars)
                            
                            final_answer = local_vars.get('answer', "I've analyzed the data for you.")
                            final_fig = local_vars.get('fig', None)
                        except Exception as e:
                            final_answer = "Oops, I ran into a little hiccup. Could you try asking in a different way?"
                            final_fig = None
                            
                    st.write(final_answer)
                    if final_fig is not None:
                        final_fig.update_layout(margin=dict(t=20, b=20, l=0, r=0), height=300)
                        st.plotly_chart(final_fig, use_container_width=True)

# 5. 主界面逻辑 (指标卡、气泡与图表)
if df is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("TOTAL REVENUE", f"${total_revenue:,.0f}")
    with col2:
        st.metric("TOTAL EXPENSES", f"${total_expense:,.0f}")
    with col3:
        st.metric("NET CASH FLOW", f"${net_cash_flow:,.0f}")

    st.write("---")
    
    # --- 业务下钻气泡 ---
    st.markdown('<h3 class="section-title">Business Distribution Preview</h3>', unsafe_allow_html=True)
    rev_counts = df_paid[df_paid['Category'].isin(revenue_categories)]['Category'].value_counts()
    exp_counts = df_paid[~df_paid['Category'].isin(revenue_categories)]['Category'].value_counts()
    
    rev_mapping = {f"{cat} ({count})": cat for cat, count in rev_counts.items()}
    exp_mapping = {f"{cat} ({count})": cat for cat, count in exp_counts.items()}

    selected_rev_tag = st.pills("💰 Revenue Streams (Click to view ledger):", options=list(rev_mapping.keys()), default=None)
    selected_exp_tag = st.pills("📉 Expense Categories (Click to view ledger):", options=list(exp_mapping.keys()), default=None)

    active_tag = selected_rev_tag or selected_exp_tag
    active_mapping = rev_mapping if selected_rev_tag else exp_mapping

    if active_tag:
        actual_category = active_mapping[active_tag]
        filtered_df = df_paid[df_paid['Category'] == actual_category]
        st.markdown(f"<div style='margin-bottom: 10px; font-weight: 600; color: #4b5563;'>👇 {actual_category} Detailed Ledger (Sorted by Date)</div>", unsafe_allow_html=True)
        st.dataframe(filtered_df.sort_values(by='Date', ascending=False).style.format({'Amount': '${:,.2f}'}), use_container_width=True, hide_index=True, height=250)



    #可视化图表
    tab_rev, tab_exp = st.tabs(["📊 Revenue Analysis", "📊 Expense Analysis"])
    editorial_colors = ['#B6CBA6', '#F0B622', '#F25822', '#3C364C', '#8EA4A1', '#d1d5db']

    def draw_charts(data_subset, title_prefix):
        if data_subset.empty:
            st.info(f"Not enough data for {title_prefix} analysis.")
            return

        c1, c2 = st.columns(2)
        with c1:
            pie_data = data_subset.groupby('Category')['Amount'].sum().reset_index()
            if not pie_data.empty:
                max_idx = pie_data['Amount'].idxmax()
                pull_list = [0.015 if i == max_idx else 0 for i in range(len(pie_data))]
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=pie_data['Category'], values=pie_data['Amount'], pull=pull_list, 
                    textposition='outside', textfont=dict(size=18, color='#333333', family='Inter'),
                    texttemplate='<span style="font-size:13px;">%{label}</span><br><b>%{percent}</b>',
                    marker=dict(colors=editorial_colors, line=dict(color='#ffffff', width=2)), sort=False, direction='clockwise', hole=0 
                )])
                fig_pie.update_layout(
                    showlegend=False, 
                    margin=dict(t=40, b=40, l=100, r=100), # 顶部缝隙收回正常值
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    hoverlabel=dict(bgcolor="white", bordercolor="#e5e7eb", font_size=14, font_family="Inter")
                )
                fig_pie.update_traces(hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>")
                st.markdown(f"<div style='text-align: center; font-size: 18px; color: #333333; font-weight: 600; margin-bottom: 5px;'>Top {title_prefix}s</div>", unsafe_allow_html=True)
                st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            if not pd.api.types.is_datetime64_any_dtype(data_subset['Date']):
                data_subset['Date'] = pd.to_datetime(data_subset['Date'], errors='coerce')
            
            data_subset['Month_Name'] = data_subset['Date'].dt.month_name()
            # 动态获取月份的正确时间顺序
            month_order = data_subset.sort_values('Date')['Month_Name'].unique().tolist()
            
            bar_data = data_subset.groupby(['Category', 'Month_Name'])['Amount'].sum().reset_index()
            
            if not bar_data.empty:
                fig_bar = px.bar(
                    bar_data, x='Category', y='Amount', color='Month_Name', barmode='group',
                    color_discrete_sequence=editorial_colors, # 动态莫兰迪取色
                    category_orders={'Month_Name': month_order}
                )
                fig_bar.update_layout(
                    # 🚨 已经删除了原有的 title=dict(...)
                    # 🚨 t=60 改成了 t=40，和左边饼图完全对齐
                    margin=dict(t=40, b=40, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_family="Inter",
                    hovermode="x unified", hoverlabel=dict(bgcolor="white", bordercolor="#e5e7eb", font_size=13, font_family="Inter"),
                    yaxis=dict(title="", showgrid=True, gridcolor='#f0f2f6', tickformat="$.2s", zeroline=False),
                    xaxis=dict(title="", showgrid=False, tickfont=dict(size=12, color='#666')),
                    legend=dict(title="", orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, font=dict(size=12, color='#666'))
                )
                fig_bar.update_traces(opacity=0.9, hovertemplate="<b>%{data.name}</b>: $%{y:,.0f}<extra></extra>") 
                
                # 【终极同步】：添加和饼图一模一样规格的 HTML 标题
                st.markdown(f"<div style='text-align: center; font-size: 18px; color: #333333; font-weight: 600; margin-bottom: 5px;'>{title_prefix} by Month</div>", unsafe_allow_html=True)
                
                st.plotly_chart(fig_bar, use_container_width=True)

    # 渲染 Tabs
    with tab_rev:
        draw_charts(df_paid[df_paid['Category'].isin(revenue_categories)].copy(), "Revenue")
    with tab_exp:
        draw_charts(df_paid[~df_paid['Category'].isin(revenue_categories)].copy(), "Expense")
