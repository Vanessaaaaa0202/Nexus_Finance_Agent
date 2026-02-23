import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
import plotly.graph_objects as go
import time
from openai import OpenAI 

# ==========================================
# 1. 页面配置与高级画廊 CSS
# ==========================================
st.set_page_config(page_title="Nexus Finance Agent", layout="wide")

# 将你找到的官方 <link> 标签和内部样式表完美结合
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
            
            # 【核心修复 1】：初始化聊天记忆库 (Session State)
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
                
            # 【核心修复 2】：每次刷新时，先把历史聊天记录渲染出来
            # 🚨 修改 1：加上 enumerate 获取序号 i
            for i, msg in enumerate(st.session_state.chat_history):
                with st.chat_message(msg["role"]):
                    st.write(msg["text"])
                    if msg.get("fig") is not None:
                        # 🚨 修改 2：加上独一无二的 key
                        st.plotly_chart(msg["fig"], use_container_width=True, key=f"history_fig_{i}")
            
            user_question = st.chat_input("Type your question...")
            
            if user_question:
                # 1. 把用户的新问题存入记忆库，并立刻显示在屏幕上
                st.session_state.chat_history.append({"role": "user", "text": user_question})
                with st.chat_message("user"):
                    st.write(user_question)
                
                # 【核心修复 3】：提取真实的业务数据上下文，彻底封杀幻觉！
                unique_categories = df['Category'].unique().tolist()
                expense_summary = df[~df['Category'].isin(revenue_categories)].groupby('Category')['Amount'].sum().to_dict()
                columns = df.columns.tolist()
                
                # 把历史聊天记录转换为大模型能听懂的格式
                api_messages = [{"role": "system", "content": "你是一个只输出 Python 代码的引擎。"}]
                for m in st.session_state.chat_history[:-1]: # 传入历史对话，让它拥有记忆
                    api_messages.append({"role": m["role"], "content": m["text"]})
                
                # 强悍的 Prompt 压制
                # 强悍的 Prompt 压制，彻底剥夺 AI 的心算权限
                prompt = f"""
                你是一个极具同理心的创业公司财务合伙人。现在正在和老板连贯对话。
                
                🚨【真实数据限制（绝对不可违背）】🚨：
                当前公司的所有业务分类有：{unique_categories}
                各项支出总计为：{expense_summary}
                你的分析必须且只能基于上述真实数据！如果用户问了不存在的类别，必须明确告知没有记录！
                
                变量：'df' (pandas), 'px' (plotly)。列名：{columns}
                老板的最新问题："{user_question}"
                
                请生成 Python 代码执行以下步骤：
                1. 数据清洗：Date 为 datetime。
                2. 画图需求：如果需要对比或趋势，用 px.bar()，赋值给 'fig'。否则 fig=None。
                3. 🚨🚨致命要求（禁止心算）🚨🚨：大语言模型极度不擅长数学计算。**绝对禁止你在脑内进行任何数值的加减乘除！**
                   遇到任何需要计算总和、差值、均值的问题，必须且只能通过 Pandas 代码去运算，赋值给 Python 变量。
                   然后，使用 f-string 将计算好的 Python 变量拼接入你要回复的字符串中。
                   错误示范：answer = "Total is " + str(100 + 200) 
                   正确示范：total_val = df['Amount'].sum(); answer = f"老板，算出来了，总计是 ${{total_val:,.2f}}哦！"
                4. 人设回复：赋值给变量 'answer'。用英语回复，像真人聊天！
                
                仅返回 Python 代码。
                """
                api_messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("assistant"):
                    with st.spinner("✨ Nexus AI is analyzing your ledger..."):
                        try:
                            # 传入完整的 api_messages，包含系统设定、历史对话和最新问题
                            response = client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=api_messages,
                                temperature=0.1
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
                        # 🚨 修改 3：给新生成的图表也加上独一无二的 key
                        st.plotly_chart(final_fig, use_container_width=True, key=f"new_fig_{len(st.session_state.chat_history)}")
                
                # 2. 把 AI 的回答也存入记忆库，完成闭环
                st.session_state.chat_history.append({"role": "assistant", "text": final_answer, "fig": final_fig})

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
