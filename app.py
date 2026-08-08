import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from lp_solver import EconomicDispatchSolver
from ga_solver import EVStationGeneticSolver
from chatbot import SmartEnergyAssistant

# تنظیمات صفحه Streamlit (باید حتماً اولین دستور Streamlit باشد)
st.set_page_config(
    page_title="سامانه توزیع هوشمند برق منطقه‌ای",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# دریافت کلید از secrets در صورت وجود، یا مقداردهی اولیه خالی برای جلوگیری از NameError
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    GEMINI_API_KEY = ""

# استایل CSS شیک، تمیز و متناسب (Light RTL Modern UI)
st.markdown("""
<style>
    /* تنظیمات جهت و فونت کلی برنامه */
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Vazirmatn', Tahoma, sans-serif;
        direction: rtl !important;
        text-align: right !important;
    }

    /* راست‌چین کردن نوار کناری (Sidebar) */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-left: 1px solid #E2E8F0;
        direction: rtl !important;
        text-align: right !important;
    }

    /* متناسب‌سازی سایز متون و عناوین */
    h1 {
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
        margin-bottom: 0.2rem !important;
    }
    h2 {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: #1E293B !important;
        margin-top: 0.5rem !important;
    }
    h3 {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
    }
    p, span, label, div {
        direction: rtl !important;
        text-align: right !important;
    }
    .stCaption {
        font-size: 0.85rem !important;
        color: #64748B !important;
    }

    /* متناسب‌سازی کارت‌های متریژ (Metrics) */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 12px 16px !important;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
        text-align: right !important;
        direction: rtl !important;
    }
    div[data-testid="stMetricValue"] {
        color: #2563EB !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        text-align: right !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        text-align: right !important;
    }

    /* راست‌چین کردن تب‌ها (Tabs) */
    div[data-baseweb="tab-list"] {
        direction: rtl !important;
        justify-content: flex-start !important;
        gap: 8px;
    }
    button[data-baseweb="tab"] {
        direction: rtl !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
    }

    /* استایل دکمه‌ها: رنگ روشن‌تر، جذاب‌تر و دارای باکس شدو */
    .stButton>button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        border-radius: 9999px !important;
        border: none !important;
        padding: 8px 20px !important;
        font-size: 0.92rem !important;
        font-weight: 600 !important;
        width: 100%;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%) !important;
        box-shadow: 0 6px 18px rgba(59, 130, 246, 0.45) !important;
        transform: translateY(-1px);
    }

    /* ورودی‌ها و اسلایدرها */
    .stSlider, .stNumberInput, .stTextInput {
        direction: rtl !important;
        text-align: right !important;
    }
</style>
""", unsafe_allow_html=True)

# عنوان اصلی
st.title("⚡ سامانه توزیع هوشمند برق منطقه‌ای")
st.caption("پروژه جامع تصمیم‌گیری مهندسی صنایع | برنامه نویسی پیشرفته")

# نوار کناری تنظیمات
st.sidebar.header("⚙️ تنظیمات پارامترهای ورود")
user_key = st.sidebar.text_input(
    "Gemini API Key (اختیاری)", 
    value="", 
    type="password", 
    help="کلید اختصاصی Gemini API را وارد کنید."
)

# اولویت با کلیدی است که کاربر در ورودی تایپ کرده، در غیر این صورت از GEMINI_API_KEY استفاده می‌شود
active_api_key = user_key if user_key else GEMINI_API_KEY

# تب‌های اصلی اپلیکیشن
tab1, tab2, tab3 = st.tabs([
    "📊 توزیع اقتصادی بار (LP)",
    "🚗 مکان‌یابی شارژر برقی (GA)",
    "🤖 چت‌بات تحلیلی هوشمند"
])

# --- TAB 1: LP Economic Dispatch ---
with tab1:
    st.header("توزیع اقتصادی بار بین نیروگاه‌ها و ذخیره‌ساز")
    
    col_input, col_chart = st.columns([1, 2])
    
    with col_input:
        st.subheader("پارامترهای شبکه")
        demand = st.slider("تقاضای کل شبکه (MW)", 50, 500, 220)
        solar = st.slider("توان خورشیدی در دسترس (MW)", 0, 100, 45)
        battery = st.slider("ذخیره باتری (MWh)", 0, 100, 30)
        
        st.subheader("هزینه نیروگاه‌های حرارتی")
        c1 = st.number_input("نیروگاه ۱ ($/MW)", value=40.0)
        c2 = st.number_input("نیروگاه ۲ ($/MW)", value=65.0)
        c3 = st.number_input("نیروگاه ۳ ($/MW)", value=90.0)
        
        run_lp = st.button("اجرای بهینه‌سازی LP", key="run_lp_btn")

    thermal_units = [
        {"name": "حرارتی 1 (پایه)", "min": 20, "max": 150, "cost_per_mw": c1},
        {"name": "حرارتی 2 (میان‌باری)", "min": 10, "max": 100, "cost_per_mw": c2},
        {"name": "حرارتی 3 (پیک)", "min": 0, "max": 60, "cost_per_mw": c3},
    ]

    solver_lp = EconomicDispatchSolver(demand, solar, battery)
    lp_res = solver_lp.solve(thermal_units)
    st.session_state['lp_results'] = lp_res

    with col_chart:
        st.subheader("نتایج تخصیص توان بهینه")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("هزینه کل ($)", f"{lp_res['total_cost']:,.1f}")
        col_m2.metric("قیمت سایه‌ای ($/MW)", f"{lp_res['shadow_price']:.2f}")
        col_m3.metric("وضعیت پاسخ", lp_res['status'])
        
        categories = list(lp_res['dispatch'].keys()) + ["خورشیدی", "تخلیه باتری"]
        values = list(lp_res['dispatch'].values()) + [lp_res['solar_generation'], lp_res['battery_discharge']]
        colors = ["#2563EB", "#0284C7", "#0D9488", "#D97706", "#DB2777"]
        
        fig = go.Figure(
            data=[
                go.Bar(
                    x=categories,
                    y=values,
                    marker=dict(color=colors[:len(categories)], cornerradius=8),
                    text=[f"{v:.1f} MW" for v in values],
                    textposition="outside",
                    textfont=dict(size=11, color="#0F172A")
                )
            ]
        )
        
        fig.update_layout(
            title={"text": "<b>میزان مشارکت و تخصیص توان هر واحد تولیدی</b>", "x": 0.5, "xanchor": "center", "font": {"size": 14, "color": "#0F172A"}},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=40),
            xaxis=dict(showgrid=False, tickfont=dict(size=10, color="#334155"), linecolor="#CBD5E1"),
            yaxis=dict(
                showgrid=True, 
                gridcolor="#E2E8F0", 
                tickfont=dict(size=10, color="#334155"), 
                title=dict(text="توان تخصیص یافته (MW)", font=dict(color="#475569", size=11))
            ),
            height=360
        )
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: GA EV Station Placement ---
with tab2:
    st.header("مکان‌یابی و ظرفیت‌سنجی ایستگاه‌های شارژ (الگوریتم ژنتیک)")
    
    col_ga_in, col_ga_out = st.columns([1, 2])
    
    with col_ga_in:
        pop_size = st.slider("اندازه جمعیت (Population)", 20, 100, 40)
        generations = st.slider("تعداد نسل‌ها (Generations)", 20, 200, 80)
        mutation_r = st.slider("نرخ جهش (Mutation Rate)", 0.01, 0.2, 0.05)
        
        np.random.seed(42)
        traffic_nodes = np.random.randint(10, 80, size=10)
        
        st.write("📊 **میزان ترافیک گره‌های کاندید:**")
        st.dataframe(pd.DataFrame({"گره": [f"N{i+1}" for i in range(10)], "ترافیک روزانه": traffic_nodes}).T)
        
        run_ga = st.button("اجرای الگوریتم ژنتیک", key="run_ga_btn")

    ga_solver = EVStationGeneticSolver(num_candidate_nodes=10, pop_size=pop_size, generations=generations, mutation_rate=mutation_r)
    best_sol, best_fit, history = ga_solver.solve(traffic_nodes)
    st.session_state['ga_results'] = {"best_sol": best_sol, "best_fit": best_fit}

    with col_ga_out:
        st.subheader("همگرایی و چیدمان بهینه ایستگاه‌ها")
        
        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(
            x=list(range(1, len(history) + 1)),
            y=history,
            mode='lines+markers',
            marker=dict(size=5, color='#059669'),
            line=dict(color='#059669', width=2.5),
            name='Fitness'
        ))
        
        fig_conv.update_layout(
            title={"text": "<b>روند همگرایی الگوریتم تکاملی</b>", "x": 0.5, "xanchor": "center", "font": {"size": 14, "color": "#0F172A"}},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            template="plotly_white",
            margin=dict(l=20, r=20, t=50, b=40),
            xaxis=dict(title=dict(text="نسل (Generation)", font=dict(size=11)), showgrid=True, gridcolor="#E2E8F0", tickfont=dict(size=10, color="#334155")),
            yaxis=dict(title=dict(text="تابع برازندگی (Fitness)", font=dict(size=11)), showgrid=True, gridcolor="#E2E8F0", tickfont=dict(size=10, color="#334155")),
            height=340
        )
        st.plotly_chart(fig_conv, use_container_width=True)
        
        res_df = pd.DataFrame({
            "گره کاندید": [f"گره {i+1}" for i in range(10)],
            "تعداد شارژر احداثی": best_sol,
            "ترافیک معبر": traffic_nodes
        })
        st.dataframe(res_df.style.highlight_max(subset=["تعداد شارژر احداثی"], color="#DBEAFE"))

# --- TAB 3: AI Assistant Chatbot ---
with tab3:
    st.header("🤖 دستیار تحلیلی هوشمند (Chatbot)")
    st.write("سوالات خود را درباره نتایج بهینه‌سازی، نرخ‌های سایه‌ای و تحلیل حساسیت بپرسید.")
    
    bot = SmartEnergyAssistant(api_key=active_api_key)
    
    user_query = st.text_input("سوال خود را وارد کنید:", value="چرا قیمت سایه‌ای در این حالت تغییر می‌کند؟")
    
    if st.button("ارسال سوال", key="send_chat_btn"):
        context = {
            "lp_results": st.session_state.get('lp_results', {}),
            "ga_results": st.session_state.get('ga_results', {})
        }
        with st.spinner("در حال تحلیل نتایج و پاسخگویی..."):
            ans = bot.ask(user_query, context)
            st.markdown("### 💬 پاسخ دستیار:")
            st.info(ans)