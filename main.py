import streamlit as st
import warnings

from src.optimization.strategy_optimization import strategy_optimization_page
from src.prediction.ai_prediction import ai_prediction_page
from src.utils.data_analysis import data_analysis_page
from src.utils.data_import import data_import_page

warnings.filterwarnings('ignore')

# ========================= 页面设置 =========================
st.set_page_config(
    page_title="风速预测与风电场优化系统",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ========================= 主程序入口 =========================
def main():
    st.sidebar.title("🌬️ 风能研究平台")

    st.sidebar.markdown("---")
    st.sidebar.info("选择功能模块开始分析")

    app_mode = st.sidebar.selectbox(
        "系统功能",
        ["🏠 系统首页", "📊 数据导入", "📈 数据分析",
         "🤖 风速预测", "⚡ 布局优化"]
    )

    if app_mode == "🏠 系统首页":
        show_home_page()
    elif app_mode == "📊 数据导入":
        data_import_page()
    elif app_mode == "📈 数据分析":
        data_analysis_page()
    elif app_mode == "🤖 风速预测":
        ai_prediction_page()
    elif app_mode == "⚡ 布局优化":
        strategy_optimization_page()


# ========================= 首页内容 =========================
def show_home_page():
    # 主标题区域
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🌬️ 风速预测与风电场优化系统")
        st.markdown("**智能风能分析与决策平台**")
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3050/3050159.png", width=100)

    st.markdown("---")

    # 系统简介
    st.subheader("📖 平台简介")
    st.markdown("""
    集成**气象数据分析**、**AI风速预测**和**空间优化算法**，
    为风电场规划提供完整的智能决策支持。
    """)

    # 功能模块卡片
    st.subheader("🔧 核心功能")


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.container(border=True):
            st.markdown("### 📊 数据导入")
            st.markdown("""
            - 数据上传验证
            - 格式自动识别
            - 质量评估报告
            """)

    with col2:
        with st.container(border=True):
            st.markdown("### 📈 数据分析")
            st.markdown("""
            - 时空可视化
            - 相关性分析
            - 模式识别
            """)

    with col3:
        with st.container(border=True):
            st.markdown("### 🤖 风速预测")
            st.markdown("""
            - 多算法对比
            - 精度评估
            - 预测可视化
            """)

    with col4:
        with st.container(border=True):
            st.markdown("### ⚡ 布局优化")
            st.markdown("""
            - 智能排布
            - 多目标优化
            - 方案可视化
            """)

    st.markdown("---")

    # 数据概览
    st.subheader("📈 系统概览")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("数据处理", "52K+ 记录", "支持大规模数据")
    with col2:
        st.metric("预测算法", "5+ 模型", "AI精准预测")
    with col3:
        st.metric("优化方案", "3+ 算法", "智能布局")

    # 快速开始指引
    st.markdown("---")
    st.subheader("🚀 快速开始")

    steps = st.columns(4)
    with steps[0]:
        st.markdown("**1. 数据导入**")
        st.markdown("上传气象CSV数据")
    with steps[1]:
        st.markdown("**2. 数据分析**")
        st.markdown("探索数据特征")
    with steps[2]:
        st.markdown("**3. 风速预测**")
        st.markdown("训练预测模型")
    with steps[3]:
        st.markdown("**4. 布局优化**")
        st.markdown("生成最优方案")


# ========================= 程序入口 =========================
if __name__ == "__main__":
    main()