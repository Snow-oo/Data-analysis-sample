# ============================================================
# Streamlit 可视化 Web 应用 - 巴西电商物流 ChatBI
# 基于 data_analysis_module.py 中的分析结果，支持全维度问答
# ============================================================

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# 导入数据分析模块
from data_analysis_module import load_analysis_data

# 设置 matplotlib 使用非交互式后端（避免 Streamlit 线程问题）
import matplotlib
matplotlib.use('Agg')

# 尝试导入字体配置（来自 viz_font.py 或模块内定义）
try:
    from viz_font import setup_matplotlib_simsun
except ImportError:
    def setup_matplotlib_simsun():
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = ["SimSun", "NSimSun", "STSong", "Songti SC", "DejaVu Serif"]
        matplotlib.rcParams["axes.unicode_minus"] = False
setup_matplotlib_simsun()

# ============================================================
# 数据加载（使用 Streamlit 缓存，避免重复计算）
# ============================================================
@st.cache_data
def get_all_analysis_data():
    """调用模块的数据加载函数，返回所有分析结果"""
    return load_analysis_data()

# ============================================================
# 辅助函数：从查询中提取数字
# ============================================================
def extract_number(query, default=5):
    numbers = re.findall(r'\d+', query)
    return int(numbers[0]) if numbers else default

# ============================================================
# ChatBI 查询引擎（扩展版，覆盖全部分析维度）
# ============================================================
def chatbi_query(query,
                 analysis_df, monthly, weekly,
                 order_items, products, category_translation,
                 review_stats, state_stats, corr_matrix,
                 coef_df, logit_coef_df,
                 bad_review_multiplier, cap_99, season_effect,
                 next_forecast, carrier_mean, seller_mean,
                 pearson_r, pearson_p, spearman_r, spearman_p,
                 f_stat, p_anova, model,
                 return_chart=False):
    """
    自然语言查询处理器。
    返回字典包含 answer 和可选的 image 路径。
    """
    q = query.lower().strip()
    output_dir = './outputs/figs/'
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 1. 评分最高的N个产品 ----------
    if ('评分最高' in q or '评分高' in q) and ('产品' in q or '商品' in q):
        top_n = extract_number(q, 5)
        prod_review = (order_items.merge(analysis_df[['order_id', 'review_score']], on='order_id')
                       .groupby('product_id')['review_score'].mean()
                       .sort_values(ascending=False).head(top_n))
        prod_name = products[['product_id', 'product_category_name']].merge(
            category_translation, on='product_category_name', how='left')
        prod_review = prod_review.reset_index().merge(prod_name, on='product_id', how='left')
        result_str = f"评分最高的{top_n}个产品:\n"
        for _, row in prod_review.iterrows():
            cat = row.get('product_category_name_english', row.get('product_category_name', '未知'))
            result_str += f"  {cat} (ID: {row['product_id'][:8]}...) - 平均评分 {row['review_score']:.2f}\n"
        return {'answer': result_str}

    # ---------- 2. 超时率最高的N个州 ----------
    elif '超时率最高' in q and ('州' in q or 'state' in q):
        top_n = extract_number(q, 5)
        state_overdue = analysis_df.groupby('customer_state')['is_overdue'].mean().sort_values(ascending=False).head(top_n)
        result_str = f"超时率最高的{top_n}个州:\n" + "\n".join(f"  {s}: {r*100:.1f}%" for s, r in state_overdue.items())
        return {'answer': result_str}

    # ---------- 3. 超时率最低的N个州 ----------
    elif '超时率最低' in q and ('州' in q or 'state' in q):
        top_n = extract_number(q, 5)
        state_low = analysis_df.groupby('customer_state')['is_overdue'].mean().sort_values(ascending=True).head(top_n)
        result_str = f"超时率最低的{top_n}个州:\n" + "\n".join(f"  {s}: {r*100:.1f}%" for s, r in state_low.items())
        return {'answer': result_str}

    # ---------- 4. 平均交付时间趋势 ----------
    elif '平均交付时间趋势' in q or '交付时间变化' in q:
        if return_chart:
            plt.figure(figsize=(10,5))
            plt.plot(monthly['purchase_month'], monthly['total_delivery_time_capped'], marker='o', color='steelblue')
            plt.title('月度平均交付时间趋势')
            plt.xlabel('月份')
            plt.ylabel('交付时间 (天)')
            plt.grid(True)
            chart_path = os.path.join(output_dir, 'chatbi_delivery_trend.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            return {'answer': '已生成月度交付时间趋势图', 'image': chart_path}
        else:
            recent = monthly.sort_values('purchase_month', ascending=False).head(12)
            result_str = "最近12个月的平均交付时间（天）:\n" + "\n".join(
                f"  {row['purchase_month'].strftime('%Y-%m')}: {row['total_delivery_time_capped']:.2f}"
                for _, row in recent.iterrows())
            return {'answer': result_str}

    # ---------- 5. 交付时间与评分的相关系数 ----------
    elif ('相关系数' in q or '相关性' in q) and ('交付时间' in q or '物流' in q) and ('评分' in q):
        result_str = f"总交付时间与用户评分的皮尔逊相关系数为 {pearson_r:.3f}，p值为 {pearson_p:.2e}。\n（负值表示交付时间越长，评分越低）"
        return {'answer': result_str}

    # ---------- 6. 指定州的物流指标 ----------
    elif '州' in q and ('平均交付时间' in q or '物流' in q or '指标' in q):
        states = re.findall(r'\b([A-Z]{2})\b', q.upper())
        if not states:
            return {'answer': '请指定具体州名（如 SP, RJ, MG）'}
        state = states[0]
        state_data = analysis_df[analysis_df['customer_state'] == state]
        if len(state_data) == 0:
            return {'answer': f'未找到州 {state} 的数据。'}
        avg_delivery = state_data['total_delivery_time_capped'].mean()
        avg_deviation = state_data['delivery_deviation_capped'].mean()
        overdue_rate = state_data['is_overdue'].mean() * 100
        avg_score = state_data['review_score'].mean()
        result_str = (f"州 {state} 的物流指标:\n"
                      f"  平均交付时间: {avg_delivery:.2f} 天\n"
                      f"  平均交付偏差: {avg_deviation:.2f} 天\n"
                      f"  超时率: {overdue_rate:.1f}%\n"
                      f"  平均评分: {avg_score:.2f}")
        return {'answer': result_str}

    # ---------- 7. 超时率与评分的关系 ----------
    elif '超时率' in q and '评分' in q:
        overdue_by_score = analysis_df.groupby('review_score')['is_overdue'].mean() * 100
        result_str = "不同评分下的超时率:\n" + "\n".join(f"  {int(score)} 星: {rate:.1f}%" for score, rate in overdue_by_score.items())
        if return_chart:
            plt.figure(figsize=(8,5))
            overdue_by_score.plot(kind='bar', color='coral')
            plt.title('各评分等级的超时率')
            plt.ylabel('超时率 (%)')
            plt.xlabel('用户评分')
            plt.xticks(rotation=0)
            chart_path = os.path.join(output_dir, 'chatbi_overdue_by_score.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            return {'answer': result_str, 'image': chart_path}
        return {'answer': result_str}

    # ---------- 8. 特定星级订单的物流指标 ----------
    elif any(str(i)+'星' in q for i in range(1,6)):
        score = None
        for s in range(1,6):
            if f"{s}星" in q:
                score = s
                break
        if score and score in review_stats.index:
            row = review_stats.loc[score]
            result_str = f"{score}星订单的物流指标:\n"
            result_str += f"  平均交付时间: {row['total_delivery_time_capped_mean']:.2f} 天\n"
            result_str += f"  交付时间中位数: {row['total_delivery_time_capped_median']:.2f} 天\n"
            result_str += f"  超时率: {row['is_overdue_mean']*100:.1f}%\n"
            result_str += f"  平均承运商运输时间: {row['carrier_transit_time_capped_mean']:.2f} 天\n"
            result_str += f"  平均卖家处理时间: {row['seller_processing_time_capped_mean']:.2f} 天\n"
            result_str += f"  平均配送距离: {row['distance_km_mean']:.1f} km"
            return {'answer': result_str}
        else:
            return {'answer': f"未找到 {score} 星订单的数据"}

    # ---------- 9. 哪个因素对评分影响最大/最小 ----------
    elif ('影响最大' in q or '影响最小' in q) and ('评分' in q or '因素' in q):
        abs_coef = coef_df['Coefficient'].abs()
        if '最大' in q:
            max_idx = abs_coef.idxmax()
            max_feat = coef_df.loc[max_idx, 'Feature']
            max_val = coef_df.loc[max_idx, 'Coefficient']
            result_str = f"对评分影响最大的因素是「{max_feat}」，标准化系数为 {max_val:.3f}。"
        else:
            min_idx = abs_coef.idxmin()
            min_feat = coef_df.loc[min_idx, 'Feature']
            min_val = coef_df.loc[min_idx, 'Coefficient']
            result_str = f"对评分影响最小的因素是「{min_feat}」，标准化系数为 {min_val:.3f}。"
        return {'answer': result_str}

    # ---------- 10. 超时订单差评倍数 ----------
    elif '超时订单' in q and '差评率' in q and '倍' in q:
        result_str = f"超时订单的差评率是准时订单的 {bad_review_multiplier:.1f} 倍。"
        return {'answer': result_str}

    # ---------- 11. 州级评分排名 ----------
    elif '评分最高' in q and '州' in q:
        top_n = extract_number(q, 1)
        top_states = state_stats.nlargest(top_n, 'review_score')[['customer_state', 'review_score']]
        result_str = f"平均评分最高的{top_n}个州:\n" + "\n".join(f"  {row['customer_state']}: {row['review_score']:.2f}" for _, row in top_states.iterrows())
        return {'answer': result_str}
    elif '评分最低' in q and '州' in q:
        top_n = extract_number(q, 1)
        bottom_states = state_stats.nsmallest(top_n, 'review_score')[['customer_state', 'review_score']]
        result_str = f"平均评分最低的{top_n}个州:\n" + "\n".join(f"  {row['customer_state']}: {row['review_score']:.2f}" for _, row in bottom_states.iterrows())
        return {'answer': result_str}

    # ---------- 12. 交付时间最长的州 ----------
    elif '交付时间最长' in q and '州' in q:
        top_n = extract_number(q, 1)
        slowest = state_stats.nlargest(top_n, 'total_delivery_time_capped')[['customer_state', 'total_delivery_time_capped']]
        result_str = f"交付时间最长的{top_n}个州:\n" + "\n".join(f"  {row['customer_state']}: {row['total_delivery_time_capped']:.2f} 天" for _, row in slowest.iterrows())
        return {'answer': result_str}

    # ---------- 13. 月度最差/最好月份 ----------
    elif '哪个月份' in q and ('交付时间最长' in q or '超时率最高' in q):
        if '交付时间最长' in q:
            worst_month = monthly.loc[monthly['total_delivery_time_capped'].idxmax(), 'purchase_month']
            worst_val = monthly['total_delivery_time_capped'].max()
            result_str = f"平均交付时间最长的月份是 {worst_month.strftime('%Y年%m月')}，达到 {worst_val:.2f} 天。"
        elif '超时率最高' in q:
            worst_month = monthly.loc[monthly['is_overdue'].idxmax(), 'purchase_month']
            worst_val = monthly['is_overdue'].max() * 100
            result_str = f"超时率最高的月份是 {worst_month.strftime('%Y年%m月')}，超时率为 {worst_val:.1f}%。"
        else:
            return {'answer': "请明确指出是交付时间还是超时率。"}
        return {'answer': result_str}

    # ---------- 14. 下个月预测 ----------
    elif '下个月' in q and ('交付时间' in q or '预测' in q):
        result_str = f"根据时间序列模型预测，下个月的平均交付时间约为 {next_forecast:.2f} 天。"
        return {'answer': result_str}

    # ---------- 15. 异常值截断阈值 ----------
    elif '截断' in q and '上限' in q:
        result_str = f"交付时间的上限截断值（99%分位数）为 {cap_99:.2f} 天。"
        return {'answer': result_str}

    # ---------- 16. 承运商 vs 卖家处理时间 ----------
    elif '哪个环节' in q and '耗时最长' in q:
        if carrier_mean > seller_mean:
            result_str = f"承运商运输时间平均为 {carrier_mean:.2f} 天，卖家处理时间平均为 {seller_mean:.2f} 天，承运商环节耗时更长。"
        else:
            result_str = f"卖家处理时间平均为 {seller_mean:.2f} 天，承运商运输时间平均为 {carrier_mean:.2f} 天，卖家环节耗时更长。"
        return {'answer': result_str}

    # ---------- 17. 年末季节效应 ----------
    elif '年末' in q and ('交付时间' in q or '物流' in q):
        result_str = f"11-12月的平均交付时间比其它月份平均长约 {season_effect:.2f} 天。"
        return {'answer': result_str}

    # ---------- 18. 其他变量与评分的相关性 ----------
    elif '相关性' in q and '评分' in q:
        var_map = {
            '距离': 'distance_km',
            '商品件数': 'items_count',
            '运费率': 'freight_rate',
            '承运商': 'carrier_transit_time_capped',
            '卖家处理': 'seller_processing_time_capped',
            '交付偏差': 'delivery_deviation_capped'
        }
        for cn, col in var_map.items():
            if cn in q:
                r = corr_matrix.loc['review_score', col]
                result_str = f"「{cn}」与评分的相关系数为 {r:.3f}。"
                return {'answer': result_str}
        return {'answer': "请指定具体变量：距离、商品件数、运费率、承运商时间等。"}

    # ---------- 19. 订单量最多的月份 ----------
    elif '订单量最多' in q or '订单最多' in q:
        max_month = monthly.loc[monthly['order_count'].idxmax(), 'purchase_month']
        max_cnt = monthly['order_count'].max()
        result_str = f"订单量最多的月份是 {max_month.strftime('%Y年%m月')}，共 {max_cnt} 单。"
        return {'answer': result_str}

    # ---------- 20. 主要结论摘要 ----------
    elif '主要结论' in q or '分析结论' in q:
        result_str = (
            "物流分析主要结论：\n"
            "1. 交付时间与评分显著负相关（r=-0.350）。\n"
            "2. 超时订单差评率是准时订单的5.9倍。\n"
            "3. 承运商运输时长对评分负面影响最大。\n"
            "4. 年末（11-12月）物流时效显著恶化。\n"
            "5. 建议优化承运商路线、动态调整承诺时间、部署旺季运力。"
        )
        return {'answer': result_str}

    # ---------- 21. ANOVA 方差分析结果 ----------
    elif '方差分析' in q or 'ANOVA' in q:
        result_str = f"不同评分组之间的总交付时间存在极显著差异（F={f_stat:.2f}, p={p_anova:.2e}）。"
        return {'answer': result_str}

    # ---------- 22. 回归系数详细列表 ----------
    elif '回归系数' in q:
        coef_str = "\n".join(f"  {row['Feature']}: {row['Coefficient']:.3f}" for _, row in coef_df.iterrows())
        result_str = f"标准化回归系数（影响大小）:\n{coef_str}"
        return {'answer': result_str}

    # ---------- 23. 默认回答 ----------
    else:
        return {'answer': "抱歉，我还无法回答这个问题。您可以尝试以下示例问题:\n"
                          "• 评分最高的5个产品\n"
                          "• 超时率最高的3个州\n"
                          "• 平均交付时间趋势\n"
                          "• 交付时间与评分的相关系数\n"
                          "• 州SP的物流指标\n"
                          "• 超时率与评分的关系\n"
                          "• 1星订单的物流指标\n"
                          "• 哪个因素对评分影响最大\n"
                          "• 超时订单的差评倍数\n"
                          "• 下个月交付时间预测\n"
                          "• 方差分析结果"}

# ============================================================
# Streamlit 界面
# ============================================================
def main():
    st.set_page_config(page_title="巴西电商物流智能问答系统", layout="wide")
    st.title("📦 巴西电商物流智能问答系统")
    st.markdown("基于 Olist 公开数据，使用自然语言查询物流时效与用户评分的关系。")

    # 加载数据（首次运行较慢，之后缓存）
    with st.spinner("正在加载数据，请稍候..."):
        (analysis_df, monthly, weekly, order_items, products, category_translation,
         review_stats, state_stats, corr_matrix, coef_df, logit_coef_df,
         bad_review_multiplier, cap_99, season_effect, next_forecast,
         carrier_mean, seller_mean, pearson_r, pearson_p, spearman_r, spearman_p,
         f_stat, p_anova, model) = get_all_analysis_data()

    st.success(f"数据加载完成！共 {len(analysis_df)} 条有效订单记录。")

    # 侧边栏：示例问题
    st.sidebar.header("💡 示例问题")
    example_queries = [
        "评分最高的10个产品",
        "超时率最高的3个州",
        "超时率最低的州",
        "平均交付时间趋势",
        "交付时间与评分的相关系数",
        "州SP的物流指标",
        "超时率与评分的关系",
        "1星订单的物流指标",
        "哪个因素对评分影响最大",
        "超时订单的差评率是准时订单的多少倍",
        "下个月的平均交付时间预测",
        "哪个月份交付时间最长",
        "方差分析结果"
    ]
    for q in example_queries:
        if st.sidebar.button(q):
            st.session_state.query = q

    # 主界面输入
    if "query" not in st.session_state:
        st.session_state.query = ""

    query = st.text_input("✍️ 请输入您的问题：", value=st.session_state.query,
                          placeholder="例如：超时率最高的5个州")
    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("🔍 查询")
    with col2:
        chart_check = st.checkbox("📊 自动生成图表（如果支持）")

    if submit and query:
        with st.spinner("思考中..."):
            result = chatbi_query(
                query,
                analysis_df, monthly, weekly,
                order_items, products, category_translation,
                review_stats, state_stats, corr_matrix,
                coef_df, logit_coef_df,
                bad_review_multiplier, cap_99, season_effect,
                next_forecast, carrier_mean, seller_mean,
                pearson_r, pearson_p, spearman_r, spearman_p,
                f_stat, p_anova, model,
                return_chart=chart_check
            )
        st.markdown("### 📋 回答")
        st.write(result['answer'])
        if 'image' in result:
            st.image(result['image'], caption="自动生成的图表")

    # 页脚
    st.markdown("---")
    st.caption("数据来源：Kaggle Olist Brazilian E-Commerce Public Dataset | 分析维度：物流时效、用户评分、地理分布、时间趋势、回归分析、时间序列预测")

if __name__ == "__main__":
    main()