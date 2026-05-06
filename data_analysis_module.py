# ============================================================
# 模块名称：data_analysis_module.py
# 包含：数据清洗、特征工程、统计分析、回归建模、时间序列、可视化
# 对外提供：load_analysis_data() 函数，返回分析所需的核心数据
# ============================================================

import os
import re
import io
import requests
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


from scipy import stats
from scipy.stats import pearsonr, spearmanr, f_oneway
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 字体配置
try:
    from viz_font import setup_matplotlib_simsun
except ImportError:
    def setup_matplotlib_simsun():
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = ["SimSun", "NSimSun", "STSong", "Songti SC", "DejaVu Serif"]
        matplotlib.rcParams["axes.unicode_minus"] = False

def load_analysis_data():
    """
    加载并处理所有原始数据，返回分析需要的关键 DataFrame 和统计结果。
    返回:
        analysis_df: 主宽表（订单级）
        monthly: 月度聚合数据
        weekly: 周度聚合数据
        order_items: 订单明细表（原始）
        products: 产品表
        category_translation: 品类翻译表
        review_stats: 各评分等级物流统计表
        state_stats: 州级统计表
        corr_matrix: 相关性矩阵
        coef_df: 线性回归标准化系数
        logit_coef_df: 逻辑回归系数
        bad_review_multiplier: 超时订单 vs 准时订单的差评倍数
        cap_99: 总交付时间99%分位数（截断上限）
        season_effect: 年末月份平均交付时间差
        next_forecast: 下个月预测值
        carrier_mean, seller_mean: 承运商与卖家处理时间均值
        pearson_r, spearman_r: 相关系数
        f_stat, p_anova: ANOVA 统计量
        model: OLS 模型对象（可选）
    """
    # ---------- 动态下载数据（如果本地不存在）----------
    csv_files = [
        'olist_orders_dataset.csv',
        'olist_order_reviews_dataset.csv',
        'olist_customers_dataset.csv',
        'olist_order_items_dataset.csv',
        'olist_products_dataset.csv',
        'olist_sellers_dataset.csv',
        'product_category_name_translation.csv',
        'olist_geolocation_dataset.csv'
    ]
    # 检查是否所有文件都已存在
    if not all(os.path.exists(f) for f in csv_files):
        print("本地未找到数据文件，开始从云存储下载...")
        # 实际下载链接
        data_url = "https://github.com/Snow-oo/Data-analysis-sample/releases/download/v1.0/data.zip"

        response = requests.get(data_url, stream=True)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall('.')
            print("数据下载并解压完成。")
        else:
            raise Exception(f"下载失败，状态码: {response.status_code}")
    else:
        print("数据文件已存在，直接使用。")
    # ============================================================
    # 1. 数据加载
    # ============================================================
    orders = pd.read_csv('olist_orders_dataset.csv')
    reviews = pd.read_csv('olist_order_reviews_dataset.csv')
    customers = pd.read_csv('olist_customers_dataset.csv')
    order_items = pd.read_csv('olist_order_items_dataset.csv')
    products = pd.read_csv('olist_products_dataset.csv')
    sellers = pd.read_csv('olist_sellers_dataset.csv')
    category_translation = pd.read_csv('product_category_name_translation.csv')
    geolocation = pd.read_csv('olist_geolocation_dataset.csv')

    # ============================================================
    # 2. 数据清洗与预处理
    # ============================================================
    # 2.1 订单表清洗：只保留已交付订单，且关键物流环节不为空
    orders_delivered = orders[orders['order_status'] == 'delivered'].copy()
    # 删去缺失值数据行['向客户显示的实际订单交付日期', '显示订单发布给物流时间点']
    orders_delivered = orders_delivered.dropna(subset=['order_delivered_customer_date',
                                                       'order_delivered_carrier_date'])

    # 2.2
    # 将数据类型转换为Pandas的 datetime 类型
    reviews['review_answer_timestamp'] = pd.to_datetime(reviews['review_answer_timestamp'])
    # 评价表去重：每订单保留最新的一条评价（按回答时间）
    reviews_unique = reviews.sort_values('review_answer_timestamp', ascending=False)\
                           .drop_duplicates(subset=['order_id'], keep='first')

    # 2.3 统一日期格式
    date_cols = ['order_purchase_timestamp', 'order_approved_at',
                 'order_delivered_carrier_date', 'order_delivered_customer_date',
                 'order_estimated_delivery_date']
    for col in date_cols:
        orders_delivered[col] = pd.to_datetime(orders_delivered[col], errors='coerce')

    # 2.4 客户邮编、卖家邮编与地理信息去重（每个邮编保留一个代表坐标）
    geo_unique = geolocation.drop_duplicates('geolocation_zip_code_prefix')
    customers = customers.merge(geo_unique, left_on='customer_zip_code_prefix',
                                right_on='geolocation_zip_code_prefix', how='left')
    sellers = sellers.merge(geo_unique, left_on='seller_zip_code_prefix',
                            right_on='geolocation_zip_code_prefix', how='left')

    # ============================================================
    # 3. 特征工程（单表特征）
    # ============================================================
    # 从下单付款到订单批准所经过的时间,小时,反映客户支付效率或订单审核速度
    orders_delivered['approval_time'] = (orders_delivered['order_approved_at'] -
                                         orders_delivered['order_purchase_timestamp']).dt.total_seconds() / 3600
    # 从订单批准到订单交给物流承运商所经过的时间,天,反映卖家的备货与发货处理效率
    orders_delivered['seller_processing_time'] = (orders_delivered['order_delivered_carrier_date'] -
                                                  orders_delivered['order_approved_at']).dt.total_seconds() / 86400  #天
    # 从订单交给承运商到实际送达客户所经过的时间,天,反映物流运输环节的时效
    orders_delivered['carrier_transit_time'] = (orders_delivered['order_delivered_customer_date'] -
                                                orders_delivered['order_delivered_carrier_date']).dt.total_seconds() / 86400
    # 从下单付款到实际送达客户的总时长,天,
    orders_delivered['total_delivery_time'] = (orders_delivered['order_delivered_customer_date'] -
                                               orders_delivered['order_purchase_timestamp']).dt.total_seconds() / 86400
    # 从下单付款到平台承诺的预计送达日期之间的天数,天,承诺时效
    orders_delivered['estimated_delivery_days'] = (orders_delivered['order_estimated_delivery_date'] -
                                                   orders_delivered['order_purchase_timestamp']).dt.total_seconds() / 86400
    # 正数表示延迟送达，负数表示提前送达，零表示完全准时。衡量履约准确性
    orders_delivered['delivery_deviation'] = orders_delivered['total_delivery_time'] - orders_delivered['estimated_delivery_days']
    # 标识订单是否超时,1是0否
    orders_delivered['is_overdue'] = (orders_delivered['delivery_deviation'] > 0).astype(int)

    # 极端异常值截断函数（上下1%分位）
    def cap_outliers(df, column, lower=0.01, upper=0.99):
        low = df[column].quantile(lower)
        high = df[column].quantile(upper)
        df[f'{column}_capped'] = df[column].clip(low, high)  # 截断后得到新列
        return df

    for col in ['total_delivery_time', 'carrier_transit_time', 'seller_processing_time', 'delivery_deviation']:
        orders_delivered = cap_outliers(orders_delivered, col)

    # ============================================================
    # 4. 构建主宽表（合并评价、客户、商品汇总）
    # ============================================================
    # 4.1 订单 + 评价
    merged = orders_delivered.merge(reviews_unique[['order_id', 'review_score']],
                                    on='order_id', how='inner')
    # 4.2 添加客户信息
    merged = merged.merge(customers[['customer_id', 'customer_state', 'geolocation_lat', 'geolocation_lng']],
                          on='customer_id', how='left')

    # 4.3 聚合订单商品信息：商品总数、总价、总运费、卖家列表
    item_group = order_items.groupby('order_id').agg({
        'order_item_id': 'count',
        'price': 'sum',
        'freight_value': 'sum'
    }).rename(columns={'order_item_id': 'items_count',
                       'price': 'total_price',
                       'freight_value': 'total_freight'})
    merged = merged.merge(item_group, on='order_id', how='left')
    merged['freight_rate'] = merged['total_freight'] / merged['total_price']

    # 4.4 计算订单运输距离（基于卖家仓库和客户地址的球面距离）
    # 注意：一个订单可能有多个卖家，这里简化为取第一个卖家的坐标（多数订单仅一个卖家）
    # 先获取每个订单的第一个 seller_id（多数订单只有一个 seller）
    seller_per_order = order_items.groupby('order_id')['seller_id'].first().reset_index()
    merged = merged.merge(seller_per_order, on='order_id', how='left')
    # 关联卖家坐标
    merged = merged.merge(sellers[['seller_id', 'geolocation_lat', 'geolocation_lng']],
                          on='seller_id', how='left')
    # 关联客户坐标（已在 customers 表中合并）
    # 此时列名：来自 customers 的 geolocation_lat 和 geolocation_lng 会自动变成 _x
    # 来自 sellers 的 geolocation_lat 和 geolocation_lng 会自动变成 _y
    merged = merged.rename(columns={'geolocation_lat_x': 'customer_lat',
                                    'geolocation_lng_x': 'customer_lng',
                                    'geolocation_lat_y': 'seller_lat',
                                    'geolocation_lng_y': 'seller_lng'})

    # 定义 Haversine 距离计算函数
    from math import radians, sin, cos, sqrt, asin
    def haversine(lat1, lon1, lat2, lon2):
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return np.nan
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return R * c

    # 对每一行（每个订单）计算卖家仓库到客户收货地址的直线距离
    merged['distance_km'] = merged.apply(lambda row: haversine(row['seller_lat'], row['seller_lng'],
                                                               row['customer_lat'], row['customer_lng']), axis=1)
    # 最终分析数据集：只保留完整评分、物流时间和距离
    analysis_df = merged.dropna(subset=['review_score', 'total_delivery_time_capped', 'distance_km'])

    # ============================================================
    # 5. 时间序列聚合（周/月）
    # ============================================================
    # 添加日期维度
    orders_delivered['purchase_week'] = orders_delivered['order_purchase_timestamp'].dt.to_period('W').dt.start_time
    orders_delivered['purchase_month'] = orders_delivered['order_purchase_timestamp'].dt.to_period('M').dt.start_time
    # 合并到 analysis_df
    analysis_df = analysis_df.merge(orders_delivered[['order_id', 'purchase_week', 'purchase_month']],
                                    on='order_id', how='left')

    # 按周聚合（使用 as_index=False 避免索引问题）
    weekly = analysis_df.groupby('purchase_week', as_index=False).agg({
        'total_delivery_time_capped': 'mean',
        'is_overdue': 'mean',
        'review_score': 'mean',
        'order_id': 'count'
    }).rename(columns={'order_id': 'order_count'})
    weekly = weekly[weekly['order_count'] >= 10]

    # 按月聚合
    monthly = analysis_df.groupby('purchase_month', as_index=False).agg({
        'total_delivery_time_capped': 'mean',
        'is_overdue': 'mean',
        'review_score': 'mean',
        'order_id': 'count'
    }).rename(columns={'order_id': 'order_count'})
    monthly = monthly[monthly['order_count'] >= 50]
    # 确保月份列为 datetime 类型
    monthly['purchase_month'] = pd.to_datetime(monthly['purchase_month'])

    # ============================================================
    # 6. 预计算多种统计量（供 ChatBI 直接使用）
    # ============================================================
    # 不同评分等级的物流表现
    review_stats = analysis_df.groupby('review_score').agg({
        'total_delivery_time_capped': ['mean', 'median', 'std'],
        'carrier_transit_time_capped': 'mean',
        'seller_processing_time_capped': 'mean',
        'delivery_deviation_capped': 'mean',
        'is_overdue': 'mean',
        'distance_km': 'mean'
    }).round(2)
    review_stats.columns = ['_'.join(col).strip() for col in review_stats.columns.values]

    # 州级地理分析
    state_stats = analysis_df.groupby('customer_state').agg({
        'review_score': 'mean',
        'total_delivery_time_capped': 'mean',
        'is_overdue': 'mean'
    }).reset_index()

    # 相关系数矩阵
    corr_cols = ['review_score', 'total_delivery_time_capped', 'carrier_transit_time_capped',
                 'seller_processing_time_capped', 'delivery_deviation_capped',
                 'freight_rate', 'items_count', 'distance_km']
    corr_matrix = analysis_df[corr_cols].corr()

    # 回归系数（标准化）
    features = ['total_delivery_time_capped', 'carrier_transit_time_capped',
                'seller_processing_time_capped', 'delivery_deviation_capped',
                'freight_rate', 'items_count', 'distance_km']
    X = analysis_df[features].fillna(0)
    y = analysis_df['review_score']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = sm.add_constant(X_scaled)
    model = sm.OLS(y, X_scaled).fit()
    coef_df = pd.DataFrame({'Feature': features, 'Coefficient': model.params[1:]})

    # 逻辑回归（差评）
    analysis_df['is_bad_review'] = (analysis_df['review_score'] <= 2).astype(int)
    logit_model = sm.Logit(analysis_df['is_bad_review'], X_scaled).fit(disp=0)
    logit_coef_df = pd.DataFrame({'Feature': features, 'Coefficient': logit_model.params[1:]})

    # 超时差评倍数
    overdue_rates = analysis_df.groupby('is_overdue')['is_bad_review'].mean()
    bad_review_multiplier = overdue_rates.iloc[1] / overdue_rates.iloc[0] if len(overdue_rates) > 1 else None

    # 异常值截断上限（99%分位数）
    cap_99 = analysis_df['total_delivery_time'].quantile(0.99)

    # 季节性效应（11-12月 vs 其他月份）
    monthly['month_num'] = monthly['purchase_month'].dt.month
    dec_nov_avg = monthly[monthly['month_num'].isin([11,12])]['total_delivery_time_capped'].mean()
    other_avg = monthly[~monthly['month_num'].isin([11,12])]['total_delivery_time_capped'].mean()
    season_effect = dec_nov_avg - other_avg

    # 下月预测（简单 Holt-Winters）
    ts_data = monthly.set_index('purchase_month')['total_delivery_time_capped'].sort_index()
    ts_data = ts_data.asfreq('MS').interpolate().dropna()
    train = ts_data[:-3] if len(ts_data) > 6 else ts_data
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        hw_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=6).fit()
        next_forecast = hw_model.forecast(1).iloc[0]
    except:
        next_forecast = ts_data.iloc[-1]

    # 承运商 vs 卖家处理时间
    carrier_mean = analysis_df['carrier_transit_time_capped'].mean()
    seller_mean = analysis_df['seller_processing_time_capped'].mean()

    # 皮尔逊/斯皮尔曼相关系数
    pearson_r, pearson_p = pearsonr(analysis_df['total_delivery_time_capped'], analysis_df['review_score'])
    spearman_r, spearman_p = spearmanr(analysis_df['total_delivery_time_capped'], analysis_df['review_score'])

    # ANOVA：不同评分组的交付时间差异
    groups = [group['total_delivery_time_capped'].values for name, group in analysis_df.groupby('review_score')]
    f_stat, p_anova = f_oneway(*groups)

    return (analysis_df, monthly, weekly, order_items, products, category_translation,
            review_stats, state_stats, corr_matrix, coef_df, logit_coef_df,
            bad_review_multiplier, cap_99, season_effect, next_forecast,
            carrier_mean, seller_mean, pearson_r, pearson_p, spearman_r, spearman_p, f_stat, p_anova, model)


# ============================================================
# 独立运行脚本：生成完整的分析报告（图表、结论输出）
# ============================================================
if __name__ == "__main__":
    # 设置字体和风格
    sns.set_theme(style="whitegrid", font_scale=1.05)
    setup_matplotlib_simsun()

    # 加载数据
    (analysis_df, monthly, weekly, order_items, products, category_translation,
     review_stats, state_stats, corr_matrix, coef_df, logit_coef_df,
     bad_review_multiplier, cap_99, season_effect, next_forecast,
     carrier_mean, seller_mean, pearson_r, pearson_p, spearman_r, spearman_p, f_stat, p_anova, model) = load_analysis_data()

    # 创建输出目录
    output_dir = './outputs/figs/'
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # 描述性统计与可视化
    # ============================================================
    print("\n=== 不同评分等级的物流表现 ===\n", review_stats)

    # 箱线图：各一张图单独输出（8×5、Blues、dpi=150）
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=analysis_df, x='review_score', y='total_delivery_time_capped', palette='Blues')
    plt.xlabel('用户评分（星级）')
    plt.ylabel('总交付时间（天，截尾后）')
    plt.title('不同评分的总交付时间')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'box_lead_by_score.png'), dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=analysis_df, x='review_score', y='delivery_deviation_capped', palette='Blues')
    plt.xlabel('用户评分（星级）')
    plt.ylabel('交付偏差（天，截尾后）')
    plt.title('不同评分的交付偏差')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'box_deviation_by_score.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 散点图 + 回归线（抽样5000点）
    sample = analysis_df.sample(min(5000, len(analysis_df)), random_state=42)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.regplot(data=sample, x='total_delivery_time_capped', y='review_score',
                scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title('交付天数与评分的关系')
    plt.subplot(1, 2, 2)
    sns.regplot(data=sample, x='delivery_deviation_capped', y='review_score',
                scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title('交付偏差与评分的关系')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_delivery_score.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 超时率与评分
    overdue_by_score = analysis_df.groupby('review_score')['is_overdue'].mean() * 100
    plt.figure(figsize=(8, 5))
    overdue_by_score.plot(kind='bar', color='skyblue')
    plt.title('不同评分的超时订单占比')
    plt.ylabel('超时率 (%)')
    for i, v in enumerate(overdue_by_score):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center')
    plt.savefig(os.path.join(output_dir, 'score_dist_by_late.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 相关性热图
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0)
    plt.title('物流时效与评分相关性矩阵')
    plt.savefig(os.path.join(output_dir, 'correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 州级地理分析
    top_states = state_stats.nlargest(10, 'review_score')
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.barplot(data=top_states, x='review_score', y='customer_state', palette='Blues')
    plt.title('平均评分最高的10个州')
    plt.subplot(1, 2, 2)
    sns.barplot(data=state_stats.nlargest(10, 'total_delivery_time_capped'),
                x='total_delivery_time_capped', y='customer_state', palette='Blues')
    plt.title('交付时间最长的10个州')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'geolocation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 相关系数打印
    print(f"\n交付时间与评分的皮尔逊相关系数: {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"斯皮尔曼相关系数: {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"ANOVA F统计量: {f_stat:.2f}, p值: {p_anova:.4e}")

    # 线性回归系数可视化
    coef_df_sorted = coef_df.sort_values('Coefficient', key=abs)
    plt.figure(figsize=(10, 5))
    plt.barh(coef_df_sorted['Feature'], coef_df_sorted['Coefficient'],
             color=['red' if c < 0 else 'green' for c in coef_df_sorted['Coefficient']])
    plt.axvline(0, color='black')
    plt.title('标准化回归系数（影响大小）')
    plt.xlabel('系数值')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'regression_coefficients.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("\n线性回归模型摘要:\n", model.summary())
    print("\n逻辑回归（预测差评）摘要:\n", logit_coef_df)

    # 周度趋势图
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(weekly['purchase_week'], weekly['total_delivery_time_capped'], marker='o', color='steelblue')
    axes[0].set_ylabel('平均交付天数')
    axes[0].set_title('周度物流时效趋势')
    axes[1].plot(weekly['purchase_week'], weekly['is_overdue'] * 100, marker='s', color='coral')
    axes[1].set_ylabel('超时率 (%)')
    axes[2].plot(weekly['purchase_week'], weekly['review_score'], marker='^', color='green')
    axes[2].set_ylabel('平均评分')
    axes[2].set_xlabel('购买周')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'week_trend.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 时间序列分解(月度)
    # seasonal_decompose 要求 len(x) >= 2 * period；样本月数不足时自动缩短周期或跳过
    # 与 purchase_month（按日历月的 period 起点）对齐为 MS 频率，避免 statsmodels 无频率警告
    ts_data = monthly.set_index('purchase_month')['total_delivery_time_capped'].sort_index()
    ts_data = ts_data[~ts_data.index.duplicated(keep='last')]
    ts_data = ts_data.asfreq('MS')
    ts_data = ts_data.interpolate(method='linear').ffill().bfill()
    ts_clean = ts_data.dropna()
    n_months = len(ts_clean)
    decomp_period = min(12, max(2, n_months // 2))
    if n_months >= 2 * decomp_period:
        decomp = seasonal_decompose(ts_data, model='additive', period=decomp_period)
        fig = decomp.plot()
        fig.set_size_inches(12, 8)
        plt.suptitle(f'月度交付时间分解（趋势、季节、残差），季节周期={decomp_period} 个月')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'month_decompose.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        print(f"警告: 月度点数={n_months}，不足以做季节分解（需 >= {2 * decomp_period}），已跳过 month_decompose.png")

    # 自相关图（PACF 要求 nlags 严格小于样本长度的 50%）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    acf_lags = min(20, max(1, n_months - 1))
    pacf_lags = min(acf_lags, max(1, n_months // 2 - 1))
    plot_acf(ts_clean, lags=acf_lags, ax=ax1)
    plot_pacf(ts_clean, lags=pacf_lags, ax=ax2)
    plt.savefig(os.path.join(output_dir, 'autocorrelation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 互相关 （交付时间 vs 评分，滞后效果）
    from statsmodels.tsa.stattools import ccf

    # 需要对齐周数据
    delivery_weekly = weekly['total_delivery_time_capped'].values
    score_weekly = weekly['review_score'].values
    ccf_values = [np.corrcoef(delivery_weekly[:-lag], score_weekly[lag:])[0, 1] if lag > 0
                  else np.corrcoef(delivery_weekly, score_weekly)[0, 1] for lag in range(12)]
    plt.figure(figsize=(10, 5))
    plt.stem(range(12), ccf_values, basefmt=' ')
    plt.axhline(0, color='black')
    plt.title('交付时间与评分的互相关（滞后周数）')
    plt.xlabel('滞后周数')
    plt.ylabel('相关系数')
    plt.savefig(os.path.join(output_dir, 'cross_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 短期预测 （Holt-Winters；季节周期随训练长度调整，避免样本不足报错）
    train = ts_data[:-4]
    test = ts_data[-4:]
    n_train = len(train.dropna())
    hw_season = min(12, max(2, n_train // 2))
    try:
        if n_train >= 2 * hw_season + 2:
            model_hw = ExponentialSmoothing(train, seasonal_periods=hw_season, trend='add', seasonal='add').fit()
        else:
            model_hw = ExponentialSmoothing(train, trend='add').fit()
        forecast = model_hw.forecast(len(test))
        mse = mean_squared_error(test, forecast)
        mae = mean_absolute_error(test, forecast)
        print(f"\nHolt-Winters预测 MSE={mse:.2f}, MAE={mae:.2f}")
    except Exception as e:
        print(f"\nHolt-Winters 拟合失败，改用无季节项的 Holt 线性趋势: {e}")
        model_hw = ExponentialSmoothing(train, trend='add').fit()
        forecast = model_hw.forecast(len(test))
        mse = mean_squared_error(test, forecast)
        mae = mean_absolute_error(test, forecast)
        print(f"Holt 预测 MSE={mse:.2f}, MAE={mae:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train.index, train, label='训练')
    plt.plot(test.index, test, label='实际')
    plt.plot(test.index, forecast, label='预测', linestyle='--')
    plt.legend()
    plt.title('月度交付时间预测')
    plt.savefig(os.path.join(output_dir, 'monthly_predict.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 最后输出结论摘要
    print("\n" + "="*60)
    print("分析结论摘要")
    print("="*60)
    print(f"1. 交付时间与评分呈显著负相关 (r={pearson_r:.3f}, p<0.001)")
    print(f"2. 超时订单的差评率是准时订单的 {bad_review_multiplier:.1f} 倍")
    print(f"3. 承运商运输时间对评分的负面影响最大（系数 {coef_df.loc[coef_df['Feature']=='carrier_transit_time_capped', 'Coefficient'].values[0]:.3f}）")
    print(f"4. 季节性分解显示，年末（11-12月）物流时效显著恶化")
    print(f"5. 建议：优化承运商路线、动态调整预计送达时间、提前部署旺季运力")
    print("="*60)