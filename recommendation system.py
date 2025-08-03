import streamlit as st
import pandas as pd

import torch
import torch.nn as nn
import networkx as nx

import json
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime

# 设置页面
st.set_page_config(
    page_title="学术合作推荐系统",
    page_icon=":mortar_board:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 学者推荐模型类（更新为5个输入特征）
class TemporalRecommendationModel(nn.Module):
    """时间敏感的学者推荐模型"""

    def __init__(self, input_dim=5):
        super(TemporalRecommendationModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.attention = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softmax(dim=1))
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))

        # 应用注意力机制
        attn_weights = self.attention(x.unsqueeze(1))
        x = torch.sum(attn_weights * x.unsqueeze(1), dim=1).squeeze()
        x = torch.sigmoid(self.fc4(x))
        return x


# 加载数据（用于UI）
@st.cache_resource(show_spinner="加载学术数据...")
def load_ui_data():
    """加载UI所需的数据"""
    data = {}

    try:
        # 加载学者特征
        with open('scholars.json', 'r', encoding='utf-8') as f:
            data['scholars_features'] = json.load(f)

        # 加载文本相似度矩阵
        with open('text_sim.json', 'r', encoding='utf-8') as f:
            data['text_similarity'] = json.load(f)

        # 加载社交相关性矩阵
        with open('social_rel.json', 'r', encoding='utf-8') as f:
            data['social_relevance'] = json.load(f)

        # 加载影响力分数
        with open('impact_scores.json', 'r', encoding='utf-8') as f:
            data['impact_scores'] = json.load(f)

        # 加载模型
        model = TemporalRecommendationModel(input_dim=5)
        model.load_state_dict(torch.load('temporal_recommendation_model.pth'))
        model.eval()
        data['model'] = model

        # 创建学者ID到姓名的映射
        data['id_to_name'] = {}
        for sid, scholar in data['scholars_features'].items():
            data['id_to_name'][sid] = scholar['name_en']

        return data
    except Exception as e:
        st.error(f"加载数据时出错: {str(e)}")
        st.stop()


# 计算研究趋势相似度
def calculate_trend_similarity(scholar1, scholar2):
    """基于研究领域变化计算趋势相似度"""
    if 'time_windows' not in scholar1 or 'time_windows' not in scholar2:
        return 0.0

    s1_windows = sorted([int(k) for k in scholar1['time_windows'].keys()])
    s2_windows = sorted([int(k) for k in scholar2['time_windows'].keys()])

    if len(s1_windows) < 2 or len(s2_windows) < 2:
        return 0.0

    def get_recent_areas(windows, scholar):
        recent_windows = sorted(windows)[-2:]
        areas = []
        for win in recent_windows:
            areas.extend(scholar['time_windows'][str(win)].get('research_areas', []))
        return set(areas)

    areas1 = get_recent_areas(s1_windows, scholar1)
    areas2 = get_recent_areas(s2_windows, scholar2)

    if not areas1 or not areas2:
        return 0.0

    intersection = areas1 & areas2
    union = areas1 | areas2
    return len(intersection) / len(union) if union else 0.0


# 计算研究领域相似度
def research_area_similarity(scholar1, scholar2):
    """计算两个学者研究领域的Jaccard相似度"""

    def extract_areas(scholar):
        areas = set()
        if 'time_windows' not in scholar:
            return areas

        for win_data in scholar['time_windows'].values():
            win_areas = win_data.get('research_areas', [])
            if isinstance(win_areas, list):
                for area in win_areas:
                    if isinstance(area, str) and area.strip():
                        areas.add(area.strip().lower())
        return areas

    areas1 = extract_areas(scholar1)
    areas2 = extract_areas(scholar2)

    stop_words = {"general", "other", "miscellaneous", "unspecified"}
    areas1 = {a for a in areas1 if a not in stop_words}
    areas2 = {a for a in areas2 if a not in stop_words}

    if not areas1 and not areas2:
        return 0.0
    elif not areas1 or not areas2:
        return 0.0

    intersection = areas1 & areas2
    union = areas1 | areas2
    return len(intersection) / len(union) if union else 0.0


# 生成推荐（UI版本）
def generate_ui_recommendations(scholar_id, data, top_n=10):
    """为目标学者生成推荐（UI版本）"""
    scholars_features = data['scholars_features']
    model = data['model']
    text_similarity = data['text_similarity']
    social_relevance = data['social_relevance']
    impact_scores = data['impact_scores']
    id_to_name = data['id_to_name']

    # 准备候选学者（排除自己）
    candidate_scholars = [sid for sid in scholars_features.keys() if sid != scholar_id]

    X_pred = []  # 存储预测特征
    # 构建每个候选学者的特征向量
    for candidate_id in candidate_scholars:
        # 获取文本相似度（处理缺失值）
        text_sim = text_similarity.get(scholar_id, {}).get(candidate_id, 0.0)
        if not text_sim:
            text_sim = text_similarity.get(candidate_id, {}).get(scholar_id, 0.0)

        # 获取社交相关性（处理缺失值）
        social_rel = social_relevance.get(scholar_id, {}).get(candidate_id, 0.0)
        if not social_rel:
            social_rel = social_relevance.get(candidate_id, {}).get(scholar_id, 0.0)

        # 计算平均影响力
        avg_impact = (impact_scores.get(scholar_id, 0.0) + impact_scores.get(candidate_id, 0.0)) / 2

        # 计算趋势相似度
        trend_sim = calculate_trend_similarity(
            scholars_features[scholar_id],
            scholars_features[candidate_id]
        )

        # 计算研究领域相似度
        area_sim = research_area_similarity(
            scholars_features[scholar_id],
            scholars_features[candidate_id]
        )

        features = [text_sim, social_rel, avg_impact, trend_sim, area_sim]
        X_pred.append(features)

    X_pred = torch.FloatTensor(X_pred)  # 转为张量

    # 预测相似度分数
    with torch.no_grad():
        scores = model(X_pred).numpy().flatten()

    # 创建结果DataFrame
    results = []
    for i, candidate_id in enumerate(candidate_scholars):
        results.append({
            '学者ID': candidate_id,
            '姓名': id_to_name.get(candidate_id, "未知学者"),
            '推荐分数': scores[i],
            '文本相似度': X_pred[i][0].item(),
            '社交相关性': X_pred[i][1].item(),
            '综合影响力': X_pred[i][2].item(),
            '趋势相似度': X_pred[i][3].item(),
            '领域相似度': X_pred[i][4].item()
        })

    df = pd.DataFrame(results)
    # 排序并获取Top-N推荐
    df = df.sort_values('推荐分数', ascending=False).head(top_n)
    df['排名'] = range(1, len(df) + 1)

    return df.set_index('排名')


def visualize_network(scholar_id, data, top_k=5):
    """可视化学者的合作网络"""
    scholars_features = data['scholars_features']
    id_to_name = data['id_to_name']

    # 创建网络图
    G = nx.Graph()

    # 添加中心节点（目标学者）
    G.add_node(scholar_id, size=30, color='gold', label=id_to_name.get(scholar_id, scholar_id))

    # 添加直接合作者
    direct_coauthors = []
    if 'coauthors' in scholars_features[scholar_id]:
        for coauthor_id, coinfo in scholars_features[scholar_id]['coauthors'].items():
            if coauthor_id in scholars_features:  # 确保合作者在系统中
                G.add_node(coauthor_id, size=20, color='lightblue',
                           label=id_to_name.get(coauthor_id, coauthor_id))
                # 边权重基于合作次数
                weight = min(10, 1 + coinfo['count'] / 2)
                G.add_edge(scholar_id, coauthor_id, weight=weight)
                direct_coauthors.append(coauthor_id)

    # 添加二级合作者（最多top_k个）
    added = 0
    for coauthor_id in direct_coauthors:
        if coauthor_id in scholars_features and 'coauthors' in scholars_features[coauthor_id]:
            for second_level, coinfo in scholars_features[coauthor_id]['coauthors'].items():
                if second_level != scholar_id and second_level in scholars_features and second_level not in G:
                    G.add_node(second_level, size=15, color='lightgreen',
                               label=id_to_name.get(second_level, second_level))
                    # 边权重基于合作次数
                    weight = min(5, 1 + coinfo['count'] / 3)
                    G.add_edge(coauthor_id, second_level, weight=weight)
                    added += 1
                    if added >= top_k:
                        break
        if added >= top_k:
            break

    # 检查是否为空图
    if len(G.nodes) == 0:
        return None

    # 使用Plotly绘制网络图
    pos = nx.spring_layout(G, seed=42)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]['label'])
        node_size.append(G.nodes[node]['size'])
        node_color.append(G.nodes[node]['color'])

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=2, color='DarkSlateGrey'))
    )

    # 创建图表并设置布局
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(
                text=f'合作网络: {id_to_name.get(scholar_id, scholar_id)}',
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    return fig


# 主UI界面
def main():
    """主UI界面"""
    # 加载数据
    data = load_ui_data()
    scholars_features = data['scholars_features']
    id_to_name = data['id_to_name']

    # 创建学者选择下拉菜单的选项
    scholar_options = {f"{name} ({sid[:10]}...)": sid for sid, name in id_to_name.items()}
    sorted_scholar_options = sorted(scholar_options.keys())

    st.title(" 时间敏感的学术合作推荐系统")
    st.markdown("""
    <style>
    .big-font {
        font-size:18px !important;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
    }
    .keyword-tag {
        background-color: #e0f0ff;
        border-radius: 5px;
        padding: 5px 10px;
        margin: 2px;
        display: inline-block;
    }
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 15px;
        margin-bottom: 20px;
        background: white;
    }
    .feature-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e7eb 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p class="big-font">本系统通过分析学者的研究内容动态变化、时间加权的社交网络关系和学术影响力，为学者推荐潜在的合作者。</p>',
        unsafe_allow_html=True)

    # 侧边栏
    with st.sidebar:
        st.header(" 搜索设置")
        st.markdown("---")

        # 学者选择
        selected_scholar = st.selectbox(
            "选择学者",
            options=sorted_scholar_options,
            index=0
        )
        scholar_id = scholar_options[selected_scholar]
        scholar_info = scholars_features[scholar_id]

        # 推荐数量设置
        top_n = st.slider("推荐数量", 5, 20, 10)

        # 显示学者信息
        st.subheader("👤 学者信息")
        st.write(f"**姓名**: {scholar_info['name_en']}")
        if scholar_info.get('name_zh'):
            st.write(f"**中文名**: {scholar_info['name_zh']}")
        st.write(f"**论文数量**: {scholar_info.get('total_paper_count', 0)}")

        # 研究方向
        research_areas = set()
        if 'time_windows' in scholar_info:
            for win_data in scholar_info['time_windows'].values():
                research_areas.update(win_data.get('research_areas', []))

        if research_areas:
            st.write(f"**研究方向**: {', '.join(list(research_areas)[:3])}...")
        else:
            st.write("**研究方向**: 暂无数据")

        st.write(f"**合作者数量**: {len(scholar_info.get('coauthors', {}))}")

        # 显示影响力分数
        impact_score = data['impact_scores'].get(scholar_id, 0.0)
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.metric(" 学术影响力", f"{impact_score:.2f}/1.0",
                  help="基于时间加权的论文数量、合作网络和研究多样性计算的综合影响力分数")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.caption(" 系统说明：本推荐系统基于时间动态的研究内容、社交网络关系和学术影响力三个维度计算合作潜力")

        # 显示时间窗口信息
        time_windows = scholar_info.get('time_windows', {})
        num_windows = len(time_windows)
        st.write(f"**时间窗口数量**: {num_windows}")

        if num_windows > 0:
            # 显示最近的时间窗口
            recent_window = sorted(time_windows.keys(), reverse=True)[0]
            window_data = time_windows[recent_window]
            st.write(f"**最近窗口**: {window_data['window'][0]}-{window_data['window'][1]}")

            # 显示研究领域
            if window_data.get('research_areas'):
                st.write(f"**研究领域**: {', '.join(window_data['research_areas'][:3])}")

            # 显示关键词
            if window_data.get('keywords'):
                st.write(f"**关键词**: {', '.join(window_data['keywords'][:5])}")
    # 主内容区 - 分为3列
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.subheader(" 合作网络")
        fig_network = visualize_network(scholar_id, data)
        st.plotly_chart(fig_network, use_container_width=True)

    with col2:
        st.subheader(f" 推荐合作者 (Top {top_n})")
        with st.spinner('正在生成推荐...'):
            recommendations = generate_ui_recommendations(scholar_id, data, top_n)

        # 显示推荐表格
        styled_df = recommendations[['姓名', '推荐分数', '文本相似度', '趋势相似度', '领域相似度']].copy()
        styled_df['推荐分数'] = styled_df['推荐分数'].apply(lambda x: f"{x:.4f}")
        styled_df['文本相似度'] = styled_df['文本相似度'].apply(lambda x: f"{x:.4f}")
        styled_df['趋势相似度'] = styled_df['趋势相似度'].apply(lambda x: f"{x:.4f}")
        styled_df['领域相似度'] = styled_df['领域相似度'].apply(lambda x: f"{x:.4f}")

        # 应用样式
        def highlight_top3(s):
            return ['background-color: #fffacd' if s.name <= 3 else '' for _ in s]

        styled_df = styled_df.style.apply(highlight_top3, axis=1)
        st.dataframe(styled_df, height=600)

    with col3:
        st.subheader(" 推荐详情")

        if not recommendations.empty:
            # 获取排名第一的推荐
            top_recommendation = recommendations.iloc[0]

            # 显示顶级推荐
            st.markdown(f"### 最佳推荐: {top_recommendation['姓名']}")
            st.markdown(f"**综合推荐分数**: {top_recommendation['推荐分数']}")

            # 创建雷达图 - 添加社交相关性和学术影响力
            metrics = [
                '文本相似度', '社交相关性', '趋势相似度',
                '领域相似度', '学术影响力'
            ]

            # 获取各项指标的值
            values = [
                float(top_recommendation['文本相似度']),
                float(top_recommendation['社交相关性']),
                float(top_recommendation['趋势相似度']),
                float(top_recommendation['领域相似度']),
                float(top_recommendation['综合影响力'])  # 学术影响力
            ]

            # 创建雷达图
            fig = go.Figure()

            # 添加指标数据
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],  # 闭合图形
                theta=metrics + [metrics[0]],  # 闭合图形
                fill='toself',
                name='相似度指标',
                line=dict(color='rgb(106,90,205)'),
                fillcolor='rgba(106,90,205,0.2)'
            ))

            # 设置布局
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickfont=dict(size=10)
                    ),
                    angularaxis=dict(
                        rotation=90,
                        direction="clockwise",
                        tickfont=dict(size=11)
                    )
                ),
                showlegend=False,
                title=f"{top_recommendation['姓名']}的相似度指标",
                height=300,
                margin=dict(l=40, r=40, t=60, b=40)
            )

            # 显示雷达图
            st.plotly_chart(fig, use_container_width=True)

            # 特征解释 - 更新为5个指标
            st.markdown("### 特征解释")
            st.markdown("""
            <div class="feature-box">
            <h4> 文本相似度</h4>
            <p>基于时间窗口的论文内容相似度，考虑近期研究的更高权重</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="feature-box">
            <h4> 社交相关性</h4>
            <p>基于时间衰减的合作网络关系强度</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="feature-box">
            <h4> 趋势相似度</h4>
            <p>比较双方学者最近两个时间窗口的研究兴趣变化趋势</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="feature-box">
            <h4> 领域相似度</h4>
            <p>基于整个研究周期的研究领域相似度</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="feature-box">
            <h4> 学术影响力</h4>
            <p>双方学者的平均学术影响力，考虑时间加权</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("没有找到推荐结果")

    # 添加底部说明（仅保留版权信息）
    st.divider()
    st.caption(f"© {datetime.now().year} 时间敏感的学术合作推荐系统 | 基于人工智能的研究合作匹配平台")


# 运行UI
if __name__ == "__main__":
    main()