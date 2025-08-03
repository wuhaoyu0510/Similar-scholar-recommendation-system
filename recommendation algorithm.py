# 导入必要的库
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from nltk.tokenize import word_tokenize
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


# 1. 加载JSON文件
def load_single_json_file(file_path):
    """从指定路径加载单个JSON文件并返回数据"""
    with open(file_path, 'r', encoding='utf-8') as f:  # 打开文件（使用utf-8编码）
        data = json.load(f)  # 加载JSON数据
    return data


# 2. 构建带时间窗口的学者动态特征

# 定义学者特征的默认结构
def scholar_default():
    """返回一个默认的学者数据结构"""
    return {
        'name_en': '',  # 英文名
        'name_zh': '',  # 中文名
        'time_windows': {},  # 按时间窗口存储的特征
        'all_papers': [],  # 所有论文列表
        'coauthors': defaultdict(coauthor_default),  # 合作者信息（使用默认字典）
        'total_paper_count': 0  # 总论文数
    }


# 定义合作者特征的默认结构
def coauthor_default():
    """返回一个默认的合作者数据结构"""
    return {'count': 0, 'last_year': 0}  # 合作次数和最近合作年份


def build_scholar_features_with_time(data_list, time_window=2):
    scholars = {}  # 初始化学者字典

    # 第一步：收集所有论文并按学者分组
    for paper in data_list:  # 遍历每篇论文
        date_parts = paper.get('date_parts', [])
        pub_year = 0  # 默认值（无有效年份时使用）
        current = date_parts
        while isinstance(current, list) and len(current) > 0:
            current = current[0]  # 取列表第一个元素，直到不是列表

        # 尝试转换为整数年份
        if isinstance(current, (int, float, str)):
            try:
                pub_year = int(current)
            except (ValueError, TypeError):
                pub_year = 0  # 转换失败则保持默认值

        for author in paper['author']:  # 遍历论文的每位作者
            # 创建学者ID（使用姓氏和名字的小写形式）
            scholar_id = f"{author['family'].lower()}_{author['given'].lower()}"

            # 如果学者不存在，初始化数据结构
            if scholar_id not in scholars:
                scholars[scholar_id] = {
                    'name_en': '',
                    'name_zh': '',
                    'time_windows': {},
                    'all_papers': [],
                    'coauthors': {},
                    'total_paper_count': 0
                }

            # 获取当前学者数据
            scholar = scholars[scholar_id]

            # 初始化学者基本信息（如果尚未设置）
            if not scholar['name_en']:
                scholar['name_en'] = f"{author['family']} {author['given']}"
                scholar['name_zh'] = author.get('chinese_name', '')

            # 存储论文信息（含年份和研究领域）
            scholar['all_papers'].append({
                'title': paper['title'],
                'abstract': paper.get('abstract', ''),
                'keywords': paper.get('keywords', []),
                'year': pub_year,  # 此处pub_year已为整数
                'authors': [f"{a['family'].lower()}_{a['given'].lower()}" for a in paper['author']],
                'class_en': paper.get('class_en', {})
            })
            scholar['total_paper_count'] += 1  # 增加论文计数

            # 更新合作者信息（记录合作次数和最近年份）
            for coauthor in paper['author']:  # 遍历所有合作者
                if coauthor == author:  # 跳过自己
                    continue
                coid = f"{coauthor['family'].lower()}_{coauthor['given'].lower()}"  # 合作者ID

                # 初始化合作者数据（如果尚未存在）
                if coid not in scholar['coauthors']:
                    scholar['coauthors'][coid] = {'count': 0, 'last_year': 0}  # 初始化计数和年份

                # 更新合作信息（此时pub_year是整数，可正常比较）
                coinfo = scholar['coauthors'][coid]
                coinfo['count'] += 1  # 增加合作次数
                if pub_year > coinfo['last_year']:  # 现在是整数与整数比较，修复错误
                    coinfo['last_year'] = pub_year  # 更新最近合作年份

    # 第二步：按时间窗口划分特征
    for scholar_id, features in scholars.items():  # 遍历每位学者
        if not features['all_papers']:  # 如果学者没有论文
            continue  # 跳过

        # 提取所有论文年份并确定窗口范围
        all_years = sorted([p['year'] for p in features['all_papers'] if p['year'] > 0])  # 获取有效年份
        if not all_years:  # 如果没有有效年份
            continue  # 跳过
        min_year, max_year = all_years[0], all_years[-1]  # 最小和最大年份
        windows = range(min_year, max_year + 1, time_window)  # 创建时间窗口列表

        # 为每个窗口提取特征
        for window_start in windows:  # 遍历每个时间窗口
            window_end = window_start + time_window - 1  # 计算窗口结束年份
            window_papers = [p for p in features['all_papers'] if window_start <= p['year'] <= window_end]  # 获取窗口内的论文
            if not window_papers:  # 如果窗口内没有论文
                continue  # 跳过

            # 提取窗口内的研究特征
            window_keywords = set()  # 初始化关键词集合
            window_areas = set()  # 初始化研究领域集合
            window_texts = []  # 初始化文本列表

            for p in window_papers:  # 遍历窗口内的每篇论文
                # 处理关键词
                if p.get('keywords'):  # 如果有关键词
                    if isinstance(p['keywords'], list):  # 如果是列表
                        # 添加所有关键词（小写、去空格）
                        window_keywords.update(kw.strip().lower() for kw in p['keywords'] if kw and isinstance(kw, str))
                    elif isinstance(p['keywords'], str):  # 如果是字符串
                        # 分割字符串并添加关键词（小写、去空格）
                        window_keywords.update(kw.strip().lower() for kw in p['keywords'].split(',') if kw)

                # 专门提取研究领域 - 针对JSON结构
                class_en = p.get('class_en', {})  # 获取研究领域信息

                # 提取研究主题（Research direction clusters）
                research_clusters = class_en.get("Research direction clusters", [])  # 获取研究主题
                if isinstance(research_clusters, list):  # 如果是列表
                    for cluster in research_clusters:  # 遍历每个主题
                        if cluster and isinstance(cluster, str):  # 如果是有效字符串
                            window_areas.add(cluster.strip().lower())  # 添加到领域集合

                # 提取二级学科（Secondary disciplines）
                secondary_disciplines = class_en.get("Secondary disciplines", [])  # 获取二级学科
                if isinstance(secondary_disciplines, list):  # 如果是列表
                    for disc in secondary_disciplines:  # 遍历每个学科
                        if disc and isinstance(disc, str):  # 如果是有效字符串
                            window_areas.add(disc.strip().lower())  # 添加到领域集合

                # 提取方法技术（Methods and technologies）
                methods_tech = class_en.get("Methods and technologies", [])  # 获取方法技术
                if isinstance(methods_tech, list):  # 如果是列表
                    for method in methods_tech:  # 遍历每个方法
                        if method and isinstance(method, str):  # 如果是有效字符串
                            window_areas.add(method.strip().lower())  # 添加到领域集合

                # 提取应用场景（Application scenarios）
                app_scenarios = class_en.get("Application scenarios", [])  # 获取应用场景
                if isinstance(app_scenarios, list):  # 如果是列表
                    for scenario in app_scenarios:  # 遍历每个场景
                        if scenario and isinstance(scenario, str):  # 如果是有效字符串
                            window_areas.add(scenario.strip().lower())  # 添加到领域集合

                # 添加文本（标题和摘要）
                window_texts.append(p['title'])  # 添加标题
                if p.get('abstract'):  # 如果有摘要
                    window_texts.append(p['abstract'])  # 添加摘要

            # 如果研究领域为空，使用关键词作为回退
            if not window_areas and window_keywords:
                window_areas = window_keywords.copy()  # 复制关键词作为领域

            # 存储窗口特征
            features['time_windows'][window_start] = {
                'window': (window_start, window_end),  # 时间窗口范围
                'paper_count': len(window_papers),  # 论文数量
                'keywords': list(window_keywords),  # 关键词列表
                'research_areas': list(window_areas),  # 研究领域列表
                'text': " ".join(window_texts).lower()  # 合并后的文本（小写）
            }

    return scholars  # 返回构建好的学者特征


# 3. 时间敏感的Doc2Vec模型（每个窗口单独编码）
def train_time_sensitive_doc2vec(scholars):
    """为每个学者的时间窗口训练Doc2Vec模型"""
    documents = []  # 初始化文档列表
    # 收集所有窗口的文本数据
    for scholar_id, features in scholars.items():  # 遍历每位学者
        for window_start, window_data in features['time_windows'].items():  # 遍历每个时间窗口
            # 生成唯一标签：学者ID_窗口起始年
            tag = f"{scholar_id}_{window_start}"
            tokens = word_tokenize(window_data['text'])  # 分词
            documents.append(TaggedDocument(tokens, [tag]))  # 创建带标签的文档

    # 训练模型（适配小样本的参数）
    model = Doc2Vec(
        vector_size=100,  # 向量维度
        window=6,  # 上下文窗口大小
        min_count=1,  # 最小词频（允许低频词）
        workers=6,  # 并行工作数
        epochs=20  # 训练轮数
    )
    model.build_vocab(documents)  # 构建词汇表
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)  # 训练模型
    return model  # 返回训练好的模型


# 4. 计算时间加权的文本相似度
def calculate_temporal_text_similarity(scholars, doc2vec_model):
    """计算学者间的时间加权文本相似度"""
    sim_dict = defaultdict(lambda: defaultdict(float))  # 初始化相似度字典（嵌套默认字典）
    current_year = 2025  # 当前年份（用于计算时间权重）

    for scholar_id1, features1 in scholars.items():  # 遍历每位学者（作为源学者）
        # 获取源学者的时间窗口（按时间倒序，近期窗口权重更高）
        windows1 = sorted(features1['time_windows'].keys(), reverse=True)

        for scholar_id2, features2 in scholars.items():  # 遍历每位学者（作为目标学者）
            if scholar_id1 == scholar_id2:  # 如果是同一个学者
                sim_dict[scholar_id1][scholar_id2] = 1.0  # 相似度为1.0
                continue  # 跳过后续处理

            windows2 = sorted(features2['time_windows'].keys(), reverse=True)  # 目标学者的时间窗口（倒序）
            total_sim = 0.0  # 总相似度
            weight_sum = 0.0  # 总权重

            # 计算所有窗口对的相似度并加权
            for w1 in windows1:  # 遍历源学者的每个窗口
                tag1 = f"{scholar_id1}_{w1}"  # 源窗口标签
                if tag1 not in doc2vec_model.dv:  # 如果标签不在模型中
                    continue  # 跳过
                vec1 = doc2vec_model.dv[tag1]  # 获取源窗口向量

                # 计算窗口1的时间权重（越近权重越高）
                w1_end = features1['time_windows'][w1]['window'][1]  # 窗口结束年份
                time_diff = current_year - w1_end  # 计算时间差
                if time_diff <= 0:  # 如果时间差为0或负数
                    time_diff = 1  # 设为1（避免除零）
                w1_weight = 1.0 / time_diff  # 时间衰减权重

                for w2 in windows2:  # 遍历目标学者的每个窗口
                    tag2 = f"{scholar_id2}_{w2}"  # 目标窗口标签
                    if tag2 not in doc2vec_model.dv:  # 如果标签不在模型中
                        continue  # 跳过
                    vec2 = doc2vec_model.dv[tag2]  # 获取目标窗口向量

                    # 计算余弦相似度
                    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)  # 向量范数乘积
                    if norm_product == 0:  # 避免除零错误
                        sim = 0.0
                    else:
                        sim = np.dot(vec1, vec2) / norm_product  # 余弦相似度

                    # 累加加权相似度
                    total_sim += sim * w1_weight
                    weight_sum += w1_weight  # 累加权重

            # 归一化相似度
            if weight_sum > 0:  # 如果有有效权重
                sim_dict[scholar_id1][scholar_id2] = total_sim / weight_sum  # 计算加权平均相似度
            else:  # 如果没有有效权重
                sim_dict[scholar_id1][scholar_id2] = 0.0  # 相似度为0

    return sim_dict  # 返回相似度字典


# 5. 带时间衰减的合作网络
def build_temporal_coauthor_network(scholars, decay_factor=0.9):
    """构建带时间衰减的合作网络"""
    G = nx.Graph()  # 创建无向图
    scholar_ids = list(scholars.keys())  # 获取所有学者ID

    # 添加节点（每位学者作为一个节点）
    for sid in scholar_ids:
        G.add_node(sid, name=scholars[sid]['name_en'])  # 添加节点（附带学者姓名）

    # 添加边（带时间衰减的权重）
    current_year = 2025  # 当前年份
    for sid, features in scholars.items():  # 遍历每位学者
        for coid, coinfo in features['coauthors'].items():  # 遍历每位合作者
            if coid not in scholars:  # 如果合作者不在学者列表中
                continue  # 跳过
            # 计算时间衰减：近期合作权重更高
            time_diff = current_year - coinfo['last_year']  # 计算时间差
            weight = coinfo['count'] * (decay_factor ** time_diff)  # 计算权重（合作次数 × 时间衰减）

            if G.has_edge(sid, coid):  # 如果边已存在
                G[sid][coid]['weight'] += weight  # 增加权重
            else:  # 如果边不存在
                G.add_edge(sid, coid, weight=weight)  # 添加新边并设置权重

    return G  # 返回构建的合作网络


# 6. 时间敏感的社交相关性（基于带权重的PageRank）
def calculate_temporal_social_relevance(network, scholars):
    """计算带时间衰减的社交相关性"""
    relevance_dict = defaultdict(dict)  # 初始化相关性字典
    scholar_ids = list(scholars.keys())  # 获取所有学者ID

    for sid in scholar_ids:  # 遍历每位学者
        # 个性化PageRank（基于带时间权重的合作网络）
        ppr = nx.pagerank(
            network,  # 合作网络
            personalization={sid: 1.0},  # 个性化向量（当前学者权重为1）
            alpha=0.85,  # 阻尼系数
            max_iter=50,  # 最大迭代次数
            weight='weight'  # 使用边的时间加权权重
        )
        # 归一化（将PageRank值缩放到0-1范围）
        max_val = max(ppr.values()) if ppr else 1.0  # 获取最大值
        for coid in scholar_ids:  # 遍历所有学者
            relevance_dict[sid][coid] = ppr.get(coid, 0.0) / (max_val + 1e-8)  # 归一化相关性

    return relevance_dict  # 返回社交相关性字典


# 7. 时间衰减的学术影响力
def calculate_temporal_impact(scholars, decay_factor=0.9):
    """学术影响力计算（三因素单独贡献）"""
    # 第一步：收集所有论文的关键数据（用于后续归一化）
    all_papers = []
    current_year = 2025
    for sid, features in scholars.items():
        all_papers.extend(features['all_papers'])

    # 预计算全局最大值（用于归一化各维度）
    max_impact_factor = max(p.get('impact_factor', 1.0) for p in all_papers) if all_papers else 1.0
    max_authors = max(len(p['authors']) for p in all_papers) if all_papers else 1.0
    max_time_diff = 10  # 假设最旧论文不超过20年前

    impact = {}  # 存储最终影响力

    for sid, features in scholars.items():  # 遍历每位学者
        # 初始化三个维度的贡献
        time_contrib = 0.0  # 时间衰减贡献
        if_contrib = 0.0  # 影响因子贡献
        author_contrib = 0.0  # 合作者数目贡献

        for paper in features['all_papers']:  # 遍历学者的每篇论文
            # 1. 提取论文基础信息（修复年份访问错误）
            pub_year = paper['year']  # 直接获取整数年份，删除[0]
            if_factor = paper.get('impact_factor', 1.0)  # 影响因子
            authors = paper['authors']  # 作者列表
            n_authors = len(authors)  # 合作者数目（含当前学者）
            n_authors = max(n_authors, 1)  # 避免除以0

            # 后续代码保持不变...
            # 2. 计算时间衰减贡献
            if pub_year <= 0:
                time_diff = max_time_diff  # 无效年份按最旧处理
            else:
                time_diff = current_year - pub_year
                time_diff = min(max(time_diff, 0), max_time_diff)
            time_score = 1 - (time_diff / max_time_diff)
            time_contrib += time_score

            # 3. 计算影响因子贡献
            if_score = if_factor / (max_impact_factor + 1e-8)
            if_contrib += if_score

            # 4. 计算合作者数目贡献
            author_score = 1 - (n_authors / (max_authors + 1e-8))
            author_contrib += author_score

        # 总影响力计算
        total_impact = time_contrib + if_contrib + author_contrib
        if not features['all_papers']:
            total_impact = 0.0
        impact[sid] = total_impact

    # 最终归一化
    max_total = max(impact.values()) if impact else 1.0
    max_total = max(max_total, 1e-8)
    return {sid: val / max_total for sid, val in impact.items()}



# 8. 研究兴趣变化趋势计算
def calculate_trend_similarity(scholar1, scholar2):
    """基于研究领域变化计算趋势相似度"""
    # 获取时间窗口列表
    s1_windows = sorted(scholar1['time_windows'].keys())
    s2_windows = sorted(scholar2['time_windows'].keys())

    # 需要至少两个时间窗口才能计算趋势
    if len(s1_windows) < 2 or len(s2_windows) < 2:
        return 0.0  # 返回0表示无法计算

    # 定义函数：获取最近两个窗口的研究领域
    def get_recent_areas(windows, scholar):
        recent_windows = sorted(windows)[-2:]  # 获取最近两个窗口
        areas = []  # 初始化领域列表
        for win in recent_windows:  # 遍历最近两个窗口
            # 获取窗口的研究领域（如果没有则使用关键词）
            win_data = scholar['time_windows'][win]
            if 'research_areas' in win_data and win_data['research_areas']:
                areas.extend(win_data['research_areas'])  # 添加研究领域
            elif 'keywords' in win_data and win_data['keywords']:
                areas.extend(win_data['keywords'])  # 使用关键词作为回退
        return set(areas)  # 返回领域集合

    # 获取最近两个窗口的研究领域
    areas1 = get_recent_areas(s1_windows, scholar1)
    areas2 = get_recent_areas(s2_windows, scholar2)

    # 检查是否有有效的研究领域
    if not areas1 or not areas2:
        return 0.0  # 返回0表示无法计算

    # 计算Jaccard相似度（交集/并集）
    intersection = areas1 & areas2  # 交集
    union = areas1 | areas2  # 并集
    return len(intersection) / len(union) if union else 0.0  # 返回相似度


# 9. 研究领域相似度计算
def research_area_similarity(scholar1, scholar2):
    """计算两个学者研究领域的Jaccard相似度"""

    # 定义函数：提取学者的研究领域
    def extract_areas(scholar):
        areas = set()  # 初始化领域集合
        if 'time_windows' in scholar:  # 检查是否有时间窗口
            for win_data in scholar['time_windows'].values():  # 遍历每个时间窗口
                # 优先使用研究领域，其次使用关键词
                if 'research_areas' in win_data and win_data['research_areas']:
                    areas.update(win_data['research_areas'])  # 添加研究领域
                elif 'keywords' in win_data and win_data['keywords']:
                    areas.update(win_data['keywords'])  # 使用关键词作为回退
        return areas  # 返回领域集合

    # 提取两位学者的研究领域
    areas1 = extract_areas(scholar1)
    areas2 = extract_areas(scholar2)

    # 移除常见停用词（无效领域）
    stop_words = {"general", "other", "miscellaneous", "unspecified"}
    areas1 = {a for a in areas1 if a not in stop_words}  # 过滤学者1的领域
    areas2 = {a for a in areas2 if a not in stop_words}  # 过滤学者2的领域

    # 检查是否有有效的研究领域
    if not areas1 and not areas2:  # 如果两者都没有领域
        return 0.0  # 返回0
    elif not areas1 or not areas2:  # 如果只有一方有领域
        return 0.0  # 返回0

    # 计算Jaccard相似度（交集/并集）
    intersection = areas1 & areas2  # 交集
    union = areas1 | areas2  # 并集
    return len(intersection) / len(union) if union else 0.0  # 返回相似度


# 10. 准备带时间特征的训练数据
def prepare_temporal_training_data(scholars, text_sim, social_rel, impact):
    """准备训练数据，包含时间相关特征"""
    X = []  # 特征向量列表
    y = []  # 标签列表
    scholar_ids = list(scholars.keys())  # 获取所有学者ID

    # 生成正样本（已有合作）
    positive_pairs = set()  # 使用集合避免重复
    for sid in scholar_ids:  # 遍历每位学者
        if 'coauthors' in scholars[sid]:  # 检查是否有合作者数据
            for coid in scholars[sid]['coauthors'].keys():  # 遍历每位合作者
                if coid in scholar_ids and sid != coid:  # 确保是有效学者且不是自己
                    # 添加排序后的元组（确保顺序一致）
                    positive_pairs.add(tuple(sorted((sid, coid))))

    # 生成负样本（无合作）
    negative_pairs = set()  # 使用集合避免重复
    while len(negative_pairs) < len(positive_pairs):  # 生成与正样本相同数量的负样本
        # 随机选择两个不同的学者
        i, j = np.random.choice(len(scholar_ids), 2, replace=False)
        sid1, sid2 = scholar_ids[i], scholar_ids[j]
        pair = tuple(sorted((sid1, sid2)))  # 排序后的学者对
        if pair not in positive_pairs:  # 如果不是合作对
            negative_pairs.add(pair)  # 添加为负样本

    # 构建特征向量
    for pair in positive_pairs.union(negative_pairs):  # 遍历所有样本对
        sid1, sid2 = pair  # 解包学者ID
        s1, s2 = scholars[sid1], scholars[sid2]  # 获取学者特征

        # 特征1：最近窗口的文本相似度
        recent_sim = text_sim.get(sid1, {}).get(sid2, 0.0)  # 尝试获取相似度
        if not recent_sim:  # 如果获取失败
            recent_sim = text_sim.get(sid2, {}).get(sid1, 0.0)  # 尝试反向获取

        # 特征2：社交相关性
        social_sim = social_rel.get(sid1, {}).get(sid2, 0.0)  # 尝试获取相关性
        if not social_sim:  # 如果获取失败
            social_sim = social_rel.get(sid2, {}).get(sid1, 0.0)  # 尝试反向获取

        # 特征3：平均学术影响力
        impact1 = impact.get(sid1, 0.0)  # 学者1的影响力
        impact2 = impact.get(sid2, 0.0)  # 学者2的影响力
        avg_impact = (impact1 + impact2) / 2  # 平均影响力

        # 特征4：研究兴趣趋势相似度
        trend_sim = calculate_trend_similarity(s1, s2)

        # 特征5：研究领域相似度
        area_sim = research_area_similarity(s1, s2)

        # 添加特征向量和标签
        X.append([recent_sim, social_sim, avg_impact, trend_sim, area_sim])  # 添加特征向量
        y.append(1 if pair in positive_pairs else 0)  # 添加标签（1=正样本，0=负样本）

    return np.array(X), np.array(y)  # 返回特征矩阵和标签数组


# 11. 带时间特征的推荐模型（神经网络）
class TemporalRecommendationModel(nn.Module):
    """融入时间特征的推荐模型"""

    def __init__(self, input_dim=5):
        """初始化模型结构"""
        super().__init__()
        # 定义全连接层
        self.fc1 = nn.Linear(input_dim, 128)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(128, 64)  # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(64, 32)  # 隐藏层2到隐藏层3

        # 注意力机制（用于强调重要特征）
        self.attention = nn.Sequential(
            nn.Linear(32, 16),  # 注意力输入层
            nn.Tanh(),  # 激活函数
            nn.Linear(16, 1),  # 注意力输出层
            nn.Softmax(dim=1)  # Softmax归一化
        )
        self.fc4 = nn.Linear(32, 1)  # 输出层
        self.dropout = nn.Dropout(0.3)  # Dropout层（防止过拟合）

    def forward(self, x):
        """前向传播"""
        x = torch.relu(self.fc1(x))  # ReLU激活
        x = self.dropout(x)  # 应用Dropout
        x = torch.relu(self.fc2(x))  # ReLU激活
        x = self.dropout(x)  # 应用Dropout
        x = torch.relu(self.fc3(x))  # ReLU激活

        # 应用注意力机制
        attn_weights = self.attention(x.unsqueeze(1))  # 计算注意力权重
        # 加权求和
        x = torch.sum(attn_weights * x.unsqueeze(1), dim=1).squeeze()

        x = torch.sigmoid(self.fc4(x))  # Sigmoid激活输出（0-1概率）
        return x  # 返回预测概率


# 12. 模型训练函数
def train_temporal_model(model, train_loader, epochs=50, lr=0.001):
    """训练推荐模型"""
    criterion = nn.BCELoss()  # 二元交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam优化器

    model.train() # 设置模型为训练模式
    for epoch in range(epochs):  # 遍历每个训练轮次
        total_loss = 0.0  # 初始化总损失
        for batch_x, batch_y in train_loader:  # 遍历每个批次
            optimizer.zero_grad()  # 梯度清零
            outputs = model(batch_x).squeeze()  # 前向传播
            loss = criterion(outputs, batch_y)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权重
            total_loss += loss.item()  # 累加损失

        # 每10轮输出一次训练进度
        if (epoch + 1) % 10 == 0:
            # 计算平均损失
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model  # 返回训练好的模型


# 13. 生成时间敏感的推荐
def generate_temporal_recommendations(target_sid, scholars, model, text_sim, social_rel, impact, top_n=5):
    """基于最新研究窗口生成推荐"""
    # 准备候选学者（排除目标学者自己）
    candidates = [sid for sid in scholars.keys() if sid != target_sid]
    if not candidates:  # 如果没有候选者
        return []  # 返回空列表

    # 构建候选者特征
    X_pred = []  # 预测特征列表
    target_features = scholars[target_sid]  # 目标学者特征

    for cid in candidates:  # 遍历每位候选学者
        c_features = scholars[cid]  # 候选学者特征

        # 特征1：最近窗口的文本相似度
        recent_sim = text_sim.get(target_sid, {}).get(cid, 0.0)
        if not recent_sim:  # 如果获取失败
            recent_sim = text_sim.get(cid, {}).get(target_sid, 0.0)  # 尝试反向获取

        # 特征2：社交相关性
        social_sim = social_rel.get(target_sid, {}).get(cid, 0.0)
        if not social_sim:  # 如果获取失败
            social_sim = social_rel.get(cid, {}).get(target_sid, 0.0)  # 尝试反向获取

        # 特征3：平均学术影响力
        target_impact = impact.get(target_sid, 0.0)  # 目标学者影响力
        cand_impact = impact.get(cid, 0.0)  # 候选学者影响力
        avg_impact = (target_impact + cand_impact) / 2  # 平均影响力

        # 特征4：研究兴趣趋势相似度
        trend_sim = calculate_trend_similarity(target_features, c_features)

        # 特征5：研究领域相似度
        area_sim = research_area_similarity(target_features, c_features)

        # 添加特征向量
        X_pred.append([recent_sim, social_sim, avg_impact, trend_sim, area_sim])

    # 预测并排序
    X_pred = torch.FloatTensor(X_pred)  # 转为张量
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        scores = model(X_pred).numpy().flatten()  # 预测分数

    # 按分数排序取Top-N
    top_indices = np.argsort(scores)[::-1][:top_n]  # 获取分数最高的索引
    recommendations = []  # 初始化推荐列表
    for idx in top_indices:  # 遍历每个推荐索引
        cid = candidates[idx]  # 候选学者ID
        # 添加推荐信息
        recommendations.append({
            'scholar_id': cid,  # 学者ID
            'name': scholars[cid]['name_en'],  # 学者姓名
            'score': scores[idx],  # 推荐分数
            'recent_sim': recent_sim,  # 文本相似度
            'trend_sim': calculate_trend_similarity(target_features, scholars[cid])  # 趋势相似度
        })

    return recommendations  # 返回推荐列表


# 14. 模型评估函数
def evaluate_temporal_model(model, X_test, y_test):
    """评估模型性能"""
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        X_tensor = torch.FloatTensor(X_test)  # 转为张量
        y_pred = model(X_tensor).numpy().flatten()  # 预测概率

    y_pred_bin = (y_pred > 0.5).astype(int)  # 二值化预测结果（0.5为阈值）

    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_bin),  # 准确率
        'recall': recall_score(y_test, y_pred_bin),  # 召回率
        'f1': f1_score(y_test, y_pred_bin),  # F1分数
        'roc_auc': roc_auc_score(y_test, y_pred)  # AUC-ROC
    }

    # 打印评估结果
    print("\n模型评估指标：")
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")

    return metrics  # 返回评估指标


# 15. 辅助函数：转换嵌套的defaultdict为普通字典
def convert_defaultdict_to_dict(d):
    """递归转换嵌套的defaultdict为普通字典"""
    if isinstance(d, defaultdict):  # 如果是defaultdict
        d = dict(d)  # 转换为普通字典
    if isinstance(d, dict):  # 如果是字典
        for k, v in d.items():  # 遍历每个键值对
            d[k] = convert_defaultdict_to_dict(v)  # 递归处理值
    return d  # 返回转换后的字典


# 16. 辅助函数：转换对象为可JSON序列化的格式
def convert_to_serializable(obj):
    """递归转换对象为可JSON序列化的格式"""
    if isinstance(obj, defaultdict):  # 如果是defaultdict
        # 递归转换每个键值对
        obj = {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, dict):  # 如果是字典
        # 递归转换每个键值对
        obj = {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):  # 如果是列表
        # 递归转换每个元素
        obj = [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):  # 如果是元组
        # 递归转换每个元素
        obj = tuple(convert_to_serializable(item) for item in obj)
    # 其他类型直接返回
    return obj




# 主函数：完整流程演示
def main():
    """主函数：执行完整的推荐系统流程"""
    # 1. 加载数据
    print("加载数据...")
    json_file = "papers_data.json"  # JSON文件路径
    all_data = load_single_json_file(json_file)  # 加载数据
    print(f"加载了{len(all_data)}条论文记录")  # 打印加载的记录数

    # 2. 构建带时间特征的学者数据
    print("\n构建学者动态特征...")
    # 构建学者特征（使用3年时间窗口）
    scholars = build_scholar_features_with_time(all_data, time_window=3)

    # 转换为可序列化结构
    serializable_scholars = convert_to_serializable(scholars)

    # 保存为JSON文件
    with open('scholars.json', 'w', encoding='utf-8') as f:
        json.dump(serializable_scholars, f, ensure_ascii=False, indent=2)
    print("已保存学者特征至 scholars.json")

    # 3. 训练时间敏感的Doc2Vec模型
    print("\n训练Doc2Vec模型...")
    doc2vec_model = train_time_sensitive_doc2vec(scholars)  # 训练模型
    print("Doc2Vec模型训练完成")

    # 4. 计算文本相似度
    print("\n计算时间加权文本相似度...")
    text_sim = calculate_temporal_text_similarity(scholars, doc2vec_model)  # 计算相似度

    # 转换defaultdict为普通dict（以便JSON序列化）
    text_sim_dict = convert_defaultdict_to_dict(text_sim)

    # 保存文本相似度
    with open('text_sim.json', 'w', encoding='utf-8') as f:
        json.dump(text_sim_dict, f, ensure_ascii=False, indent=2)
    print("已保存文本相似度至 text_sim.json")

    # 5. 构建合作网络并计算社交相关性
    print("\n构建时间衰减的合作网络...")
    coauthor_network = build_temporal_coauthor_network(scholars)  # 构建合作网络
    print(f"合作网络包含{len(coauthor_network.nodes)}个节点，{len(coauthor_network.edges)}条边")  # 打印网络规模

    print("计算社交相关性...")
    social_rel = calculate_temporal_social_relevance(coauthor_network, scholars)  # 计算社交相关性

    # 转换defaultdict为普通dict
    social_rel_dict = convert_defaultdict_to_dict(social_rel)

    # 保存社交相关性
    with open('social_rel.json', 'w', encoding='utf-8') as f:
        json.dump(social_rel_dict, f, ensure_ascii=False, indent=2)
    print("已保存社交相关性至 social_rel.json")

    # 6. 计算学术影响力
    print("\n计算时间敏感的学术影响力...")
    impact_scores = calculate_temporal_impact(scholars)  # 计算影响力

    # 保存影响力分数
    with open('impact_scores.json', 'w', encoding='utf-8') as f:
        json.dump(impact_scores, f, ensure_ascii=False, indent=2)
    print("已保存学术影响力分数至 impact_scores.json")

    # 7. 准备训练数据
    print("\n准备训练数据...")
    # 准备特征和标签
    X, y = prepare_temporal_training_data(scholars, text_sim, social_rel, impact_scores)

    # 分割训练集和测试集（80%训练，20%测试）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 转换为PyTorch张量数据集
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))

    # 创建数据加载器（批量大小32，打乱数据）
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 8. 训练模型
    print("\n训练推荐模型...")
    model = TemporalRecommendationModel(input_dim=5)  # 创建模型（5个输入特征）
    trained_model = train_temporal_model(model, train_loader, epochs=50)  # 训练模型

    # 保存模型权重
    torch.save(trained_model.state_dict(), 'temporal_recommendation_model.pth')
    print("模型训练完成并已保存")

    # 9. 评估模型
    print("\n评估模型性能...")
    evaluate_temporal_model(trained_model, X_test, y_test)  # 评估模型性能

    # 10. 生成推荐示例
    print("\n生成推荐示例...")
    target_sid = next(iter(scholars.keys()))  # 取第一个学者作为目标
    # 生成推荐（Top 5）
    recommendations = generate_temporal_recommendations(
        target_sid, scholars, trained_model, text_sim, social_rel, impact_scores, top_n=5
    )

    # 打印推荐结果
    print(f"\n为学者{scholars[target_sid]['name_en']}推荐的合作者：")
    for i, rec in enumerate(recommendations, 1):  # 遍历推荐结果（从1开始编号）
        print(f"{i}. {rec['name']} (推荐分数: {rec['score']:.4f}, 近期相似度: {rec['recent_sim']:.4f})")


# 程序入口
if __name__ == "__main__":
    main()  # 执行主函数