'''
利用图特征来做数据增强：获取更多的训练数据
数据增强原则：
（1）如果q1，q2相似，且q1,q2在同一个连通图，则连通图的问题都相似 
   —— 利用connected components得到
   —— 组合后一共3796821个问题对，太多了
   —— 我们取 1<graph_distance<5 的问题对 （之所以要大于1，是因为等于1的是从训练数据来的）
   
（2）如果q1, q2不相似，且存在连通图cc1包含q1，和cc2包含q2，则cc1和cc2的任意组合均不相似
   —— 利用independent groups得到
   —— 注意一共有119475717个,太多了!!! 限制集合大小小于50，仍有1577269个问题对。选100万个。
   ---- 取跟相似问题一样多的量就好，构建平衡数据集

我们把数据增强的结果保存为一个单独的文件,包含3列 label, q1, q2

用了效果不好耶...
'''

import numpy as np
import pandas as pd
from collections import deque
from post_processing_with_graph_features import *


#char_embed =  pd.read_csv('datasets/char_embed.txt', sep=' ', header=None, index_col=0)
#word_embed = pd.read_csv('datasets/word_embed.txt', sep=' ', header=None, index_col=0)
#question = pd.read_csv('datasets/question.csv',index_col=0)
#train = pd.read_csv('datasets/train.csv')
#test = pd.read_csv('datasets/test.csv')


def all_pair_Dijkstra(train_graph, connected_component, max_distance):
    '''
    计算连通图中所有节点对的距离, 比Floyd算法快2.7倍左右
    '''
    m = len(connected_component)
    cc = list(connected_component)
    distance = {}
    
    # 初始化距离矩阵
    matrix = [[0 for j in range(m)] for i in range(m)]
    
    for i in range(m):
        dist = Dijkstra(train_graph, connected_component, cc[i])
        for j,node in enumerate(cc):
            matrix[i][j] = dist[node]
            # 保存我们想要的距离
            if matrix[i][j] > 1 and matrix[i][j] <= max_distance \
                          and (cc[i], cc[j]) not in distance.keys() \
                          and (cc[j], cc[i]) not in distance.keys():
                distance[(cc[i], cc[j])] = matrix[i][j]
                    
    return distance, matrix, cc


def gen_similar_data(train_graph, connected_components, max_cc_size, max_distance):
    '''
    对每个连通图，计算连通图中任意两点的距离
    注意：
    如果连通图节点只有2个，直接break
    如果连通图计算的距离为1，不存储
    '''
    
    distance = {}
    
    for cc in connected_components:
        if len(cc) > 2 and len(cc) <= max_cc_size:
            cc_distance, _, _ = all_pair_Dijkstra(train_graph, cc, max_distance)
            distance.update(cc_distance)
        else:
            continue
    
    return distance


def gen_dissimilar_data(independent_groups, max_group_size):
    '''
    如果q1, q2不相似，且存在连通图cc1包含q1，和cc2包含q2，则cc1和cc2的任意组合均不相似
    max_group_size用来控制返回的问题对数量，设为46，对应100万左右的问题对
    '''
    dissimilar_pairs = set()  
    for ig in independent_groups:
        cc1 = ig[0]
        cc2 = ig[1]
        # 限制连通图大小，不然太多了
        if len(cc1) < max_group_size and len(cc2) < max_group_size:
            for q1 in cc1:
                for q2 in cc2:
                    dissimilar_pairs.add((q1, q2))
    return dissimilar_pairs
 
    
    
def data_augmentation(train, similar_data, dissimilar_data):
    '''
    与train数据去重，生成平衡数据集
    similar_data: dict,{(q1,q2): d(q1,q2)}
    dissimilar_data: set, {(q1,q2)}
    '''
   
    #问题对转化为set格式
    similar_pairs = set(similar_data.keys())
    train_data1 = set([(train.loc[i,'q1'], train.loc[i,'q2']) for i in train.index])
    train_data2 = set([(train.loc[i,'q2'], train.loc[i,'q1']) for i in train.index])
    
    # 查看（q1,q2）组合是否与train数据重复,如重复，则去掉
    similar_pairs = similar_pairs - train_data1
    similar_pairs = list(similar_pairs - train_data2)
    dissimilar_pairs = dissimilar_data - train_data1
    dissimilar_pairs = list(dissimilar_pairs - train_data2)
    
    
    # 生成新的训练数据并导出
    new_data = []
    new_data.extend(similar_pairs)
    new_data.extend(dissimilar_pairs)
    new_data = pd.DataFrame(np.array(new_data))
    new_data.columns = ['q1','q2']
    new_data['label'] = 0
    new_data.loc[0:len(similar_pairs)-1, 'label'] = 1
    
    return new_data