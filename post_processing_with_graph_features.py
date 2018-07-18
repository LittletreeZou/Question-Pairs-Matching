'''
后处理
'''
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import deque


#char_embed =  pd.read_csv('datasets/char_embed.txt', sep=' ', header=None, index_col=0)
#word_embed = pd.read_csv('datasets/word_embed.txt', sep=' ', header=None, index_col=0)
#question = pd.read_csv('datasets/question.csv',index_col=0)
#train = pd.read_csv('datasets/train.csv')
#test = pd.read_csv('datasets/test.csv')



#求q1,q2的图上距离（最短路径）
'''
step1 : 利用训练数据生成无向图，求各个连通分量
step2 : 求q1,q2的距离d
     （1）如果q1,q2在一个连通图上：求q1,q2的距离d 
     （2）如果q1,q2不在一个连通图上，令d(q1,q2) = 1000
'''
# 生成无向图
def gen_graph(train):
    """
    把输入数据转化为以字典表示的无向图
    """ 
    data = train[train['label']==1][['q1','q2']]
    graph = {}
    for i in range(len(data)):
        if data.iloc[i,0] not in graph.keys():
            graph[data.iloc[i,0]] = set([data.iloc[i,1]])
        else:
            graph[data.iloc[i,0]].add(data.iloc[i,1])
    
        if data.iloc[i,1] not in graph.keys():
            graph[data.iloc[i,1]] = set([data.iloc[i,0]])
        else:
            graph[data.iloc[i,1]].add(data.iloc[i,0])
    
    return graph


def bfs_visited(ugraph, start_node):
    """
    输入无向图ugraph和一个节点start_node
    返回从这个节点出发，通过广度优先搜索访问的所有节点的集合
    """
    # initialize Q to be an empty queue
    que = deque()
    # initialize visited
    visited = [start_node]
    # enqueue(que, start_node)
    que.append(start_node)
    while len(que) > 0:
        current_node = que.popleft()
        neighbours = ugraph[current_node]
        for nei in neighbours:
            if nei not in visited:
                visited.append(nei)
                que.append(nei) 
    return set(visited)


def cc_visited(ugraph):
    """
    输入无向图ugraph
    返回一个list，list的元素是每个连通分量的节点构成的集合
    """
    remaining_nodes = list(ugraph.keys())
    connected_components = []
    while len(remaining_nodes) > 0 :
        # choose the first element in remaining_nodes to be the start_node
        start_node = remaining_nodes[0]
        # use bfs_visited() to get the connected component containing start_node
        con_component = bfs_visited(ugraph, start_node)
        # update connected_components
        connected_components.append(con_component)
        # update remaining_nodes
        remaining_nodes = list(set(remaining_nodes) - con_component)
    return connected_components


# 单源最短路径
def Dijkstra(ugraph, connected_component, start_node):
    '''
    返回start_node到connected_component所有节点的最短距离
    '''
    # 初始化
    minv = start_node
    visited = set()
    
    # 源顶点到其余各顶点的初始路程
    dist = dict([(node,np.float('inf')) for node in connected_component])
    dist[minv] = 0
    
    # 遍历集合V中与A直接相邻的顶点，找出当前与A距离最短的顶点
    while len(visited) < len(connected_component):
        visited.add(minv)
        # 确定当期顶点的距离
        for v in ugraph[minv]:
            if dist[minv] + 1 < dist[v]:   # 如果从当前点扩展到某一点的距离小与已知最短距离 
                dist[v] = dist[minv] + 1   # 对已知距离进行更新
        
        # 从剩下的未确定点中选择最小距离点作为新的扩散点
        new = np.float('inf')                                      
        for w in connected_component - visited:   
            if dist[w] < new: 
                new = dist[w]
                minv = w  
    return dist
            

## 先生成图
#print('Generating Graph...')
#start = time.time()
#train_graph = gen_graph(train)
#end = time.time()
#print('Graph generated. Time used {:0.1f} mins'.format((end-start)/60))

## 寻找各连通分项（大概7分钟）
#print('Searching Connected Components...')
#start = time.time()
#connected_components = cc_visited(train_graph)
#end = time.time()
#print('Search finished. Time used {:0.1f} mins'.format((end-start)/60))

def get_graph_distance(data, train_graph, connected_components, training_data=True):
    '''
    1. 如果q1,q2在一个连通图上：返回q1,q2的距离d
    2. 如果q1,q2不在一个连通图上: 令d(q1, q2) = 1000
    '''
    n = data.shape[0]
    
    # 初始化
    record_distance = {}  #用来记录已经计算过的距离
    result_distance = [1000 for i in range(n)]
    
    for i in range(n):
        q1 = data.loc[i,'q1']
        q2 = data.loc[i,'q2']

        # 如果是训练数据的相似问题，则dist=1
        if training_data and data.loc[i,'label'] == 1:
            result_distance[i] = 1
        
        # 如果已经计算过，直接取出计算过的值
        elif (q1,q2) in record_distance.keys():
            result_distance[i] = record_distance[(q1,q2)]

        elif (q2,q1) in record_distance.keys():
            result_distance[i] = record_distance[(q2,q1)]

        else:       
            # check whether q1,q2 are in one connected_componets
            for cc in connected_components:
                if (q1 in cc) and (q2 in cc):
                    # 连通图cc,q1到其它节点的距离
                    q1_dist = Dijkstra(train_graph, cc, q1)
                    # 把计算过的距离保存起来
                    new_dict = dict([((q1,node),q1_dist[node]) for node in q1_dist.keys()])
                    record_distance.update(new_dict)          
                    result_distance[i] = q1_dist[q2]            
                    break

    result_distance = pd.DataFrame(np.array(result_distance), index=data.index)
    result_distance.columns = ['graph_distance']
    
    return result_distance
                
'''
通过训练数据得到问题之间的距离，进行统计发现：
label = 1 : graph_distance = 1
label = 0 : graph_distance = 1000（表示不连通）
说明：不相似的问题不可能在一个连通图里
推断：q1与q2不相似，则q1与q2的连通图G(q2)的所有顶点都不相似，q2与q1的连通图G(q1)的所有顶点都不相似
另有一个不太充分的结论： 相似问题具有传递性，而且可以传递很远。

算法：区分确定的不相似和不确定的不相似
input: (q1, q2) , connected component
return: graph_feature（gf for short）
step1: 先利用训练数据中问题对的不相似，找出相互独立的连通子图对
step2: 对于测试数据中的问题对（q1, q2）,如果q1存在连通图cc(q1), q2存在连通图cc(q2)，且cc(q1)与cc(q2)独立，则q1,q2不相似。
'''

def get_independent_groups(train, train_graph_distance, connected_components):
    
    # 找出不相似的问题对
    data = train[train.label == 0]
    
    independent_groups = []
       
    for i in data.index:
        q1 = data.loc[i,'q1']
        q2 = data.loc[i,'q2']
        
        if train_graph_distance.loc[i, 'graph_distance'] == 1000:
            # 查看它们是否有连通图
            cc1 = set([])
            cc2 = set([])
            for cc in connected_components:
                if q1 in cc:
                    cc1 = cc
                if q2 in cc:
                    cc2 = cc
            if len(cc1) > 0 and len(cc2) > 0 and (cc1,cc2) not in independent_groups and (cc2,cc1) not in independent_groups:
                independent_groups.append((cc1,cc2))
                
    return independent_groups



def get_graph_features(test, test_graph_distance, independent_groups):
    
    n = test.shape[0]
    
    # 初始化, 0 表示从训练集的graph无法确定是否相似， 1表示确定相似，-1表示确定不相似
    graph_features = [0 for i in range(n)]
    
    for i in range(n):
        q1 = test.loc[i,'q1']
        q2 = test.loc[i,'q2']

        if test_graph_distance.loc[i,'graph_distance'] < 1000:
            graph_features[i] = 1
        else:
            # 看看q1和q2是否在independent group里面，如果在，则q1，q2确定不相似
            for ig in independent_groups:
                if (q1 in ig[0] and q2 in ig[1]) or (q1 in ig[1] and q2 in ig[0]):
                    graph_features[i] = -1
      
    graph_features = pd.DataFrame(np.array(graph_features), index=test.index)
    graph_features.columns = ['graph_features']
    
    return graph_features

