'''
特征构造
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


'''
求q1,q2长度差异，normalize by the max len of question pairs
'''
def get_len_diff(merge_data, word_level=True):
    if word_level:
        merge = merge_data[['words_x','words_y']]
    else:
        merge = merge_data[['chars_x','chars_y']]
    
    merge.columns = ['q1','q2']
   
    q1_len = merge.q1.apply(lambda x: len(x.split(' '))).values
    q2_len = merge.q2.apply(lambda x: len(x.split(' '))).values
 
    len_diff = np.abs((q1_len - q2_len) / np.max([q1_len, q2_len],axis=0))
    
    return len_diff


'''
# 取q1,q2中相同词的个数
'''
def get_num_common_words(question, data):
    # merge data
    merge = pd.merge(data,question,left_on=['q1'],right_on=['qid'],how='left')
    merge = pd.merge(merge,question,left_on=['q2'],right_on=['qid'],how='left')
    merge = merge[['words_x','words_y']]
    merge.columns = ['q1','q2']
    
    q1_word_set = merge.q1.apply(lambda x: x.split(' ')).apply(set).values
    q2_word_set = merge.q2.apply(lambda x: x.split(' ')).apply(set).values
           
    result = [len(q1_word_set[i] & q2_word_set[i]) for i in range(len(q1_word_set))]
    result = pd.DataFrame(result, index=data.index)
    result.columns = ['num_common_words']
    return result

'''
计算共现词比例
'''
def get_common_word_ratio(merge_data, data, word_level=True):
    
    if word_level:
        merge = merge_data[['words_x','words_y']]
    else:
        merge = merge_data[['chars_x','chars_y']]
    merge.columns = ['q1','q2']
    
    q1_word_set = merge.q1.apply(lambda x: x.split(' ')).apply(set).values
    q2_word_set = merge.q2.apply(lambda x: x.split(' ')).apply(set).values
    q1_word_len = merge.q1.apply(lambda x: len(x.split(' '))).values
    q2_word_len = merge.q2.apply(lambda x: len(x.split(' '))).values
           
    result = [len(q1_word_set[i] & q2_word_set[i])/max(q1_word_len[i],q2_word_len[i]) for i in range(len(q1_word_set))]
    result = pd.DataFrame(result, index=data.index)
    result.columns = ['common_word_ratio']
    return result

'''
计算tf-idf向量
'''
def get_tfidf_vector(question, merge_data, word_level=True):
  
    # use the question corpus to train tf-idf vec
    if word_level:
        vectorizer = TfidfVectorizer().fit(question.words.values)   #max_features=1000
        merge = merge_data[['words_x','words_y']]
    else:
        vectorizer = TfidfVectorizer().fit(question.chars.values)
        merge = merge_data[['chars_x','chars_y']]
    merge.columns = ['q1','q2']
        
    q1_tfidf = vectorizer.transform(merge.q1.values)
    q2_tfidf = vectorizer.transform(merge.q2.values)
   
    return vectorizer.vocabulary_,q1_tfidf, q2_tfidf

'''
用tfidf作为系数，调整共现词比例
'''
def common_word_ratio_adjust_with_tfidf(merge_data, word_to_index, q1_tfidf, q2_tfidf, word_level=True):
    
    if word_level:
        merge = merge_data[['words_x','words_y']]
        merge.columns = ['q1','q2']
    else:
        merge = merge_data[['chars_x','chars_y']]
        merge.columns = ['q1','q2']
    
    adjusted_common_word_ratio = []
    
    for i in range(q1_tfidf.shape[0]):
        q1words = {}
        q2words = {}
        for word in merge.loc[i,'q1'].lower().split():
            q1words[word] = q1words.get(word, 0) + 1
        for word in merge.loc[i,'q2'].lower().split():
            q2words[word] = q2words.get(word, 0) + 1
        
        sum_shared_word_in_q1 = sum([q1words[w] * q1_tfidf[i,word_to_index[w]] for w in q1words if w in q2words])
        sum_shared_word_in_q2 = sum([q2words[w] * q2_tfidf[i,word_to_index[w]] for w in q2words if w in q1words])
        sum_tol = sum(q1words[w] * q1_tfidf[i,word_to_index[w]] for w in q1words) + sum(q2words[w] * q2_tfidf[i,word_to_index[w]] for w in q2words)
        if 1e-6 > sum_tol:
            adjusted_common_word_ratio.append(0.)
        else:
            adjusted_common_word_ratio.append(1.0 * (sum_shared_word_in_q1 + sum_shared_word_in_q2) / sum_tol)
     
    return adjusted_common_word_ratio


"""
计算数据中词语的影响力，格式如下：
词语 --> [0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
"""
def generate_powerful_word(merge_data, word_level=True):
    
    if word_level:
        train_subset_data = merge_data[['label','words_x','words_y']]
    else:
        train_subset_data = merge_data[['label','chars_x','chars_y']]
        
    train_subset_data.columns = ['label','q1','q2']
    
    words_power = {}
    
    for i in train_subset_data.index:
        label = int(train_subset_data.loc[i,'label'])
        q1_words = train_subset_data.loc[i,'q1'].lower().split()
        q2_words = train_subset_data.loc[i,'q2'].lower().split()
        all_words = set(q1_words + q2_words)
        q1_words = set(q1_words)
        q2_words = set(q2_words)
        for word in all_words:
            if word not in words_power:
                words_power[word] = [0. for i in range(7)]
                # 计算出现语句对数量
            words_power[word][0] += 1.
            words_power[word][1] += 1.

            if ((word in q1_words) and (word not in q2_words)) or ((word not in q1_words) and (word in q2_words)):
                # 计算单侧语句数量
                words_power[word][3] += 1.
                if 0 == label:
                    # 计算正确语句对数量
                    words_power[word][2] += 1.
                    # 计算单侧语句正确比例
                    words_power[word][4] += 1.
                    
            if (word in q1_words) and (word in q2_words):
                # 计算双侧语句数量
                words_power[word][5] += 1.
                if 1 == label:
                    # 计算正确语句对数量
                    words_power[word][2] += 1.
                    # 计算双侧语句正确比例
                    words_power[word][6] += 1.
    
    for word in words_power:
        # 计算出现语句对比例
        words_power[word][1] /= train_subset_data.shape[0]
        # 计算正确语句对比例
        words_power[word][2] /= words_power[word][0]
        # 计算单侧语句对正确比例
        if words_power[word][3] > 1e-6:
            words_power[word][4] /= words_power[word][3]
        # 计算单侧语句对比例
        words_power[word][3] /= words_power[word][0]
        # 计算双侧语句对正确比例
        if words_power[word][5] > 1e-6:
            words_power[word][6] /= words_power[word][5]
        # 计算双侧语句对比例
        words_power[word][5] /= words_power[word][0]
    
    sorted_words_power = sorted(words_power.items(), key=lambda d: d[1][0], reverse=True)
        
    return sorted_words_power


'''
若问题两侧存在有预测力的powerful words,则设置标签为1，否则为0
'''
def powerful_words_dside_tag(pword, merge_data,thresh_num, thresh_rate, word_level=True):
    #筛选powerful words (有预测力的)
    pword_dside = []
    pword = filter(lambda x: x[1][0] * x[1][5] >= thresh_num, pword)   #保证统计可靠性
    pword_sort = sorted(pword, key=lambda d: d[1][6], reverse=True)
    pword_dside.extend(map(lambda x: x[0], filter(lambda x: x[1][6] >= thresh_rate, pword_sort)))  #保证抽取到真正powerful的word
    
    if word_level:
        merge = merge_data[['words_x','words_y']]
    else:
        merge = merge_data[['chars_x','chars_y']]
        
    merge.columns = ['q1','q2']
    
    pword_dside_tags = []
    
    for i in merge_data.index:
        tags = []
        q1_words = set(merge.loc[i,'q1'].lower().split())
        q2_words = set(merge.loc[i,'q2'].lower().split())
        for word in pword_dside:
            if (word in q1_words) and (word in q2_words):
                tags.append(1.0)
            else:
                tags.append(0.0)
                
        pword_dside_tags.append(tags)
                
    return pword_dside, pword_dside_tags
	
	
def powerful_words_oside_tag(pword, merge_data,thresh_num, thresh_rate, word_level=True):

    pword_oside = []
    pword = filter(lambda x: x[1][0] * x[1][3] >= thresh_num, pword)
    pword_oside.extend(map(lambda x: x[0], filter(lambda x: x[1][4] >= thresh_rate, pword)))

    if word_level:
        merge = merge_data[['words_x','words_y']]
    else:
        merge = merge_data[['chars_x','chars_y']]
        
    merge.columns = ['q1','q2']
    
    pword_oside_tags = []
    
    for i in merge_data.index:
        tags = []
        q1_words = set(merge.loc[i,'q1'].lower().split())
        q2_words = set(merge.loc[i,'q2'].lower().split())
        for word in pword_oside:
            if (word in q1_words) and (word not in q2_words):
                tags.append(1.0)
            elif (word not in q1_words) and (word in q2_words):
                tags.append(1.0)
            else:
                tags.append(0.0)
                
        pword_oside_tags.append(tags)
        
    return pword_oside, pword_oside_tags


def powerful_word_dside_rate(sorted_words_power, pword_dside, merge_data, word_level=True):
    '''
    注意rate是指label=0的可能性，question pair中两侧powerful word越多，power越大，则rate越小
    '''
    num_least = 300
    
    if word_level:
        merge = merge_data[['words_x','words_y']]
    else:
        merge = merge_data[['chars_x','chars_y']]
        
    merge.columns = ['q1','q2']
        
    words_power = dict(sorted_words_power)  #转化为字典格式

    pword_dside_rate = []

    for i in merge.index:
        rate = 1.0    # 指labei=0的可能性,先初始化为1
        q1_words = set(merge.loc[i,'q1'].lower().split())
        q2_words = set(merge.loc[i,'q2'].lower().split())
        share_words = list(q1_words.intersection(q2_words))
        for word in share_words:
            if word in pword_dside:
                rate *= (1.0 - words_power[word][6])    #uestion pair中两侧powerful word越多，power越大，则rate越小
        pword_dside_rate.append(1-rate)
    return pword_dside_rate
    
def powerful_word_oside_rate(sorted_words_power, pword_oside, merge_data, word_level=True):
    '''
    注意rate是指label=1的可能性，question pair中单侧powerful word越多，power越大，则rate越小
    '''
    num_least = 300
    
    if word_level:
        merge = merge_data[['words_x','words_y']]
    else:
        merge = merge_data[['chars_x','chars_y']]
        
    merge.columns = ['q1','q2']
    words_power = dict(sorted_words_power)  #转化为字典格式
        
    pword_oside_rate = []
        
    for i in merge.index:
        rate = 1.0    # 指labei=1的可能性,先初始化为1
        q1_words = set(merge.loc[i,'q1'].lower().split())
        q2_words = set(merge.loc[i,'q2'].lower().split())
        q1_diff = list(set(q1_words).difference(set(q2_words)))
        q2_diff = list(set(q2_words).difference(set(q1_words)))
        all_diff = set(q1_diff + q2_diff)
        for word in all_diff:
            if word in pword_oside:
                rate *= (1.0 - words_power[word][4])    #question pair中单侧powerful word越多，power越大，则rate越小
        pword_oside_rate.append(1-rate)
            
    return pword_oside_rate
    


'''
扩展的编辑距离(Damerau-Levenshtein Distance)
扩展的编辑距离在思想上与编辑距离一样，只是除插入、删除和替换操作外，还支持 相邻字符的交换 这样一个操作，增加这个操作的考虑是人们在计算机上输入文档时的错误情况中，因为快速敲击而前后两个字符的顺序被输错的情况很常见。
'''
def edit_distance(q1, q2):
    
    str1 = q1.split(' ')
    str2 = q2.split(' ')
    matrix = [[i+j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
 
    for i in range(1,len(str1)+1):
        for j in range(1,len(str2)+1):
            if str1[i-1] == str2[j-1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i-1][j]+1,matrix[i][j-1]+1,matrix[i-1][j-1]+d)
 
        if i > 1 and j > 1 and str1[i-1] == str2[j-2] and str1[i-2] == str2[j-1]:
            d = 0   # d=0表示允许交换，d =1表示不允许交换
            matrix[i][j] = min(matrix[i][j], matrix[i-2][j-2] + d)   # allow transposition
 
    return matrix[len(str1)][len(str2)]

def get_edit_distance(merge_data, word_level=True):

    if word_level:
        merge = merge_data[['words_x','words_y']]
        merge.columns = ['q1','q2']
    else:
        merge = merge_data[['chars_x','chars_y']]
        merge.columns = ['q1','q2']
        
    q1_len = merge['q1'].apply(lambda x: len(x.split(' '))).values
    q2_len = merge['q2'].apply(lambda x: len(x.split(' '))).values
    
    # normalize the edit_distance by the max(len(q1),len(q2))
    dist =[edit_distance(merge.loc[i,'q1'],merge.loc[i,'q2'])/ np.max([q1_len,q2_len],axis=0)[i] for i in merge.index]
    
    return dist
  