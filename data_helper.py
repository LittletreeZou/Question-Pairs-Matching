import numpy as np
import pandas as pd
import os
import math


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` 
    
    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 
    
    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    m = X.shape[0]            # number of training examples 
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros((m, max_len))
    for i in range(m):                               # loop over training examples
        # split the sentences into words
        sentence_words =X[i].split(' ') 
        # Loop over the words of sentence_words
        for j,w in enumerate(sentence_words):
            if j >= max_len:
                break
            # Set the (i,j)th entry of X_indices to the index of the correct word.
            X_indices[i, j] = word_to_index[w]
            
    return X_indices


def load_dataset(max_seq_len, embed_dim, word_level=True):
    '''
    读取数据，对数据进行预处理，并生成embed_matrix
    '''
    #1、读取数据，数据预处理
    #数据路径
    question_path = os.path.join('datasets', 'question.csv')
    train_path = os.path.join('datasets', 'train.csv')
    if word_level:
        embed_path = os.path.join('datasets', 'word_embed.txt')
    else:
        embed_path = os.path.join('datasets', 'char_embed.txt')
    
    #读取数据
    question = pd.read_csv(question_path)
    
    train = pd.read_csv(train_path)
    # 把train里面的问题id匹配到句子
    train = pd.merge(train,question,left_on=['q1'],right_on=['qid'],how='left')
    train = pd.merge(train,question,left_on=['q2'],right_on=['qid'],how='left')
    
    if word_level:
        train = train[['label','words_x','words_y']]
    else:
        train = train[['label','chars_x','chars_y']]
    train.columns = ['label','q1','q2']
    
    # 读取word_to_vec_map，注意这里的index是word id
    word_to_vec_map = pd.read_csv(embed_path, sep=' ', header=None, index_col=0)
    
    # 先定义两个字典，实现wid与(positive) index的相互转换,注意index从1开始
    word = word_to_vec_map.index.values
    word_to_index = dict([(word[i],i+1) for i in range(len(word))])
    index_to_word = dict([(i+1, word[i]) for i in range(len(word))])
    
    # 把句子转换成int indices,并zero pad the sentance to max_seq_len
    train_q1_indices = sentences_to_indices(train.q1.values, word_to_index, max_seq_len)
    train_q2_indices = sentences_to_indices(train.q2.values, word_to_index, max_seq_len)
    label = train.label.values
    
    #3、生成embeding_matrix, index为整数，其中index=0,对应的是np.zeros(300),0向量，对应我们padding的值
    vocab_len = len(word_to_index) + 1                                   
    # Initialize the embedding matrix as numpy arrays of zeros
    embed_matrix = np.zeros((vocab_len, embed_dim))
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        embed_matrix[index, :] = word_to_vec_map.loc[word].values

    return (train_q1_indices,train_q2_indices, label, embed_matrix)

	

def load_test_data(max_seq_len, word_level=True):
    '''
    读取测试数据
    '''
    #1、读取数据，数据预处理
    #数据路径
    question_path = os.path.join('datasets', 'question.csv')
    test_path = os.path.join('datasets', 'test.csv')
    if word_level:
        embed_path = os.path.join('datasets', 'word_embed.txt')
    else:
        embed_path = os.path.join('datasets', 'char_embed.txt')
    
    #读取数据
    question = pd.read_csv(question_path)
    test = pd.read_csv(test_path)
    # 把train里面的问题id匹配到句子
    test = pd.merge(test,question,left_on=['q1'],right_on=['qid'],how='left')
    test = pd.merge(test,question,left_on=['q2'],right_on=['qid'],how='left')
    if word_level:
        test = test[['words_x','words_y']]
    else:
        test = test[['chars_x','chars_y']]
    test.columns = ['q1','q2']
    # 读取word_to_vec_map，注意这里的index是word id
    word_to_vec_map = pd.read_csv(embed_path, sep=' ', header=None, index_col=0)
    
    # 先定义两个字典，实现wid与(positive) index的相互转换,注意index从1开始
    word = word_to_vec_map.index.values
    word_to_index = dict([(word[i],i+1) for i in range(len(word))])
    index_to_word = dict([(i+1, word[i]) for i in range(len(word))])
    
    # 把句子转换成int indices,并zero pad the sentance to max_seq_len
    test_q1_indices = sentences_to_indices(test.q1.values, word_to_index, max_seq_len).astype(np.int32)
    test_q2_indices = sentences_to_indices(test.q2.values, word_to_index, max_seq_len).astype(np.int32)
    
    
    return test_q1_indices,test_q2_indices



