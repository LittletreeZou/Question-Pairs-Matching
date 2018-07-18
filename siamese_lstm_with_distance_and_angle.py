import numpy as np
import pandas as pd
np.random.seed(0)

from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, BatchNormalization,concatenate,Subtract, Dot, Multiply,Bidirectional,Lambda
from keras.layers.embeddings import Embedding
from keras.initializers import glorot_uniform
from keras.layers.noise import GaussianNoise
from keras import backend as K
from keras import optimizers
import tensorflow as tf

import keras.callbacks as kcallbacks
np.random.seed(1)

from data_helper import *

import warnings
warnings.filterwarnings('ignore')

# jupyter magic commands，自动重新加载更改的模块
%load_ext autoreload
%autoreload 2


MAX_SEQUENCE_LENGTH = 15  # 20 for character level and 15 for word level
EMBEDDING_DIM = 300

# 读取数据
train_q1, train_q2, train_label, embed_matrix = load_dataset(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, word_level=True)
print('train_q1: ',train_q1.shape)
print('train_q2: ', train_q2.shape)
print('train_label: ',train_label.shape)
print('embed_matrix: ',embed_matrix.shape)

# 加载test 数据
test_q1, test_q2 = load_test_data( MAX_SEQUENCE_LENGTH, word_level=True)
print('test_q1: ',test_q1.shape)
print('test_q2: ', test_q2.shape)


# 读取手工特征
train_features = pd.read_csv('features/0714_all_train_features_17.csv')
test_features = pd.read_csv('features/0714_all_test_features_17.csv')

train_moka_features = pd.read_csv('features/non_nlp_features_train.csv')
test_moka_features = pd.read_csv('features/non_nlp_features_test.csv')

train_features = pd.merge(train_features, train_moka_features, left_index=True, right_index=True)
test_features = pd.merge(test_features, test_moka_features, left_index=True, right_index=True)

pick_columns = ['adjusted_common_word_ratio', 'edit_distance','len_diff', 'pword_dside_rate', 'pword_oside_rate',
                'adjusted_common_char_ratio', 'pchar_dside_rate', 'pchar_oside_rate',
                'coo_max_degree_(0, 5]','coo_max_degree_(5, 30]', 'coo_max_degree_(30, 130]',
                 'coo_q1_q2_degree_diff','common_neighbor_ratio']

train_features = train_features[pick_columns]
test_features = test_features[pick_columns]

train_features.info()


# 读取数据分裂index
split_index = {}
for i in range(10):
    split_index[i]= pd.read_csv('features/0714_train_split_index/vali_idx_'+str(i)+'.csv').idx.values
	

# define model, 10-fold cv

best_vali_score ={}

def trainLSTM(train_q1, train_q2, train_label, embed_matrix, test_q1, test_q2, train_features, test_features, split_index):
   
    lstm_num = 75
    lstm_drop = 0.5
    BATCH_SIZE = 256  # 128
    
    for model_count in range(10):
        
        print("MODEL:", model_count)
            
        # split data into train/vali set
        idx_val = split_index[model_count]
        idx_train = []
        for i in range(10):
             if i != model_count:
                    idx_train.extend(list(split_index[i]))

        q1_train = train_q1[idx_train]
        q2_train = train_q2[idx_train]
        y_train = train_label[idx_train]
        f_train = train_features[idx_train]
       
        q1_val = train_q1[idx_val]
        q2_val = train_q2[idx_val]
        y_val = train_label[idx_val]
        f_val = train_features[idx_val]
    
        # Define the model
        question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
        question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

        embed_layer = Embedding(embed_matrix.shape[0], EMBEDDING_DIM, weights=[embed_matrix],
                                input_length=MAX_SEQUENCE_LENGTH, trainable=False)

        q1_embed = embed_layer(question1)
        q2_embed = embed_layer(question2)

        shared_lstm_1 = LSTM(lstm_num, return_sequences=True)
        shared_lstm_2 = LSTM(lstm_num)

        q1 = shared_lstm_1(q1_embed)
        q1 = Dropout(lstm_drop)(q1)
        q1 = BatchNormalization()(q1)
        q1 = shared_lstm_2(q1)
        # q1 = Dropout(0.5)(q1)

        q2 = shared_lstm_1(q2_embed)
        q2 = Dropout(lstm_drop)(q2)
        q2 = BatchNormalization()(q2)
        q2 = shared_lstm_2(q2)
        # q2 = Dropout(0.5)(q2)   # of shape (batch_size, 128)

        # 求distance (batch_size,1)
        d = Subtract()([q1, q2])
        #distance = Dot(axes=1, normalize=False)([d, d])
        #distance = Lambda(lambda x: K.abs(x))(d)
        distance = Multiply()([d, d])
        # 求angle (batch_size,1)
        # angle = Dot(axes=1, normalize=False)([q1, q2])
        angle = Multiply()([q1, q2])
        # merged = concatenate([distance,angle])

        # magic featurues
        magic_input = Input(shape=(train_features.shape[1],))
        magic_dense = BatchNormalization()(magic_input)
        magic_dense = Dense(64, activation='relu')(magic_dense)
        #magic_dense = Dropout(0.3)(magic_dense)
        
        merged = concatenate([distance,angle,magic_dense])
        merged = Dropout(0.3)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(256, activation='relu')(merged)  # 64
        merged = Dropout(0.3)(merged)
        merged = BatchNormalization()(merged)

        merged = Dense(64, activation='relu')(merged)  # 64
        merged = Dropout(0.3)(merged)
        merged = BatchNormalization()(merged)

        is_duplicate = Dense(1, activation='sigmoid')(merged)

        model = Model(inputs=[question1, question2, magic_input], outputs=is_duplicate)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        model.summary()
        
        # define save model 
        best_weights_filepath = 'models/0715 lstm keras/word_lstm_with_magics_' + str(model_count) + '.hdf5'
        earlyStopping = kcallbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
        saveBestModel = kcallbacks.ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1,\
                                                   save_best_only=True, mode='auto')

        hist = model.fit([q1_train, q2_train, f_train], 
                         y_train,
                         validation_data=([q1_val, q2_val, f_val], y_val),
                         epochs=30, 
                         batch_size=BATCH_SIZE, 
                         shuffle=True,
                         callbacks=[earlyStopping, saveBestModel], 
                         verbose=1)

        model.load_weights(best_weights_filepath)
        print(model_count, "validation loss:", min(hist.history["val_loss"]))
        best_vali_score[model_count] = min(hist.history["val_loss"])
        
        # predict on the val set
        preds = model.predict([q1_val, q2_val, f_val], batch_size=1024, verbose=1)
        val_preds = pd.DataFrame({"y_pre": preds.ravel()})
        val_preds['val_index'] = idx_val
        save_path = 'features/0715_lstm_word_with_magic/vali_' + str(model_count) + '.csv'
        val_preds.to_csv(save_path, index=0)
        print(model_count, "val preds saved.")
        
        # predict on the test set
        preds1 = model.predict([test_q1, test_q2, test_features], batch_size=1024, verbose=1)
        test_preds = pd.DataFrame({"y_pre": preds1.ravel()})
        save_path1 = 'features/0715_lstm_word_with_magic/test_' + str(model_count) + '.csv'
        test_preds.to_csv(save_path1, index=0)
        print(model_count, "test preds saved.")
		

# run the model and predict		
import time
start = time.time()

trainLSTM(train_q1, train_q2, train_label, embed_matrix, test_q1, test_q2, \
          train_features.values, test_features.values, split_index)

end = time.time()
print('Training time {0:.3f} 分钟'.format((end-start)/60))