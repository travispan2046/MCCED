import argparse
import json
import numpy as np
import os
from keras.layers import Input, ConvLSTM2D, BatchNormalization, Conv2D, Reshape, Conv3D, Flatten, RepeatVector
from keras.models import Sequential
from keras.models import Model
import scipy.io as scio
import random
from keras.models import Model
from keras.layers import Input, LSTM, Dense, ConvLSTM2D, BatchNormalization, Conv3D, TimeDistributed
from sklearn import metrics
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.layers import Lambda
import tensorflow as tf
from sklearn import preprocessing
import pandas as pd
from sklearn import metrics

import conditional_test_add_pure_merge_interval


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

'''function'''
def options_parser():
    parser = argparse.ArgumentParser(description='Train a neural network to handle real-valued data.')
    # meta-option
    parser.add_argument('--settings_file', help='json file of settings, overrides everything else', type=str, default='')
    return parser

def load_settings_from_file(settings):
    # settings可以是任何一个txt形式的字典文件
    settings_path = "./"+settings['settings_file'] + ".txt"
    print('Loading settings from', settings_path)
    settings_loaded = json.load(open(settings_path, 'r',encoding='utf-8'))
    # check for settings missing in file
    return settings_loaded



def get_settings_and_files():
    parser = options_parser()
    settings = vars(parser.parse_args())

    if settings['settings_file']:
        settings = load_settings_from_file(settings)

    result_path = "../seconddata/" + settings['preprocessing_timepath'] + "ConvLstm预处理结果(conditional版本)/"
    list = os.listdir(result_path)
    total_result = []
    for i in range(0, len(list)):
        path = os.path.join(result_path, list[i])
        if os.path.isfile(path):
            total_result.append(np.load(path)['result'][()])
    choice_result = total_result[0]
    train_input = choice_result['train_input']
    train_predict = choice_result["train_predict"]
    test_input = choice_result["test_input"]
    test_predict = choice_result["test_predict"]
    ground_truth=choice_result["ground_truth"]
    params = settings
    return params,ground_truth,train_input,train_predict,test_input,test_predict



def sort_model_by_time(model_path):
    models = os.listdir(model_path)
    if not models:
        return
    else:
        files = sorted(models, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
        return files


def one_model_operation_conditional(encoder_model,past_decoder_model,fu_decoder_model,model_path):
    encoder_model.load_weights(model_path, by_name=True)
    past_decoder_model.load_weights(model_path, by_name=True)
    fu_decoder_model.load_weights(model_path, by_name=True)
    _,_, train_input, train_predict, test_input, test_predict = get_settings_and_files()
    train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true = network.NN_predict(encoder_model,
                                                                                        past_decoder_model,
                                                                                        fu_decoder_model, train_input,
                                                                                        train_predict)
    test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true = network.NN_predict(encoder_model, past_decoder_model,
                                                                                    fu_decoder_model, test_input,
                                                                                    test_predict)
    return train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true,test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true

def one_model_operation_unconditional(model,model_path):
    model.load_weights(model_path, by_name=True)
    _,_, train_input, train_predict, test_input, test_predict = get_settings_and_files()
    train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true = network.NN_predict(model,train_input,train_predict)
    test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true = network.NN_predict(model, test_input, test_predict)
    return train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true,test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true


def get_test_results(model_path):
    params, ground_truth, train_input, train_predict, test_input, test_predict=get_settings_and_files()
    
  
    window = 120
    ground_truth = ground_truth[:-window]
    
    if network.NN_predict.__code__.co_argcount>4:
        encoder_model, past_decoder_model, fu_decoder_model = network.network(params, train_input, train_predict,False)
        train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true, test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true=one_model_operation_conditional(encoder_model,past_decoder_model,fu_decoder_model, model_path)
    else:
        model = network.network(params, train_input, train_predict)
        train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true, test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true= one_model_operation_unconditional(model, model_path)

    '''prediction + reconstruction'''
    beta = 1.0
    alpha = 0.0
    train_predict_NN = beta * train_predict_NN[:-window] + alpha * train_rcstr_NN[window:]
    train_predict_true = beta * train_predict_true[:-window] + alpha * train_rcstr_true[window:]
    test_predict_NN = beta * test_predict_NN[:-window] + alpha *test_rcstr_NN[window:]
    test_predict_true = beta * test_predict_true[:-window] + alpha * test_rcstr_true[window:]

    
    
    start_t1 = '2015-12-28 10:02:00 AM'
    end_t1 = '2016-1-2 2:59:59 PM'
    # start_t1 = '2017-10-9 06:02:00 PM'
    # end_t1 = '2017-10-11 06:00:00 PM'

    '''single_point'''
    test_obj = conditional_test_add_pure_merge_interval.analysis(train_predict_true, train_predict_NN, test_predict_true,
                                                  test_predict_NN,
                                                  ground_truth, start_t1, end_t1, 120, weight_or_not=True,
                                                  weight_type=1,
                                                  norm_or_not=False,
                                                  specific= model_path)
    print('max',np.max(test_obj.wgt_error_out_ewma))
    print('min',np.min(test_obj.wgt_error_out_ewma))
    my_dict = test_obj.threshold_grid_search_no_plotting(weight_or_not=True, number=100, lower=0.01, upper=0.15,
                                                         consecutive_or_not=False, fine_or_not=True)
    print(my_dict)
    
    
    
    from sklearn import metrics
    seq_anomaly = list(np.argwhere(ground_truth == 1).reshape(-1))
    seq_norm = list(np.argwhere(ground_truth == 0).reshape(-1))
    test_pure_anomaly_error = metrics.mean_squared_error(test_predict_NN[seq_anomaly, :],
                                                         test_predict_true[seq_anomaly, :])
    test_pure_norm_error = metrics.mean_squared_error(test_predict_NN[seq_norm, :],
                                                      test_predict_true[seq_norm, :])
    my_dict2 = {
        'prediction error of normal series in test dataset': test_pure_norm_error,
        'prediction error of abnornal series in test dataset': test_pure_anomaly_error,
        'gap': test_pure_anomaly_error - test_pure_norm_error
    }
    print(my_dict2)



if __name__ == '__main__':
    
    import advanced_mem_无卷积_内积式attention_1layer_非teacher_forcing as network
    model_path="../resultdata/11_17_21_24conditional_training(conditional)/models/11_17_21_24model-ep073-loss0.21003-val_loss0.16002.h5"
    get_test_results(model_path=model_path)

