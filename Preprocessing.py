# coding=UTF-8
import pandas as pd
import numpy as np
import datetime
#from prettytable import PrettyTable
from sklearn import preprocessing
import os
import matplotlib.pyplot as plt
import argparse
import json

#####################################################  读入数据  ##########################################################
dataset_station_train="../../dataset/SWaT_Dataset_Normal_v0.csv"
"""E:\\yuting\\SWAT\\SWaT_Dataset_Normal_v0.csv"""
df_train=pd.read_csv(dataset_station_train,sep=',',header=1)
dataset_station_test="../../dataset/SWaT_Dataset_Attack_v0.csv"
"""E:\\yuting\\SWAT\\SWaT_Dataset_Attack_v0.csv"""
df_test=pd.read_csv(dataset_station_test,sep=',')


#####################################################  初始化  ##########################################################
global df_mat_train
df_mat_train=df_train.as_matrix(columns=None)
df_mat_train=np.array(df_mat_train[16000:,1:52],dtype=np.float64)
global df_mat_test
df_mat_test=df_test.as_matrix(columns=None)
df_mat_test[:,-1]=np.where(df_mat_test[:,-1]=="Normal",0,1)
ground_truth = df_mat_test[:,-1][:,np.newaxis]
df_mat_test=np.array(df_mat_test[:,1:52],dtype=np.float64)

print("ground_truth 此时的类型和shape是：", type(ground_truth),ground_truth.shape)
print("test 此时的类型和shape是：",type(df_mat_test),df_mat_test.shape)
'''
加入删除长攻击的代码
'''
'''
index = range(225919, 263729)
df_mat_test = np.delete(df_mat_test, index, axis=0)
ground_truth = np.delete(ground_truth, index)
print("删除长攻击之后！")
print("ground_truth 此时的类型和shape是：", type(ground_truth),ground_truth.shape)
print("test 此时的类型和shape是：",type(df_mat_test),df_mat_test.shape)
'''


'''训练集 中画图确定前16000个点系统不稳定，没有意义，因此修剪掉'''

'''yuting 下采样 2019/8/3'''
#生成时间序列
'''
mat_train=[]
time = pd.date_range('9/10/2017', periods=df_mat_train.shape[0], freq='S')
for i in range(df_mat_train.shape[1]):
    data=pd.Series(df_mat_train[:,i],time)
    data=data.resample(rule='5S').median()
    data=np.array(data)
    mat_train.append(data)
mat_train=np.array(mat_train).T
df_mat_train=mat_train
'''

'''yuting 下采样  2019/8/3'''

###################################################  数据归一化  ########################################################
def Z_ScoreNorm(column,_train_,train_scaler):
    if _train_==1:
    #     if np.std(column)==0:
    #         return [(1.0/(1 + np.exp(column[row]))) for row in range(column.shape[0])]
    #     else:

        return preprocessing.scale(column)
    else:
        return np.array(train_scaler.transform(column.reshape(-1,1))).reshape(-1)

def dataset_norm(minmax_or_not,train_set,test_set):
    print("正在使用对训练集和测试集进行归一化")
    train_set1=[]
    test_set1=[]
    if minmax_or_not==True:
        
        for i in range(train_set.shape[1]):
            min_max_scaler = preprocessing.MinMaxScaler()
            c=min_max_scaler.fit_transform(train_set[:, i].reshape(-1,1)).reshape(-1)
            d=min_max_scaler.transform(test_set[:,i].reshape(-1,1)).reshape(-1)
            train_set1.append(c)
            test_set1.append(d)
    else:
        
        for i in range(train_set.shape[1]):
            standard_scaler = preprocessing.StandardScaler()
            c=standard_scaler.fit_transform(train_set[:, i].reshape(-1,1)).reshape(-1)
            d=standard_scaler.transform(test_set[:,i].reshape(-1,1)).reshape(-1)
            train_set1.append(c)
            test_set1.append(d)
    return np.array(train_set1,dtype=np.float64).T,np.array(test_set1,dtype=np.float64).T

#####################################################  Train/Test无偏处理  ##########################################################
def Dataset_Processing(dataset, sample_size, input_win, predict_win, step, in_frame_size, sensor_id0, sensor_ide,
                       _unfold_):
    row = dataset.shape[0]
    # overlap_rate=0.5
    # step=int((1-overlap_rate)*sample_size)
    sample_num = (row - sample_size) // step + 1
    row_modified = sample_num * sample_size
    '''在一个样本之内分为input、horizon和predict3个部分，其中对input部分进行了frame分割，而predict作为完整的一块儿不进行分割'''
    horizon_win = sample_size - input_win - predict_win
    in_frame_num = input_win // in_frame_size
    pre_frame_num = predict_win // in_frame_size
    tmp_result = dataset
    '''试验点2'''
    time_stamp = np.array([i for i in range(0, row)], np.float32)[:, np.newaxis]
    tmp_result = np.concatenate((tmp_result, time_stamp), axis=1)
    sensor_scope = [sensor_id for sensor_id in range(sensor_id0 - 1, sensor_ide)]
    sensor_time_dim = len(sensor_scope)
    '''[1,51]是传感器变量'''
    # 卷的操作
    sample_generation = []

    def gen_sample(total_array):
        for start, stop in zip(range(0, row - sample_size + 1, step), range(sample_size, row + 1, step)):
            yield total_array[start:stop, :]

    for sample_arr in gen_sample(tmp_result):
        sample_generation.extend(sample_arr)
    sample_generation = np.array(sample_generation, np.float32)

    sample_divided_mat = []
    for sample_id in range(0, sample_num):
        sample_divided_mat.append(
            sample_generation[sample_id * sample_size:(sample_id + 1) * sample_size, sensor_scope])
    sample_divided_mat = np.array(sample_divided_mat)

    def sample_inside(a_sample, _input_):
        # a_sample是2维数组
        # for sample_id in range(0,sample_num):
        framed_sample = []
        if _input_ == 1:
            a_sample = a_sample[-in_frame_num * in_frame_size:, :]
            for frame_id in range(0, in_frame_num):
                framed_sample.append(a_sample[frame_id * in_frame_size:(frame_id + 1) * in_frame_size, :])
        else:
            a_sample = a_sample[:in_frame_num * in_frame_size, :]
            for frame_id in range(0, pre_frame_num):
                framed_sample.append(a_sample[frame_id * in_frame_size:(frame_id + 1) * in_frame_size, :])

        '''frame修剪,对除不尽部分进行舍弃,对于预测窗来说且除不尽的部分在前，除得尽的部分在后'''
        return np.array(framed_sample)

    sample_id = 0
    input_generation = []
    for sample_id in range(sample_num):
        input_generation.append(sample_inside(sample_divided_mat[sample_id, :input_win, :], 1))
    input_generation = np.array(input_generation)

    if _unfold_ == True:
        predict_generation = []
        sample_id = 0
        for sample_id in range(sample_num):
            predict_generation.append(sample_inside(sample_divided_mat[sample_id, -predict_win:, :], 0))
        predict_generation = np.array(predict_generation)
    else:
        predict_generation = sample_divided_mat[:, -predict_win:, :]

    return input_generation, predict_generation


def dataset_skip_norm_orno(mat_train, mat_test, norm_method, minmax_or_not):
    '''
    :param 对于settings['test_norm_with_train']参数:
    :param 为true时我们使用训练集参数归一化测试集，为false时我们将训练集和测试集进行统一归一化:
    :return:
    '''

    if norm_method == 'test_norm_with_train':
        normed_train_set = "normed_data_with_train.npz"
        if os.path.exists(normed_train_set):
            print("Find cache file %s" % normed_train_set)
            c = np.load('normed_data_with_train.npz')
            mat_train = c['mat_train']
            mat_test = c['mat_test']
        else:
            mat_train, mat_test = dataset_norm(minmax_or_not=minmax_or_not, train_set=mat_train, test_set=mat_test)
            print(mat_train.shape)
            print(mat_test.shape)
            #np.savez("normed_data_with_train.npz", mat_train=mat_train, mat_test=mat_test)
        print("#以训练集参数归一化测试集#结束")
        return mat_train, mat_test
    elif norm_method == 'norm_together':
        normed_train_set = "normed_data_together.npz"
        if os.path.exists(normed_train_set):
            print("Find cache file %s" % normed_train_set)
            c = np.load('normed_data_together.npz')
            mat_train = c['mat_train']
            mat_test = c['mat_test']
        else:
            aa = np.append(mat_train, mat_test, axis=0)
            aa, aa = dataset_norm(minmax_or_not=minmax_or_not, train_set=aa, test_set=aa)
            mat_train = aa[:mat_train.shape[0], :]
            mat_test = aa[mat_train.shape[0]:, :]
            print(mat_train.shape)
            print(mat_test.shape)
            # mat_train, train_scaler1 = dataset_norm(mat_train, 1, [])
            # mat_test = dataset_norm(mat_test, 0, train_scaler1)
            #np.savez("normed_data_together.npz", mat_train=mat_train, mat_test=mat_test)
        print("训练集和测试集#统一归一化#结束")
        return mat_train, mat_test

    elif norm_method == 'norm_respectively':
        normed_train_set = "normed_data_respectively.npz"
        if os.path.exists(normed_train_set):
            print("Find cache file %s" % normed_train_set)
            c = np.load('normed_respectively.npz')
            mat_train = c['mat_train']
            mat_test = c['mat_test']
        else:
            mat_train, mat_train = dataset_norm(minmax_or_not=minmax_or_not, train_set=mat_train, test_set=mat_train)
            mat_test, mat_test = dataset_norm(minmax_or_not=minmax_or_not, train_set=mat_test, test_set=mat_test)
            print(mat_train.shape)
            print(mat_test.shape)
            #np.savez("normed_data_respectively.npz", mat_train=mat_train, mat_test=mat_test)
        print("训练集和测试集#各自归一化#结束")
        return mat_train, mat_test


def interval_generation(sequence, proportion=None):
    if proportion == None:
        flag0 = 0
        flage = 0
        interval = []

        for i in range(len(sequence)):
            if i != len(sequence) - 1:  # 如果不等于最后一个元素的话
                if sequence[i + 1] - sequence[i] > 1:
                    flage = i
                    interval.append((sequence[flag0], sequence[flage]))
                    flag0 = i + 1
            else:
                interval.append((sequence[flag0], sequence[i]))
        return interval
    else:
        flag0 = 0
        flage = 0
        interval = []

        for i in range(len(sequence)):
            if i != len(sequence) - 1:  # 如果不等于最后一个元素的话
                if sequence[i + 1] - sequence[i] > 1:
                    flage = i
                    add_len = int((sequence[flage] - sequence[flag0] + 1) * proportion)
                    if sequence[flage] + add_len < sequence[i + 1] - 2:
                        interval.append((sequence[flag0], sequence[flage] + add_len))
                    else:
                        interval.append((sequence[flag0], sequence[i + 1] - 2))
                    flag0 = i + 1
            else:
                interval.append((sequence[flag0], sequence[i]))
        return interval


def test_split(ground_truth, test_data, choice):
    seq_ground = np.argwhere(ground_truth == 0).reshape(-1)  # 选取正常的数据点
    ground_interval = interval_generation(seq_ground)  # 通过interval_generation生成ground_truth中为0的区间
    test_data_set = []
    # print(ground_interval)
    for i in range(len(ground_interval)):

        if ground_interval[i][0] != ground_interval[i][1]:
            test_data_set.append(test_data[ground_interval[i][0]:ground_interval[i][1] + 1, :])
        else:
            test_data_set.append(test_data[ground_interval[i][0]:ground_interval[i][1] + 1, :])
    # print(seq_ground)
    test_pure_normal_input = []
    test_pure_normal_predict = []
    for i in range(len(test_data_set)):
        a, b = Dataset_Processing(test_data_set[i], choice.sample_size, choice.input_win, choice.predict_win,
                                  choice.step, choice.frame_size, choice.sensor_id0, choice.sensor_ide,
                                  choice.conditional)
        test_pure_normal_input.extend(a)
        test_pure_normal_predict.extend(b)
    test_pure_normal_input = np.array(test_pure_normal_input)
    test_pure_normal_predict = np.array(test_pure_normal_predict)
    # print("新分出的测试集正常序列shape为",test_pure_normal_input.shape)
    return test_pure_normal_input, test_pure_normal_predict


def main(df_mat_train, df_mat_test, ground_truth, choice):
    
    
    train_data, test_data = dataset_skip_norm_orno(df_mat_train, df_mat_test, choice.norm_method, choice.minmax_or_not)
    '''先归一化后加时间戳'''
    train_input, train_predict = Dataset_Processing(dataset=train_data, sample_size=choice.sample_size,
                                                    input_win=choice.input_win, predict_win=choice.predict_win,
                                                    step=choice.step,
                                                    in_frame_size=choice.frame_size, sensor_id0=choice.sensor_id0,
                                                    sensor_ide=choice.sensor_ide, _unfold_=choice.conditional)
    test_input, test_predict = Dataset_Processing(test_data, choice.sample_size, choice.input_win, choice.predict_win,
                                                  choice.step, choice.frame_size, choice.sensor_id0, choice.sensor_ide,
                                                  choice.conditional)
    # ground_truth=ground_truth[choice.input_win:(ground_truth.shape[0]//choice.predict_win)*choice.predict_win].reshape(-1)
    horizon = choice.sample_size - choice.predict_win
    ground_truth = ground_truth[horizon:(ground_truth.shape[0] // choice.predict_win) * choice.predict_win].reshape(-1)

    # test_pure_normal_input, test_pure_normal_predict=test_split(ground_truth, test_data, choice)  # 保证提取的是已经归一化之后的test数据
    # 根据删减传感器组进行切片
    if choice.sensor_list != "default":
        train_input = train_input[:, :, :, choice.sensor_list]
        train_predict = train_predict[:, :, :, choice.sensor_list]
        test_input = test_input[:, :, :, choice.sensor_list]
        test_predict = test_predict[:, :, :, choice.sensor_list]
        # test_pure_normal_input=test_pure_normal_input[:,:,:,choice.sensor_list]
        # test_pure_normal_predict=test_pure_normal_predict[:,:,:,choice.sensor_list]
    # 根据删减传感器组进行切片

    ######################################################### 为输入数据加入噪声 #############################################################
    # train_max = np.max(train_input)
    # test_max = np.max(test_input)
    # train_min = np.min(train_input)
    # test_min = np.min(test_input)
    # train_mean = np.mean(train_input)
    # test_mean = np.mean(test_input)
    # train_var = np.var(train_input)
    # test_var = np.var(test_input)
    # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 加入噪声之前数据集的统计学特性 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print("<<<train dataset>>> 形状:{4}, 最大值：{0}, 最小值：{1}, 均值:{2}, 方差:{3}".format(train_max, train_min, train_mean,
    #                                                                             train_var, train_input.shape))
    # print("<<<test dataset>>> 形状:{4}, 最大值：{0}, 最小值：{1}, 均值:{2}, 方差:{3}".format(test_max, test_min, test_mean, test_var,
    #                                                                            test_input.shape))
    #
    # # wight = 0.5
    # mean = 0.0
    # scale = 1.0
    # train_input_noise = train_input + wight * np.random.normal(loc=mean, scale=scale, size=train_input.shape)
    # test_input_noise = test_input + wight * np.random.normal(loc = mean, scale = scale, size = test_input.shape)
    #
    # # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 加入噪声之后数据集的统计学特性 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # train_max = np.max(train_input_noise)
    # test_max = np.max(test_input_noise)
    # train_min = np.min(train_input_noise)
    # test_min = np.min(test_input_noise)
    # train_mean = np.mean(train_input_noise)
    # test_mean = np.mean(test_input_noise)
    # train_var = np.var(train_input_noise)
    # test_var = np.var(test_input_noise)
    # print("<<<train dataset>>> 形状:{4}, 最大值：{0}, 最小值：{1}, 均值:{2}, 方差:{3}".format(train_max, train_min, train_mean,
    #                                                                             train_var, train_input.shape))
    # print("<<<test dataset>>> 形状:{4}, 最大值：{0}, 最小值：{1}, 均值:{2}, 方差:{3}".format(test_max, test_min, test_mean, test_var,
    #                                                                            test_input.shape))


    # my_dict={"tag":choice.tag,"train_input":train_input,"train_predict":train_predict,"test_input":test_input,"test_predict":test_predict,
    #          "ground_truth":ground_truth,"test_pure_normal_input":test_pure_normal_input,"test_pure_normal_predict":test_pure_normal_predict}
    my_dict = {"tag": choice.tag, "train_input": train_input, "train_predict": train_predict, "test_input": test_input,
               "test_predict": test_predict,
               "ground_truth": ground_truth}
    return my_dict


class hyper_tuning_suite:
    def __init__(self, choice_descrip, sensor_id0, sensor_ide, input_win, predict_win, sample_size, step, frame_size,
                 conditional, norm_method, minmax_or_not, sensor_list):
        self.choice_descrip = choice_descrip
        if norm_method == 'test_norm_with_train':
            self.tag = "训练归一测试_" + self.choice_descrip[0] + "_" + self.choice_descrip[1]  # 这个[0]和[1]是不会变的 这是test文件中的
        elif norm_method == 'norm_together':
            self.tag = "训练测试统一归一化_" + self.choice_descrip[0] + "_" + self.choice_descrip[1]  # 这个[0]和[1]是不会变的 这是test文件中的
        else:
            if minmax_or_not == True:
                self.tag = "训练测试各自归一化_minmax_" + self.choice_descrip[0] + "_" + self.choice_descrip[
                    1]  # 这个[0]和[1]是不会变的 这是test文件中的
            else:
                self.tag = "训练测试各自归一化_standard_" + self.choice_descrip[0] + "_" + self.choice_descrip[
                    1]  # 这个[0]和[1]是不会变的 这是test文件中的
        self.sensor_id0 = sensor_id0
        self.sensor_ide = sensor_ide
        self.input_win = input_win
        self.predict_win = predict_win
        self.sample_size = sample_size
        self.frame_size = frame_size
        self.conditional = conditional
        self.norm_method = norm_method
        self.minmax_or_not = minmax_or_not
        self.sensor_list = sensor_list
        self.step = step

    def display(self):
        print("*" * 20 + "\n" + "此次传感器组合为:" + self.choice_descrip[0] + "\n此次窗口组合为:" + self.choice_descrip[
            1] + "\n输入窗口长度为" + str(self.input_win) + "\n输出窗口长度为" + str(self.predict_win) + "\n窗口内帧长为" + str(
            self.frame_size) +
              "\n是否conditional" + str(self.conditional) + "\n归一化方法:" + self.norm_method)


def options_parser():
    parser = argparse.ArgumentParser(description='Train a neural network to handle real-valued data.')
    # meta-option
    parser.add_argument('--settings_file', help='json file of settings, overrides everything else', type=str,
                        default='')
    return parser


def load_settings_from_file(settings):
    # settings可以是任何一个txt形式的字典文件
    settings_path = "./" + settings['settings_file'] + ".txt"
    print('Loading settings from', settings_path)
    settings_loaded = json.load(open(settings_path, 'r', encoding='utf-8'))
    # check for settings missing in file
    return settings_loaded


if __name__ == '__main__':

    parser = options_parser()
    settings = vars(parser.parse_args())

    if settings['settings_file']:
        settings = load_settings_from_file(settings)
    #
    #####################################################  对象构造  ##########################################################
    choices = []

    for i in range(len(settings["conditional_window_choices"])):
        choices.append(hyper_tuning_suite([settings["sensor_choices"][0]["choice_descrip"],
                                           settings["conditional_window_choices"][i]["choice_descrip"]],
                                          settings["sensor_choices"][0]["sensor_id0"],
                                          settings["sensor_choices"][0]["sensor_ide"],
                                          settings["conditional_window_choices"][i]["input_win"],
                                          settings["conditional_window_choices"][i]["predict_win"],
                                          settings["conditional_window_choices"][i]["sample_size"],
                                          settings["conditional_window_choices"][i]["step"],
                                          settings["conditional_window_choices"][i]["frame_size"],
                                          settings["conditional"], settings["norm_method"], settings["minmax_or_not"],
                                          settings["sensor_list"]))
    
        choices[i].display()
        

    #####################################################  存入字典  ##########################################################
    # 通过yield将结果变为列表线性存储
    nowTime = datetime.datetime.now().strftime('%m_%d_%H_%M')  # 现在
    pathc = "../seconddata/" + nowTime + "ConvLstm预处理结果" + "(conditional版本)/"
    os.makedirs(pathc)
    for i in range(len(settings["conditional_window_choices"])):
        res_dict = main(df_mat_train, df_mat_test, ground_truth, choices[i])
        time_result = pathc + "Conditional预处理结果" + res_dict['tag'] + ".npz"
        np.savez(time_result, result=res_dict)
        shape_predict = res_dict["test_predict"].shape
        shape_input = res_dict["test_input"].shape
        print("test predict_shape", shape_predict)
        print("test input_shape", shape_input)




