import numpy as np
import pandas as pd
import datetime

import os

from sklearn import preprocessing

import metric_utils_merge_interval



import heapq
import sys


nowtime = datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')
from sklearn import metrics


# f = open('./a'+nowtime+'.log', 'a')
# sys.stdout = f
# sys.stderr = f

##########################################################################  一些基础函数  ##########################################################################
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


def bianchenglabel(some_interval):
    label = np.zeros((cum.ground_truth.shape[0],), dtype=np.int)

    for i in range(len(some_interval)):
        for j in range(some_interval[i][1] - some_interval[i][0] + 1):
            label[some_interval[i][0] + j] = 1
    return label


def sort_model_by_time(model_path):
    models = os.listdir(model_path)
    if not models:
        return
    else:
        files = sorted(models, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
        return files


##########################################################################  类定义  ##########################################################################

class analysis:
    def __init__(self, test_pure_norm_true, test_pure_norm_NN, test_predict_true, test_predict_NN, ground_truth,
                 start_t,
                 end_t, input_win, weight_or_not, weight_type, norm_or_not, specific=None):

        self.test_pure_norm_true = test_pure_norm_true
        self.test_pure_norm_NN = test_pure_norm_NN
        self.test_predict_true = test_predict_true
        self.test_predict_NN = test_predict_NN

        self.power = 6  # power改变点
        self.ground_truth = ground_truth

        if specific != None:
            self.specific = specific
        self.start_strptime = datetime.datetime.strptime(start_t, '%Y-%m-%d %I:%M:%S %p')
        self.end_strptime = datetime.datetime.strptime(end_t, '%Y-%m-%d %I:%M:%S %p')
        self.basetime = datetime.datetime.strptime('2017-10-9 06:02:00 PM', '%Y-%m-%d %I:%M:%S %p')
        self.input_win = input_win

        self.start = int((self.start_strptime - self.basetime).total_seconds() - self.input_win)
        end_index = int((self.end_strptime - self.basetime).total_seconds() - self.input_win)

        self.weight_or_not = weight_or_not
        self.weight_type = weight_type
        self.norm_or_not = norm_or_not

        if end_index > ground_truth.shape[0]:
            self.end = ground_truth.shape[0]
        else:
            self.end = end_index + 1
        if weight_or_not == True:
            if weight_type == 1:
                self.wgt_error_out_ewma, self.wgt_error_thre_ewma = metric_utils_merge_interval.error_weighting(
                    self.test_pure_norm_true, self.test_pure_norm_NN, self.test_predict_true, self.test_predict_NN,
                    self.power,
                    self.input_win, norm_or_not)
            else:
                self.wgt_error_out_ewma, self.wgt_error_thre_ewma = metric_utils_merge_interval.error_weighting_second(
                    self.test_pure_norm_true, self.test_pure_norm_NN, self.test_predict_true, self.test_predict_NN,
                    self.power,
                    self.input_win, norm_or_not)
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<self.wgt_error_thre_ewma>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>:")
            print(self.wgt_error_thre_ewma)
        else:
            self.nwgt_error_out_ewma, self.nwgt_error_thre_ewma = metric_utils_merge_interval.error_no_weighting(
                self.test_pure_norm_true, self.test_pure_norm_NN, self.test_predict_true, self.test_predict_NN,
                self.power,
                self.input_win, norm_or_not)
            self.nwgt_threshold = np.percentile(self.nwgt_error_thre_ewma, 99)
        # self.wgt_threshold =0.0018016102467660112
        self.offset = self.start + input_win
        self.start_time = datetime.timedelta(seconds=self.offset) + self.basetime
        print(self.start_time)
        self.datetime1 = pd.date_range(self.start_time, periods=self.end - self.start,
                                       freq='S')  # test_pure_norm_true.shape[0]
    

    def threshold_grid_search_no_plotting(self, weight_or_not, number, lower, upper, consecutive_or_not, fine_or_not):

            if consecutive_or_not == True:
                if weight_or_not == True:
                    percents, thre_x, precision, recall, f1 = metric_utils_merge_interval.threshold_finetuning_for_consecutive1(
                        self.wgt_error_out_ewma, self.wgt_error_thre_ewma, number, lower, upper, self.ground_truth,
                        self.input_win, fine_or_not)
                    thre_x=np.array(thre_x)
                    f1 = np.array(f1)
                    precision = np.array(precision)
                    recall = np.array(recall)
                    best = np.argwhere(f1 == max(f1))[0][0]
                    result_dict = {"模型": self.specific + "(加权的event-based的统计)", '阈值':thre_x[best],"最好的F1值": max(f1),
                                   "此时的Precision": precision[best], "此时的Recall": recall[best]}
                    return result_dict

                else:
                    percents, thre_x, precision, recall, f1 = metric_utils_merge_interval.threshold_finetuning_for_consecutive1(
                        self.nwgt_error_out_ewma, self.nwgt_error_thre_ewma, number, lower, upper, self.ground_truth,
                        self.input_win, fine_or_not)
                    thre_x = np.array(thre_x)
                    f1 = np.array(f1)
                    precision = np.array(precision)
                    recall = np.array(recall)
                    best = np.argwhere(f1 == max(f1))[0][0]
                    result_dict = {"模型": self.specific + "(不加权的event-based的统计)",'阈值':thre_x[best], "最好的F1值": max(f1),
                                   "此时的Precision": precision[best], "此时的Recall": recall[best]}
                    return result_dict
            else:  # OUTLIER
                
                windows = list(range(150,350,50))
                #windows =[300]
                print("window range：", windows)
                model_dict_for_window = {"model": [], "window":[],'thres': [], "F1": [], "Precision": [], "Recall": [],
                                          "ratio": []}
                start_a_model_time = datetime.datetime.now()
                for window in windows:
                    print("################## window = {0} ##############".format(window))
                    start_a_window_model_time = datetime.datetime.now()
                    if weight_or_not == True:  # weighted
                        
                        percents, thre_x, precision, recall, f1 = metric_utils_merge_interval.threshold_finetuning_for_outlier1(
                            self.wgt_error_out_ewma, self.wgt_error_thre_ewma, number, lower, upper, self.ground_truth,
                            fine_or_not=fine_or_not, window = window)
                        thre_x = np.array(thre_x)
                        f1 = np.array(f1)
                        precision = np.array(precision)
                        recall = np.array(recall)
                        best = np.argwhere(f1 == max(f1))[0][0]
                        
                        
                        ############################################################################################################
                        my_dict = {"model": self.specific + "(weighted single point)",'thres':thre_x[best], "F1": max(f1), "Precision": precision[best],
                                       "Recall": recall[best]}
                       
                        


                    else:  # unweighted
                        percents, thre_x, precision, recall, f1 = metric_utils_merge_interval.threshold_finetuning_for_outlier1(
                            self.nwgt_error_out_ewma, self.nwgt_error_thre_ewma, number, lower, upper,
                            self.ground_truth,window = window)
                        thre_x = np.array(thre_x)
                        f1 = np.array(f1)
                        precision = np.array(precision)
                        recall = np.array(recall)
                        best = np.argwhere(f1 == max(f1))[0][0]
                        my_dict = {"model": self.specific + "(weighted single point)", 'thres': thre_x[best],
                                   "F1": max(f1), "Precision": precision[best],
                                   "Recall": recall[best]}

                    # save results
                    model_dict_for_window["model"].append(my_dict["model"])
                    model_dict_for_window["thres"].append(my_dict["thres"])
                    model_dict_for_window["F1"].append(my_dict["F1"])
                    model_dict_for_window["Precision"].append(my_dict["Precision"])
                    model_dict_for_window["Recall"].append(my_dict["Recall"])
                    model_dict_for_window["window"].append(window)
                    model_dict_for_window["ratio"].append(my_dict["thres"] / max(self.wgt_error_thre_ewma))
                   
               
                # get the best f1 under the range of window
                f1max = max(model_dict_for_window["F1"])
                best = np.argwhere(model_dict_for_window["F1"] == f1max)[0][0]
                # print all results
                for _ in range(len(windows)):
                    my_dict = {"model": model_dict_for_window["model"][_], "window":model_dict_for_window["window"][_],"thres": model_dict_for_window["thres"][_], "F1": model_dict_for_window["F1"][_],
                           "Precision": model_dict_for_window["Precision"][_], "Recall": model_dict_for_window["Recall"][_],
                           "ratio": model_dict_for_window["ratio"][_]}
                    print(my_dict)
                result_dict = {"model": model_dict_for_window["model"][best], "window":model_dict_for_window["window"][best],"thres": model_dict_for_window["thres"][best], "F1": model_dict_for_window["F1"][best],
                           "Precision": model_dict_for_window["Precision"][best], "Recall": model_dict_for_window["Recall"][best],
                           "ratio": model_dict_for_window["ratio"][best]}
                return result_dict


    
    

    

    

