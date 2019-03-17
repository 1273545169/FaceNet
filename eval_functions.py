# 评估过程需要用到的函数

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import interpolate

def calculate_dist(thresholds,embeddings,):

    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])

    nrof_thresholds = len(thresholds)

    # dist为path0和path1集合的欧式距离，每一行的结果与issame每行一一对应
    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    return dist , nrof_thresholds

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    # tp统计predict_issame和actual_issame均为True的个数，即true posotive.A-P中预测对的
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    # false positive，将A-N预测为A-P
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    # true negtive，A-N中预测对的
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    # false negtive，将A-P预测为A-N
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # ROC曲线,横轴为fpr，纵轴为tpr，曲线越接近左上角效果越好
    # 当测试集中的正负样本的分布变化的时候，ROC曲线能够保持不变
    # true positive rate  tp/p   A-P中预测对的
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    # false positive rate  fp/n  将A-N预测为A-P
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    #
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def calculate_roc_acc(thresholds, embeddings, actual_issame):
    dist, nrof_thresholds = calculate_dist(thresholds, embeddings)

    acc_val = np.zeros((nrof_thresholds))
    tprs = np.zeros((nrof_thresholds))
    fprs = np.zeros((nrof_thresholds))

    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], acc_val[threshold_idx] = calculate_accuracy(threshold, dist,
                                                                                              actual_issame)
    best_threshold = thresholds[np.argmax(acc_val)]
    accuracy = np.max(acc_val)

    return tprs, fprs, accuracy



def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    # A-P对中预测对的
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    # 将A-N预测为A-P
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    # A-P对总数
    n_same = np.sum(actual_issame)
    # A-N对总数
    n_diff = np.sum(np.logical_not(actual_issame))

    val = float(tp) / float(n_same)
    far = float(fp) / float(n_diff)

    return val, far

def calculate_val(thresholds, embeddings, actual_issame, far_target):
    dist, nrof_thresholds = calculate_dist(thresholds, embeddings)

    # Find the threshold that gives FAR = far_target
    far_val = np.zeros(nrof_thresholds)
    for threshold_idx, threshold in enumerate(thresholds):
        _, far_val[threshold_idx] = calculate_val_far(threshold, dist, actual_issame)
    if np.max(far_val) >= far_target:
        # interp1d一维线性插值，它可通过函数在有限个点处的取值状况，估算出函数在其他点处的近似值
        f = interpolate.interp1d(far_val, thresholds, kind='slinear')
        threshold = f(far_target)
    else:
        threshold = 0.0

    val, far = calculate_val_far(threshold, dist, actual_issame)

    return val, far , threshold


def evaluate(embeddings_val, actual_issame_val):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    # embeddings1和embeddings2 分别对应 path0和path1

    tpr, fpr, acc_val = calculate_roc_acc(thresholds,  embeddings_val,
                                      np.asarray(actual_issame_val))

    thresholds = np.arange(0, 4, 0.001)

    val, far, threshold_far = calculate_val(thresholds,embeddings_val,
                             np.asarray(actual_issame_val),1e-3)

    return tpr, fpr, acc_val, val, far ,threshold_far
