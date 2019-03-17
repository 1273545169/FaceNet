
# 此文件包含训练测试过程需要用到的各个函数
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import skimage
import skimage.io
import skimage.transform
import tensorflow as tf
import numpy as np
from scipy import interpolate
from six import iteritems


# 模型改变部分
def model_add(input, embedding_size):

        # 构造计算图
        # 加入一个4096维的全连接的层
        fc6 = tf.layers.dense(input, 4096, tf.nn.relu, name='conv6')
        # 加入dropout层
        dropout = tf.layers.dropout(fc6, 0.5, name="dropout")
        # 加入一个128维的全连接的层
        prelogits = tf.layers.dense(dropout, embedding_size, tf.nn.relu, name="prelogits")

        # L2范化函数
        # embeddings = tf.nn.l2_normalize(输入向量, L2范化的维数（取0（列L2范化）或1（行L2范化））, 泛化的最小值边界, name='embeddings')
        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        return embeddings


#加载改变图片大小
# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img

# 三元组损失计算
def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):

        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)  # tf.square:平方。tf.subtract::减法
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss

# 得到learning_rate
def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                    return learning_rate

# 梯度算法
def train(total_loss, global_step, learning_rate, moving_average_decay, update_gradient_vars,):

        opt = tf.train.AdamOptimizer(learning_rate)
        grads = opt.compute_gradients(total_loss, update_gradient_vars)

        # Apply gradients.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

# 数据整理
class ImageClass():
    "Stores the codes，label to images for a given class"

    def __init__(self, codes, labels):
        self.codes = codes
        self.labels = labels

    def __str__(self):
        return self.labels + ', ' + str(len(self.codes)) + ' images'

    def __len__(self):
        return len(self.codes)

def get_dataset(codes, labels, has_class_directories=True):
    dataset = []

    code_set=None
    label_list = set(labels)
    nrof_images = codes.shape[0]

    dimension=codes.shape[1]

    for i in label_list:
        for idx in range(nrof_images):
            if i == labels[idx]:

                if code_set is None:
                    #这部分要改变，因为得出的codes[idx].shape=（25088，），要改为（1,25088）
                    code_set = np.reshape(codes[idx],(1,dimension))
                else:
                    code_set = np.concatenate((code_set, np.reshape(codes[idx],(1,dimension))))

        dataset.append(ImageClass(code_set, i))
        code_set = None

    return dataset

# 得到A-P对和A-N对
def get_pairs(dataset , nrof_classes ):
    # pairs中存储的是(path0,path1)，即A-P对和A-N对的code值
    pairs_list = []
    # True表示两张图片是一只羊即为A-P对，False表示两张图片不是一只羊即为A-N对
    actual_issame = []

    # 挑选A-P对
    for i in range(nrof_classes):
        for j in range(len(dataset[i])):
            path0 = dataset[i].codes[j]
            for k in range(j+1, len(dataset[i])):
                path1 = dataset[i].codes[k]
                same = True
                pairs_list.append((path0,path1))
                actual_issame.append(same)

    # A-P对的总数
    nrof_pos=np.sum(actual_issame)

    # 挑选A-N对
    for i in range(nrof_classes):

        if i == nrof_classes - 1:
            break

        for j in range(len(dataset[i])):
            path0 = dataset[i].codes[j]
            nrof = len(dataset[i])-j-1
            for k in range(nrof):
                if len(dataset[i+1]) == k :
                    break

                # if i == nrof_classes-1:
                #     path1 = dataset[0].codes[k]
                # else:
                #     path1 = dataset[i + 1].codes[k]

                path1 = dataset[i + 1].codes[k]
                same = False
                pairs_list.append((path0, path1))
                actual_issame.append(same)
    # A-N对的总数
    nrof_neg = np.sum(np.logical_not(actual_issame))

    return pairs_list , actual_issame

def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))



# 模型评估函数
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

def calculate_roc_train(thresholds, embeddings, actual_issame):

    dist, nrof_thresholds = calculate_dist(thresholds,embeddings)

    # Find the best threshold
    acc_train = np.zeros((nrof_thresholds))
    for threshold_idx, threshold in enumerate(thresholds):
        _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist, actual_issame)
    best_threshold_index = np.argmax(acc_train)
    accuracy = np.max(acc_train)

    return best_threshold_index, accuracy


def calculate_roc(thresholds, best_threshold_index, embeddings, actual_issame):

    dist, nrof_thresholds = calculate_dist(thresholds, embeddings)

    tprs = np.zeros((nrof_thresholds))
    fprs = np.zeros((nrof_thresholds))

    for threshold_idx, threshold in enumerate(thresholds):
        tprs[threshold_idx], fprs[threshold_idx], _ = calculate_accuracy(threshold,dist,
                                                                         actual_issame)
    _, _, accuracy = calculate_accuracy(thresholds[best_threshold_index],dist,
                                        actual_issame)

    return tprs, fprs, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    # tp统计predict_issame和actual_issame均为True的个数，即true posotive
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # ROC曲线,横轴为fpr，纵轴为tpr，曲线越接近左上角效果越好
    # 当测试集中的正负样本的分布变化的时候，ROC曲线能够保持不变
    # true positive rate  tp/p
    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    # false positive rate  fp/n
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    #
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc


def calculate_val_threshold(thresholds, embeddings, actual_issame, far_target):

    dist, nrof_thresholds = calculate_dist(thresholds, embeddings)

    # Find the threshold that gives FAR = far_target
    far_train = np.zeros(nrof_thresholds)
    for threshold_idx, threshold in enumerate(thresholds):
        _, far_train[threshold_idx] = calculate_val_far(threshold, dist, actual_issame)
    if np.max(far_train) >= far_target:
        # interp1d一维线性插值，它可通过函数在有限个点处的取值状况，估算出函数在其他点处的近似值
        f = interpolate.interp1d(far_train, thresholds, kind='slinear')
        threshold = f(far_target)
    else:
        threshold = 0.0

    return threshold

def calculate_val(thresholds,threshold, embeddings, actual_issame):

    dist, nrof_thresholds = calculate_dist(thresholds, embeddings)

    val, far = calculate_val_far(threshold, dist, actual_issame)

    return val , far

def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far

def evaluate(embeddings_train, actual_issame_train,embeddings_val,actual_issame_val):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    # embeddings1和embeddings2 分别对应 path0和path1

    best_threshold_index, acc_train = calculate_roc_train(thresholds, embeddings_train,
                                                          np.asarray(actual_issame_train))

    tpr,fpr,acc_val = calculate_roc(thresholds, best_threshold_index,embeddings_val,
                                    np.asarray(actual_issame_val))

    thresholds = np.arange(0, 4, 0.001)
    threshold = calculate_val_threshold(thresholds, embeddings_train,
                                              np.asarray(actual_issame_train), 1e-3)
    val,far=calculate_val(thresholds,threshold,embeddings_val,
                         np.asarray(actual_issame_val))

    return  acc_train, tpr, fpr, acc_val , val , far




