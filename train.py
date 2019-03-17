# 此文件用于模型的训练，评估
# 主要包括5个函数，sample_people、select_triplets、train、evaluate、main
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import time
import sys
import tensorflow as tf
import numpy as np
import argparse
from six.moves import xrange
from tensorflow.contrib.layers import l2_regularizer

import functions
import eval_functions
import input_data

from val_set import get_codes
import os
import shutil
from datetime import datetime


def main(args):

    # 用当前日期来命名模型
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    # 日志保存在c:\\users\\Administrator\logs\ 文件夹里
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)  # 没有日志文件就创建一个
    functions.write_arguments_to_file(args, os.path.join(log_dir, 'train_arguments.txt'))

    # 获取训练集和测试集的codes
    train_x, train_y, code_size  = input_data.get_codes()
    val_x, val_y = get_codes(args)

    # 用于判断是训练还是测试
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    # 输入数据的维度(之前得到的特征值)
    inputs_placeholder = tf.placeholder(tf.float32, shape=[None, code_size], name="flatten")
    # 学习率 Placeholder for the learning rate
    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

    embeddings=functions.model_add(inputs_placeholder,args.embedding_size)
    # Split embeddings into anchor, positive and negative and calculate triplet loss
    anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, args.embedding_size]), 3, 1)
    triplet_loss = functions.triplet_loss(anchor, positive, negative, args.alpha)

    # L2正则化将网络中所有权重的平方和加到损失函数。较小权重w，保持模型简单，防止过拟合
    # https://blog.csdn.net/u012162613/article/details/44261657
    l2_regularizer(0.5)
    # 获取正则项损失列表
    # tf.GraphKeys.REGULARIZATION_LOSSES是一个tensor，保存了计算正则项损失的方法，tensorflow后端就通过此方法计算出Tensor对应的值
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # 损失函数加上正则项损失即为总的损失
    total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

    tf.summary.scalar('loss', total_loss)

    global_step = tf.Variable(0, trainable=False)
    # 学习率衰减
    learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                               args.learning_rate_decay_epochs,
                                               args.learning_rate_decay_factor, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters
    # 确定优化方法并根据损失函数求梯度，在这里，每更行一次参数，global_step会加1
    optimizer = functions.train(total_loss, global_step,
                             learning_rate, args.moving_average_decay, tf.global_variables())

    # 采用用得最广泛的AdamOptimizer优化器
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss , global_step = global_step)

    # Create a saver创建一个saver用来保存或者从内存中回复一个模型参数
    saver = tf.train.Saver()

    sess = tf.Session()
    # tensorboard计算图显示
    summary_op = tf.summary.merge_all()
    # 写log文件
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # 训练测试过程=============================
    # 获取训练集
    train_set = functions.get_dataset(train_x, train_y)
    # 获取验证集
    val_set = functions.get_dataset(val_x, val_y)

    # 获取参数
    if os.path.isdir(args.model_path):
        saver.restore(sess, args.model_path)

    if not os.path.isdir(args.model_path):
        sess.run(tf.global_variables_initializer(), feed_dict={phase_train_placeholder: args.phase_train})

    # 保存准确率
    accuracy = []
    # 最高准确率
    max_acc = 0
    # 获取之前训练完的准确率
    if os.path.exists('model/max_accuracy.txt') and os.path.getsize('model/max_accuracy.txt'):
        with open('model/max_accuracy.txt') as f:
            max_acc = float(f.readline())

    # 训练多少轮
    for epoch in range(args.max_nrof_epochs):
        step = sess.run(global_step, feed_dict=None)
        # 训练
        train(args, train_set, len(set(train_y)) ,code_size ,epoch,
              inputs_placeholder, learning_rate_placeholder, phase_train_placeholder,
              embeddings, total_loss, optimizer,saver,summary_op,summary_writer,step,sess)

        # 评估，并保存准确率最高的那组参数
        if epoch % args.epoch_size == 0 :
            acc = evaluate(args, sess, summary_op, summary_writer, step,
                 inputs_placeholder, learning_rate_placeholder, phase_train_placeholder,
                 embeddings, code_size, train_set , val_set)

            # 将accuracy存入文件中
            accuracy.append((epoch, acc))

            with open('model/accuracy.txt', 'w') as f:
                writer = csv.writer(f, delimiter='\n')
                writer.writerow(accuracy)

            # 初始化参数
            if  epoch == args.epoch_size and max_acc==0:
                max_acc = acc
                saver.save(sess, args.model_path)

            if acc > max_acc:
                max_acc = acc
                if os.path.isdir(args.model_path):
                    # 删除旧的参数
                    shutil.rmtree(args.model_path)
                # 保存新的参数
                saver.save(sess, args.model_path)

            # 将max_acc存入文件中
            with open('model/max_accuracy.txt', 'w') as f:
                f.write(str(max_acc))


def train(args, dataset, nrof_classes, code_size, epoch ,
          inputs_placeholder,learning_rate_placeholder,phase_train_placeholder,
          embeddings,total_loss, optimizer,saver,summary_op,summary_writer ,step,sess):

    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = functions.get_learning_rate_from_file(args.learning_rate_train_tripletloss, epoch)
    batch_number = 0

    # Sample people randomly from the dataset
    sample_codes, num_per_class = sample_people(dataset, nrof_classes, code_size, args.people_per_batch,
                                                args.images_per_person)

    ##计算embedding
    # print('Running forward pass on sampled images: ')
    start_time = time.time()
    # 每个大batch中图片数量
    nrof_images_per_batch = sum(num_per_class)

    # nrof_batches为计算embedding时batch的个数，batch_size为每个batch中图片数量，默认是900/90=10
    nrof_batches = int(np.ceil(nrof_images_per_batch / args.batch_size))  # 向上取整
    # print('计算embedding时的batch个数：', nrof_batches)
    emb_array = None
    start_id_per_batch = 0

    # 批处理求特征，默认为10个批
    for i in range(nrof_batches):
        batch_size = min(nrof_images_per_batch - i * args.batch_size, args.batch_size)

        emb = sess.run(embeddings, feed_dict={
            inputs_placeholder: sample_codes[start_id_per_batch:start_id_per_batch + batch_size],
            learning_rate_placeholder: lr,
            phase_train_placeholder: args.phase_train})

        if emb_array is None:
            emb_array = np.reshape(emb, (-1, args.embedding_size))
        else:
            emb_array = np.concatenate((emb_array, np.reshape(emb, (-1, args.embedding_size))))

        start_id_per_batch += batch_size

    # print('计算embedding所用时间%.3f' % (time.time() - start_time))

    # Select triplets based on the embeddings
    # print('Selecting suitable triplets for training')
    print('开始训练:epoch[%d]=========================================================' % epoch )
    start_time = time.time()
    triplets, nrof_triplets = select_triplets(emb_array, num_per_class,
                                              sample_codes, args.people_per_batch, args.alpha)
    selection_time = time.time() - start_time
    print('triplet个数 = %d: time=%.3f seconds' %
          (nrof_triplets, selection_time))

    # 在三元组上正式开始训练
    triplets_array = np.reshape(np.array(triplets), (-1, code_size))

    nrof_batches = int(np.ceil(nrof_triplets * 3 / args.train_batch_size))
    print('计算triplet损失函数进行优化时batch的个数：', nrof_batches)

    train_time = 0
    start_id_per_batch = 0

    # 根据求出的特征计算triplet损失函数并进行优化
    for i in range(nrof_batches):
        start_time = time.time()
        batch_size = min(nrof_triplets * 3 - i * args.train_batch_size, args.train_batch_size)
        feed_dict = {inputs_placeholder: triplets_array[start_id_per_batch:start_id_per_batch + batch_size],
                     learning_rate_placeholder: lr,
                     phase_train_placeholder: args.phase_train}

        summary__op, loss, _ = sess.run([summary_op, total_loss, optimizer], feed_dict=feed_dict)

        duration = time.time() - start_time
        print('Epoch:[%d],batch_num:[%d]\tTime %.3f\tLoss %2.3f' %
              (epoch, batch_number + 1, duration, loss))
        if not batch_number:
            summary_writer.add_summary(summary__op, epoch)

        start_id_per_batch += batch_size
        batch_number += 1
        train_time += duration



# # 从数据集中进行抽样图片，输入参数：
# # 1、训练数据集  2、每一个batch抽多少只羊people_per_batch
# 3、每只羊抽样多少张照片images_per_person
# people_per_batch：30  images_per_person：30
# 选择一个用于训练的三元组
def select_triplets(embeddings, nrof_images_per_people, sample_codes, people_per_batch, alpha):
    #每只羊编码起始位置
    emb_start_idx = 0
    triplets = []
    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.
    """ VGGFace：选择好的三元组是至关重要的，应该选择对于深度学习网络具有挑战的例子。
    """
    # 遍历每一只羊
    for i in xrange(people_per_batch):

        # 这只羊有几张图片
        nrof_images = int(nrof_images_per_people[i])

        # 遍历第i只羊的所有图片
        for j in xrange(1, nrof_images):
            # 第j张图的embedding在emb_arr 中的位置
            a_idx = emb_start_idx + j - 1
            # 第j张图跟其他所有图片的欧氏距离
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            # 同一只羊的图片不作为negative，所以将距离设为无穷大
            neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
            # 遍历每一对可能的(anchor,postive)图片，记为(a,p)吧
            for pair in xrange(j, nrof_images):  # For every possible positive pair.
                # 第p张图片在emb_arr中的位置
                p_idx = emb_start_idx + pair
                # (a,p)之前的欧式距离
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                # all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                # 挑选出符合条件的negative，并返回行的索引
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]  # VGG Face selecction
                # 所有可能的negative
                nrof_random_negs = all_neg.shape[0]
                # 如果有满足条件的negative
                if nrof_random_negs > 0:
                    # 从中随机选取一个作为n
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    # 选到(a,p,n)作为三元组
                    triplets.append((sample_codes[a_idx], sample_codes[p_idx], sample_codes[n_idx]))
                    # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))


        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets,len(triplets)


# 从数据集中进行抽样图片，输入参数：
# 1、训练数据集 2.# 数据集中一共有多少只羊的图像  3、每一个batch抽样多少 4、每只羊抽样多少张
# 默认：选择30张羊脸图片作为正样本，随机筛选其他羊脸图片作为负样本
def sample_people(dataset, nrof_classes ,code_size,people_per_batch, images_per_person):
    # 总共应该抽取多少张    people_per_batch：30  images_per_person：30
    nrof_images = people_per_batch * images_per_person

    # 每只羊的索引
    class_indices = np.arange(nrof_classes)
    # 随机打乱一下
    np.random.shuffle(class_indices)

    # 保存抽样出来的图片的特征
    sample_codes = None
    # 存放每只羊抽样数
    num_per_class = []
    # 抽样的样本是属于哪一只羊的，作为label
    sampled_labels = []

    # Sample images from these classes until we have enough
    # 不断抽样直到达到指定数量
    for i in range(people_per_batch) :
        # 从第i只羊开始抽样
        class_index = class_indices[i]
        # 第i只羊有多少张图片
        nrof_images_in_class = len(dataset[class_index])
        # 这些图片的索引
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        # 从第i只羊中抽样的图片数量
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - sum(num_per_class))
        idx = image_indices[0:nrof_images_from_class]
        #抽样出来的羊的特征值
        sample_codes_from_class = [dataset[class_index].codes[j] for j in idx]

        # # 抽样出来的羊的label
        # sample_labels_from_class = [dataset[class_index].labels]
        # sampled_labels.append(sample_labels_from_class)

        if sample_codes is None:
            sample_codes = np.reshape(sample_codes_from_class, (-1, code_size))
        else:
            sample_codes = np.concatenate((sample_codes, np.reshape(sample_codes_from_class, (-1, code_size))))

        # 第i只羊抽样了多少张
        num_per_class.append(nrof_images_from_class)

    return sample_codes, num_per_class


# def evaluate(args, sess, summary_op,summary_writer,step,
#              inputs_placeholder,learning_rate_placeholder,phase_train_placeholder,
#              embeddings, code_size, train_set, val_set):
#     # start_time = time.time()
#     print('开始评估：')
#
#
#     # 获取pairs_list和issame_list
#     pairs_list_train,issame_list_train = functions.get_pairs(train_set, len(train_set)
#     pairs_list_val ,issame_list_val = functions.get_pairs(val_set, len(val_set))
#
#
#     emb_array_train = calculate_embeddings(args, sess, pairs_list_train ,issame_list_train , code_size,embeddings,
#                         inputs_placeholder ,learning_rate_placeholder,phase_train_placeholder)
#
#     emb_array_val =  calculate_embeddings(args, sess, pairs_list_val ,issame_list_val , code_size,embeddings,
#                         inputs_placeholder ,learning_rate_placeholder,phase_train_placeholder)
#
#     print("embedings计算完毕")
#     # 根据得到的embedings开始评估
#     acc_train, tpr, fpr, acc_val, val, far= functions.evaluate(emb_array_train, issame_list_train,emb_array_val,issame_list_val)
#
#     print("acc_train:"+str(np.mean(acc_train)))
#     print("acc_val:"+str(np.mean(acc_val)))
#     # lfw_time = time.time() - start_time
#
#
#     # tf.summary.scalar('accuracy', acc_train)
#     # tf.summary.scalar('accuracy', acc_val)
#
#     # summary = tf.Summary()
#     # summary.value.add(tag='accuracy', simple_value=np.mean(acc_train))
#     # summary.value.add(tag='accuracy', simple_value=np.mean(acc_val))
#
#     # summary = sess.run(summary_op, feed_dict=None)
#     # summary_writer.add_summary(summary, step)

def evaluate(args, sess, summary_op,summary_writer,step,
             inputs_placeholder,learning_rate_placeholder,phase_train_placeholder,
             embeddings, code_size, train_set, val_set):

    print('开始评估：')

    # 获取pairs_list和issame_list
    pairs_list_val ,issame_list_val = functions.get_pairs(val_set, len(val_set))

    emb_array_val =  calculate_embeddings(args, sess, pairs_list_val ,issame_list_val , code_size,embeddings,
                        inputs_placeholder ,learning_rate_placeholder,phase_train_placeholder)

    print("embedings计算完毕")
    # 根据得到的embedings开始评估
    tpr, fpr, acc_val, val, far, best_threshold= eval_functions.evaluate(emb_array_val,issame_list_val)

    print("acc_val:"+str(np.mean(acc_val)))

    return acc_val


def calculate_embeddings(args, sess, pairs_list, actual_issame, code_size, embeddings,
                         inputs_placeholder, learning_rate_placeholder, phase_train_placeholder):
    nrof_images = len(actual_issame) * 2
    # 计算embeddings
    pairs_list_array = np.reshape(np.array(pairs_list), (-1, code_size))
    emb_array = None
    nrof_batches = int(np.ceil(nrof_images / args.batch_size))
    print(nrof_batches)

    start_id_per_batch = 0
    for i in xrange(nrof_batches):
        # print(i)
        batch_size = min(nrof_images - i * args.batch_size, args.batch_size)
        feed_dict = {inputs_placeholder: pairs_list_array[start_id_per_batch:start_id_per_batch + batch_size],
                     learning_rate_placeholder: 0,
                     phase_train_placeholder: False}
        emb = sess.run(embeddings, feed_dict=feed_dict)
        if emb_array is None:
            emb_array = np.reshape(emb, (-1, args.embedding_size))
        else:
            emb_array = np.concatenate((emb_array, np.reshape(emb, (-1, args.embedding_size))))

        start_id_per_batch += batch_size

    return emb_array


# 所有的训练参数都在这里定义，更改对应的参数即可
def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='~/logs/detailed_graph')

    parser.add_argument('--model_path', type=str,
                        help='Directory where to write trained models and checkpoints.', default='model/checkpoints/')

    parser.add_argument('--code_dir', type=str,
                        help='Path to the data directory containing  codes.',
                        default='data/')

    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=40)
    parser.add_argument('--epoch_size', type=int,
                        help='evaluate once after epoch_size epoch.', default=10)

    parser.add_argument('--train_batch_size', type=int,
                        help='Number of batches per epoch and the value is a multiple of three', default=270)

    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)

    parser.add_argument('--people_per_batch', type=int,
                        help='Number of people per batch.', default=15)
    parser.add_argument('--images_per_person', type=int,
                        help='Number of images per person.', default=20)

    parser.add_argument('--alpha', type=float,
                        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)

    parser.add_argument('--phase_train', type=bool,
                        help='判断是训练还是测试.', default=True)

    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                             'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.01)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=0.96)
    parser.add_argument('--learning_rate_train_tripletloss', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='data/learning_rate_schedule.txt')
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)

    parser.add_argument('--codes_path', type=str,
                        help='Path to the data directory containing val_codes.',
                        default='data/val/codes.npy')
    parser.add_argument('--labels_path', type=str,
                        help='Path to the data directory containing val_labels.',
                        default='data/val/labels.npy')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
