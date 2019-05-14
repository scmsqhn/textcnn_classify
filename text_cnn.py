#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import gensim
import pdb

DEBUG = False#True

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, sequence_length, num_classes, vocab_size,embedding_size, filter_sizes, num_filters, l2_reg_lambda):
      # Placeholders for input, output and dropout
      self.preloss = 10.0
      self.count = 0
      self.sumacc =0.0
      self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
      self.input_y = tf.placeholder(tf.int32, [None], name="input_y")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
      # Keeping track of l2 regularization loss (optional)
      l2_loss = tf.constant(0.0)
      w2vmodel = gensim.models.word2vec.Word2Vec.load("w2vmodel")
      poolsz = 9
      """
      #self.embedded_chars = self.input_x
      #embedded_chars = w2vmodel[j]
      #self.embedded_chars = np.array(embedded_chars).reshape(32, sequence_length, embedding_size)
      """
      with tf.device('/gpu:0'),tf.name_scope("embedding"):
          self.W= tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
          self.embedded_chars= tf.nn.embedding_lookup(self.W, tf.add(self.input_x,10))
          if DEBUG:
              pdb.set_trace()
          self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
          print(self.embedded_chars_expanded)
          # Create a convolution + maxpool layer for each filter size
      pooled_outputs_conv = []
      pooled_outputs_pool = []
      """
      开始　第一层卷积池化
      """
      for i, filter_size in enumerate(filter_sizes):
          with tf.name_scope("conv-maxpool-%s" % filter_size):
              filter_shape = [filter_size, embedding_size, 1, num_filters]
              W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
              conv = tf.nn.conv2d(
                  self.embedded_chars_expanded,
                  tf.cast(W, tf.float32),
                  strides= [1,1,1,1],
                  padding="VALID",
                  name="conv")
              b = tf.Variable(tf.constant(0.1, shape=[num_filters]))
              #b = tf.Variable(tf.random_uniform([num_filters], 0.0, 1.0))
              conv = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
              pooled = tf.nn.max_pool(
                  conv,
                  ksize=[1,sequence_length-filter_size+1,1,1],
                  strides=[1,1,1,1],
                  padding='VALID',
                  name="pool")
              pooled_outputs_conv.append(conv)
              pooled_outputs_pool.append(pooled)
      num_filters_total = num_filters*len(filter_sizes)
      self.h_pool = tf.concat(pooled_outputs_pool,3)
      self.h_pool_final = tf.reshape(self.h_pool, [-1,num_filters_total])

      with tf.name_scope("dropout"):
          self.h_pool_final = tf.nn.dropout(self.h_pool_final, self.dropout_keep_prob)

          shape3 = int(self.h_pool_final.shape[1])
          self.h_pool_final = tf.reshape(self.h_pool_final,(-1, shape3))
      with tf.name_scope("output"):
          Wout= tf.get_variable(
              "Wout",
              shape=[shape3, num_classes],
              initializer=tf.contrib.layers.xavier_initializer())
          bout= tf.Variable(tf.constant(0.0, shape=[num_classes]), name="bout")
          print(self.h_pool_final)
          self.scores = tf.nn.xw_plus_b(self.h_pool_final, Wout, bout, name="scores")
          print(self.scores)
          l2_loss += tf.nn.l2_loss(Wout)
          l2_loss += tf.nn.l2_loss(bout)
          self.predictions = tf.argmax(self.scores, 1, name='predictions')
          # Calculate mean cross-entropy loss
      with tf.name_scope("loss"):
          """
          将其他滤出,修改loss值的分布
          因为 其他类 样本过多,使用tf.where过滤掉其他类样本的loss共享度,也就是让模型只学习普通分类如盗窃抢劫诈骗等的分类,但是这样会导致模型预测的全部是普通分类
          所以,引入fscore,将对于其他类的辨析也引入训练
          秦海宁 2018.07.20
          """
          class_filter = tf.cast(tf.where(tf.equal(self.input_y, tf.constant([4]*64)), tf.constant([0]*64), tf.constant([1]*64)), tf.float32)
          #cnt_class = tf.reduce_sum(class_filter)
          self.scores = tf.clip_by_value(self.scores,-3.0,3.0)
          #class_filter = tf.cast(tf.where(tf.equal(self.input_y, tf.constant([4]*64)), tf.constant([1]*64), tf.constant([100]*64)), tf.float32)
          self.softmax = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.scores))
          self.softmax = tf.multiply(class_filter, self.softmax)
          #self.softmax = tf.multiply(softmax, weight_loss)
          #cls_flt = tf.reduce_sum(class_filter)
          #loss_2 = tf.cast(tf.div(loss_2,cls_flt),tf.float32)
          self.fsc4,self.fsc1,self.fsc2,self.fsc3,self.fsc0 = 0.0,0.0,0.0,0.0,0.0

          _,_,_,_,_,_,self.fsc0  = self.fscore(self.input_y, self.scores, 0)
          _,_,_,_,_,_,self.fsc1  = self.fscore(self.input_y, self.scores, 1)
          _,_,_,_,_,_,self.fsc2  = self.fscore(self.input_y, self.scores, 2)
          _,_,_,_,_,_,self.fsc3  = self.fscore(self.input_y, self.scores, 3)
          _,_,_,_,self.precision4,self.recall4,self.fsc4 = self.fscore(self.input_y, self.scores, 4)
          lst = [self.fsc0,self.fsc1,self.fsc2,self.fsc3,self.fsc4]
          #lst = [self.fsc0,self.fsc1,self.fsc2,self.fsc3]
          self.sum_fsc =tf.reduce_sum(lst)
          #mean_fsc = tf.reduce_mean(self.sum_square(lst))
          #loss_2 = self.softmax
          #loss_2 = tf.add(tf.cast(self.softmax,tf.float32),tf.multiply(tf.cast(self.fsc1,tf.float32),1.0))
          loss_2 = tf.add(tf.cast(self.softmax,tf.float32),tf.multiply(tf.cast(self.fsc4,tf.float32),0.5))
          self.loss_2 = loss_2 + (l2_reg_lambda * l2_loss)
          self.loss = self.loss_2
          #self.loss = tf.sqrt(tf.add(self.loss_2,1.0))
          # Accuracy
      with tf.name_scope("accuracy"):
          self.correct_predictions = tf.equal(tf.cast(self.predictions, tf.int32), tf.cast(self.input_y, tf.int32))
          self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

      self.losses = self.loss

    def sum_square(self, inlst):
        res = 0
        for item in inlst:
            res+=tf.square(item)
        return res

    def add_lst(self, lst):
        item0 = lst[0]
        for item in lst[1:]:
            item0 = tf.add(item0, item)
        return item0

    def filter_class(self, y, n, h, l):
        """
        将其他类至为0, 普通类至为1
        """
        _y = tf.cast(tf.where(tf.equal(tf.cast(y,tf.int32),tf.constant([n]*64)),tf.constant([h]*64),tf.constant([l]*64)),tf.int32)
        return tf.multiply(y, _y)

    def fscore(self, labels, logits, n): # y y_
        y = tf.cast(tf.reshape(labels, [-1]), tf.int32)
        y_ = tf.cast(tf.argmax(logits, 1), tf.int32)
        """
        对label进行过滤，使用
        """
        y = self.filter_class(y,n,1,0)
        y_ = self.filter_class(y_,n,1,0)
        """
        label, 非本类均为零
        """
        #y_filter = tf.where(tf.equal(y,[0]*64),tf.constant([0]*64),tf.constant([1]*64))
        #y_ = tf.multiply(y_, y_filter)


        T = tf.cast(tf.where(tf.equal(y,y_),tf.constant([1]*64), tf.constant([0]*64)), tf.int32)
        F = tf.cast(tf.where(tf.equal(y,y_),tf.constant([0]*64), tf.constant([1]*64)), tf.int32)
        TP = tf.cast(tf.multiply(T,y_),tf.int32)
        FP = tf.cast(tf.multiply(F,y_),tf.int32)

        sum_one = tf.reduce_sum(tf.constant([1]*64))
        sum_tp = tf.reduce_sum(TP)
        sum_fp = tf.reduce_sum(FP)
        sum_tn = tf.subtract(tf.reduce_sum(T),sum_tp)
        sum_fn = tf.subtract(tf.reduce_sum(F),sum_fp)

        Precision = tf.divide(sum_tp,tf.add(sum_tp,sum_fp))
        Recall= tf.divide(sum_tp,tf.add(sum_tp,sum_fn))
        Fscore = tf.divide(tf.multiply(tf.multiply(Precision,Recall),1.0),tf.add(Precision,Recall))
        #squareFscore = tf.multiply(Fscore, Fscore)
        #threeMulFscore = tf.multiply(squareFscore, Fscore)
        #forMulFscore = tf.multiply(threeMulFscore, Fscore)
        #fourMulFscore = tf.multiply(threeMulFscore, Fscore) # if u wanna fscore more effection modify here
        """ this to make the fscore more important before close enought to zero """
        #floss= tf.divide(1, Fscore)
        #floss= Fscore
        floss= tf.square(tf.subtract(1.5, tf.cast(Fscore,tf.float32)))
        return TP, FP, T, F, Precision, Recall, floss



