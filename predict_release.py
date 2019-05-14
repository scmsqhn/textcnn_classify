#! /usr/bin/env python
import tensorflow as tf
import gensim
import jieba
import jieba.seg
import numpy as np
import re
import os

global predictions
global dct

CURPATH = os.path.dirname(os.path.realpath(__file__))
PARPATH = os.path.dirname(CURPATH)

"""
文件功能:加载模型进行预测
"""

"""
模块功能:设置
"""
# Flag Parameters
tf.flags.DEFINE_string("debug", "True", "is now we are debug.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("line_lenth", 1000, "length of per sentence")
tf.flags.DEFINE_string("checkpoint_dir", str(CURPATH), "the path of checkpoint dir is the cur dir")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#path Parameters
tf.flags.DEFINE_boolean("dictionary", "dictionary.dct", "dictionary.dct name")

FLAGS = tf.flags.FLAGS

dct = gensim.corpora.dictionary.Dictionary.load(os.path.join(CURPATH,FLAGS.dictionary))
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))

"""
输出映射表
"""
out_lst = {"抢劫","抢夺","诈骗","盗窃","其他"}
c2n,n2c = {},{}

for i,j in enumerate(out_lst):
    c2n[j]=i
    n2c[i]=j

"""
处理一条文本
"""
def line_2_words(line):
    res = []
    words = list(jieba.seg.cut(re.sub("[^\u4e00-\u9fa5a-zA-Z0-9]",",",line)))
    for word in words:
        if not word.flag == "x":
            res.append(word.word)
        res.extend(list("".join(res)))
        if not len(res)>FLAGS.line_lenth:
            res.extend([" "]*FLAGS.line_lenth)
        return res[:FLAGS.line_lenth]

"""
文本转batch
"""
def text_2_batch(textLst):
    resLst = []
    for j in range(FLAGS.batch_size):
        res = []
        i = j%(len(textLst))
        wordLst = line_2_words(textLst[i])
        for word in wordLst:
            try:
                res.append(dct.token2id[word])
            except:
                res.append(dct.token2id[" "])
        resLst.append(res)
    return resLst

"""
预测
cont 输入文本list
predicionts 模型输出
"""
def pred(cont,predictions):
    batch_predictions = []
    for i in range(len(cont)//FLAGS.batch_siza+1):
        batch = text_2_batch(cont[i*FLAGS.batch_size:i*FLAGS.batch_size+FLAGS.batch_size])#输入文本转为模型输入batch
        batch_predictions += sess.run(predictions, {input_x:batch, dropout_keep_prob:1.0})
    return batch_predictions[:len(cont)]

"""
加载模型
"""
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        #sess = tf.Session(session_conf)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        input_x = sess.graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = sess.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        all_operate = sess.graph.get_operations()
        output_operate = []
        for i in all_operate:
            if "output" in i.name:
                output_operate.append(i)
        predictions = sess.graph.get_operation_by_name("output/predictions").outputs[0]

if __name__ == "__main__":
    pred(["抢劫抢夺","诈骗盗窃"],predictions)

