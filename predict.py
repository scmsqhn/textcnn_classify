#! /usr/bin/env python
from multiprocessing import Queue
import tensorflow as tf
from sklearn.metrics import classification_report
import numpy as np
import re
import os
import pdb
import time
import datetime
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import datahelper
dh = datahelper.Data_Helper()
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
tf.flags.DEFINE_string("checkpoint_dir", str(os.path.join(CURPATH, "runs/target/checkpoints")), "Checkpoint directory from training run path")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

kv_out = {"0":"抢劫","1":"抢夺","2":"诈骗","3":"盗窃","4":"其他"}

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
breakLoop = False

print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def write_pred_wrong(x, y_pred, y_lb):
    with open(os.path.join(CURPATH, './pred.txt'), "a+") as f:
        f.write("\n pred:%s | lb:%s | cont:%s \n"%(y_pred,y_lb,x))

print(FLAGS.checkpoint_dir)
#checkpoint_file = os.path.join(CURPATH, "runs/target/checkpoints/model-24000")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

# CHANGE THIS: Load data. Load your own data here
#if FLAGS.eval_train:
#    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
#    y_test = np.argmax(y_test, axis=1)
#else:
#    x_raw = ["a masterpiece four years in the making", "everything is off."]
#    y_test = [1, 0]

# Map data into vocabulary
#vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
#vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
#x_test = np.array(list(vocab_processor.transform(x_raw)))

def assification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))


filepath = os.path.join(CURPATH, "lb.txt")
f = open(filepath)
lines=f.readlines()
lbs = [re.sub("[^\u4e00-\u9fa5a-zA-Z]","",i) for i in lines]
c2n={}
n2c={}
for i,j in enumerate(lbs):
        c2n[j]=i
        n2c[i]=j
all_predictions = []
y_test = []
import pdb
#pdb.set_trace()
def gene(item):
            x = item[0]
            y = item[1]
            for i,j in zip(x,y):
                yield (i,j)
print(checkpoint_file)

def getNextData(q):
    if not q.empty():
        return q.get()
    else:
        return -1

def set_break():
    breakLoop = True

def reset_break():
    breakLoop = False

def classify_5_predict(cont):
    batch = dh.text_2_batch(cont)
    batch_predictions = sess.run(predictions, {input_x: batch, dropout_keep_prob: 1.0})
    return batch_predictions


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

def main():
    tt3 = open("./tt3","r")
    tt2 = open("./tt2","r")
    cont = tt2.read()
    pdb.set_trace()
    cont3 = tt3.read()
    lns = cont.split("\n")
    lns3 = cont3.split("\n")
    pdb.set_trace()
    #classify_5_predict(["家中被盗","路边被抢","2018年3月16日22时21分，李小金(男，32岁，汉族，个体，高中文化，身份证号：511023198602063298，户籍地址：四川省资阳市安岳县林凤镇山湾村5组，现住址：贵阳市南明区大南门护国路152号，电话：15885108678，)报称其停放在贵阳市云岩区贝蒂领航F栋地下停车场的一辆二轮红色立马电瓶车(车架号：185121701001766，电机号：不详，2017年3月以4600元购买，安装万物互联，车卡编号：30069404，人卡编号：31066537)被人以“搭线发车”的方式盗走。"])
    a = classify_5_predict(lns)
    pdb.set_trace()
    c = classify_5_predict(lns3)
    pdb.set_trace()

def main2():
    lst = ['双抢','双抢','诈骗','盗窃','其他']
    f = open("tri.txt","r")
    lines = f.readlines()
    np.random.shuffle(lines)
    for i in range(4):
        cur_lines = lines[i*64:i*64+64]
        batch = dh.text_2_batch(cur_lines)
        batch_predictions = sess.run(predictions, {input_x: batch, dropout_keep_prob: 1.0})
        for i,j in zip(cur_lines,batch_predictions):
            with open("result.txt","a+") as g:
                g.write("%s\t%s\t%s\n"%(i.split("\t")[0],j,i.split("\t")[-1]))
    #return batch_predictions


"""
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(y_test), all_predictions))
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(predictions_human_readable)
"""

if __name__ == "__main__":
    main2()
