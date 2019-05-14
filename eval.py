#! /usr/bin/env python

import gensim
import tensorflow as tf
import numpy as np
import re
import os
import pdb
import time
import datetime
import datahelper
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv

import datahelper
CURPATH = os.path.dirname(os.path.realpath(__file__))

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", str(os.path.join(CURPATH,"runs/target/checkpoints")), "help")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def write_pred_wrong(x, y_pred, y_lb):
    with open('./pred.txt', "a+") as f:
        f.write("\n pred:%s | lb:%s | cont:%s \n"%(y_pred,y_lb,x))


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

from sklearn.metrics import classification_report
def assification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))

filepath = "/home/distdev/iba/dmp/gongan/labelmarker/data/lb.txt"
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
# Evaluation
# ==================================================
print(FLAGS.checkpoint_dir)
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
dct = gensim.corpora.dictionary.Dictionary.load("./dictionary.dct")
with graph.as_default():

    global batches
    global evalbatches
    dh=datahelper.Data_Helper()
    #GEN= dh.dataGen(begin_cursor=0, dirpath='/home/distdev/src/iba/dmp/gongan/labelmarker/data', filename='eval.txt.bak', textcol='text', targetcol='addrcrim', funcname='gen_train_text_classify_from_text')

    def gene(item):
        x = item[0]
        y = item[1]
        for i,j in zip(x,y):
            yield (i,j)

    #_evalbatches = dh.getCorpus("eval.txt.bak",1) # every class sample num
    #dh.sav2Arctic(_evalbatches, "lb64")
    #_evalbatches = dh.loadFromArctic("lb64")
    #_testbatches= dh.getCorpus("five_classify_test_200.txt",1) # every class sample num
    #dh.sav2Arctic(_testbatches, "test200")
    _testbatches = dh.loadFromArctic("test200")
    #pdb.set_trace()
    batches = gene(_testbatches)
    #pdb.set_trace()
    #batches = datahelper.batches_iter(100, gen, 96, dh.tsne)

    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        print(checkpoint_file)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        all_operate = graph.get_operations()
        output_operate = []
        for i in all_operate:
            if "output" in i.name:
                output_operate.append(i)
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        #accuracy = graph.get_operation_by_name("accuracy").outputs[0]
        acc_sum = 0
        ts = 0
        while(1):
            batch = ""
            try:
                batch = batches.__next__()
            except:
                break
            batch_predictions = sess.run(predictions, {input_x: batch[0], dropout_keep_prob: 1.0})
            #batch_acc = sess.run(accuracy, {input_x: batch[0], dropout_keep_prob: 1.0})
            ts+=1
            #acc_sum+=batch_acc
            for i,j,k in zip(batch_predictions, batch[1],batch[0]):
                #pdb.set_trace()
                xstr = ""
                for c in k:
                    try:
                        xstr+=dh.dct.get(c)
                    except:
                        continue
                print(xstr)
                write_pred_wrong(xstr,i,j)
            for csr in range(len(batch_predictions)):
                all_predictions.append(batch_predictions[csr])
            y_batch = batch[1]
            for csr in  range(len(y_batch)):
                y_test.append(y_batch[csr])
            #assification_report(y_batch, batch_predictions)
            print("acc_avg", acc_sum/ts)
assification_report(y_test, all_predictions)
tp,fp,tn,fn = 0,0,0,0

dctpred = {}
dctlb = {}
wrongDct = {}
totalDct = {}

for i in range(0,5):
    wrongDct[str(i)] = 1
    totalDct[str(i)] = 1

for i,j in zip(all_predictions,y_test):
    #pdb.set_trace()
    if i == j:
        tp+=1
    else:
        fp+=1
        wrongDct[str(j)]+=1
    totalDct[str(j)]+=1
    if not i == j:
        print(i,j, "not equal")
        k = "%s"%i
        k2 = "%s"%j
        if k in dctpred.keys():
            dctpred[k]+=1
        else:
            dctpred[k]=1
        if k2 in dctlb.keys():
            dctlb[k2]+=1
        else:
            dctlb[k2]=1
pdb.set_trace()
if tp==0 and fp==0:
    pdb.set_trace()
acc = tp/(tp+fp)
print("dctpred:", dctpred)
print("dctlb", dctlb)
print("> Final acc is :", acc)
for i in  range(5):
    pdb.set_trace()
    rens = wrongDct[str(i)]/totalDct[str(i)]
    print(str(i), "wrong percent is ",rens)
pdb.set_trace()
# Print accuracy if y_test is defined

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

