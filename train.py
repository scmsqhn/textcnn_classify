#! /use/bin/env python3
import os
import tensorflow as tf
import numpy as np
import time
import datetime
import sys
CURPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(CURPATH)
from text_cnn import TextCNN
import pdb
# ==================================================
def gen(src):
    x = src[0]
    y = src[1]
    for i,j in zip(x,y):
        yield (i,j)

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

tf.flags.DEFINE_string("train_file", "./data/train.dat", "Data source for the train.")
tf.flags.DEFINE_string("eval_file", "./data/eval.dat", "Data source for the eval.")
tf.flags.DEFINE_string("test_file", "./data/test.dat", "Data source for the test.")

# Model Hyperparameters
tf.flags.DEFINE_integer("num_classes", 5, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("vocab_size", 100000, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_integer("sequence_length", 1000, "Number of filters per filter size (default: 128)")

tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2,3", "Comma-separated filter sizes (default: '1,2,3')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Learning Rate (default:1e-3)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

import datahelper

#data_helpers = data_helpers.Data_Helper()

def train():
    # Training
    # ==================================================
    pass  #  print("train")
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement
        )
        pass  #  print("> new a sess")
        sess = tf.Session(config=session_conf)
        pass  #  print("> new a sess finish")
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=FLAGS.sequence_length,
                num_classes= FLAGS.num_classes,
                vocab_size = FLAGS.vocab_size,
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
            pass  #  print("train4")

            # Define Training procedure
            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=tf.train.get_or_create_global_step())
            timestamp = str(int(time.time()))

            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", "target"))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            sess.run(tf.global_variables_initializer())
            dh=datahelper.Data_Helper()
            global batches
            global evalbatches
            global _batchres
            global _evalbatches

            _batches = dh.loadFromArctic("train_file")
            _evalbatches = dh.loadFromArctic("eval_file")
            _testbatches = dh.loadFromArctic("test_file")

            #_batches = dh.dataGen(begin_cursor=0, dirpath='/home/distdev/src/iba/dmp/gongan/labelmarker/data', filename='train.txt.bak', textcol='text', targetcol='addrcrim', funcname='gen_train_text_classify_from_text')
            #_evalbatches = dh.dataGen(begin_cursor=0, dirpath='/home/distdev/src/iba/dmp/gongan/labelmarker/data', filename='eval.txt.bak', textcol='text', targetcol='addrcrim', funcname='gen_train_text_classify_from_text')
            #dh.sav2Arctic(_batches, "train640")
            #dh.sav2Arctic(_evalbatches, "eval640")
            #_batches = dh.getCorpus("classify_traintrain.txt.bak",1) # every class sample num
            #_evalbatches = dh.getCorpus("five_classify_eval.txt",1) # every class sample num
            #_testbatches= dh.getCorpus("five_classify_test_200.txt",1) # every class sample num
            #_evalbatches = dh.getCorpus("eval.txt.bak",1) # every class sample num
            maxacc = 0.0

            def train_step(x_batch, y_batch):
                print(x_batch)
                print(y_batch)
                """
                A single training step
                """
                # y_batch = [(i-1)/2 for i in y_batch]
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }

                _, step, loss, softmax, accuracy, y_pred , scores, corr, embed_chars, h_pool, h_pool_final,losses= sess.run(
                    [train_op, global_step,  cnn.loss, cnn.softmax, cnn.accuracy, cnn.predictions, cnn.scores, cnn.correct_predictions, cnn.embedded_chars_expanded, cnn.h_pool, cnn.h_pool_final, cnn.losses],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                current_step = tf.train.global_step(sess, global_step)
                if current_step % 10 == 0:
                    sent = ("train {}: step {}, loss {}, acc {}, losses{} ".format(time_str, step, loss, accuracy,losses))
                    #sent = ("{}: step {}, loss {:g}, acc {:g}, losses{:g},  softmax{} ".format(time_str, step, loss, accuracy,losses,softmax))
                    print(sent)
                    with open('./sumf.txt', 'a+') as sumf:
                        sumf.write(sent+"\n")
                    #print("{}: step {}, correct_predict{}, ".format(time_str, step, corr))
                    #print("{}: step {}, input_y{}, scores{}, ".format(time_str, step, y_batch, scores))
                    #print(type(scores[0][0]))
                    #print("{}: step {}, \ninput_y{}, \npred_y{}".format(time_str, step, y_batch, y_pred))
                    #print("{}: step {}, \nh_pool_plat{}".format(time_str, step, h_pool_plat))
                #train_summary_writer.add_summary(summaries, step)

            pass  #  print("trainstep")
            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                #y_batch = [(i-1)/2 for i in y_batch]
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                time_str = datetime.datetime.now().isoformat()
                #[accc]= sess.run([cnn.accuracy], feed_dict)
                [step, sum_fsc,accc]= sess.run([global_step,cnn.sum_fsc,cnn.accuracy], feed_dict)
                #[step, pred_y,accc]= sess.run([global_step, cnn.predictions, cnn.accuracy], feed_dict)
                sent = "eval {}: step {}\n, sum_fsc{}, \naccc {} ".format(time_str, step, sum_fsc, accc)
                #sent = "eval {}: step {}\n, sum_fsc{}, \naccc {} fsc4 {} fsc1 {} fsc2 {} fsc3 {} ".format(time_str, step, sum_fsc, accc, fsc4, fsc1, fsc2, fsc3)
                with open('./sumf.txt', 'a+') as sumf:
                    sumf.write(sent+"\n")
                return accc
            # Generate batches
            # Training loop. For each batch...
            #for batch in batches:
            for i in range(FLAGS.num_epochs):
                _batches = dh.loadFromArctic("train")
                _evalbatches = dh.loadFromArctic("eval")
                _testbatches = dh.loadFromArctic("test")
                #_evalbatches = dh.loadFromArctic("lb64")
                #_batches = dh.loadFromArctic("tfidf64")
                batches = gen(_batches)
                evalbatches = gen(_evalbatches)
                while(1):
                    batch =0.0
                    try:
                        batch = batches.__next__()
                    except:
                        break
                    x_batch, y_batch = batch[0], batch[1]
                    x_batch = x_batch.astype(np.int32)
                    y_batch = y_batch.astype(np.int32)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0:
                        x_dev = np.array([])
                        y_dev = np.array([])
                        accsum = 0
                        ts = 0
                        for i in range(5):
                            try:
                                x_dev, y_dev= evalbatches.__next__()
                                x_dev = x_dev.astype(np.int32)
                                y_dev = y_dev.astype(np.int32)
                                accsum += dev_step(x_dev, y_dev)
                                ts+=1
                            except StopIteration:
                                evalbatches = gen(_evalbatches)
                                x_dev, y_dev= evalbatches.__next__()
                                x_dev = x_dev.astype(np.int32)
                                y_dev = y_dev.astype(np.int32)
                                accsum += dev_step(x_dev, y_dev)
                                ts+=1
                        time_str = datetime.datetime.now().isoformat()
                        save_step = tf.train.global_step(sess, global_step)
                        #current_step = tf.train.global_step(sess, global_step)
                        sent = ("{}: step {}, accsum {:g}, ".format(time_str, save_step, accsum/ts))
                        print(sent)
                        with open('./sumf.txt', 'a+') as sumf:
                            sumf.write(sent+"\n")
                    if current_step % FLAGS.checkpoint_every == 0:
                        maxacc = accsum
                        path = saver.save(sess, checkpoint_prefix, global_step=save_step)
                        """
                        将模型保存为pb格式的文件

                        """
                        #builder = tf.saved_model.builder.SavedModelBuilder('./model.pb')
                        #builder.add_meta_graph_and_variables(sess, ["mytag"])
                        #builder.save()

                #if current_step % FLAGS.checkpoint_every == 0:
                #    path = saver.save(sess, checkpoint_prefix, global_step=save_step)
                #    pass  #  print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    #x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train()

if __name__ == '__main__':
    tf.app.run()
    #!/usr/bin/env python
