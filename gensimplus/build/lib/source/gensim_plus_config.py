#!/usr/bin/env python3
import tensorflow as tf
import os

CURPATH = os.path.dirname(os.path.realpath(__file__))
DATAPATH = os.path.join(os.pardir, CURPATH, "data")
dictionary_file_path = os.path.join(os.pardir, CURPATH, "data", "dictionary.dct")


tf.flags.DEFINE_string("current_path", CURPATH, "current path")
tf.flags.DEFINE_string("data_path", DATAPATH, "data path")
tf.flags.DEFINE_string("dictionary_file_path", dictionary_file_path, "data path")

tf.flags.DEFINE_string("word2vec_4_word", "word2vec_4_word", "word2vec_4_word")
tf.flags.DEFINE_string("word2vec_4_word_complex", "word2vec_4_word_complex", "word2vec_4_word_complex")
tf.flags.DEFINE_string("word2vec_4_char", "word2vec_4_char", "word2vec_4_char")


tf.flags.DEFINE_string("nous", "nr,ns,nz,n", "nous word flags") # 名词
tf.flags.DEFINE_string("verb", "v,vn", "verb word falgs")       # 动词
tf.flags.DEFINE_string("adj", "a", "adj word")                  #形容词
tf.flags.DEFINE_string("pj", "p", "pj word")                    #介词

#tf.flags.DEFINE_string("DEBUG", "True", "whether is debug mode")
tf.flags.DEFINE_string("DEBUG", "False", "whether is debug mode")
tf.flags.DEFINE_string("ERROR", "except a error", "except a error")

FLAGS = tf.flags.FLAGS

print("\n",FLAGS)
