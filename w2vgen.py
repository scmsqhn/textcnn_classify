import jieba
import jieba.posseg as seg
#jieba.load_userdict("/home/distdev/bilstm/model/all_addr_dict.txt")
import re
import gensim
w2v = gensim.models.word2vec.Word2Vec()
f = open("/home/distdev/iba/dmp/gongan/labelmarker/data/train.txt.bak","r")
g = open("/home/distdev/iba/dmp/gongan/labelmarker/data/eval.txt.bak","r")
ls = f.readlines()
gs = g.readlines()
ls.extend(gs)
sents = []
count = []

def segc(l):
    words = list(seg.cut(l))
    return words
print(segc(ls[0]))
for i in ls:
    #j = re.sub("[^\u4e00-\u9fa5a-zA-Z0-9]","",i)
    #sents.append(list(i))
    sents.append(list(jieba.cut(i)))
    sents.append(list(jieba.cut(re.sub("[^a-zA-Z]",",",i))))
    wds = []
    for word in list(seg.cut(i)):
        wds.append(word.word)
    sents.append(wds)
    sents.append(list(i))

    count.extend(list(jieba.cut(i)))
    count.extend(list(i))
import collections
dct = dict(collections.Counter(count))
print(dct['给'])
print(dct['个'])
print(dct['让'])
import pdb
pdb.set_trace()
w2v = gensim.models.word2vec.Word2Vec(sents, size=128, min_count=0)
w2v.save("./w2vmodel")
w2v['买']
