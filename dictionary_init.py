import gensim
import data_helper
dh = data_helper.Data_Helper()

def genDct():
        dirpath = "/home/distdev/src/iba/dmp/gongan/labelmarker/data"
        filename = "train.txt.bak"
        res , reslb = dh.read_file_2_possge(dirpath,filename,1)
        res = [i[-1] for i in res]
        dictionary = gensim.corpora.Dictionary(res)
        dictionary.save('./dct')
dh.genDct()
