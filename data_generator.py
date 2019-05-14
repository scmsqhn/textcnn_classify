import datahelper

dh=datahelper.Data_Helper()

batches = dh.dataGen(begin_cursor=0, dirpath='./data', filename='classify_train.txt', textcol='text', targetcol='addrcrim', funcname='gen_train_text_classify_from_text')
evalbatches = dh.dataGen(begin_cursor=0, dirpath='./data', filename='classify_eval.txt', textcol='text', targetcol='addrcrim', funcname='gen_train_text_classify_from_text')
testbatches = dh.dataGen(begin_cursor=0, dirpath='./data', filename='classify_test.txt', textcol='text', targetcol='addrcrim', funcname='gen_train_text_classify_from_text')

dh.sav2Arctic(batches, "train_file")
dh.sav2Arctic(evalbatches, "eval_file")
dh.sav2Arctic(testbatches, "test_file")

