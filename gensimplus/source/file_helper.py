#!/usr/bin/env python3
import codecs
import sys
import gensim
import os
import pdb
from gensim_plus_config import FLAGS
import traceback
from base_interface import DictionaryHelperInterface
import sys
import re
import string
from base_interface import FileHelperInterface
import docx
import json
import traceback
import subprocess
class FileHelper(FileHelperInterface):

    def __init__(self):
        pass

    def renameDoc2Docx(self,inpath):
        print(type(inpath))
        print(inpath)
        cmds = ["libreoffice","--headless","--invisible","--convert-to","docx",inpath,"--outdir","/home/siyuan/guangxi_liuzhou/data/Untitled Folder"]
        print(str(cmds))
        output = subprocess.check_output(cmds)

    def isDoc(self,inpath):
        append = re.findall("\.doc$",inpath)
        print("isDoc",append)
        if len(append)>0:
            return True
        else:
            return False

    def isDocx(self,inpath):
        append = re.findall("\.docx$",inpath)
        print("isDocx",append)
        if len(append)>0:
            return True
        else:
            return False

    def readDoc(self,inpath):
        #pdb.set_trace()
        sentences = ""
        if not self.isDocx(inpath):
            return -1
        cont = docx.Document(inpath)
        #pdb.set_trace()
        texts = cont.paragraphs
        for text in texts:
            sentence = text.text
            sentences+=sentence
            sentences+="&"
        return sentences

    def readDocDir(self,dirpath):
        res_json = {}
        pathLst = os.walk(dirpath)
        print(pathLst)
        pathLst = list(pathLst)
        for path in pathLst:
            for filename in path[2]:
                try:
                    js = {}
                    cont = self.readDoc(os.path.join(path[0],filename))
                    js['doc'] = path[0]
                    js['title'] = filename
                    js['content'] = cont
                    res_json[len(res_json)]= js
                except:
                    traceback.print_exc()
                    continue
        return res_json

    def docx(self,infile):
        outfile = re.sub("\.doc",".docx",infile)
        return outfile

    def savDoc2DocxDir(self,dirpath):
        pathLst = os.walk(dirpath)
        print(pathLst)
        pathLst = list(pathLst)
        for path in pathLst:
            for filename in path[2]:
                #pdb.set_trace()
                flag = self.isDoc(os.path.join(path[0],filename))
                if flag:
                    print("savDoc2DocxDir: ",filename,self.docx(filename))
                    self.renameDoc2Docx(os.path.join(path[0],filename))

    def savJson(self,js,filename):
        with codecs.open(filename,"a+","utf-8") as f:
            f.write(json.dumps(js))

    def loadJson(self,filename):
        with codecs.open(filename,"r","utf-8") as f:
            return json.loads(f.read())


if __name__ == "__main__":
    pass
    fileHelperInstance = FileHelper()
    #fileHelperInstance.savDoc2DocxDir(os.path.join("/home/siyuan/guangxi_liuzhou/data","guangxi_liuzhou"))
    res_json = fileHelperInstance.readDocDir("/home/siyuan/guangxi_liuzhou/data/Untitled Folder")
    fileHelperInstance.savJson(res_json,"./guangxi_liuzhou.json")
    js = fileHelperInstance.loadJson("./guangxi_liuzhou.json")
