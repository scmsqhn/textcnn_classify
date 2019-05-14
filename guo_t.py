#/usr/bin/bash
#/coding = 'utf-8'
import requests
import json
import random
import traceback
import pdb

url1="http://localhost:8888/guizhou/classify/case-type-five"
headers = {'content-type': 'application/json', "Accept": "application/json"}

columns=["carinfo", "creditCard", "identifier", "phoneNum", 'criminalphone', "qq", "weibo", "wx","momo","nickname","web","mail",'qqname','wxname','criminalidentifier','criminalqq','criminalwx']

def _post(url):
  l2 = ["元人民币","家中被抢劫"]
  l1 = ["贵阳观山湖","贵阳龙洞堡"]
  data = {
        'messageid': "12",
        'clientid': "13",
        'text':l2,
        #'funcname':columns[random.randint(0,len(columns)-1)],
        'encrypt':'false',
        }
  print(url)
  print(data['text'])
  try:
     res = requests.post(url,data=json.dumps(data),headers=headers,verify=False,timeout=200)
     print('>>>> res')
     print(res)
     print(res.status_code)
     print(res.headers)
     print(res.text)
     #pdb.set_trace()
  except:
      traceback.print_exc()
      #pdb.set_trace()


def _post_baidu():
    res = requests.get(url="https://www.baidu.com/",verify='false')

while(1):
  try:
    _post(url1)
    #pdb.set_trace()
  except:
    traceback.print_exc()
