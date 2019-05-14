#/usr/bin/bash
#/coding = 'utf-8'
import requests
import json
import random
import traceback
import pdb

url1="http://60.247.77.171:8888/guizhou/classify/case-type-five"
#url1="http://localhost:8888/guizhou/classify/case-type-five"
#url1="http://localhost:8888/guizhou/classify/case-type-five"
#url1="http://0.0.0.0:9999/classify/case-type-five"
#url2="http://127.0.0.1:9999/predict/110/identifier"
#url3="http://127.0.0.1:9999/predict/110/weixin"
#url4="http://127.0.0.1:9999/predict/110/carinfo"
#url2="https://113.204.229.74:18100/guizhou/method/carinfo"
#url3="https://113.204.229.74:18100/guizhou/loc/predict"
#url1="http://127.0.0.1:9999/predict/110/carinfo"
#url2="http://127.0.0.1:9999/predict/110/phoneNum"
#url3="http://127.0.0.1:9999/predict/110/ner/carinfo"
#url1="http://127.0.0.1:5556/predict/110/phoneNum"
#url1="http://127.0.0.1:23578/ner/carinfo"
#url1="http://0.0.0.0:9999/ner/carinfo"
#url2="http://113.204.229.74:9999/ner/carinfo"
#url3="http://127.0.0.1:9999/ner/phoneNum"
#url3="http://113.204.229.74:9999/method/weixin"
headers = {'content-type': 'application/json', "Accept": "application/json"}

columns=["carinfo", "creditCard", "identifier", "phoneNum", 'criminalphone', "qq", "weibo", "wx","momo","nickname","web","mail",'qqname','wxname','criminalidentifier','criminalqq','criminalwx']

def _post(url):
  l2 = ["校门口被盗","家中被抢劫","2018年3月16日22时21分，李小金（男，32岁，汉族，个体，高中文化，身份证号：511023198602063298，户籍地址：四川省资阳市安岳县林凤镇山湾村5组，现住址：贵阳市南明区大南门护国路152号，电话：15885108678，）报称其停放在贵阳市云岩区贝蒂领航F栋地下停车场的一辆二轮红色立马电瓶车（车架号：185121701001766，电机号：不详，2017年3月以4600元购买，安装万物互联，车卡编号：30069404，人卡编号：31066537）被人以“搭线发车”的方式盗走。"]
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
