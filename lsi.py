from gensim.test.utils import common_corpus, common_dictionary, get_tmpfile
from gensim.models import LsiModel
import gensim.corpora
import gensim.similarities as similarities
import gensim.models
import gensim.corpora.dictionary as dictionary
import jieba
import jieba.posseg
# 训练样本
INIT_DICT=True
INIT_CORPUS=True
corpus = ""
dictionary= ""

raw_documents = [
    '0南京江心洲污泥偷排”等污泥偷排或处置不当而造成的污染问题，不断被媒体曝光',
    '1面对美国金融危机冲击与国内经济增速下滑形势，中国政府在2008年11月初快速推出“4万亿”投资十项措施',
    '2全国大面积出现的雾霾，使解决我国环境质量恶化问题的紧迫性得到全社会的广泛关注',
    '3大约是1962年的夏天吧，潘文突然出现在我们居住的安宁巷中，她旁边走着40号王孃孃家的大儿子，一看就知道，他们是一对恋人。那时候，潘文梳着一条长长的独辫',
    '4坐落在美国科罗拉多州的小镇蒙特苏马有一座4200平方英尺(约合390平方米)的房子，该建筑外表上与普通民居毫无区别，但其内在构造却别有洞天',
    '5据英国《每日邮报》报道，美国威斯康辛州的非营利组织“占领麦迪逊建筑公司”(OMBuild)在华盛顿和俄勒冈州打造了99平方英尺(约9平方米)的迷你房屋',
    '6长沙市公安局官方微博@长沙警事发布消息称，3月14日上午10时15分许，长沙市开福区伍家岭沙湖桥菜市场内，两名摊贩因纠纷引发互殴，其中一人被对方砍死',
    '7乌克兰克里米亚就留在乌克兰还是加入俄罗斯举行全民公投，全部选票的统计结果表明，96.6%的选民赞成克里米亚加入俄罗斯，但未获得乌克兰和国际社会的普遍承认',
    '8京津冀的大气污染，造成了巨大的综合负面效应，显性的是空气污染、水质变差、交通拥堵、食品不安全等，隐性的是各种恶性疾病的患者增加，生存环境越来越差',
    '9 1954年2月19日，苏联最高苏维埃主席团，在“兄弟的乌克兰与俄罗斯结盟300周年之际”通过决议，将俄罗斯联邦的克里米亚州，划归乌克兰加盟共和国',
    '10北京市昌平区一航空训练基地，演练人员身穿训练服，从机舱逃生门滑降到地面',
    '11腾讯入股京东的公告如期而至，与三周前的传闻吻合。毫无疑问，仅仅是传闻阶段的“联姻”，已经改变了京东赴美上市的舆论氛围',
    '12国防部网站消息，3月8日凌晨，马来西亚航空公司MH370航班起飞后与地面失去联系，西安卫星测控中心在第一时间启动应急机制，配合地面搜救人员开展对失联航班的搜索救援行动',
    '13新华社昆明3月2日电，记者从昆明市政府新闻办获悉，昆明“3·01”事件事发现场证据表明，这是一起由新疆分裂势力一手策划组织的严重暴力恐怖事件',
    '14在即将召开的全国“两会”上，中国政府将提出2014年GDP增长7.5%左右、CPI通胀率控制在3.5%的目标',
    '15中共中央总书记、国家主席、中央军委主席习近平看望出席全国政协十二届二次会议的委员并参加分组讨论时强调，团结稳定是福，分裂动乱是祸。全国各族人民都要珍惜民族大团结的政治局面，都要坚决反对一切危害各民族大团结的言行'
]
with open("/home/distdev/iba/dmp/gongan/labelmarker/data/eval.txt.bak","r") as f:
    raw_documents = f.read().split("\n")
raw_documents = [i.split("\t")[1] for i in raw_documents[:-1]]
print(raw_documents)
import pdb
pdb.set_trace()
corpora_documents = []
"""
将文章分解成，由词组成的二维数组 corpora_documents
"""
for item_text in raw_documents:
    item_segs = list(jieba.posseg.cut(item_text))
    words = []
    for item in item_segs:
        if item.flag == "x":
            continue
        words.append(item.word)
    corpora_documents.append(words)
print(corpora_documents[:2])
"""
生成词典
"""
dct = gensim.corpora.dictionary.Dictionary(corpora_documents)
dct.save('dct') #保存生成的词典
# print(dictionary)
dct=gensim.corpora.dictionary.Dictionary.load('dct')#加载
# 通过下面一句得到语料中每一篇文档对应的稀疏向量（这里是bow向量）
corpus = [dct.doc2bow(text) for text in corpora_documents]
# 向量的每一个元素代表了一个word在这篇文档中出现的次数
# print(corpus)
gensim.corpora.MmCorpus.serialize('corpuse.mm',corpus)#保存生成的语料
corpus=gensim.corpora.MmCorpus('corpuse.mm')#加载

# corpus是一个返回bow向量的迭代器。下面代码将完成对corpus中出现的每一个特征的IDF值的统计工作
tfidf_model = gensim.models.TfidfModel(corpus)
corpus_tfidf = tfidf_model[corpus]

'''
#查看model中的内容
for item in corpus_tfidf:
    print(item)
    # tfidf.save("data.tfidf")
    # tfidf = models.TfidfModel.load("data.tfidf")
# print(tfidf_model.dfs)
'''

"""

similarity = gensim.similarities.Similarity('Similarity-tfidf-index', corpus_tfidf, num_features=800)
test_data_1 = '北京雾霾红色预警'
test_cut_raw_1 = list(jieba.cut(test_data_1))  # ['北京', '雾', '霾', '红色', '预警']
print(test_cut_raw_1)
import pdb
pdb.set_trace()
test_corpus_1 = dct.doc2bow(test_cut_raw_1)  # [(51, 1), (59, 1)]，即在字典的56和60的地方出现重复的字段，这个值可能会变化
similarity.num_best = 5
print(test_corpus_1)
test_corpus_tfidf_1=tfidf_model[test_corpus_1]  # 根据之前训练生成的model，生成query的IFIDF值，然后进行相似度计算
print(test_corpus_tfidf_1)
# [(51, 0.7071067811865475), (59, 0.7071067811865475)]
print(test_data_1)
print(similarity[test_corpus_tfidf_1])  # 返回最相似的样本材料,(index_of_document, similarity) tuples
print('-'*40)
test_data_2 = '成都街头发生砍人事件致6人死亡'
test_cut_raw_2 = list(jieba.cut(test_data_2))
test_corpus_2 = dct.doc2bow(test_cut_raw_2)
test_corpus_tfidf_2=tfidf_model[test_corpus_2]
similarity.num_best = 3
print(test_data_2)
print(similarity[test_corpus_tfidf_2])  # 返回最相似的样本材料,(index_of_document, similarity) tuples
"""
# 使用LSI模型进行相似度计算
lsi = gensim.models.LsiModel(corpus_tfidf)
corpus_lsi = lsi[corpus_tfidf]
similarity_lsi=gensim.similarities.Similarity('Similarity-LSI-index', corpus_lsi, num_features=400,num_best=10)
# save
# LsiModel.load(fname, mmap='r')#加载
test_data_3 = '报称：李哥庄周臣屯银行，嫌疑人拿刀威胁并殴打报警人，耳环跟现金等物品被抢走。嫌疑人特征：骑黑色踏板摩托车，身高170左右带墨镜平头往铁路往南逃跑。 报警人在周臣屯167号。2017-7-23 13:06:54，接指挥中心派警：报称：李哥庄周臣屯引黄济青南岸，嫌疑人持刀威胁并殴打受害人，耳环跟现金等物品被抢走。嫌疑人特征：骑黑色踏板摩托车，40多岁本地口音身高170左右带墨镜平头往铁路往南逃跑。 报警人在周臣屯167号，受害人孙珍萍 女 53岁。出警到达现场后，电话联系报警人（孙珍萍，女，身份证号370。2017年7月23日 13左右，孙珍萍（女，51岁，身份证号:370222196606285320，户籍地：山东省胶州市李哥庄镇周臣屯村，电话15864241991）报称：在李哥庄周臣屯引黄济青南岸，嫌疑人持刀威胁并殴打自己，耳环跟现金等物品被抢走，要求处理。'
test_cut_raw_3 = list(jieba.cut(test_data_3))         # 1.分词
test_corpus_3 = dct.doc2bow(test_cut_raw_3)  # 2.转换成bow向量
test_corpus_tfidf_3 = tfidf_model[test_corpus_3]  # 3.计算tfidf值
test_corpus_lsi_3 = lsi[test_corpus_tfidf_3]  # 4.计算lsi值
#lsi.add_documents(test_corpus_lsi_3) #更新LSI的值
print('-'*40)
print(test_data_3)
print(similarity_lsi[test_corpus_lsi_3])
for i in range(10):
    _id = similarity_lsi[test_corpus_lsi_3][i][0]
    print(raw_documents[_id])
    print("\n")

