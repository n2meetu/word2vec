# -*- coding: utf-8-*-
import jieba
from gensim.models import word2vec

#  去掉中英文状态下的逗号、句号
def clearSen(comment):
    comment = comment.strip(' ')
    comment = comment.replace('、','')
    comment = comment.replace('~','。')
    comment = comment.replace('～','')
    comment = comment.replace('{"error_message": "EMPTY SENTENCE"}','')
    comment = comment.replace('…','')
    comment = comment.replace('\r', '')
    comment = comment.replace('\t', ' ')
    comment = comment.replace('\f', ' ')
    comment = comment.replace('/', '')
    comment = comment.replace('、', ' ')
    comment = comment.replace('/', '')
    comment = comment.replace(' ', '')
    comment = comment.replace(' ', '')
    comment = comment.replace('_', '')
    comment = comment.replace('?', ' ')
    comment = comment.replace('？', ' ')
    comment = comment.replace('了', '')
    comment = comment.replace('➕', '')
    return comment

# 用jieba进行分词
comment = open('./corpus/comment.txt').read()
comment = clearSen(comment)
jieba.load_userdict('./user_dict/userdict_food.txt')
comment = ' '.join(jieba.cut(comment))

# 分完词的文件写到新的txt中去
fo = open("./corpus/afterSeg.txt","w")
fo.write(comment)
print("finished!")
fo.close()

sentences=word2vec.Text8Corpus(u'./corpus/afterSeg.txt')
# 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5, 第三个参数是神经网络的隐藏层单元数，默认为100
model=word2vec.Word2Vec(sentences,min_count=3, size=50, window=5, workers=4)

y2=model.similarity(u"不错", u"好吃") #计算两个词之间的余弦距离
print(y2)

for i in model.most_similar(u"好吃"): #计算余弦距离最接近“滋润”的10个词
    print(i[0],i[1])

# 存储和加载咱们辛辛苦苦训练好的模型：
# model.save('/model')
# new_model=gensim.models.Word2Vec.load('/model/word2vec_model')
# 获取每个词的词向量
# model['computer']

# 训练词向量时传入的两个参数也对训练效果有很大影响，需要根据语料来决定参数的选择，好的词向量对NLP的分类、聚类、相似度判别等任务有重要意义