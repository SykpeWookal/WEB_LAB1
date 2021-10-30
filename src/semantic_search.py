import json
import nltk
import os

from nltk import word_tokenize,pos_tag   #分词、词性标注
from nltk.corpus import stopwords    #停用词
from nltk.stem import PorterStemmer    #词干提取
from nltk.stem import WordNetLemmatizer    #词性还原

import sklearn.decomposition as Spca
import sklearn.feature_extraction.text as ft #tfidf模型训练器
import numpy as np
import scipy.spatial



InvertedIndex = {}
SearchTerm = set()

TfIdf_OnlySearchWordArray = []

with open("../实验一查询词表.txt", "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  #去掉列表中每一个元素的换行符
        #print(line)
        line_cg = PorterStemmer().stem(line)
        line_final = WordNetLemmatizer().lemmatize(line_cg, pos='v')
        SearchTerm.add(line_final)
        InvertedIndex[line_final] = set()

#print(SearchTerm)



PROJECT_DIR_PATH = os.path.dirname(os.path.abspath(os.path.abspath(__file__)))
#print(PROJECT_DIR_PATH)
#DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'US_Financial_News_Articles/2018_01')
DIR_PATH = os.path.join(PROJECT_DIR_PATH, '../dataset')
#print(DIR_PATH)
files = os.listdir(DIR_PATH)
#print(len(files))


interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '|', '\'\'', '``', '/']   #定义标点符号列表

DocNameDict = {}
DocIndex = 0

for file in files:
    DocNameDict[DocIndex] = file
    DocIndex += 1
    #print(file)
    #f = open('US_Financial_News_Articles/2018_01/' + file, 'rb')
    f = open('../dataset/' + file, 'rb')
    x = json.load(f)
    #print(type(x))
    #print(x['text'])
    #y = x['text'].split()
    #print(y)

    tokens = word_tokenize(x['text'])
    # print('【NLTK分词结果：】')
    # print(tokens)

    #interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '|', '\'\'', '``', '/']   #定义标点符号列表
    tokens_x = [word for word in tokens if word not in interpunctuations]   #去除标点符号
    # print('\n【NLTK分词后去除符号结果：】')
    # print(tokens_x)

    stops = set(stopwords.words("english"))
    tokens_stops = [word for word in tokens_x if word not in stops]
    # print('\n【NLTK分词后去除停用词结果：】')
    # print(tokens_stops)

    # print('\n【NLTK分词进行词干提取：】')
    tokens_cg = []
    for i in tokens_stops:
        tokens_cg.append(PorterStemmer().stem(i))    #词干提取
    # print(cutwords4)

    # print('\n【NLTK分词进行词形还原：】')
    tokens_finial = []
    for i in tokens_cg:
        tokens_finial.append(WordNetLemmatizer().lemmatize(i, pos='v'))   #指定还原词性为动词
    #print(tokens_finial)

    # #接下来生成倒排索引
    # for i in tokens_finial:
    #     #print(i)
    #     if i in SearchTerm:
    #         InvertedIndex[i].add(file)
    #

    #########tfidf部分
    str = ''
    for i in tokens_finial:
        if i in SearchTerm:
            str += i
            str += ' '
    if str != '':
        TfIdf_OnlySearchWordArray.append(str)


#print(TfIdf_OnlySearchWordArray)

cv = ft.CountVectorizer()           # 构建词袋模型
bow = cv.fit_transform(TfIdf_OnlySearchWordArray)       # 训练词袋模型
#print(cv.get_feature_names_out())       # 获取所有特征名

# TFIDF
tt = ft.TfidfTransformer()          # 获取TF-IDF模型训练器
Doc_tfidf = tt.fit_transform(bow)       # 训练

#print(np.round(Doc_tfidf.toarray(), 4))       # 精确到小数点后4位
#   Tfidf矩阵结构：
#              查询词1         查询词2         查询词3        ······      查询词M
#   文档1                                 {v1}
#   文档2                                 {v2}
#   文档3                                 {v3}
#   ·····                               ······
#   文档N                                 {vn}

Doc_featureNames = (cv.get_feature_names_out()).tolist() #语料库内所有的查询词列表，按字母升序
TfMat = Doc_tfidf.toarray()
# print(TfMat)
# print(TfMat.shape)
# print(Doc_featureNames)
# print(len(featureNames))

###################输出文件#############################
TfOutPut_PATH = os.path.join(PROJECT_DIR_PATH, '../output/semantic_search_TfMat.txt')
# fresult = open(OutPut_PATH,'w+')
np.savetxt(TfOutPut_PATH,TfMat,fmt='%.2e',newline='\n')
###################输出文件#############################

#####以下处理查询输入
finnal_searchWordsList = []

with open("../语义查询输入.txt", "rb") as f:
    searchWords = f.read()

tokens = word_tokenize(bytes.decode(searchWords))
#print(tokens)
tokens_x = [word for word in tokens if word not in interpunctuations]   #去除标点符号
stops = set(stopwords.words("english"))
tokens_stops = [word for word in tokens_x if word not in stops]
tokens_cg = []
for i in tokens_stops:
    tokens_cg.append(PorterStemmer().stem(i))    #词干提取
tokens_finial = []
for i in tokens_cg:
    tokens_finial.append(WordNetLemmatizer().lemmatize(i, pos='v'))   #指定还原词性为动词

str = ''
for i in tokens_finial:
    if i in Doc_featureNames:
        str += i
        str += ' '
if str != '':
    finnal_searchWordsList.append(str)

#print(finnal_searchWordsList)

cv = ft.CountVectorizer()                           # 构建词袋模型
bow = cv.fit_transform(finnal_searchWordsList)      # 训练词袋模型
tt = ft.TfidfTransformer()                          # 获取TF-IDF模型训练器
tfidf_Inquire = tt.fit_transform(bow)               # 训练
Inquire_TfMat = tfidf_Inquire.toarray()
#print(tfidf.toarray())
Inquire_featureNames = (cv.get_feature_names_out()).tolist() #查询输入的所有的查询词列表，按字母升序
#print(Inquire_featureNames)
#print(len(Inquire_featureNames))
columns = len(Inquire_featureNames)
rows = TfMat.shape[0]
New_TfMat = np.zeros([rows,columns])    #去除了查询输入中没出现过的查询词的所有文档的TFIDF矩阵
for i in range (0,columns):
    New_TfMat[:,i] = TfMat[:,Doc_featureNames.index(Inquire_featureNames[i])]
#print(New_TfMat)
distances = scipy.spatial.distance.cdist(Inquire_TfMat, New_TfMat, "cosine")[0]
#print(distances)#nan分母为0，认为极大

#选10个最小的出来，结束
ListDistance = distances.tolist()
#print(distances.tolist())
#print(type(distances.tolist()))
#print(min(ListDistance))

OutPut_PATH = os.path.join(PROJECT_DIR_PATH, '../output/semantic_search_Result.txt')
fresult = open(OutPut_PATH,'w+')
fresult.write('与输入查询最相近的10条结果如下：\n')
print("与输入查询最相近的10篇文章为：")
for i in range(0,10):
    print(DocNameDict[ListDistance.index(min(ListDistance))])
    fresult.write(DocNameDict[ListDistance.index(min(ListDistance))])
    fresult.write('\n')
    ListDistance[ListDistance.index(min(ListDistance))] = 1

fresult.close()