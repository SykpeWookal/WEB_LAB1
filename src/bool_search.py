import json
import nltk
import os

from nltk import word_tokenize,pos_tag   #分词、词性标注
from nltk.corpus import stopwords    #停用词
from nltk.stem import PorterStemmer    #词干提取
from nltk.stem import WordNetLemmatizer    #词性还原

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
DIR_PATH = os.path.join(PROJECT_DIR_PATH, '../US_Financial_News_Articles/TestSource')
files = os.listdir(DIR_PATH)



interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '|', '\'\'', '``', '/']   #定义标点符号列表

DocNameDict = {}
DocIndex = 0

for file in files:
    DocNameDict[DocIndex] = file
    DocIndex += 1
    f = open('../US_Financial_News_Articles/TestSource/' + file, 'rb')
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

    #接下来生成倒排索引
    for i in tokens_finial:
        #print(i)
        if i in SearchTerm:
            InvertedIndex[i].add(file)

print(InvertedIndex)
