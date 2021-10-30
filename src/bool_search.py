import json
import os
from nltk import word_tokenize           # 分词、词性标注
from nltk.corpus import stopwords        # 停用词
from nltk.stem import PorterStemmer      # 词干提取
from nltk.stem import WordNetLemmatizer  # 词性还原


if __name__ == '__main__':
    InvertedIndex = {}
    SearchTerm = set()
    with open("../实验一查询词表.txt", "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            line_cg = PorterStemmer().stem(line)
            line_final = WordNetLemmatizer().lemmatize(line_cg, pos='v')
            SearchTerm.add(line_final)
            InvertedIndex[line_final] = set()
    PROJECT_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname('__file__'), os.path.pardir))
    DIR_PATH = os.path.join(PROJECT_DIR_PATH, 'dataset\\')
    files = os.listdir(DIR_PATH)
    # 定义标点符号列表
    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '|', '\'\'', '``', '/']
    DocNameDict = {}
    DocIndex = 0
    DocCompleteSet = set()#所有文档的全集
    for file in files:
        DocCompleteSet.add(file)
        DocNameDict[DocIndex] = file
        DocIndex += 1
        f = open(DIR_PATH + file, 'rb')
        x = json.load(f)
        tokens = word_tokenize(x['text'])
        # 定义标点符号列表
        tokens_x = [word for word in tokens if word not in interpunctuations]   # 去除标点符号
        stops = set(stopwords.words("english"))
        tokens_stops = [word for word in tokens_x if word not in stops]         # 去除停顿词
        tokens_cg = []
        for i in tokens_stops:
            tokens_cg.append(PorterStemmer().stem(i))    # 词干提取
        tokens_finial = []
        for i in tokens_cg:
            tokens_finial.append(WordNetLemmatizer().lemmatize(i, pos='v'))   # 指定还原词性为动词
        # 接下来生成倒排索引
        for i in tokens_finial:
            if i in SearchTerm:
                InvertedIndex[i].add(file)

    InvertedIndexOutPut_PATH = os.path.join(PROJECT_DIR_PATH, 'output/bool_search_InvertedIndex.txt')
    fInvertedIndex = open(InvertedIndexOutPut_PATH, 'w+')
    for key in InvertedIndex.keys():
        fInvertedIndex.write( key + ':  ' )
        if len(InvertedIndex[key]) != 0:
            fInvertedIndex.write(str(InvertedIndex[key]))
        fInvertedIndex.write('\n')
    fInvertedIndex.close()

    #######以下处理查询输入，返回输出结果###########
    ##检查是否为操作符,操作符返回1，操作数返回0
    def check_op(str):
        ops = ['(', ')', 'AND', 'NOT', 'OR']
        for op in ops:
            if op in str:
                return 1
        return 0
    ##检查操作符优先级
    def priority_op(input):
        if input == 'OR':
            return 1
        elif input == 'AND':
            return 2
        elif input == 'NOT':
            return 3
        elif input =='(':
            return 4
        elif input == ')':
            return 0
        else:
            return -1
    #处理查询输入
    with open("../布尔查询输入.txt", "r") as f:
        InputWords = f.read()
        f.close()
    SearchList = InputWords.split()
    #对查询词进行词干提取与词性还原
    for i in SearchList:
        if check_op(i) == 0:
            i_cg = PorterStemmer().stem(i)
            i_finial = WordNetLemmatizer().lemmatize(i_cg, pos='v')
            SearchList[SearchList.index(i)] = i_finial
    #print(SearchList)
    #符号栈和后缀结果栈
    SymbolStack = []
    FinialList = []
    for i in SearchList:
        if check_op(i) == 0:#操作数直接入栈
            FinialList.append(i)
        else: #操作符
            if len(SymbolStack) == 0:
                SymbolStack.append(i)
            else:
                if priority_op(i) > priority_op(SymbolStack[-1]): #新操作符优先级更高，入栈
                    SymbolStack.append(i)
                else:
                    if i == ')':#右括号，特殊处理
                        while SymbolStack[-1] != '(' and len(SymbolStack) != 0:
                            FinialList.append(SymbolStack[-1])
                            SymbolStack.pop()
                        SymbolStack.pop()
                    else:#新入操作数优先级更低，pop所有更高优先级的符号
                        while len(SymbolStack) != 0 and priority_op(i) <= priority_op(SymbolStack[-1]): #先判断是否为空，再判断优先级
                            FinialList.append(SymbolStack[-1])
                            SymbolStack.pop()
                        SymbolStack.append(i)
    while len(SymbolStack) != 0:
        FinialList.append(SymbolStack[-1])
        SymbolStack.pop()
    #print(FinialList)
    #操作数栈和查询结果集合
    OperandStack = []
    ResultSet = set()
    for i in FinialList:
        if check_op(i) == 0: #操作数
            OperandStack.append(InvertedIndex[i])
        else:
            if i == 'AND':
                set1 = OperandStack[-1]
                OperandStack.pop()
                set2 = OperandStack[-1]
                OperandStack.pop()
                OperandStack.append(set1.intersection(set2))
            elif i == 'OR':
                set1 = OperandStack[-1]
                OperandStack.pop()
                set2 = OperandStack[-1]
                OperandStack.pop()
                OperandStack.append(set1.union(set2))
            elif i == 'NOT':
                set1 = OperandStack[-1]
                OperandStack.pop()
                OperandStack.append(DocCompleteSet.intersection(set1))
    #print(OperandStack[0])

    OutPut_PATH = os.path.join(PROJECT_DIR_PATH, 'output/bool_search_Result.txt')
    fresult = open(OutPut_PATH, 'w+')
    fresult.write('满足输入的布尔查询结果如下：\n')
    print("满足输入的布尔查询结果如下：")
    print(OperandStack[0])
    fresult.write(str(OperandStack[0]))
    fresult.close()