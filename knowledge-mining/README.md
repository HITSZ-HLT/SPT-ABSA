## 代码结构

数据标注分为4步：
1. `spacy_tokenize.py` 使用spacy解析文本
2. 挖掘单词情感极性
    1. `build_vocab_for_polarity_assign.py` 在大量文本语料上构建词典
    2. `polarity_assign.py` 为语料中的所有形容词赋予情感极性
3. 挖掘aspect和opinion词
    1. `build_vocab_pathmanager.py` 为500k文本构建词典和路径词典
    2. `mine.py` 运行情感知识挖掘算法
4. `annotate.py` 通过上两步的挖掘结果，对文本进行知识标注

具体的命令，见`make_bash.py`文件。
