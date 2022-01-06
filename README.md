![architecture](./simMatFigure.png)

This code is part of the paper: 본문-질의 비교를 활용한 오답 질의 분류 http://koreascience.or.kr/article/CFKO201930060759842.pdf

This code compares query and context by similarity matrix and classify whether is solvable or not.

#embedding dir<br />
data/glove/300d.txt<br />
<br /><br />
#data dir<br />
data/tenFold/test_0<br />
data/tenFold/train_0<br />
<br /><br />
#data format<br />
qid\tquery\tT\tsource.txt<br />
qid\tquery\tF\tsource.txt,source2.txt
