#!/usr/bin/env python
# coding: utf-8

# In[1]:


# CountVectorizer 클래스로 BoW 만들기
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()

# 코퍼스로부터 각 단어의 빈도수를 기록
print('bag of words vector :', vector.fit_transform(corpus).toarray())

# 각 단어의 인덱스가 어떻게 부여되었는지를 출력
print('vocabulary :', vector.vocabulary_)


# In[ ]:




