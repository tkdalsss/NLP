#!/usr/bin/env python
# coding: utf-8

# In[3]:


# 불용어를 제거한 BoW 만들기
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

# (1) 사용자가 직접 정의한 불용어 사용
text = ["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"]) # 직접 불용어 지정
print('bag of words vector :', vect.fit_transform(text).toarray())
print('vocabulary :', vect.vocabulary_)

# (2) CountVectorizer에서 제공하는 자체 불용어 사용
text = ["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words="english")
print('bag of words vector :', vect.fit_transform(text).toarray())
print('vocabulary :', vect.vocabulary_)

# (3) NLTK에서 지원하는 불용어 사용
text = ["Family is not an important thing. It's everything."]
stop_wrods = stopwords.words("english")


# In[ ]:




