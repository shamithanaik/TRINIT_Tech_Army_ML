import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob
import pickle
import numpy as np
import os
train_File = pd.read_csv("train_project.csv")
test_File=pd.read_csv("test_project.csv")
def text_splitter(text):

    return text.split()
tfidf = TfidfVectorizer(
                        min_df=0.00009,
                        max_features=2000,
                        smooth_idf=True,
                        norm="l2",
                        tokenizer = text_splitter,
                        sublinear_tf=False,
                        ngram_range=(1,3)
                       )

tfidf_train = tfidf.fit_transform(train_File.Description.apply(lambda x: np.str_(x)))
tfidf_test = tfidf.transform(test_File.Description)
train_label = train_File[['Commenting','Ogling/Facial Expressions/Staring','Touching /Groping']]
test_label = test_File[['Commenting','Ogling/Facial Expressions/Staring','Touching /Groping']]
with open("tfidf.pkl", "wb") as g:
    pickle.dump((tfidf), g)
x_train=tfidf_train
y_train=train_label
x_test=tfidf_test
y_test=test_label

classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)
y_pred=classifier.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)

with open("model.pkl", "wb") as f:
    pickle.dump((classifier, x_test, y_test), f)
y_train_pol=[]
y_test_pol=[]
for item in train_File.Description:
  blob=TextBlob(item)
  y_train_pol.append(blob.sentiment.polarity)
for item in test_File.Description:
    blob=TextBlob(item)
    y_test_pol.append(blob.sentiment.polarity)
for i in range(len(y_train_pol)):
  if y_train_pol[i]<=-0.4:
    y_train_pol[i]=1   #Sexual
  else:
    y_train_pol[i]=0   #Non-Sexual
for i in range(len(y_test_pol)):
  if y_test_pol[i]<=-0.4:
    y_test_pol[i]=1   #Sexual
  else:
    y_test_pol[i]=0   #Non-Sexual
clf1=RandomForestClassifier()
clf1.fit(x_train,y_train_pol)
with open("model_1.pkl", "wb") as i:
    pickle.dump((clf1), i)
clf2=DecisionTreeClassifier()
clf2.fit(x_train,y_train_pol)
with open("model_2.pkl", "wb") as j:
    pickle.dump((clf2), j)
clf3=SVC()
clf3.fit(x_train,y_train_pol)
with open("model_3.pkl", "wb") as k:
    pickle.dump((clf3), k)


