import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.svm import LinearSVC
import time
import matplotlib.pyplot as plt
import numpy as np
import gensim 
from pyvi import ViTokenizer, ViPosTagger


#Phân loại một câu bình luận, 
def com_classify(comment_file_path, model_file_path, tfidf_vect_path):
    tfidf_vect = pickle.load(open(tfidf_vect_path, 'rb'))
    model = pickle.load(open(model_file_path, 'rb'))
    with open(comment_file_path, 'r', encoding= 'utf-8', errors = 'ignore') as f:
        comment =f.read().splitlines()
    comment = ' '.join(comment)
    comment = gensim.utils.simple_preprocess(comment)
    comment = ' '.join(comment)
    comment = ViTokenizer.tokenize(comment)
    # comment = [w for w in comment if w not in stopwords]
    # comment = ' '.join(comment)
    print("comment: ",comment)
    comment_tfidf = tfidf_vect.transform([comment])
    status = model.predict(comment_tfidf)
    if status[0] == 'pos':
        print("status of comment is: position")
    else:
        print("status of comment is: negative")
    

comment_file_path = "./data/Comment.txt"    #Chọn file chứa bình luận muốn phân loại
model_file_path = "./tfidf_finalized_model.sav"     #Chọn mô hình
tfidf_vect_path = "./tfidf_vect.sav"                #Chọn vector
com_classify(comment_file_path, model_file_path, tfidf_vect_path)