import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
import time
import matplotlib.pyplot as plt
import numpy as np
import gensim 
from pyvi import ViTokenizer, ViPosTagger

with open("./stopwords-nlp-vi.txt", 'r', encoding= 'utf-8') as f:
    stopwords = set([w.strip().replace(' ', '_') for w in  f.readlines()])

print("Loading data...")
X_data = pickle.load(open('data/X_data.pkl', 'rb'))
y_data = pickle.load(open('data/y_data.pkl', 'rb'))

X_test = pickle.load(open('data/X_test.pkl', 'rb'))
y_test = pickle.load(open('data/y_test.pkl', 'rb'))

X_val = pickle.load(open('data/X_val.pkl', 'rb'))
y_val = pickle.load(open('data/y_val.pkl', 'rb'))

print("loading data is done.")

#Vector hóa chuyển dữ liệu thành dạng ma trận thư tfidf với số chiều là (số điểm dữ liệu, 30000)
print("Tfidf vectorizing...")
tfidf_vect = TfidfVectorizer(analyzer = 'word', max_features = 30000)
tfidf_vect.fit(X_data)
X_data_tfidf = tfidf_vect.transform(X_data)
X_test_tfidf = tfidf_vect.transform(X_test)
X_val_tfidf = tfidf_vect.transform(X_val)
pickle.dump(tfidf_vect, open("tfidf_vect.sav", 'wb'))
print("Tfidf vectorizing is done")


#Chuyển label từ chữ sang ký tự 
encoder = LabelEncoder()
y_data_n = encoder.fit_transform(y_data)
y_test_n = encoder.fit_transform(y_test)
y_val_n = encoder.fit_transform(y_val)

print("Label")
print(encoder.classes_)




def train_model(classifier, X_data, y_data, X_val, y_val, X_test, y_test):
    t0 = time.time()
    #fit dữ liệu (training mô hình)
    classifier.fit(X_data, y_data)

    #Dự đoán 
    predicted_val = classifier.predict(X_val)
    predicted_test = classifier.predict(X_test)

    run_time = time.time() - t0
    #Tính accuracy scores
    test_accuracy = metrics.accuracy_score(y_test, predicted_test)
    val_accuracy = metrics.accuracy_score(y_val, predicted_val)

    print("time: ", run_time)
    print("test accuracy: ", test_accuracy)
    print("validation accuracy: ", val_accuracy)

    return run_time, test_accuracy, val_accuracy



test_score = []
val_score = []
run_time = []
clfs = [LinearSVC(), LogisticRegression(), BernoulliNB()]
name_model = ["Linear SVM", "Logistic Regression", "Bernoulli Naive Bayes"]
for i in range(len(clfs)):
    print("="*40)
    print("Model: ", name_model[i])
    t, test_accuracy, val_accuracy = train_model(clfs[i], X_data_tfidf, y_data_n, X_val_tfidf, y_val_n, X_test_tfidf, y_test_n)
    test_score.append(test_accuracy)
    val_score.append(val_accuracy)
    run_time.append(t)

def plot_result(test_score, val_score, run_time, name_model):
    test_score = np.array(test_score)
    val_score = np.array(val_score)
    run_time = np.array(run_time)
    index = np.arange(len(name_model))

    #Vẽ biểu đồ so sánh
    plt.bar(index, test_score, width = 0.3)
    plt.title("test accuracy with models")
    plt.xlabel("model",  fontsize = 10)
    plt.ylabel("Accuracy", fontsize = 10)
    plt.xticks(index, name_model, fontsize=10, rotation=15)
    for i in range(len(name_model)):
        plt.text(x= index[i], y = test_score[i] ,s= round(test_score[i], 4), fontsize= 10,horizontalalignment='center')
    plt.show()
    

    plt.bar(index, val_score, width = 0.3)
    plt.title("validation accuracy with models")
    plt.xlabel("model",  fontsize = 10)
    plt.ylabel("Accuracy", fontsize = 10)
    plt.xticks(index, name_model, fontsize=10, rotation=15)
    for i in range(len(name_model)):
        plt.text(x= index[i], y = val_score[i] ,s= round(val_score[i], 4), fontsize= 10,horizontalalignment='center')
    plt.show()
    

    plt.bar(index, run_time, width = 0.3)
    plt.title("Run time with models")
    plt.xlabel("Model",  fontsize = 10)
    plt.ylabel("Run time", fontsize =10)
    plt.xticks(index, name_model, fontsize=10, rotation=15)
    for i in range(len(name_model)):
        plt.text(x= index[i], y = run_time[i] ,s= round(run_time[i], 4), fontsize= 10,horizontalalignment='center')
    plt.show()



#Vẽ biểu đồ so sánh
plot_result(test_score, val_score, run_time, name_model)



#Chạy mô hình linear
clf = LinearSVC()
clf.fit(X_data_tfidf, y_data)
y_predictions = clf.predict(X_test_tfidf)
print("accuracy: ", metrics.accuracy_score(y_test, y_predictions))


#Lưu mô hình 
filename = "tfidf_finalized_model.sav"
pickle.dump(clf, open(filename, 'wb'))



