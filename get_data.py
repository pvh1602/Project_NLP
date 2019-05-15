from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tqdm import tqdm
import numpy as np
import gensim # thư viện NLP
import pickle
import os 


data_dir = "./foody_review"     
with open("./stopwords-nlp-vi.txt", 'r', encoding= 'utf-8') as f:
    stopwords = set([w.strip().replace(' ', '_') for w in  f.readlines()])



def get_data(data_path):
    X = []
    y = []
    dirs = os.listdir(data_path)    #Liệt kê thư mục trong data path ví dụ vs train_path ở dưới dirs = [neg, pos]
    for folder_path in tqdm(dirs):  #Duyệt dirs
        file_paths = os.listdir(os.path.join(data_path, folder_path))   #Lấy danh sách các file 
        for file_path in tqdm(file_paths):                              #Duyệt file 
            with open(os.path.join(data_path, folder_path, file_path), 'r', encoding = 'utf-8') as f:
                lines = f.readlines()   #Đọc từng dòng của file
                lines = ' '.join(lines)     #Ghép các dòng lại thành 1 đoạn
                lines = ViTokenizer.tokenize(lines)     #Tách từ văn bản
                lines = gensim.utils.simple_preprocess(lines)   #Chuyển chữ hoa thành chữ thường, loại bỏ dấu câu, chữ số
                lines = [word for word in lines if word not in stopwords]   #loại bỏ stopwords
                lines = ' '.join(lines)

                X.append(lines)
                y.append(folder_path)

    return X, y

train_path = os.path.join(data_dir, 'data_train/train')
X_data, y_data = get_data(train_path)

pickle.dump(X_data, open('data/X_data.pkl', 'wb'))
print("dump X_data done")
pickle.dump(y_data, open('data/y_data.pkl', 'wb'))
print("dump y_data done")


val_path = os.path.join(data_dir, 'data_train/val')
X_val, y_val = get_data(val_path)

pickle.dump(X_val, open('data/X_val.pkl', 'wb'))
print("dump X_val done")
pickle.dump(y_val, open('data/y_val.pkl', 'wb'))
print("dump y_val done")


test_path = os.path.join(data_dir, 'data_test/test')
X_test, y_test = get_data(test_path)

pickle.dump(X_test, open('data/X_test.pkl', 'wb'))
print("dump X_test done")
pickle.dump(y_test, open('data/y_test.pkl', 'wb'))
print("dump y_test done")



