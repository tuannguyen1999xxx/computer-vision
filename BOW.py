#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split


# In[2]:


start = datetime.datetime.now()
sift = cv2.xfeatures2d.SIFT_create()

# Đọc ảnh từ file:
# Cấu trúc file: Folder chính// các folder con được gán nhãn// tên từng ảnh
image_path_list = []
for label in os.listdir("BOW"):
    image_path_list.append(os.path.join("BOW", label))
    
des_list = []
des_list1 = []
labels = []
count = 0
for path in image_path_list:
    for file_name in os.listdir(path):
        count = count +1
        img = cv2.imread(os.path.join(path,file_name),2)
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp,des = sift.detectAndCompute(img,None)
        des_list1.append(des)
        labels.append(path)
        # Lấy 1/5 số lượng descriptors trích xuất được từ mỗi ảnh
        for i in range(len(des)):
            if i%5 == 0: 
                des_list.append(des[i])

end = datetime.datetime.now()
print(end-start)


# In[5]:



import pickle
with open('test2.pkl', 'wb') as handle:
    pickle.dump((labels), handle)


# In[4]:


start = datetime.datetime.now()
# Phân tất cả các descriptors thành 250 cụm (tùy số lượng cụm mình chọn, vector mã hóa cuối cùng sẽ có chiều = số cụm)
kmeans = KMeans(n_clusters=250).fit(des_list)
end = datetime.datetime.now()
print(end-start)


# In[5]:


visual_word = kmeans.cluster_centers_


# In[6]:

# Tạo histogram cho từng ảnh
def build_histogram(descriptor_list, cluster_alg):
    histogram = np.zeros(len(cluster_alg.cluster_centers_))
    # Phân các descriptors vào từng cụm và lấy tần suất xuất hiện của chúng.
    cluster_result = cluster_alg.predict(descriptor_list)
    for i in cluster_result:
        histogram[i] += 1.0
    return histogram


# In[9]:


preprocessed_image = []
label = []
# Đọc lại từng ảnh và mã hóa thành các vector histogram
# (phần này có thể tối ưu được, không cần trích xuất lại các descriptors vì ở trên đã trích xuất rồi nhưng mà lười làm :(( )

for path in image_path_list:
    for file_name in os.listdir(path):
        img = cv2.imread(os.path.join(path,file_name))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kp,des = sift.detectAndCompute(gray,None)
        if (des is not None):
            histogram = build_histogram(des, kmeans)
            preprocessed_image.append(histogram)
            label.append(path)
# Ảnh được mã hóa thành các vector có chiều dài tùy thuộc số lượng visual word mình chọn, bài này chọn 250 visual word(tương đương 250 cụm)

# In[10]:


from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Từ tập ảnh đã được mã hóa, chia thành các list train, test
X_train, X_test, y_train, y_test = train_test_split(preprocessed_image,label, test_size=0.3, random_state=0)

clf = neighbors.KNeighborsClassifier(n_neighbors = 10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy1 %.2f %%" %(100*accuracy_score(y_test, y_pred)))


# In[11]:


from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)
print("Accuracy2 %.2f %%" %(100*accuracy_score(y_test, y_predict)))


# In[12]:


#import pickle
with open('test1.pkl', 'wb') as handle:
    pickle.dump((kmeans,visual_word,preprocessed_image), handle)

