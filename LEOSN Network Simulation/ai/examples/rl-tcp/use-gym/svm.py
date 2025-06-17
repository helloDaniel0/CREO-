import numpy as np
import torch
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random


ori_data = []
Y = []
for i in range(4000):
    data = []
    label = False
    for j in range(10):
        mean = 10  # 均值
        std_dev = 1  # 标准差
        random_vector = np.random.normal(loc=mean, scale=std_dev, size=8)
        random_float = random.random()
        if random_float < 0.05:
            random_vector[5] = 0
            random_vector[6] *= 2
            random_vector[7] *= 10
            label = True
        data.append(random_vector)
    if label is False:
        Y.append(0)
    else:
        Y.append(1)
    ori_data.append(np.array(data).reshape(1,-1))

gen_data = np.array(ori_data).reshape(4000, 80)
y = np.array(Y)

print(gen_data.shape)
print(y.shape)


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(gen_data, y, test_size=0.2, random_state=42)

# 创建SVM分类器
classifier = svm.SVC(kernel='linear')

# 训练模型
classifier.fit(X_train, y_train)

# 进行预测
y_pred = classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

