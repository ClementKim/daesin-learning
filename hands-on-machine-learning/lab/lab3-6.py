import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
from sklearn.multioutput import ClassifierChain

# MNIST 784 : Open ML에서 MNIST 데이터셋의 이름 또는 ID
# as_frame = False : numpy 배열 형식으로 반환
mnist = fetch_openml('mnist_784', as_frame = False)

# X에는 데이터셋의 feature를, y에 타겟 레이블 저장
X, y = mnist.data, mnist.target

some_digit = X[0]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_large = (y_train >= '7')
y_train_odd = (y_train.astype('int8') % 2 == 1)

# np.c_ : 두 배열을 합치는 명령어
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

print(knn_clf.predict([some_digit]))

y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)

# 각 레이블의 F1 점수를 구하고 (또는 어떤 이진 분류기 지표 사용) 평균 점수 계산
print(f1_score(y_multilabel, y_train_knn_pred, average="macro"))
print(f1_score(y_multilabel, y_train_knn_pred, average = "weighted"))

chain_clf = ClassifierChain(SVC(), cv = 3, random_state = 42)
chain_clf.fit(X_train[:2000], y_multilabel[:2000])

print(chain_clf.predict([some_digit]))