import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay


# MNIST 784 : Open ML에서 MNIST 데이터셋의 이름 또는 ID
# as_frame = False : numpy 배열 형식으로 반환
mnist = fetch_openml('mnist_784', as_frame = False)
print('fetch done')

# X에는 데이터셋의 feature를, y에 타겟 레이블 저장
X, y = mnist.data, mnist.target

some_digit = X[0]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# 다중 분류 데이터셋에서 SGDClassifier를 훈련하고 예측 (OvR 전략 사용)
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train)
print('fit done')

sgd_clf.predict([some_digit])
print('predict done')


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype("float64"))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy", n_jobs=-1)
print('cross val score done')

# 오차 행렬 사용 : 정규화 X
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv = 3, n_jobs=-1)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred)
plt.show()

print('confusion matrix done')

# 각 값을 해당 클래스(True label)의 총 이미지 수로 나누어 오차 행렬 정규화
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, normalize = "true", values_format = ".0%")
plt.show()

print('confusion matrix done')

# 오차 행렬: 올바른 예측 제외
sample_weight = (y_train_pred != y_train)
ConfusionMatrixDisplay.from_predictions(y_train, y_train_pred, sample_weight = sample_weight, normalize = "true", values_format = ".0%")
plt.show()

print('confusion matrix done')
