from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier

# MNIST 784 : Open ML에서 MNIST 데이터셋의 이름 또는 ID
# as_frame = False : numpy 배열 형식으로 반환
mnist = fetch_openml('mnist_784', as_frame = False)

# X에는 데이터셋의 feature를, y에 타겟 레이블 저장
X, y = mnist.data, mnist.target

some_digit = X[0]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train_5)
print(sgd_clf.predict([some_digit]))