from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score

# MNIST 784 : Open ML에서 MNIST 데이터셋의 이름 또는 ID
# as_frame = False : numpy 배열 형식으로 반환
mnist = fetch_openml('mnist_784', as_frame = False)

# X에는 데이터셋의 feature를, y에 타겟 레이블 저장
X, y = mnist.data, mnist.target

some_digit = X[0]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

## 서포트 벡터 머신 분류기
svm_clf = SVC(random_state = 42)
svm_clf.fit(X_train[:2000], y_train[:2000])

svm_clf.predict([some_digit])

some_digit_scores = svm_clf.decision_function([some_digit])
print(some_digit_scores.round(2))

class_id = some_digit_scores.argmax()
print(class_id)

# SVC 기반 OvR 전략 사용 다중 분류기
ovr_clf = OneVsRestClassifier(SVC(random_state = 42))
ovr_clf.fit(X_train[:2000], y_train[:2000])

# 훈련된 분류기 개수 확인 코드
ovr_clf.predict([some_digit])
print(len(ovr_clf.estimators_))

# 다중 분류 데이터셋에서 SGDClassifier를 훈련하고 예측 (OvR 전략 사용)
sgd_clf = SGDClassifier(random_state = 42)
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

# 10개 클래스의 이진 분류기가 각 클래스에 부여한 점수 확인
print(sgd_clf.decision_function([some_digit]).round())

# 모델 평가
print(cross_val_score(sgd_clf, X_train, y_train, cv = 3, scoring = "accuracy"))
