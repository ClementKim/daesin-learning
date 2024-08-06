import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score

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

# cv = 3 : 3-폴드 교차 검증 수행
# scoring = "accuracy" : 모델의 성능 지표로 정확도 사용
print(cross_val_score(sgd_clf, X_train, y_train, cv = 3, scoring = "accuracy"))

# 교차 검증 구현
skfolds = StratifiedKFold(n_splits = 3)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

# 더미 분류기로 만들어 평가
dummy_clf = DummyClassifier()
dummy_clf.fit(X_train, y_train)

print(any(dummy_clf.predict(X_train)))

print(cross_val_score(dummy_clf, X_train, y_train, cv = 3, scoring = "accuracy"))

## 오차 행렬 확인
# cross_val_predict()를 k-폴드 검증 수행해서 예측값 얻기
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3)
cm = confusion_matrix(y_train_5, y_train_pred)
print(cm)

y_train_perfect_predictions = y_train_5  # pretend we reached perfection
print(confusion_matrix(y_train_5, y_train_perfect_predictions))

## 정밀도와 재현율
print(precision_score(y_train_5, y_train_pred)) # == 3530 / (687 + 3530)

print(recall_score(y_train_5, y_train_pred)) # == 3530 / (1891 + 3530)

print(f1_score(y_train_5, y_train_pred))

## 정밀도 / 재현율 trade-off
# 결정 점수로 y_score 생성
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3, method = "decision_function")

threshold = 0
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)


# 임계값을 3000으로 설정
threshold = 3000
y_some_digit_pred = (y_scores > threshold)
print(y_some_digit_pred)

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# 결정임계값에 대한 정밀도와 재현율 그래프
plt.figure(figsize = (8, 4))
plt.plot(thresholds, precisions[:-1], "b--", label = "Precision", linewidth = 2)
plt.plot(thresholds, recalls[:-1], "g-", label = "Recall", linewidth = 2)
plt.vlines(threshold, 0, 1.0, "k", "dotted", label = "threshold")

# formatting
idx = (thresholds >= threshold).argmax()
plt.plot(thresholds[idx], precisions[idx], "bo")
plt.plot(thresholds[idx], recalls[idx], "go")
plt.axis([-50000, 50000, 0, 1])
plt.grid()
plt.xlabel("Threshold")
plt.legend(loc = "center right")
plt.show()

## 재현율에 대한 정밀도 곡선 그리기
plt.figure(figsize = (6, 5))
plt.plot(recalls, precisions, linewidth = 2, label = "Precision / Recall curve")

# 화살표와 점
plt.plot([recalls[idx], recalls[idx]], [0., precisions[idx]], "k:")
plt.plot([0.0, recalls[idx]], [precisions[idx], precisions[idx]], "k:")
plt.plot([recalls[idx]], [precisions[idx]], "ko", label = "Point at threshold 3,000")
plt.gca().add_patch(patches.FancyArrowPatch((0.79, 0.60), (0.61, 0.78), connectionstyle="arc3, rad=0.2",
                                            arrowstyle = "Simple, tail_width = 1.5, head_width = 8, head_length = 10",
                                            color = "#444444"))
plt.text(0.56, 0.62, "Higher\nthreshold", color = "#333333")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc = "lower left")

plt.show()


## 정밀도 90% 달성 코드
idx_for_90_precision = (precisions >= .90).argmax()
threshold_for_90_precision = thresholds[idx_for_90_precision]
print(threshold_for_90_precision)

y_train_pred_90 = (y_scores >= threshold_for_90_precision)

print(precision_score(y_train_5, y_train_pred_90))

recall_at_90_precision = recall_score(y_train_5, y_train_pred_90)
print(recall_at_90_precision)

## ROC 곡선
#여러 임계값에서 값계산하기
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


idx_for_threshold_at_90 = (thresholds <= threshold_for_90_precision).argmax()
tpr_90, fpr_90 = tpr[idx_for_threshold_at_90], fpr[idx_for_threshold_at_90]

#그래프 그리기
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, linewidth=2, label="ROC curve")
plt.plot([0, 1], [0, 1], 'k:', label="Random classifier's ROC curve")
plt.plot([fpr_90], [tpr_90], "ko", label="Threshold for 90% precision")
plt.xlabel('False Positive Rate (Fall-Out)')
plt.ylabel('True Positive Rate (Recall)')
plt.grid()
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower right", fontsize=13)

#화살표
plt.gca().add_patch(patches.FancyArrowPatch(
    (0.20, 0.89), (0.07, 0.70),
    connectionstyle="arc3,rad=.4",
    arrowstyle="Simple, tail_width=1.5, head_width=8, head_length=10",
    color="#444444"))
plt.text(0.12, 0.71, "Higher\nthreshold", color="#333333")

plt.show()

# AUC값 계산
print(roc_auc_score(y_train_5, y_scores))

# RandomForestClassifier를 만들어 PR곡선그리기
#모델 만들기
forest_clf = RandomForestClassifier(random_state = 42)


#양성클래스에 대한 확률을 점수로 사용하기
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv = 3, method = "predict_proba")

print(y_probas_forest[:2])

#PR곡선 그리기
y_scores_forest = y_probas_forest[:, 1]
precisions_forest, recalls_forest, thresholds_forest = precision_recall_curve(y_train_5, y_scores_forest )

plt.figure(figsize=(6, 5))
plt.plot(recalls_forest, precisions_forest, "b-", linewidth=2,
         label="Random Forest")
plt.plot(recalls, precisions, "--", linewidth=2, label="SGD")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.axis([0, 1, 0, 1])
plt.grid()
plt.legend(loc="lower left")

plt.show()

# RandomForestClassifier를 만들어  F1 계산하기
y_train_pred_forest = y_probas_forest[:, 1] >= 0.5
print(f1_score(y_train_5, y_train_pred_forest))

print(roc_auc_score(y_train_5, y_scores_forest))

print(precision_score(y_train_5, y_train_pred_forest))

print(recall_score(y_train_5, y_train_pred_forest))

