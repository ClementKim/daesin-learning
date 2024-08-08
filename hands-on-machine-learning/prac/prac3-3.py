import tarfile
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC

def load_titanic_data():
    tarball_path = Path("datasets/titanic.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as titanic_tarball:
            titanic_tarball.extractall(path="datasets")
    return [pd.read_csv(Path("datasets/titanic") / filename)
            for filename in ("train.csv", "test.csv")]

train_data, test_data = load_titanic_data()

'''
PassengerId: a unique identifier for each passenger
Survived: that's the target, 0 means the passenger did not survive, while 1 means he/she survived.
Pclass: passenger class.
Name, Sex, Age: self-explanatory
SibSp: how many siblings & spouses of the passenger aboard the Titanic.
Parch: how many children & parents of the passenger aboard the Titanic.
Ticket: ticket id
Fare: price paid (in pounds)
Cabin: passenger's cabin number
Embarked: where the passenger embarked the Titanic
'''
print(train_data.head())

# PassengerID를 인덱스로 설정
train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")

print(train_data.info())

# 여성의 연령 중앙값
print(train_data[train_data["Sex"]=="female"]["Age"].median())

# 분포 확인
print(train_data.describe())

print(train_data["Survived"].value_counts())

print(train_data["Pclass"].value_counts())

print(train_data["Sex"].value_counts())

'''
C: Cherbourg
Q: Queenstown
S: Southampton
'''
print(train_data["Embarked"].value_counts())


# 숫자형 데이터 전처리: 비어있는 값 중앙값으로 대체 및 스케일링
num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
])

# 범주형 데이터 전처리: 순서형 인코딩 후 one-hot 인코딩
cat_pipeline = Pipeline([
        ("ordinal_encoder", OrdinalEncoder()),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse_output=False)),
    ])


num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

# 학습 데이터 전처리
X_train = preprocess_pipeline.fit_transform(train_data)
print(X_train)

# Label
y_train = train_data["Survived"]

# Random Forest Classifier 사용 학습
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)

# 테스트 데이터 전처리 및 예측
X_test = preprocess_pipeline.transform(test_data)
y_pred = forest_clf.predict(X_test)

# 교차 검증
# cv=10: 10-폴드 교차 검증, 데이터셋을 10개의 폴드로 나누고, 각 폴드를 한 번씩 검증용 데이터로 사용하여 총 10번의 학습과 평가 수행
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
print(forest_scores.mean())

# SVC: SVM 분류기 생성
# gamma="auto" : 커널 함수에서 사용하는 감마 파라미터 1/n_features로 설정
svm_clf = SVC(gamma="auto")

# 교차 검증
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
print(svm_scores.mean())

# 그래프 생성
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))
plt.ylabel("Accuracy")
plt.show()
