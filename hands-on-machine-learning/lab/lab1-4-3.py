import matplotlib.pyplot as plt 
import pandas as pd 

from sklearn.linear_model import LinearRegression

# 데이터를 다운로드하고 준비합니다.
data_root = "https://github.com/ageron/data/raw/main/"

# CSV 파일 읽기
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")

# GDP per capita 데이터를 불러온 후, 변수에 저장
x = lifesat[["GDP per capita (USD)"]].values

# Life satisfaction 데이터를 불러온 후, 변수에 저장
y = lifesat[["Life satisfaction"]].values

# 데이터를 그래프로 나타냅니다.
# lifesat.plot : 데이터 프레임을 시각화
# kind = 'scatter' : 산점도 그래프 사용
# grid=True : 그래프에 격자를 추가
# x="GDP per capita (USD)" : x축의 이름을 GDP per capita (USD)로 설정
# y="Life satisfaction" : y축의 이름을 Life satisfaction으로 설정
lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")

# plt.axis([x_axis_start, x_axis_end, y_axis_start, y_axis_end]) : x축과 y축의 범위를 지정
plt.axis([23_500, 62_500, 4, 9])

# plt.show() : 그래프를 화면에 표시
plt.show()

# 선형 회귀 모델을 선택합니다.
model = LinearRegression()

# 모델을 훈련합니다.
model.fit(x, y)

# 키프로스에 대해 예측을 만듭니다.
# X_new는 예측하려는 새로운 입력 데이터
X_new = [[37_655.2]]  # 2020년 키프로스 1인당 GDP
print(model.predict(X_new))  # 출력: [[6.30165767]]

#해당 선형회귀 모델 생성 및 훈련, 예측 과정을 통해 GDP가 삶의 만족도에 어떤 영향을 미치는지 분석하고, 특정 국가의 GDP에 기반한 삶의 만족도 예측 가능
