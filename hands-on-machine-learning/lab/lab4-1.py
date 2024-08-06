import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import add_dummy_feature

np.random.seed(42)
m = 100                               # 샘플 개수
X = 2 * np.random.rand(m, 1)          # 열 벡터
y = 4 + 3 * X + np.random.randn(m, 1) # 열 벡터

X_b = add_dummy_feature(X) # 각 샘플에 x0 = 1 추가
theta_best = np.linalg.inv(X_b.T @ X_b ) @ X_b.T @ y # A @ B = np.matmul(A, B)

X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new) # 각 샘플에 X0 = 1 추가
y_predict = X_new_b @ theta_best


plt.plot(X_new, y_predict, "r-", label = "prediction")
plt.plot(X, y, "b.")

# add label, axis, grid, range
plt.xlabel("$x_1$")
plt.ylabel("$y$", rotation=0)
plt.axis([0, 2, 0, 15])
plt.grid()
plt.legend(loc="upper left")

plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)
print(lin_reg.predict(X_new))

theta_best_svd, residuals, rank, s = np.linalg.lstsq(X_b, y, rcond=1e-6)
print(theta_best_svd)

