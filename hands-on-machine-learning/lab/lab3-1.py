import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml

# MNIST 784 : Open ML에서 MNIST 데이터셋의 이름 또는 ID
# as_frame = False : numpy 배열 형식으로 반환
mnist = fetch_openml('mnist_784', as_frame = False)

# X에는 데이터셋의 feature를, y에 타겟 레이블 저장
X, y = mnist.data, mnist.target

## data_check
# imshow() : 변환된 2차원 배열 image 시각화
# cmap = "binary" : 이미지 흑백으로 표시
def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap = "binary")
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
plt.show()
