from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import time

fashion_mnist = fetch_openml('Fashion-MNIST',version=1, as_frame=False)
fashion_mnist.keys()
    
X,y = fashion_mnist["data"], fashion_mnist["target"]
# X is data
# y is label

y=y.astype(np.uint8) # X.dtype = float64 , y.dtype = uint8


labels=["T-shirt/top", #0
        "Trouser",     #1
        "Pullover",    #2
        "Dress",       #3
        "Coat",        #4
        "Sandal",      #5
        "Shirt",       #6
        "Sneaker",     #7
        "Bag",         #8
        "Ankle boot"]  #9

def IndexToLabel(index):
    return labels[index]


X_train, X_test, y_train, y_test = X[:60000], X[60000:],y[:60000],y[60000:]

#nomalize
X_train=X_train/255.0
X_test=X_test/255.0


# KNN, KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

test_score=[]
StartTime=time.time()
model=KNeighborsClassifier(4)
model.fit(X_train,y_train)
EndTime=time.time()
acc_test=model.score(X_test,y_test)
print("학습에 걸린 시간 : ",EndTime-StartTime)
print("test score : ",acc_test)



from PIL import Image

# 이미지 로드 및 전처리 함수
def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # 흑백 이미지로 변환
    image = image.resize((28, 28))  # 28x28 크기로 조정
    image_array = np.array(image)  # 이미지를 배열로 변환
    image_array = image_array.reshape(1, -1)  # 1차원 배열로 펼치기
    image_array = image_array / 255.0  # 정규화
    return image_array


def show_image(image_array, original_shape=(28, 28),title=""):
    image_array = image_array.reshape(original_shape)
    plt.imshow(image_array, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

# 이미지를 로드하고 전처리
image_path = "D:/trouser image.png"  # 실제 파일 경로로 변경

input_image = load_and_preprocess_image(image_path)


# 예측 함수
def predict_label(model, image_path):
    input_image = load_and_preprocess_image(image_path)
    predicted_label = model.predict(input_image)
    return IndexToLabel(predicted_label[0])


# 예측 및 출력
predicted_label = predict_label(model, image_path)
show_image(input_image,title=predicted_label)