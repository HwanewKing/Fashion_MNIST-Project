from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import time

fashion_mnist = fetch_openml('Fashion-MNIST',version=1, as_frame=False)
fashion_mnist.keys()
    
X,y = fashion_mnist["data"], fashion_mnist["target"]
# X is data
# y is label

print(X.shape)
print(y.shape)
y=y.astype(np.uint8) # X.dtype = float64 , y.dtype = uint8
print(X.dtype)
print(y.dtype)


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

print(X_train[0])

print(np.unique(y_test, return_counts = True)) #각 레이블의 개수 파악

# 그림 예시 출력

fig, axs = plt.subplots(3, 3, figsize=(28,28))
fig.subplots_adjust(wspace=0.5, hspace=0.5)

X_train_image = []
X_train_index = []

for i in range(9): 
    X_train_image.append(X_train[18+i].reshape(28, 28))
    axs[i // 3, i % 3].imshow(X_train_image[i], cmap='binary')
    axs[i // 3, i % 3].axis('off')
    X_train_index.append(IndexToLabel(y_train[18+i]))
    axs[i // 3, i % 3].set_title(IndexToLabel(y_train[18+i]),fontsize=50)

plt.show()



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



# SGDclassifier
from sklearn.linear_model import SGDClassifier

#model 만들기
model = SGDClassifier(loss = 'log') # loss 함수를 log로 지정
test_score = []
classes = np.unique(y_train)

TimeArray=[]
TotalTime=0.00

#epoch n으로 학습 (n번 학습)
for epoch in range(300):
    StartTime=time.time()
    model.partial_fit(X_train, y_train, classes=classes)
    EndTime=time.time()
    TimeToTrain=EndTime-StartTime
    TimeArray.append(TotalTime+TimeToTrain)
    TotalTime=TotalTime+TimeToTrain
    test_score.append(model.score(X_test, y_test))

maximum=max(test_score)
print("학습에 걸린 시간 : ",TimeArray[-1],"sec")
print("test score : ",maximum)
print("index : ",test_score.index(maximum))



# 로지스틱 회귀
from sklearn.linear_model import LogisticRegression

StartTime=time.time()
model = LogisticRegression()
model.fit(X_train , y_train)
EndTime=time.time()
print("학습에 걸린 시간 : ",EndTime-StartTime,"sec")
print("test score: ",model.score(X_test , y_test))










