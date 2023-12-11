from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = fetch_openml('Fashion-MNIST',version=1, as_frame=False)
fashion_mnist.keys()
    
X,y = fashion_mnist["data"], fashion_mnist["target"]
# X is data
# y is label

y=y.astype(np.uint8) # X.dtype = float64 , y.dtype = uint8


X_train, X_test, y_train, y_test = X[:60000], X[60000:],y[:60000],y[60000:]

#nomalize
X_train=X_train/255.0
X_test=X_test/255.0

from sklearn.linear_model import SGDClassifier
import time

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
print("test scores : ",maximum)
print("index : ",test_score.index(maximum))

x=np.arange(0,300)

fig, ax1 = plt.subplots()
ax1.plot(x, test_score)
plt.xlabel('epoch')
plt.ylabel('moodel score')

f,a=plt.subplots()
a.plot(x,TimeArray)
plt.xlabel('epoch')
plt.ylabel('time')

plt.show()