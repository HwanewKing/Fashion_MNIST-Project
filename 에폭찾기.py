from sklearn.datasets import fetch_openml
import numpy as np

fashion_mnist = fetch_openml('Fashion-MNIST',version=1, as_frame=False)
fashion_mnist.keys()
    
X,y = fashion_mnist["data"], fashion_mnist["target"]

y=y.astype(np.uint8) # X.dtype = float64 , y.dtype = uint8

X_train, X_test, y_train, y_test = X[:60000], X[60000:],y[:60000],y[60000:]

#nomalize
X_train=X_train/255.0
X_test=X_test/255.0

# SGDclassifier

from sklearn.linear_model import SGDClassifier

#model 만들기
model = SGDClassifier(loss = 'log') # loss 함수를 로지스틱 손실 함수로 지정
classes = np.unique(y_train)

import time

MaxData=[]
MaxIndex=[]
TimeData=[]

for n in range(20):
    
    TimeArray=[]
    test_score=[]
    
    for epoch in range(300):
        StartTime=time.time()
        model.partial_fit(X_train, y_train, classes=classes)
        EndTime=time.time()
        TimeToTrain=EndTime-StartTime
        TimeArray.append(TimeToTrain)
        test_score.append(model.score(X_test, y_test))
        
    mean=np.mean(TimeArray)
    maximum=max(test_score)
    index=test_score.index(maximum)
    MaxData.append(maximum)
    MaxIndex.append(index)
    TimeData.append(mean)

print("평균 정확도의 최대값 : ",np.mean(MaxData))
print("최대값이 나오는 평균적인 epoch 수 : ",np.mean(MaxIndex))
print("평균적으로 한 epoch 당 걸리는 시간 : ",np.mean(TimeData) ,"sec")