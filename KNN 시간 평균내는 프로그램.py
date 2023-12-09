from sklearn.datasets import fetch_openml
import numpy as np

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

import time

from sklearn.neighbors import KNeighborsClassifier

Times=[]
for i in range(100):
    StartTime=time.time()
    model=KNeighborsClassifier(4)
    model.fit(X_train,y_train)
    EndTime=time.time()
    print(i+1,": 학습에 걸린 시간 : ",EndTime-StartTime,"sec")
    Times.append(EndTime-StartTime)
    
average=np.mean(Times)
print("평균적으로 걸린 시간 : ",average,"sec")
