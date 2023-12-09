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

# KNN, KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
knns=[]
test_score=[]

for i in range(2,10):
    KNC=KNeighborsClassifier(i)
    KNC.fit(X_train,y_train)
    print("이웃수 : {}인 모델 학습 완료".format(i))
    knns.append(KNC)
    
    n=2
for model in knns:
    acc_test=model.score(X_test,y_test)
    test_score.append(acc_test)
    print(n)
    print(acc_test)
    n=n+1
    
x=np.arange(2,10)

fig, ax1 = plt.subplots()
ax1.plot(x, test_score)
plt.xlabel('i')
plt.ylabel('moodel score')

plt.show()
