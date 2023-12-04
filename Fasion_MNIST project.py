from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

fashion_mnist = fetch_openml('Fashion-MNIST',version=1, as_frame=False)
fashion_mnist.keys()

X,y = fashion_mnist["data"], fashion_mnist["target"]
# X data
# y label
print(X.shape)
print(y.shape)
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

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)
print(IndexToLabel(y[0]))

plt.imshow(some_digit_image, cmap='binary')
plt.axis('off')
plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:],y[:60000],y[60000:]

#nomalize
X_train=X_train/255.0
X_test=X_test/255.0

'''
# 로지스틱 회귀
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train , y_train)
print("train score: " ,model.score(X_train , y_train))
print("test score: ",model.score(X_test , y_test))
'''

# SGDclassifier
from sklearn.linear_model import SGDClassifier

#model 만들기
model = SGDClassifier(loss = 'log') # loss 함수를 log로 지정
train_score = []
test_score = []
classes = np.unique(y_train)

#epoch 300으로 학습 (300번 학습)
for epoch in range(20):
  model.partial_fit(X_train, y_train, classes=classes)
  train_score.append(model.score(X_train, y_train))
  test_score.append(model.score(X_test, y_test))


x=np.arange(0,20)

fig, ax1 = plt.subplots()
ax1.plot(x, train_score, color='r')
ax2 =ax1.twinx()
ax2.plot(x, test_score)
plt.xlabel('epoch')
plt.ylabel('moodel score')

plt.show()

print("train score: " ,train_score[-1])
print("test score: ",test_score[-1])