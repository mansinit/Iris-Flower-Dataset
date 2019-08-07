import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib

dataset=pd.read_csv("C:\\Users\\Mansi Dhingra\\PycharmProjects\\try\\copy\\iris.data")
x=dataset.iloc[:,0:4].values
y=dataset.iloc[:,[4]].values

dataset.head()

sn.countplot(x='classes',data=dataset)
matplotlib.use('Agg')  #When using Matplotlib versions older than 3.1, it is necessary to explicitly instantiate an Agg canvas
plt.savefig('count species in trained data.png')


species={'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3}
dataset=dataset.replace({'classes':species})
dataset.head()

dataset.info()

dataset.describe()

print(dataset.corr()[['classes']].sort_values(by='classes',ascending=False))


sn.regplot(x='pl',y='classes',data=dataset)
plt.savefig('dependance of petal length of iris flowers on prediction.png')

sn.regplot(x='pw',y='classes',data=dataset)
plt.savefig('dependance of petal width of iris flowers on prediction.png')

sn.regplot(x='sw',y='classes',data=dataset)
plt.savefig('dependance of sepal width of iris flowers on prediction.png')


sn.regplot(x='sl',y='classes',data=dataset)
plt.savefig('dependance of sepal length of iris flowers on prediction.png')

X_train=dataset.drop('classes',axis=1)
y_train=dataset[['classes']]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.2,random_state=18)

'''
LOGISTIC REGRESSION
 
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=18)
lr.fit(X_train,np.array(y_train).ravel())

y_pred=lr.predict(X_test)

 NAIVE BAYES
 
from sklearn.naive_bayes import GaussianNB
gb=GaussianNB()
gb.fit(X_train,y_train)

y_pred=gb.predict(X_test)
'''
#DECISION TREE CLASSIFICATION

from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(random_state=18)
dtc.fit(X_train,np.array(y_train))

y_pred=dtc.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
x=accuracy_score(y_test,y_pred)

