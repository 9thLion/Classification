import ClassifPackage as cla
import PCApackage as pac
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#os.system('wget http://mlearn.ics.uci.edu/databases/yeast/yeast.data')

temp=[]
vector=[]
with open('yeast.data') as f:
	for l in f:
		temp2=[]
		for x in l.split()[1:]:
			try:
				temp2.append(float(x))
			except ValueError: #Categorical Values, the classes
				vector.append(x)
				temp2.append(x)
		temp.append(temp2)

Data=np.array(temp).T
L=set(vector) #Labels

np.random.shuffle(Data.T) #Shuffle the order of the columns

#Save a test set for later
Split1 = Data[:,round(Data.shape[1]/10)+1:]
X = Split1[:-1,:].astype(float) #Data
L = Split1[-1,:] #Labels

Split2 = Data[:,:round(Data.shape[1]/10)+1]
test = Split2[:-1,:].astype(float)
V = Split2[-1,:]

#-----------------------------------
Data=np.array(temp).T

DataPC = pac.MyFirstAlgorithm(Data[:-1,:].astype(float), k=2)[0]
Labels = Data[-1,:]

fig = plt.figure()
ax = fig.add_subplot(111)
d={}
counter = 0
for x,l in zip(DataPC.T,Labels):
	if l in d.keys():
		ax.scatter(x[0], x[1], color=d[l]) 
	else:
		np.random.seed(counter)
		d[l] = (np.random.rand(),np.random.rand(),np.random.rand())
		ax.scatter(x[0], x[1], color=d[l], label=l) 
		counter+=3

plt.title('Data')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()


#-----------------------------------
#10-fold validation
K = list(range(1,19,3))

plt.title('KNN tuning')
plt.xlabel('K Hyperparameter')
plt.ylabel('Accuracy')
for dm in range(0,4): #4 different distance metrics

	configurations = []
	accuracies = []
	for i in K: #Different values of k
		#By appending at the same times, 
		#the index of the configuration will match
		#the correct index on the accuracy list.
		print('...')
		configurations.append('{}NN_{}'.format(i,dm)) 
		accuracies.append(cla.KFold(X, L, 10, method='KNN',hp=i, dist_metric=dm))
	#labels for the legend
	metrics = ['euclidean','manhattan','chebyshev','cosine'] 
	plt.plot(K, accuracies, label=metrics[dm])
plt.legend()
plt.show()

#Why is euclidean skipped if I use 4 distance metrics? if dm range is 0,3 it works 

configurations.append('NBayes') 
accuracies.append(cla.KFold(X, L, K=10,method='NaiveBayes'))

#Now to find the configuration with the maximum accuracy,
#this should be enough:
configurations = np.array(configurations)
accuracies = np.array(accuracies)
print(configurations[accuracies==max(accuracies)])

#Leave one out validation (just set the fold number equal to size of samples)
print(cla.KFold(X,L,K=X.shape[1],method='KNN',k=10, dist_metric='manhattan'))
print(cla.KFold(X,L,K=X.shape[1],method='NaiveBayes'))


#------------------SVM--------------
configurations = []
accuracies = []
C=[]
for i in range(1,500,50):
	configurations.append('LinearSVM_C{}'.format(i)) 
	accuracies.append(cla.KFold(X, L, K=10,method='SVM',hp=i))
	C.append(i)
plt.plot(C, accuracies)
plt.title('Linear SVM tuning')
plt.xlabel('C Hyperparameter')
plt.ylabel('Accuracy')
plt.show()
#--------------

TestLabels = cla.KNN(X, L, test, k=10, distance_metric='cosine')
print(cla.Acc(V, TestLabels))

model=svm.LinearSVC()
clf=model.fit(X.T,L)
print(cla.Acc(V, clf.predict(test.T)))


#NBclasses = NaiveBayes(X=X,test=test)
#Knnclasses = KNN(train=X,validation=test,k=5)
#print(KFold(X=X, K=10, method = 'NaiveBayes'))