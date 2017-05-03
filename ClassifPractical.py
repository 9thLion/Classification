import ClassifPackage as cla
import os
import numpy as np

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

#10-fold validation
configurations = []
accuracies = []
for i in range(1,25): #Different values of k
	for dm in range(0,4): #4 different distance metrics
		configurations.append('{}NN_{}'.format(i,dm)) 
		accuracies.append(cla.KFold(X, L, 10, method='KNN',k=i, dist_metric=dm))
configurations.append('NBayes') 
accuracies.append(cla.KFold(X, L, K=10,method='NaiveBayes'))

#By appending at the same times, the index of the configuration will match
#the correct index on the accuracy list.
#Now to find the configuration with the maximum accuracy,
#this should be enough:
configurations = np.array(configurations)
accuracies = np.array(accuracies)
print(configurations[accuracies==max(accuracies)])

#Leave one out validation (just set the fold number equal to size of samples)
print(cla.KFold(X,L,K=X.shape[1],method='KNN',k=10, dist_metric='manhattan'))
print(cla.KFold(X,L,K=X.shape[1],method='NaiveBayes'))

#-----------

TestLabels = cla.KNN(X, L, test, k=10, distance_metric='manhattan')
print(cla.Acc(V, TestLabels))




#NBclasses = NaiveBayes(X=X,test=test)
#Knnclasses = KNN(train=X,validation=test,k=5)
#print(KFold(X=X, K=10, method = 'NaiveBayes'))