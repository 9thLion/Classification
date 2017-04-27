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
L=np.array(set(vector)) #Labels

k=5
np.random.shuffle(Data.T) #Shuffle the order of the columns

#Save a test set for later
test = Data[:,:round(Data.shape[1]/10)+1]
X = Data[:,round(Data.shape[1]/10)+1:]



#---------------------KNN---------------------
def KNN(train=train,validation=validation,length=X.shape[0]-1,k=30):
	#Isolate the numeric part
	x=train[:length].astype(float)
	y=validation[:length].astype(float)
	KnnClasses=[]

	for j in range(y.shape[1]):
		labels=[]
		dists=[]
		for i in range(x.shape[1]):
			labels.append(train[train.shape[0]-1,i])
			dists.append((sum((x[:,i]-y[:,j])**2))**(1/2))
		labels = np.array(labels)
		dists = np.array(dists)
		indices = dists.argsort()[:k] #sort in ascending order
		#k are the indexes that match the lowest distance values
		labels = labels[indices] 
		
		#Keep the most frequent one
		ClassVotes = {}
		for l in labels:
			if l in ClassVotes:
				ClassVotes[l]+=1
			else:
				ClassVotes[l]=1

		check = 0
		for c in ClassVotes.items():
			#To initialize
			if check == 0:
				check = 1
				FinalClass = c[0]
				previous = c[1]
				continue
			#Replace the label if the new one has more votes
			if c[1]>previous:
				FinalClass = c[0]
				previous = c[1]

		KnnClasses.append(FinalClass)
	KnnClasses=np.array(KnnClasses)
	return(KnnClasses)

K = KNN()

#Accuracy Test
def Acc(Known, New):
	return(New[New==Known].size / New.size)

Acc(V,K)

#---------------------10-fold CV------------------
def TenFold (X=X, method='KNN'):
	folds = [X[:,i:i+round(X.shape[1]/10)+1] for i in range(0, X.shape[1], round(X.shape[1]/10) + 1)] #The 10th array will be ~5 smaller

	#Use the commented function below to check that no sample is lost
	#       vvvv 
	#sum([folds[i].shape[1] for i in range(len(folds))])
	AccList = []
	for num in range(10):
		#Pick another fold for validation in each iteration
		validation_fold = num
		validation = folds[num]
		#the train will consist of all the folds except the validation:
		#to do that we iterate over the range of fold indices and skip 
		#the one used on the validation
		train_folds = [folds[i] for i in range(len(folds)) if i !=validation_fold]
		train = np.concatenate(train_folds,axis=1)

		if method == 'KNN':
			K = KNN(train,validation,X.shape[0]-1,k=15)
			V = validation[validation.shape[0]-1,:]
			AccList.append(Acc(V,K))
	return(sum(AccList)/len(AccList))

print(TenFold())