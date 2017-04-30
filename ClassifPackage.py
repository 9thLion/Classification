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

k=5
np.random.shuffle(Data.T) #Shuffle the order of the columns

#Save a test set for later
test = Data[:,:round(Data.shape[1]/10)+1]
V=test[test.shape[0]-1,:]
X = Data[:,round(Data.shape[1]/10)+1:]



#---------------------KNN---------------------
def KNN(train, validation, k=10, distance_metric='euclidean'):

	from sklearn import preprocessing

	#Isolate and standardize the numeric part
	length=train.shape[0]-1 #Remove the class part

	x = preprocessing.scale(train[:length].astype(float).T).T
	y = preprocessing.scale(validation[:length].astype(float).T).T
	KnnClasses=[]

	#Feature Selection should be added. Our dataset had only 7 features so it wasnt required

	#4 distance metric options
	import scipy.spatial.distance as scp
	def Dist(x,y):
		if distance_metric == 'euclidean':
			return(sum((x[:,i]-y[:,j])**2)**(1/2))
		if distance_metric == 'manhattan':
			return(sum(abs(x[:,i]-y[:,j])))
		if distance_metric == 'chebyshev':
			return(scp.chebyshev(x[:,i],y[:,j]))
		if distance_metric == 'cosine':
			return(scp.cosine(x[:,i],y[:,j]))

	for j in range(y.shape[1]):
		labels=[]
		dists=[]
		for i in range(x.shape[1]):
			labels.append(train[train.shape[0]-1,i])
			dists.append(Dist(x,y))
		labels = np.array(labels)
		dists = np.array(dists)
		indices = dists.argsort()[:k] #sort in ascending order
		#k are the indexes that match the lowest distance values
		labels = labels[indices] 
		
		#Keep the most frequent one
		#Count Votes
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

#Accuracy Test
def Acc(Known, New):
	return(New[New==Known].size / New.size)

#---------------------10-fold CV------------------
def KFold (X, K, method='KNN'):
	if K == X.shape[1]: #LOO validation, fold number equals total sample size
		folds = [X[:,i:i+1] for i in range(0, X.shape[1])]
	else: #KFold validation
		folds = [X[:,i:i+round(X.shape[1]/K)+1] for i in range(0, X.shape[1], round(X.shape[1]/K) + 1)] #The 10th array will be ~5 smaller
	#Use the commented function below to check that no sample is lost
	#       vvvv 
	#sum([folds[i].shape[1] for i in range(len(folds))])
	AccList = []
	for fold_i in range(K):
		#Pick another fold for validation in each iteration
		validation_fold = fold_i
		validation = folds[fold_i]
		#the train will consist of all the folds except the validation:
		#to do that we iterate over the range of fold indices and skip 
		#the one used on the validation
		train_folds = [folds[i] for i in range(len(folds)) if i !=validation_fold]
		train = np.concatenate(train_folds,axis=1)

		if method == 'KNN':
			K = KNN(train,validation,k=10)
			V = validation[validation.shape[0]-1,:]
			AccList.append(Acc(V,K))
		elif method == 'NaiveBayes':
			K = NaiveBayes(X=train,test=validation)
			V = validation[validation.shape[0]-1,:]
			AccList.append(Acc(V,K))
	return(sum(AccList)/len(AccList))

#10-fold validation
#print(KFold(X,10))

#Leave one out validation (just set the fold number to size of samples)
#print(KFold(X,K=X.shape[1]))

#print(KNN(X,test))


def NaiveBayes(X=X, test=test):
	#Lay the data on a dictionary, where each class will correspond to
	#the appropriate vectors
	separated = {}
	for i in range(X.shape[1]):
		v = X[:,i]
		if v[-1] not in separated:
			separated[v[-1]] = []
		separated[v[-1]].append(v[:-1].astype(float))
	#Each value of separated will be one sample

	#define two useful functions, a gaussian PDF and a product function
	def Gauss(x,mu,var):
		return( (1/(2*np.pi*var)**(1/2)) * np.exp(-((x-mu)**2)/(2*var)) )

	def Product(l):
		x = l[0]*l[1]
		i=2
		while True:
			if i == len(l):
				break
			x = x*l[i]
			i+=1
		return(x)

	#Keep a summary of the training data, hence means and variances 
	#of each feature for each class
	datasum = {}
	gg=[]
	marginals = {}
	for group in separated:
		datasum[group] = []
		#Make a temporary array, where the columns are the samples 
		#that belong to the class specified by the variable "group"
		temp = np.column_stack(separated[group])

		for feature in temp:
			m = sum(feature)/len(feature)
			v = np.var(feature)
			datasum[group].append((m,v))

		#change to np.array, column 1 will be the means, column 2 the variance
		datasum[group]=np.array(datasum[group])
		marginals[group]=(len(separated[group])/X.shape[1])

	#The total needs to be equal to 1:
	#sum(marginals.values())

	t = test[:-1,:].T.astype(float)
	NBclasses = []
	for x in t:
		check = 0
		for label in datasum.keys():
			temp = []
			index=0
			for i in x:
				#Calculate the PDF for the corresponding variance and mean
				#To parse properly, we increase the index in each loop
				Mean = datasum[label][index,0]
				Variance = datasum[label][index,1]

				if Variance == 0:
					#If the variance is 0 then this feature value should be 
					#either equal to the mean and have a high likelihood of
					#belonging to that class or be not equal and have a lower 
					#likelihood
					if Mean == i:
						Laplace = (len(separated['MIT'])+1)/len(separated.values())
						temp.append(Laplace)
					if Mean != i:
						Laplace = 1/len(separated.values())
						temp.append(Laplace)
				else:
					temp.append(Gauss(i,Mean,Variance))
				index+=1

			Prob = Product(temp)*marginals[label]
			if check == 0:
				check+=1
				PreviousHigh = Prob
				Chosen = label
				continue

			if Prob > PreviousHigh:
				PreviousHigh = Prob
				Chosen = label

		NBclasses.append(Chosen)
	return(np.array(NBclasses))

NBclasses = NaiveBayes(X=X,test=test)
Knnclasses = KNN(train=X,validation=test,k=5)
print(KFold(X=X, K=10, method = 'NaiveBayes'))