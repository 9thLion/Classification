import numpy as np

#It is important for the input to have random order

#-----------------------------------------------------
#-------------------------KNN-------------------------
#-----------------------------------------------------

def KNN(X, L, Xnew, k=10, distance_metric='euclidean'):

	from sklearn import preprocessing

	x = preprocessing.scale(X)
	y = preprocessing.scale(Xnew)
	KnnClasses=[]

	#4 distance metric options
	import scipy.spatial.distance as scp

	#Define a distance function, based on the argument given. I also use
	#Numbers to be able to iterate over the function when validating
	#to find the best distance metric
	def Dist(x,y):
		if distance_metric == 'euclidean' or distance_metric == 0:
			return(sum((x[:,i]-y[:,j])**2)**(1/2))
		if distance_metric == 'manhattan' or distance_metric == 1:
			return(sum(abs(x[:,i]-y[:,j])))
		if distance_metric == 'chebyshev' or distance_metric == 2:
			return(scp.chebyshev(x[:,i],y[:,j]))
		if distance_metric == 'cosine'or distance_metric == 3:
			return(scp.cosine(x[:,i],y[:,j]))

	for j in range(y.shape[1]):
		labels=[]
		dists=[]
		for i in range(x.shape[1]):
			labels.append(L[i])
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


#-----------------------------------------------------
#---------------------Naive Bayes---------------------
#-----------------------------------------------------

def NaiveBayes(X, L, Xnew):
	#Lay the data on a dictionary, where each class will correspond to
	#the appropriate vectors
	separated = {}
	for i in range(X.shape[1]):
		if L[i] not in separated:
			separated[L[i]] = []
		separated[L[i]].append(X[:,i])
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

	NBclasses = []
	for x in Xnew.T:
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

#----------------------Accuracy Test--------------------
def Acc(Known, New):
	return(New[New==Known].size / New.size)

#-------------------------------------------------------
#---------------------Cross Validation------------------
#-------------------------------------------------------

def KFold (X, L, K, method='KNN', hp=5, dist_metric='euclidean'):
	# K is the number of folds. If it is defined as equal to X.shape[1], then
	# it will run Leave One Out validation
	#hp is the hyperparameter for the selected method, 
	#(Naive Bayes doesnt have one)
	from sklearn import svm

	A = np.vstack((X,L))
	if K == A.shape[1]: #LOO validation, fold number equals total sample size
		folds = [A[:,i:i+1] for i in range(0, A.shape[1])]
	else: #KFold validation
		folds = [A[:,i:i+round(A.shape[1]/K)+1] for i in range(0, A.shape[1], round(A.shape[1]/K) + 1)] #The 10th array will be ~5 smaller
	#Use the commented function below to check that no sample is lost
	#       vvvv 
	#sum([folds[i].shape[1] for i in range(len(folds))])
	AccList = []
	for fold_i in range(K):
		#Pick another fold for validation in each iteration
		validation_fold = fold_i
		validation = folds[fold_i][:-1,:].astype(float)
		V = folds[fold_i][-1,:]
		#the train will consist of all the folds except the validation:
		#to do that we iterate over the range of fold indices and skip 
		#the one used on the validation
		train_folds = [folds[i] for i in range(len(folds)) if i !=validation_fold]
		train = np.concatenate(train_folds,axis=1)[:-1,:].astype(float)
		L = np.concatenate(train_folds,axis=1)[-1,:]
		
		if method == 'KNN':
			K = KNN(train, L, validation,k=hp, distance_metric=dist_metric)
			AccList.append(Acc(V,K))
		elif method == 'NaiveBayes':
			K = NaiveBayes(train, L, validation)
			AccList.append(Acc(V,K))
		elif method == 'SVM':
			model=svm.LinearSVC(C=hp)
			clf=model.fit(train.T,L)
			K = clf.predict(validation.T)
			AccList.append(Acc(V,K))

	return(sum(AccList)/len(AccList))
