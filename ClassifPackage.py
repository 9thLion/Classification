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

X=np.array(temp).T
L=np.array(set(vector)) #Labels

k=5


#---------------------10-fold CV------------------
np.random.shuffle(X.T) #Shuffle the order of the columns
folds = [X[:,i:i+round(X.shape[1]/10)+1] for i in range(0, X.shape[1], round(X.shape[1]/10) + 1)] #The 10th array will be ~5 smaller

#Use the commented function below to check that no sample is lost
#       vvvv 
#sum([folds[i].shape[1] for i in range(len(folds))])

test_fold = 0
test = folds[0]
#the train will consist of all the folds except the test:
train_folds = [folds[i] for i in range(len(folds)) if i !=test_i]
train = np.concatenate(train_folds,axis=1)

#---------------------Distance---------------------
def euc(x,y,length=X.shape[0]-1):
	x=x[:length].astype(float)
	y=y[:length].astype(float)
	for i in range(length):
		return ((sum((x[i]-y[i])**2))**(1/2))

Dict = {}
for te in test.T:
	for tr in train.T:
		tr=tr[:X.shape[0]-1].astype(float)
		
r=2

for x in range(k):
	dist = euc(test[r])

np.argsort()

	