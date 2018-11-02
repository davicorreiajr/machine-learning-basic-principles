from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

# load the data
train_data = pd.read_csv('train_data.csv',header=None)
train_labels = pd.read_csv('train_labels.csv',header=None)
test_data = pd.read_csv('test_data.csv',header=None)

# plot the histogram showing class distribution
hist_data = np.array(train_labels)
plt.hist(hist_data, bins=range(1,12,1))
plt.xlabel('Classes')
plt.ylabel('Occurences')
plt.title("Class distribution")
minor_ticks = np.arange(1, 10, 1)
plt.axis([1, 11, 0, 2500])

# split data into training and validation sets: 6/7 training and 1/7 validation
train_set, val_set, train_lbl, val_lbl = train_test_split(train_data, train_labels, test_size = 1/7, random_state = 0)

# standardize the data: mean = 0 and variance = 1
scaler = StandardScaler()
scaler.fit(train_set)
scaler.fit(test_data)
scaler.fit(val_set)
train_set = scaler.transform(train_set)
val_set = scaler.transform(val_set)
test_data = scaler.transform(test_data)

# apply PCA to the data: 
pca = PCA(.98) # choose the minimum number of principal components such that 98% of the variance is retained
pca.fit(train_set)
pca.fit(test_data)
val_set = pca.transform(val_set)
train_set = pca.transform(train_set)
test_data = pca.transform(test_data)

# apply logistic regression to the transformed data
logisticRegr = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=10000)
logisticRegr.fit(train_set, train_lbl.values.ravel())
print("Accuracy on the given validation data and labels is", logisticRegr.score(val_set, val_lbl)) #

# predict the labels of the test data
labels = logisticRegr.predict(test_data) # predict labels
liklhds = logisticRegr.predict_proba(test_data) # predict likelihoods

# write the results
ids = np.arange(1,len(labels)+1) # sample IDs
df = pd.DataFrame({"Sample_id" : ids, "Sample_label" : labels})
df.to_csv("labels.csv", index=False)
df = pd.DataFrame({"Sample_id" : ids, "Class_1" : liklhds[:,0], "Class_2" : liklhds[:,1], "Class_3" : liklhds[:,2], \
                    "Class_4" : liklhds[:,3], "Class_5" : liklhds[:,4], "Class_6" : liklhds[:,5], "Class_7" : liklhds[:,6], \
                    "Class_8" : liklhds[:,7], "Class_9" : liklhds[:,8], "Class_10" : liklhds[:,9]})
df.to_csv("liklhds.csv", index=False)
plt.show()