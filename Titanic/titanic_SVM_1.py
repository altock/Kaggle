import numpy as np
import pandas as pd

from sklearn import preprocessing, cross_validation, neighbors, svm
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.cluster import KMeans
from scipy.stats import mode
import csv as csv


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def handle_non_numerical_data(df): 
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)

            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int,df[column] ))
    
    return df



def cross_train_and_test(clf,X,y):
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)


	# test
	train_accuracy = 0.0
	test_accuracy = 0.0
	

	n = 40

	
	for i in range(n):
		X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

		clf.fit(X_train, y_train)
		train_accuracy += clf.score(X_train, y_train)
		test_accuracy += clf.score(X_test, y_test)

	train_accuracy /= n
	test_accuracy /= n
	print(train_accuracy, test_accuracy)



def prepare_dataframe(df):
	# Deal with empty

	# All missing Embarked -> just make them embark from most common place
	if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
		df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values

	# All the ages with no data -> make the median of all Ages
	median_age = df['Age'].dropna().median()
	if len(df.Age[ df.Age.isnull() ]) > 0:
		df.loc[ (df.Age.isnull()), 'Age'] = median_age


	# All the missing Fares -> assume median of their respective class
	if len(df.Fare[ df.Fare.isnull() ]) > 0:
		median_fare = np.zeros(3)
		for f in range(0,3):
			# loop 0 to 2
			median_fare[f] = df[ df.Pclass == f+1 ]['Fare'].dropna().median()
		for f in range(0,3):
			# loop 0 to 2
			df.loc[ (df.Fare.isnull()) & (df.Pclass == f+1 ), 'Fare'] = median_fare[f]

	# Assume room number on floor does not matter, change cabin to just be the floor
	df.Cabin[ df.Cabin.notnull()] = df.Cabin[ df.Cabin.notnull()].str[0]
	# All the missing Cabins -> assume median of their respective class
	if len(df.Cabin[ df.Cabin.isnull() ]) > 0:
		median_fare = ["","",""]
		for f in range(0,3):
			# loop 0 to 2
			z = df[ df.Pclass == f+1 ]['Cabin'].dropna().value_counts().index.values[0]#.median().astype(str)
			median_fare[f] = z
		for f in range(0,3):
			# loop 0 to 2
			df.loc[ (df.Cabin.isnull()) & (df.Pclass == f+1 ), 'Cabin'] = median_fare[f]


	# get rid of useless fields
	df.drop(['Name', 'PassengerId'], 1, inplace=True)
	df.convert_objects(convert_numeric=True)
	df.fillna(0,inplace=True)

	df = handle_non_numerical_data(df)
	return df

def get_prediction(X,y, test_X):
	
	svm_kernels = [ "rbf"]
	accuracies = {x : [0.0,0.0, [],[]]  for x in svm_kernels}
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)
	mat = []
	for kernel in svm_kernels:
		clf = BaggingClassifier(svm.SVC(kernel=kernel, C = 2))
		model = BaggingClassifier(svm.SVC(kernel=kernel, C = 2))

		clf.fit(X_train, y_train)
		model.fit(X,y)

		accuracies[kernel][0] += clf.score(X_train, y_train)
		accuracies[kernel][1] += clf.score(X_test, y_test)

		accuracies[kernel][2] = clf.predict(X_train)
		accuracies[kernel][3] = clf.predict(X_test)

		print(kernel, ": train accuracy", accuracies[kernel][0], ", test accuracy", accuracies[kernel][1])
		mat.append(model.predict(test_X).astype(int))
		# mat = [np.array(x[2]) for x in accuracies]

	clf = BaggingClassifier( neighbors.KNeighborsClassifier(n_jobs=-1))
	model =  BaggingClassifier( neighbors.KNeighborsClassifier(n_jobs=-1))
	clf.fit(X_train, y_train)
	model.fit(X,y)
	test_accuracy = clf.score(X_test, y_test)
	print("KNN Accuracy: ", test_accuracy)

	test_output = model.predict( test_X).astype(int)


	mat.append(test_output)


	forest = RandomForestClassifier(n_estimators=200)
	model = RandomForestClassifier(n_estimators=200)
	model.fit(X,y)
	forest = forest.fit( X_train, y_train )


	test_output = model.predict( test_X).astype(int)


	mat.append(test_output)

	accuracy = forest.score(X_test,y_test)
	print("Forest accuracy: ", accuracy)

	weights = np.array([x[1] for x in accuracies.values()])
	weights = np.append(weights,accuracy)
	weights = np.append(weights,test_accuracy)
	print(weights)
	print(np.average(mat, axis = 0, weights = weights))
	print(np.average(mat, axis = 0))
	mat = np.array(np.average(mat, axis = 0, weights = weights)) #basically mode
	print(mat.shape)
	mat = np.round(mat).astype(int)
	print("Shape : ", mat)
	return mat


def get_prediction_new(X,y,test_X):
	num_folds = 10
	num_instances = len(X)
	seed = 7
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	# create the sub models
	estimators = []
	model1 = LogisticRegression()
	estimators.append(('logistic', model1))
	model2 = DecisionTreeClassifier()
	estimators.append(('cart', model2))
	model3 = svm.SVC(C=2)
	estimators.append(('svm', model2))
	forest = RandomForestClassifier(n_estimators=200)
	estimators.append((('forest'),forest))
	clf = neighbors.KNeighborsClassifier(n_jobs=-1)
	estimators.append(('KNN',clf))
	# create the ensemble model
	ensemble = VotingClassifier(estimators)
	results = cross_validation.cross_val_score(ensemble, X, y, cv=kfold)
	print(results.mean())
	ensemble.fit(X,y)
	return ensemble.predict(test_X)


if __name__ == '__main__':

	# set up training data

	df_train = pd.read_csv('data/train.csv')

	df_train = prepare_dataframe(df_train)

	X_train = np.array(df_train.drop(['Survived'], 1).astype(float))
	X_train = preprocessing.scale(X_train)
	y_train = np.array(df_train['Survived'])
	


	# test
	df_test = pd.read_csv('data/test.csv')

	PassengerId = df_test['PassengerId'].values
	df_test = prepare_dataframe(df_test)




	df_test = handle_non_numerical_data(df_test)


	X_test = np.array(df_test.astype(float))
	X_test = preprocessing.scale(X_test)


	output = get_prediction_new(X_train, y_train, X_test)


	predictions_file = open("data/titanicSVM1.csv", "w")
	open_file_object = csv.writer(predictions_file)
	open_file_object.writerow(["PassengerId","Survived"])
	open_file_object.writerows(zip(PassengerId, output))
	predictions_file.close()
	print('Done.')

