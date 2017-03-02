Title: HR Analytics
Date: 2010-12-03 10:20
Category: Review

```python
import pandas as pd
import numpy as np
# import xgboost as xgb
from sklearn import preprocessing, neighbors, svm
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, \
  RandomForestRegressor, AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV

from sklearn.decomposition import PCA

%matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import sqrt

import itertools
from itertools import cycle

import pickle
DIR_DATA = "data"
DIR_PROCESSED = "processed"
LABEL = "rating"
NON_PREDICTORS = [LABEL]#, "name","anime_id"]
CV_FOLDS = 5
```


```python
# HR_comma_sep.csv

# Employee satisfaction level
# Last evaluation
# Number of projects
# Average monthly hours
# Time spent at the company
# Whether they have had a work accident
# Whether they have had a promotion in the last 5 years
# Department
# Salary
# Whether the employee has left
```


```python
hr = pd.read_csv(DIR_DATA + '/HR_comma_sep.csv')
hr.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>sales</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>sales</td>
      <td>low</td>
    </tr>
  </tbody>
</table>
</div>




```python
print('Percent who left: {:.2f}'.format(np.sum(hr.left) / len(hr.left) * 100))
```

    Percent who left: 23.81



```python
print(list(hr.sales.astype('category').cat.categories))
hr.sales = hr.sales.astype('category').cat.codes
hr.salary = hr.salary.astype('category').cat.codes
# hr.Work_accident = hr.Work_accident.astype('category')
# hr.promotion_last_5years = hr.promotion_last_5years.astype('category')
# hr.left = hr.left.astype('category')

```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



```python
# Department metrics don't work

# hr["is_4"] = hr.sales.apply(lambda x: x== 4)

# depProperties = hr.groupby('sales').agg({'promotion_last_5years':np.mean}).values
# hr["avg_per_team"] = hr.sales.apply(lambda x: depProperties[x])
```


```python
hr.salary.dtype
```




    dtype('int8')




```python
hr.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>satisfaction_level</th>
      <th>last_evaluation</th>
      <th>number_project</th>
      <th>average_montly_hours</th>
      <th>time_spend_company</th>
      <th>Work_accident</th>
      <th>left</th>
      <th>promotion_last_5years</th>
      <th>sales</th>
      <th>salary</th>
      <th>is_4</th>
      <th>avg_per_team</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.38</td>
      <td>0.53</td>
      <td>2</td>
      <td>157</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>False</td>
      <td>[0.024154589372]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.80</td>
      <td>0.86</td>
      <td>5</td>
      <td>262</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>False</td>
      <td>[0.024154589372]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.11</td>
      <td>0.88</td>
      <td>7</td>
      <td>272</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>2</td>
      <td>False</td>
      <td>[0.024154589372]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.72</td>
      <td>0.87</td>
      <td>5</td>
      <td>223</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>False</td>
      <td>[0.024154589372]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.37</td>
      <td>0.52</td>
      <td>2</td>
      <td>159</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>False</td>
      <td>[0.024154589372]</td>
    </tr>
  </tbody>
</table>
</div>




```python
def predict_left(df, clf, test_size=0.2):
    X = df.drop(['left'],1)
    y = df.left 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    
    clf.fit(X_train, y_train)
    
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    print('Training Score: {:.3f}'.format(clf.score(X_train, y_train)))
    print('Testing Score: %.3f' % (clf.score(X_test, y_test)))
    
    print()
```


```python
classifiers = [RandomForestClassifier(n_jobs=-1), RandomForestClassifier(criterion='entropy', n_jobs=-1), svm.SVC(), LogisticRegressionCV(), LinearRegression(), AdaBoostClassifier(),GradientBoostingClassifier(), neighbors.KNeighborsClassifier(n_jobs=-1)] #, GradientBoostingClassifier(),  neighbors.KNeighborsClassifier(n_jobs=-1)]
# classifiers = [RandomForestClassifier(n_estimators=500 ,n_jobs=-1), RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1), xgb.XGBClassifier(n_estimators=500, nthread=-1)]# svm.SVC()]

for i, clf in enumerate(classifiers):
    print('Classifier ', i)
    
    predict_left(hr, clf, test_size=0.4)
```

    Classifier  0
    Training Score: 0.998
    Testing Score: 0.988
    
    Classifier  1
    Training Score: 0.998
    Testing Score: 0.985
    
    Classifier  2
    Training Score: 0.956
    Testing Score: 0.944
    
    Classifier  3
    Training Score: 0.776
    Testing Score: 0.778
    
    Classifier  4
    Training Score: 0.200
    Testing Score: 0.189
    
    Classifier  5
    Training Score: 0.960
    Testing Score: 0.960
    
    Classifier  6
    Training Score: 0.979
    Testing Score: 0.975
    
    Classifier  7
    Training Score: 0.950
    Testing Score: 0.930
    



```python
classifiers = [RandomForestClassifier(n_jobs=-1), RandomForestClassifier(criterion='entropy', n_jobs=-1), svm.SVC(), LogisticRegressionCV(), LinearRegression(), AdaBoostClassifier(),GradientBoostingClassifier(), neighbors.KNeighborsClassifier(n_jobs=-1)] #, GradientBoostingClassifier(),  neighbors.KNeighborsClassifier(n_jobs=-1)]
# classifiers = [RandomForestClassifier(n_estimators=500 ,n_jobs=-1), RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1), xgb.XGBClassifier(n_estimators=500, nthread=-1)]# svm.SVC()]

for i, clf in enumerate(classifiers):
    print('Classifier ', i)
    
    predict_left(hr, clf)
```

    Classifier  0
    Training Score: 0.999
    Testing Score: 0.987
    
    Classifier  1
    Training Score: 0.998
    Testing Score: 0.988
    
    Classifier  2
    Training Score: 0.958
    Testing Score: 0.953
    
    Classifier  3
    Training Score: 0.775
    Testing Score: 0.763
    
    Classifier  4
    Training Score: 0.197
    Testing Score: 0.193
    
    Classifier  5
    Training Score: 0.959
    Testing Score: 0.962
    
    Classifier  6
    Training Score: 0.976
    Testing Score: 0.980
    
    Classifier  7
    Training Score: 0.953
    Testing Score: 0.929
    



```python
def cross_val_left(hr, clf, cv_folds=CV_FOLDS, drop=['left']):
    X = hr.drop(drop, 1)#, 'sales', 'salary'],1)
    y = hr.left 
    scores = cross_val_score(clf, X, y, cv=cv_folds, n_jobs=-1)#, scoring='roc_auc')
    
    
    
    print('Cross val score: ', sum(scores) / cv_folds )
    print(scores)
    
    print()
    
```


```python
classifiers = [RandomForestClassifier(n_estimators=500 ,n_jobs=-1), RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1)]#, svm.SVC()]#xgb.XGBClassifier(n_estimators=500, nthread=-1)]# svm.SVC()]

for i, clf in enumerate(classifiers):
    print('Classifier ', i)
    
    cross_val_left(hr, clf)
```

    Classifier  0
    Cross val score:  0.991866333178
    [ 0.9990003332  0.9806666667  0.9856666667  0.999333111   0.9946648883]
    
    Classifier  1
    Cross val score:  0.991666333163
    [ 0.998667111   0.9806666667  0.9853333333  0.9989996666  0.9946648883]
    



```python
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
X = hr.drop(['left'],1)
y = hr.left 
train_sizes, train_scores, valid_scores = learning_curve(clf, X, y, train_sizes=np.linspace(.1, 1.0, 10), cv=5, n_jobs=-1)

```


```python
print(valid_scores)

```

    [[ 0.9773408864  0.9776666667  0.9816666667  0.9943314438  0.9886628876]
     [ 0.9783405531  0.9783333333  0.9816666667  0.9963321107  0.9913304435]
     [ 0.9783405531  0.979         0.9826666667  0.9973324441  0.9929976659]
     [ 0.9783405531  0.979         0.9813333333  0.9976658886  0.9936645549]
     [ 0.9946684439  0.979         0.9823333333  0.9983327776  0.9939979993]
     [ 0.9993335555  0.9803333333  0.9823333333  0.999333111   0.9943314438]
     [ 0.9990003332  0.9803333333  0.985         0.9989996666  0.9946648883]]



```python
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
test_scores_std = np.std(valid_scores, axis=1)
train_sizes = np.linspace(.1,1.0,10)
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

plt.xlabel("Training examples")
plt.ylabel("Score")

plt.grid()
plt.legend(loc="best")
plt.show()
```


![png](HR%20Exploration_files/HR%20Exploration_15_0.png)



```python
train_scores_mean = np.mean(train_scores, axis=1)[3:] # 0.4 on
train_scores_std = np.std(train_scores, axis=1)[3:]
test_scores_mean = np.mean(valid_scores, axis=1)[3:]
test_scores_std = np.std(valid_scores, axis=1)[3:]
train_sizes = np.linspace(.1,1.0,10)[3:]

plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

plt.xlabel("Training examples")
plt.ylabel("Score")

plt.grid()
plt.legend(loc="best")
plt.show()
```


![png](HR%20Exploration_files/HR%20Exploration_16_0.png)



```python
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
X = hr.drop(['left'],1)
y = hr.left 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
clf.fit(X_train, y_train)
print(X.columns)
clf.feature_importances_
# Drop all lower than 0.01 relevance
```

    Index(['satisfaction_level', 'last_evaluation', 'number_project',
           'average_montly_hours', 'time_spend_company', 'Work_accident',
           'promotion_last_5years', 'sales', 'salary', 'is_4', 'avg_per_team'],
          dtype='object')





    array([ 0.32805276,  0.11853322,  0.18245216,  0.14757901,  0.186516  ,
            0.00783057,  0.00129782,  0.00892283,  0.00954625,  0.00080433,
            0.00846506])




```python

```


```python
# Dropping all with <0.01 relevance seems to not affect score much (.9920->.9916)
drop = ['left', 'promotion_last_5years', 'Work_accident', 'sales', 'salary']
for i, clf in enumerate(classifiers):
    print('Classifier ', i)
    
    cross_val_left(hr, clf, drop=drop)
```

    Classifier  0
    Cross val score:  0.99450795097
    [ 0.99989599  0.98623671  0.98705094  0.99963132  0.99972479]
    
    Classifier  1
    Cross val score:  0.994485333896
    [ 0.99990578  0.98579436  0.98744887  0.99962519  0.99965246]
    



```python
hr.corr()["left"]
```




    satisfaction_level      -0.388375
    last_evaluation          0.006567
    number_project           0.023787
    average_montly_hours     0.071287
    time_spend_company       0.144822
    Work_accident           -0.154622
    left                     1.000000
    promotion_last_5years   -0.061788
    sales                    0.032105
    salary                  -0.001294
    is_4                    -0.046035
    Name: left, dtype: float64




```python
# Drop everything with corr to left of < 0.005
# Makes it worse
drop = ['left',  'sales', 'salary']
classifiers = [RandomForestClassifier(n_estimators=500 ,n_jobs=-1), RandomForestClassifier(n_estimators=500, criterion='entropy', n_jobs=-1)]#, svm.SVC()]#xgb.XGBClassifier(n_estimators=500, nthread=-1)]# svm.SVC()]

for i, clf in enumerate(classifiers):
    print('Classifier ', i)
    
    cross_val_left(hr, clf, drop=drop)
```

    Classifier  0
    Cross val score:  0.994111289948
    [ 0.99968583  0.98524786  0.98646125  0.99962488  0.99953662]
    
    Classifier  1
    Cross val score:  0.994280805443
    [ 0.99987427  0.98537989  0.98719829  0.99943763  0.99951394]
    



```python
from sklearn.feature_selection import RFE
model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
clf = RFE(model,7 )

X = hr.drop(['left'],1)
y = hr.left 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
clf.fit(X_train, y_train)
print(clf.support_)
print(clf.ranking_)
# Gets same result as feature_importance, which makes sense
```

    [ True  True  True  True  True False False  True  True False False]
    [1 1 1 1 1 3 4 1 1 5 2]



```python
# from sklearn.feature_selection import RFE
# model = svm.SVC(kernel='linear')
# clf = RFE(model,5 )

# X = hr.drop(['left'],1)
# y = hr.left 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
# clf.fit(X_train, y_train)
# print(clf.support_)
# print(clf.ranking_)
# # Gets same result as feature_importance
```


```python
# PCA on relevant features
drop = ['left',  'sales', 'salary']

y = np.array(hr.left)

X = np.array(hr.drop(drop,1))
pca = PCA(n_components=2).fit(X)
X_pca = pca.transform(X)
```


```python
print(pca.components_)
```

    [[ -1.00023272e-04   1.16454622e-03   1.03016962e-02   9.99939260e-01
        3.73903656e-03  -7.14279268e-05  -1.02210886e-05   3.40923099e-06
       -1.97930557e-06]
     [ -2.15441037e-02   1.53800123e-02   2.72326123e-01  -6.42153324e-03
        9.61671473e-01   7.51616199e-04   6.30737234e-03   1.55475088e-02
        1.56786116e-03]]



```python
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
```

    [  9.98549061e-01   8.69441180e-04]
    0.999418502162



```python
# Still get good accuracy, ~97%
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
y = np.array(hr.left) 

scores = cross_val_score(clf, X_pca, y, cv=CV_FOLDS, n_jobs=-1)
    
    
    
print('Cross val score: ', sum(scores) / CV_FOLDS )
print(scores)
```

    Cross val score:  0.971332998837
    [ 0.97634122  0.953       0.956       0.98466155  0.98666222]



```python
colors = cycle('rb')
target_ids = range(2)
plt.figure()
for i, c, label in zip(target_ids, colors, ["stay","left"]):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                c=c, label=label)
plt.legend()
plt.show()
```


![png](HR%20Exploration_files/HR%20Exploration_28_0.png)



```python

```


```python

```


```python

```


```python

```


```python
# 3D PCA, Clear plane difference between them
drop = ['left',  'sales', 'salary']

y = np.array(hr.left)

X = np.array(hr.drop(drop,1))
pca = PCA(n_components=3).fit(X)
X_pca = pca.transform(X)

colors = cycle('rb')
target_ids = range(2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, c, label in zip(target_ids, colors, ["stay","left"]):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1],X_pca[y == i, 2],
                c=c, label=label)
    
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
    
plt.legend()
plt.show()
```

    /home/altock/anaconda3/envs/py35/lib/python3.5/site-packages/matplotlib/collections.py:865: RuntimeWarning: invalid value encountered in sqrt
      scale = np.sqrt(self._sizes) * dpi / 72.0 * self._factor



![png](HR%20Exploration_files/HR%20Exploration_33_1.png)



```python
# 3D accuracy is the same as 2D
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
y = hr.left 

scores = cross_val_score(clf, X_pca, y, cv=CV_FOLDS, n_jobs=-1)
    
    
    
print('Cross val score: ', sum(scores) / CV_FOLDS )
print(scores)
```

    Cross val score:  0.970999798837
    [ 0.97534155  0.95033333  0.957       0.98366122  0.98866289]



```python
# Attempt to find department level features, doesn't seem to matter
hr_corr = hr.corr()
hr_corr["sales"]
```




    satisfaction_level       0.003153
    last_evaluation          0.007772
    number_project           0.009268
    average_montly_hours     0.003913
    time_spend_company      -0.018010
    Work_accident            0.003425
    left                     0.032105
    promotion_last_5years   -0.027336
    sales                    1.000000
    salary                   0.000685
    is_4                    -0.136533
    Name: sales, dtype: float64




```python
hr.columns.values
```




    array(['satisfaction_level', 'last_evaluation', 'number_project',
           'average_montly_hours', 'time_spend_company', 'Work_accident',
           'left', 'promotion_last_5years', 'sales', 'salary', 'is_4',
           'avg_per_team'], dtype=object)




```python
for col in hr.columns:
    depProperties = hr.groupby('sales').agg({col:[np.size,np.mean]})
    print(depProperties)
```

          satisfaction_level          
                        size      mean
    sales                             
    0                 1227.0  0.618142
    1                  787.0  0.619822
    2                  767.0  0.582151
    3                  739.0  0.598809
    4                  630.0  0.621349
    5                  858.0  0.618601
    6                  902.0  0.619634
    7                 4140.0  0.614447
    8                 2229.0  0.618300
    9                 2720.0  0.607897
          last_evaluation          
                     size      mean
    sales                          
    0              1227.0  0.716830
    1               787.0  0.712122
    2               767.0  0.717718
    3               739.0  0.708850
    4               630.0  0.724000
    5               858.0  0.715886
    6               902.0  0.714756
    7              4140.0  0.709717
    8              2229.0  0.723109
    9              2720.0  0.721099
          number_project          
                    size      mean
    sales                         
    0               1227  3.816626
    1                787  3.853875
    2                767  3.825293
    3                739  3.654939
    4                630  3.860317
    5                858  3.687646
    6                902  3.807095
    7               4140  3.776329
    8               2229  3.803948
    9               2720  3.877941
          average_montly_hours            
                          size        mean
    sales                                 
    0                     1227  202.215974
    1                      787  200.800508
    2                      767  201.162973
    3                      739  198.684709
    4                      630  201.249206
    5                      858  199.385781
    6                      902  199.965632
    7                     4140  200.911353
    8                     2229  200.758188
    9                     2720  202.497426
          time_spend_company          
                        size      mean
    sales                             
    0                   1227  3.468623
    1                    787  3.367217
    2                    767  3.522816
    3                    739  3.355886
    4                    630  4.303175
    5                    858  3.569930
    6                    902  3.475610
    7                   4140  3.534058
    8                   2229  3.393001
    9                   2720  3.411397
          Work_accident          
                   size      mean
    sales                        
    0              1227  0.133659
    1               787  0.170267
    2               767  0.125163
    3               739  0.120433
    4               630  0.163492
    5               858  0.160839
    6               902  0.146341
    7              4140  0.141787
    8              2229  0.154778
    9              2720  0.140074
           left          
           size      mean
    sales                
    0      1227  0.222494
    1       787  0.153748
    2       767  0.265971
    3       739  0.290934
    4       630  0.144444
    5       858  0.236597
    6       902  0.219512
    7      4140  0.244928
    8      2229  0.248991
    9      2720  0.256250
          promotion_last_5years          
                           size      mean
    sales                                
    0                      1227  0.002445
    1                       787  0.034307
    2                       767  0.018253
    3                       739  0.020298
    4                       630  0.109524
    5                       858  0.050117
    6                       902  0.000000
    7                      4140  0.024155
    8                      2229  0.008973
    9                      2720  0.010294
          sales     
           size mean
    sales           
    0      1227    0
    1       787    1
    2       767    2
    3       739    3
    4       630    4
    5       858    5
    6       902    6
    7      4140    7
    8      2229    8
    9      2720    9
          salary          
            size      mean
    sales                 
    0       1227  1.368378
    1        787  1.407878
    2        767  1.340287
    3        739  1.424899
    4        630  1.000000
    5        858  1.344988
    6        902  1.349224
    7       4140  1.363043
    8       2229  1.359354
    9       2720  1.347794
           is_4       
           size   mean
    sales             
    0      1227  False
    1       787  False
    2       767  False
    3       739  False
    4       630   True
    5       858  False
    6       902  False
    7      4140  False
    8      2229  False
    9      2720  False



    ---------------------------------------------------------------------------

    DataError                                 Traceback (most recent call last)

    <ipython-input-29-c7f435eb979a> in <module>()
          1 for col in hr.columns:
    ----> 2     depProperties = hr.groupby('sales').agg({col:[np.size,np.mean]})
          3     print(depProperties)


    /home/altock/anaconda3/envs/py35/lib/python3.5/site-packages/pandas/core/groupby.py in aggregate(self, arg, *args, **kwargs)
       3702     @Appender(SelectionMixin._agg_doc)
       3703     def aggregate(self, arg, *args, **kwargs):
    -> 3704         return super(DataFrameGroupBy, self).aggregate(arg, *args, **kwargs)
       3705 
       3706     agg = aggregate


    /home/altock/anaconda3/envs/py35/lib/python3.5/site-packages/pandas/core/groupby.py in aggregate(self, arg, *args, **kwargs)
       3191 
       3192         _level = kwargs.pop('_level', None)
    -> 3193         result, how = self._aggregate(arg, _level=_level, *args, **kwargs)
       3194         if how is None:
       3195             return result


    /home/altock/anaconda3/envs/py35/lib/python3.5/site-packages/pandas/core/base.py in _aggregate(self, arg, *args, **kwargs)
        547 
        548                 try:
    --> 549                     result = _agg(arg, _agg_1dim)
        550                 except SpecificationError:
        551 


    /home/altock/anaconda3/envs/py35/lib/python3.5/site-packages/pandas/core/base.py in _agg(arg, func)
        498                 result = compat.OrderedDict()
        499                 for fname, agg_how in compat.iteritems(arg):
    --> 500                     result[fname] = func(fname, agg_how)
        501                 return result
        502 


    /home/altock/anaconda3/envs/py35/lib/python3.5/site-packages/pandas/core/base.py in _agg_1dim(name, how, subset)
        481                     raise SpecificationError("nested dictionary is ambiguous "
        482                                              "in aggregation")
    --> 483                 return colg.aggregate(how, _level=(_level or 0) + 1)
        484 
        485             def _agg_2dim(name, how):


    /home/altock/anaconda3/envs/py35/lib/python3.5/site-packages/pandas/core/groupby.py in aggregate(self, func_or_funcs, *args, **kwargs)
       2654         if hasattr(func_or_funcs, '__iter__'):
       2655             ret = self._aggregate_multiple_funcs(func_or_funcs,
    -> 2656                                                  (_level or 0) + 1)
       2657         else:
       2658             cyfunc = self._is_cython_func(func_or_funcs)


    /home/altock/anaconda3/envs/py35/lib/python3.5/site-packages/pandas/core/groupby.py in _aggregate_multiple_funcs(self, arg, _level)
       2716                 obj._reset_cache()
       2717                 obj._selection = name
    -> 2718             results[name] = obj.aggregate(func)
       2719 
       2720         if isinstance(list(compat.itervalues(results))[0],


    /home/altock/anaconda3/envs/py35/lib/python3.5/site-packages/pandas/core/groupby.py in aggregate(self, func_or_funcs, *args, **kwargs)
       2658             cyfunc = self._is_cython_func(func_or_funcs)
       2659             if cyfunc and not args and not kwargs:
    -> 2660                 return getattr(self, cyfunc)()
       2661 
       2662             if self.grouper.nkeys > 1:


    /home/altock/anaconda3/envs/py35/lib/python3.5/site-packages/pandas/core/groupby.py in mean(self, *args, **kwargs)
       1017         nv.validate_groupby_func('mean', args, kwargs)
       1018         try:
    -> 1019             return self._cython_agg_general('mean')
       1020         except GroupByError:
       1021             raise


    /home/altock/anaconda3/envs/py35/lib/python3.5/site-packages/pandas/core/groupby.py in _cython_agg_general(self, how, numeric_only)
        806 
        807         if len(output) == 0:
    --> 808             raise DataError('No numeric types to aggregate')
        809 
        810         return self._wrap_aggregated_output(output, names)


    DataError: No numeric types to aggregate



```python
clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
X = hr.drop(drop,1)
y = hr.left 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```


```python
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=10)

class_names = ["Stay","Left"]

plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix')


plt.show()
```


![png](HR%20Exploration_files/HR%20Exploration_39_0.png)



```python
print(classification_report(y_test,y_pred, target_names=class_names))
```

                 precision    recall  f1-score   support
    
           Stay       0.99      1.00      0.99      2278
           Left       0.99      0.96      0.98       722
    
    avg / total       0.99      0.99      0.99      3000
    



```python

```
