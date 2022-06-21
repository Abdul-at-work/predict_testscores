import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression




import xgboost as xgb


scores=pd.read_csv("test_scores.csv")


le = preprocessing.LabelEncoder()
scores["school"]=le.fit_transform(scores["school"])
scores["school_setting"]=le.fit_transform(scores["school_setting"])
scores["school_type"]=le.fit_transform(scores["school_type"])
scores["classroom"]=le.fit_transform(scores["classroom"])
scores["teaching_method"]=le.fit_transform(scores["teaching_method"])
scores["gender"]=le.fit_transform(scores["gender"])
scores["lunch"]=le.fit_transform(scores["lunch"])


target=scores["posttest"]
cols=['pretest']
data=scores[cols]
data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30)

df_sel_var = scores[['pretest', 'teaching_method', 'posttest']]
X = df_sel_var.drop(columns=['posttest'])
y = df_sel_var['posttest']
X_training_data, X_testing_data, y_training_data, y_testing_data = train_test_split(X, y, test_size=0.3)
final_model =  LinearRegression()
final_model.fit(X_training_data, y_training_data)
r_sq = final_model.score(X_training_data, y_training_data)
print('Coefficient of determination for training data:', r_sq)

gnb = GaussianNB()
#train the algorithm on training data and predict using the testing data
gnb.fit(X_training_data, y_training_data)
pred=gnb.predict(X_testing_data)
r_sq = gnb.score(X_training_data, y_training_data)
print('Coefficient of determination for training data:', r_sq)

#print(pred.tolist())
#print the accuracy score of the model
print("Naive-Bayes accuracy : ",accuracy_score(y_testing_data, pred, normalize = True))
print('Validation MAE', mean_absolute_error(target_test, pred))
print('Validation RMSE', sqrt(mean_squared_error(target_test, pred)))
print('Validation R^2', gnb.score(data_test, target_test))


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#create object of the lassifier
neigh = KNeighborsClassifier(n_neighbors=3)
#Train the algorithm
neigh.fit(X_training_data, y_training_data)
pred=neigh.predict(X_testing_data)
r_sq = neigh.score(X_training_data, y_training_data)
print('Coefficient of determination for training data:', r_sq)
pred=neigh.fit(data_train, target_train).predict(data_test)
# predict the response
# evaluate accuracy
print ("KNeighbors accuracy score : ",accuracy_score(target_test, pred))
print('Validation MAE', mean_absolute_error(target_test, pred))
print('Validation RMSE', sqrt(mean_squared_error(target_test, pred)))
print('Validation R^2', neigh.score(data_test, target_test))

svc_model = LinearSVC(random_state=0)
svc_model.fit(X_training_data, y_training_data)
pred=svc_model.predict(X_testing_data)
r_sq = svc_model.score(X_training_data, y_training_data)
print('Coefficient of determination for training data:', r_sq)

pred = svc_model.fit(data_train, target_train).predict(data_test)

print("LinearSVC accuracy : ",accuracy_score(pred,target_test, normalize = True))
print('Validation MAE', mean_absolute_error(target_test, pred))
print('Validation RMSE', sqrt(mean_squared_error(target_test, pred)))
print('Validation R^2', svc_model.score(data_test, target_test))

