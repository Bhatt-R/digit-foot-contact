import math
from turtle import color
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression


col_names=['lf_grf_z','rf_grf_z','lf_spring0','lf_spring1','rf_spring0','rf_spring1','lf_label','rf_label','imu_z','imu_y','imu_x','time']
data = pd.read_csv(r"C:\Users\rajar\OneDrive\Desktop\digit_data_updated.csv", header=0, usecols=col_names)

feature_cols_lf = ['lf_spring0','imu_z','imu_y','imu_x']
feature_cols_rf = ['rf_spring0','imu_z','imu_y','imu_x']


Xlf = data[feature_cols_lf]
ylf = data.lf_label
Xrf = data[feature_cols_rf]
yrf = data.rf_label
Xlf_train, Xlf_test, ylf_train, ylf_test = train_test_split(Xlf, ylf, test_size=0.2, random_state=16)
Xrf_train, Xrf_test, yrf_train, yrf_test = train_test_split(Xrf, yrf, test_size=0.2, random_state=16)

#-----------------Logistic Regression----------------
clf = LogisticRegression(random_state=16)
crf = LogisticRegression(random_state=16)
lf_fit = clf.fit(Xlf_train, ylf_train)
rf_fit = crf.fit(Xrf_train, yrf_train)

print("lf_fit_coeff: ", lf_fit.coef_)
print("lf_fit_intercept: ", lf_fit.intercept_)

print("rf_fit_coeff: ", rf_fit.coef_)
print("rf_fit_intercept: ", rf_fit.intercept_)

ylf_pred = clf.predict(Xlf_test)
print(clf.score(Xlf_test, ylf_test))
yrf_pred = crf.predict(Xrf_test)
print(crf.score(Xrf_test, yrf_test))
# print(ylf_pred)
# print(ylf_test)
# print(yrf_pred)
# print(yrf_test)
lf_grf_z  = data['lf_grf_z'].tolist()
lf_label  = data['lf_label'].tolist()
lf_label = [i*400 for i in lf_label]
rf_grf_z  = data['rf_grf_z'].tolist()
rf_label  = data['rf_label'].tolist()
rf_label = [i*400 for i in rf_label]
time = data['time'].tolist()
# # print(lf_spring_1)
plt.plot(time,lf_grf_z,'red')
plt.plot(time,lf_label,'blue')
plt.plot(time,rf_grf_z,'green')
plt.plot(time,rf_label,'yellow')
plt.gca().grid(True)
plt.gca().legend(('left foot z axis grf ','left foot contact','right foot z axis grf','right foot contact'))
plt.xlabel('time')
plt.show()

ylf_pred = clf.predict(Xlf_test)
#-------------------------test-data-------------------

# test_col_names=['lf_spring0','lf_spring1','rf_spring0','rf_spring1','lf_label','rf_label','time']
# tdata = pd.read_csv(r"C:\Users\rajar\OneDrive\Desktop\real_data.csv", header=0, usecols=test_col_names)

# ft_cols = ['lf_spring0','lf_spring1','rf_spring0','rf_spring1']

# Xlf_t = tdata[ft_cols]
# Xrf_t = tdata[ft_cols]
# real_time = tdata['time'].tolist()
# lf_sp_0 = tdata['lf_spring0'].tolist()
# rf_sp_0 = tdata['rf_spring0'].tolist()
# # ylf_t = tdata.lf_label
# # yrf_t = tdata.rf_label
# # tXlf_train, tXlf_test, tylf_train, tylf_test = train_test_split(Xlf_t, ylf_t, test_size=0.95, random_state=16)
# # tXrf_train, tXrf_test, tyrf_train, tyrf_test = train_test_split(Xrf_t, yrf_t, test_size=0.95, random_state=16)
# ylf_pred = clf.predict(Xlf_t)
# yrf_pred = crf.predict(Xrf_t)
# print(ylf_pred)
# print(yrf_pred)
# plt.plot(lf_sp_0,'red')
# plt.plot(rf_sp_0,'blue')
# plt.show()