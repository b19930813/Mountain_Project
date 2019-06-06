import pandas as pd
from pandas import DataFrame
from sklearn import datasets
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression 

from sklearn import metrics
from sklearn.model_selection import train_test_split



need_data = pd.read_csv('D:\\Frank\\Mountain_Project\\train.csv')

data_result = need_data[['total_price']]
need_data = need_data.drop(['total_price','building_id','doc_rate','master_rate','bachelor_rate','jobschool_rate','highschool_rate','junior_rate','elementary_rate','born_rate','death_rate','marriage_rate','divorce_rate'],axis=1)
#need_data = need_data[['building_material','city','txn_dt','total_floor','building_type','building_use','building_complete_dt','parking_way','parking_area','parking_price','txn_floor','land_area','building_area','lat','lon','village_income_median','town_population','town_area','town_population_density','village','N_10000','I_5000','XIV_MIN']]
need_data = need_data.fillna(0)

return_data = pd.read_csv('D:\\Frank\\Mountain_Project\\test.csv')
building_data = return_data['building_id']
return_data = return_data.drop(['building_id','doc_rate','master_rate','bachelor_rate','jobschool_rate','highschool_rate','junior_rate','elementary_rate','born_rate','death_rate','marriage_rate','divorce_rate'],axis=1)
#return_data = return_data[['building_material','city','txn_dt','total_floor','building_type','building_use','building_complete_dt','parking_way','parking_area','parking_price','txn_floor','land_area','building_area','lat','lon','village_income_median','town_population','town_area','town_population_density','village','N_10000','I_5000','XIV_MIN']]
return_data = return_data.fillna(0)

#get building 
#轉換格式
building_df = pd.DataFrame(building_data)
#print(type(building_df))


#劃分訓練集
X_train, X_test, y_train, y_test = train_test_split(need_data, data_result, random_state=1)

#print(X_train.shape)
#print(y_train.shape)
#print(X_test.shape)
#print(y_test.shape)


linreg = LinearRegression()  #線性回歸
#linreg = LogisticRegression()  #
linreg.fit(X_train, y_train)

#print(linreg.intercept_)
#print(linreg.coef_)
y_pred = linreg.predict(return_data)

#print(y_pred)
#寫入Excel
print("MSE:",metrics.mean_squared_error(y_test, y_pred))
#head = ['building_id','total_price']
#df = pd.DataFrame(y_pred,columns=['total_price'])
#print(df)
#df.to_csv ('D:\\Frank\\Mountain_Project\\result.csv', index = False,mode='w')
#building_data.to_csv(y_pred,columns=['total_price'])
#write_data = pd.read_csv('D:\\Frank\\Mountain_Project\\submit_test.csv')
#write_data.update(column="total_price",value=y_pred)
# 用scikit-learn计算RMSE

#print("MSE:",metrics.mean_squared_error(data_result, data_pred))

#print(data_pred)
