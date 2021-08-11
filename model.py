import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
#import seaborn as sns
import pickle

df = pd.read_csv("C:\\Users\\saurav\\Downloads\\excelR_Files_datas\\Data science Bateches\\Assignments All\\4 assign_simple linear\\Salary_Data.csv")
df.head()

df.shape

df.isnull().sum()

X = np.array(df['YearsExperience']).reshape(-1, 1)
y = np.array(df['Salary']).reshape(-1, 1)


# Dropping any rows with Nan values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
  
# Splitting the data into training and testing data
regr = LinearRegression()
  
regr.fit(x_train, y_train)

pickle.dump(regr, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[2,9,6]]))
