import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


houses_data = pd.read_csv('houses_info.csv')

features = [
    'OverallCond', 'GrLivArea', 'GarageCars', 'BsmtFinSF1',
    'FullBath', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
    'TotRmsAbvGrd', 'Fireplaces'
]
# - OverallQual, 1stFLrSF, TotalBsmtSF
# + OverallCond, MasVnrArea, BsmtFinSF1


X = houses_data[features]
y = houses_data['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, max_depth=7)

model.fit(X_train, y_train)


all_test_data = pd.read_csv('test_for_houses.csv')
test_data = all_test_data[features]

test_data['GarageCars'] = test_data['GarageCars'].fillna(0)
test_data['BsmtFinSF1'] = test_data['BsmtFinSF1'].fillna(0)

prices = model.predict(test_data)

output = pd.DataFrame({
    'Id': all_test_data['Id'],
    'SalePrice': prices
})

output.to_csv('submission_house.csv', index=False)