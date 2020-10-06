import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

if len(sys.argv) != 3:
    print('Usage: python3 nfl_regression.py <abbv_team> <degree>')
    print('Exiting.')
    sys.exit(1)

script, team, d = sys.argv
d = int(d)

df = pd.read_csv('https://raw.githubusercontent.com/fivethirtyeight/nfl-elo-game/master/data/nfl_games.csv')
array = df.values
y = [x[9] for x in array if x[4] == team]
y += [x[10] for x in array if x[5] == team]

X = [i for i in range(len(y))]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=5)
X_train = np.array(X_train).reshape(-1, 1)

pf = PolynomialFeatures(degree=d)
X_poly = pf.fit_transform(X_train)
test = np.array([len(y) + 1] * (d + 1)).reshape(1, -1)
model = LinearRegression()
model.fit(X_poly, y_train)
prediction = model.predict(test)
print(np.round(prediction))