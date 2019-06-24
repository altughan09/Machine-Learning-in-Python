import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

dates = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
    return

def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))

    svr_rbf = SVR(kernel= 'rbf', C=1e3, gamma=0.1)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='green', label='RBF Model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('test regression analysis')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0]

get_data('testdata.csv')

predicted_price = predict_prices(dates, prices, 100)

print(predicted_price)