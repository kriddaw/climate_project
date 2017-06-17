import re
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# oldest data at Year: 1880, model accuracy improves with less data because of exponential nature of PPM growth
# toggle this value for different graphs - range (1880 - 2016)
YEAR_START = 1986


# this variable plugs into linear regression method, prints PPM/Temperature prediction to console
# note how using more than 30 years of data adds noise to the model
# Sample Output:
# Using 30 years of historical data.
# Confidence Score: 0.997150953255
# Prediction for year 2050 - PPM: 463.8395869911533, Temp: 1.4417232145883432
PREDICTION_YEAR = 2050

""" Prepare data for machine learning algorithms """


def co2_ppm_data():
    """ load the co2 data, use regular expressions to parse data from text file, output pandas DataFrame"""
    regex = re.compile(r'\d{4}\s{2}\d{3}\.\d\d?')

    with open('datasets/Fig1A.ext.txt', 'r') as f:
        data = f.read()

    data_points = re.findall(regex, str(data))
    # print(data_points)

    year_ppm = []
    for point in sorted(data_points):
        year, ppm = point.split()
        year_ppm.append({'Year': int(year), 'ppm': float(ppm)})

    df = pd.DataFrame(year_ppm)
    # range from 1850 - 2016
    df = (df.ix[df['Year'] >= YEAR_START])
    # print(df.head(10))
    return df


def global_mean_temp():
    """load Goddard Institute land/ocean global mean temperature data, output pandas DataFrame"""
    df = pd.read_csv('datasets/GISS-land-ocean-GM-temp.csv')
    # range from 1880 - 2016
    df = (df.ix[df['Year'] >= YEAR_START])
    # print(df.head(10))
    return df


def merged_data():
    """merge the Data Frames for plotting and multivariate linear regression output new DataFrame"""
    concat = pd.merge(co2_ppm_data(), global_mean_temp(), on='Year')
    return concat


def visualize_climate_data(climate_data):
    # visualize the data
    # 3d graph
    fig = plt.figure()
    fig.set_size_inches(12.5, 7.5)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xs=climate_data['Year'], ys=climate_data['J-D'], zs=climate_data['ppm'])

    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature Anomaly')
    ax.set_zlabel('CO2: Atmosphere Concentration (PPM)')
    ax.view_init(10, -60)

    # 2d graph sharing 'Year' axis
    f, axarr = plt.subplots(2, sharex='all')
    f.set_size_inches(12.5, 7.5)

    axarr[0].plot(climate_data['Year'], climate_data['ppm'])
    axarr[0].set_ylabel('CO2: Atmosphere Concentration (PPM)')

    axarr[1].plot(climate_data['Year'], climate_data['J-D'])
    axarr[1].set_xlabel('Year')
    axarr[1].set_ylabel('Temperature Anomaly')

    plt.show()


def linear_regression(climate_data):
    X = climate_data.as_matrix(['Year'])
    y = climate_data.as_matrix(['ppm', 'J-D'])

    X_train, X_test, y_train, y_test = np.asarray(train_test_split(X, y, test_size=0.1))
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)

    # predictive capacity of model
    print('Using {} years of historical data.' .format(2016 - YEAR_START))
    print('Confidence Score:', lin_reg.score(X_test, y_test))
    prediction = lin_reg.predict([[PREDICTION_YEAR]])
    # print(prediction[0][0])
    print('Prediction for year {} - PPM: {}, Temp: {}' .format(PREDICTION_YEAR,
                                                               prediction[0][0], prediction[0][1]))

    # visualize results
    x_vis = np.arange(YEAR_START, 2016).reshape(-1, 1)
    p = lin_reg.predict(x_vis).T

    # 3d graph with linear regression line
    fig = plt.figure()
    fig.set_size_inches(12.5, 7.5)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=climate_data['Year'], ys=climate_data['J-D'], zs=climate_data['ppm'], c='b')

    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature Anomaly')
    ax.set_zlabel('CO2: Atmosphere Concentration (PPM)')

    ax.plot(xs=x_vis, ys=p[1], zs=p[0], color='red', linewidth=2)
    ax.view_init(10, -60)

    # 2d graph sharing 'Year' axis with linear regression line
    f, axarr = plt.subplots(2, sharex='all')
    f.set_size_inches(12.5, 7.5)

    axarr[0].plot(climate_data['Year'], climate_data['ppm'])
    axarr[0].plot(x_vis, p[0], color='red')
    axarr[0].set_ylabel('CO2: Atmosphere Concentration (PPM)')

    axarr[1].plot(climate_data['Year'], climate_data['J-D'])
    axarr[1].plot(x_vis, p[1], color='red')
    axarr[1].set_xlabel('Year')
    axarr[1].set_ylabel('Temperature Anomaly')

    plt.show()


if __name__ == '__main__':

    # load the data
    climate_data = merged_data()

    # graph data only
    # visualize_climate_data(climate_data)

    # build machine learning model using sklearn LinearRegression class and graph
    linear_regression(climate_data)
