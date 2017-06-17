This is a simple data science project to play with multivariate linear regression.
I thought it would be interesting to plot some actual climate data using matplotlib, so that's what I did.
Disclaimer: I am not a scientist nor a programmer, I just really like python and have been teaching myself for the last 5 months.

Imports:
re, pandas (0.20.2), numpy (1.13.0), matplotlib (2.0.2), scikit-learn (0.18.1)

It's written using python 3 syntax. See climate.py for source code.
Read the comments and play around with the variables! It's pretty neat running this for yourself and moving around the 3d graphs. :D

Prediction Output:

```
Using 30 years of historical data.
Confidence Score: 0.997150953255
Prediction for year 2050 - PPM: 463.8395869911533, Temp: 1.4417232145883432
```

Graph Output:
2D Graphs
![alt text](https://raw.githubusercontent.com/kriddaw/climate_project/master/2d-plt.png)
3D Graph
![alt text](https://raw.githubusercontent.com/kriddaw/climate_project/master/3d-plt.png)

Regression Output:
2D Regression (30 years)
![alt text](https://raw.githubusercontent.com/kriddaw/climate_project/master/2d-regr-30yr.png)
3D Regression (30 years)
![alt text](https://raw.githubusercontent.com/kriddaw/climate_project/master/3d-regr-30yr.png)

Data Sources:

Atmospheric CO2 Concentrations: https://data.giss.nasa.gov/modelforce/ghgases/
Global Mean Land/Ocean Temperatures: https://data.giss.nasa.gov/gistemp/
