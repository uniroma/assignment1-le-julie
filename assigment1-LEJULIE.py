## Importing libraries in Python allows us to expand the functionality
## of the language. These libraries contain pre-written code that help
## us perform various tasks: i.e. Numpy and Pandas are for data manipulation,
## and Matplotlib for data visualization.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

## THE FRED-MD DATASET is a datased provided by the Federal Reserve Bank
## of St.Louise including a wide range of macrovariables.
## We will manipulate these datas in order to develop a forecasting model

df = pd.read_csv("/Users/biancadiveroli/Downloads/current (1).csv")
# Clean the DataFrame by removing the row with transformation codes
df_cleaned = df.drop(index=0)
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned['sasdate'] = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')

# Extract transformation codes
transformation_codes = df.iloc[0, 1:].to_frame().reset_index()
transformation_codes.columns = ['Series', 'Transformation_Code']



# Function to apply transformations based on the transformation code
def apply_transformation(series, code):
    if code == 1:
        # No transformation
        return series
    elif code == 2:
        # First difference
        return series.diff()
    elif code == 3:
        # Second difference
        return series.diff().diff()
    elif code == 4:
        # Log
        return np.log(series)
    elif code == 5:
        # First difference of log
        return np.log(series).diff()
    elif code == 6:
        # Second difference of log
        return np.log(series).diff().diff()
    elif code == 7:
        # Delta (x_t/x_{t-1} - 1)
        return series.pct_change()
    else:
        raise ValueError("Invalid transformation code")

# Applying the transformations to each column in df_cleaned based on transformation_codes
for series_name, code in transformation_codes.values:
    df_cleaned[series_name] = apply_transformation(df_cleaned[series_name].astype(float), float(code))


df_cleaned = df_cleaned[2:]
df_cleaned.reset_index(drop=True, inplace=True)
df_cleaned.head()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

series_to_plot = ['INDPRO', 'CPIAUCSL', 'TB3MS']
series_names = ['Industrial Production',
                'Inflation (CPI)',
                '3-month Treasury Bill rate']


# Create a figure and a grid of subplots
fig, axs = plt.subplots(len(series_to_plot), 1, figsize=(8, 15))

# Iterate over the selected series and plot each one
for ax, series_name, plot_title in zip(axs, series_to_plot, series_names):
    if series_name in df_cleaned.columns:
        dates = pd.to_datetime(df_cleaned['sasdate'], format='%m/%d/%Y')
        ax.plot(dates, df_cleaned[series_name], label=plot_title)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title(plot_title)
        ax.set_xlabel('Year')
        ax.set_ylabel('Transformed Value')
        ax.legend(loc='upper left')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax.set_visible(False)  # Hide plots for which the data is not available

plt.tight_layout()
plt.show()

## FORECASTING IN TIME SERIES

Yraw = df_cleaned['INDPRO']
Xraw = df_cleaned[['CPIAUCSL', 'TB3MS']]

num_lags  = 4  ## this is p
num_leads = 1  ## this is h
X = pd.DataFrame()
## Add the lagged values of Y
col = 'INDPRO'
for lag in range(0,num_lags+1):
        # Shift each column in the DataFrame and name it with a lag suffix
        X[f'{col}_lag{lag}'] = Yraw.shift(lag)

for col in Xraw.columns:
    for lag in range(0,num_lags+1):
        # Shift each column in the DataFrame and name it with a lag suffix
        X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
## Add a column on ones (for the intercept)
X.insert(0, 'Ones', np.ones(len(X)))


## X is now a DataFrame
X.head()


y = Yraw.shift(-num_leads)
y


## Save last row of X (converted to numpy)
X_T = X.iloc[-1:].values
## Subset getting only rows of X and y from p+1 to h-1
## and convert to numpy array
y = y.iloc[num_lags:-num_leads].values
X = X.iloc[num_lags:-num_leads].values
X_T

## ESTIMATION

from numpy.linalg import solve
# Solving for the OLS estimator beta: (X'X)^{-1} X'Y
beta_ols = solve(X.T @ X, X.T @ y)

## Produce the One step ahead forecast
## % change month-to-month INDPRO
forecast = X_T@beta_ols*100
forecast
print(forecast)

def calculate_forecast(df_cleaned, p = 4, H = [1,4,8], end_date = '12/1/1999',target = 'INDPRO', xvars = ['CPIAUCSL', 'TB3MS']):

    ## Subset df_cleaned to use only data up to end_date
    rt_df = df_cleaned[df_cleaned['sasdate'] <= pd.Timestamp(end_date)]
    ## Get the actual values of target at different steps ahead
    Y_actual = []
    for h in H:
        os = pd.Timestamp(end_date) + pd.DateOffset(months=h)
        Y_actual.append(df_cleaned[df_cleaned['sasdate'] == os][target]*100)
        ## Now Y contains the true values at T+H (multiplying * 100)

    Yraw = rt_df[target]
    Xraw = rt_df[xvars]

    X = pd.DataFrame()
    ## Add the lagged values of Y
    for lag in range(0,p):
        # Shift each column in the DataFrame and name it with a lag suffix
        X[f'{target}_lag{lag}'] = Yraw.shift(lag)

    for col in Xraw.columns:
        for lag in range(0,p):
            X[f'{col}_lag{lag}'] = Xraw[col].shift(lag)
    
    ## Add a column on ones (for the intercept)
    X.insert(0, 'Ones', np.ones(len(X)))
    
    ## Save last row of X (converted to numpy)
    X_T = X.iloc[-1:].values

    ## While the X will be the same, Y needs to be leaded differently
    Yhat = []
    for h in H:
        y_h = Yraw.shift(-h)
        ## Subset getting only rows of X and y from p+1 to h-1
        y = y_h.iloc[p:-h].values
        X_ = X.iloc[p:-h].values
        # Solving for the OLS estimator beta: (X'X)^{-1} X'Y
        beta_ols = solve(X_.T @ X_, X_.T @ y)
        ## Produce the One step ahead forecast
        ## % change month-to-month INDPRO
        Yhat.append(X_T@beta_ols*100)

    ## Now calculate the forecasting error and return

    return (np.array(Y_actual) - np.array(Yhat), np.array(Yhat), np.array(Y_actual))

t0 = pd.Timestamp('12/1/1999')
e = []
T = []
Yhat_plot = []
Y_actual_plot = []

for j in range(0, 281):
    t0 = t0 + pd.DateOffset(months=1)
    print(f'Using data up to {t0}')
    ehat, yh, ya = calculate_forecast(df_cleaned, p = 4, H = [1, 4, 8], end_date = t0)
    e.append(ehat.flatten())
    Yhat_plot.append(yh.flatten())
    Y_actual_plot.append(ya.flatten())
    T.append(t0)

edf = pd.DataFrame(e)
yhdf = pd.DataFrame(Yhat_plot)
yadf = pd.DataFrame(Y_actual_plot)

fig, axs = plt.subplots(3, 1, figsize=(8, 6))

H = [1, 4, 8]
dates = pd.to_datetime(df_cleaned['sasdate'][489:770], format='%m/%d/%Y')
for i, h in enumerate(H):
    axs[i].plot(dates, yadf[i], color='blue', label='Actual data')
    axs[i].plot(dates, yhdf[i], color='red', label=f'Prediction for t+{h}')
    axs[i].legend()
    axs[i].set_title(f'h={h}')

plt.tight_layout()
plt.show()

## Calculate the RMSFE
## We use the Root Mean Squared Forecast Error to evaluate the accuracy
## and reliability of our forecasting model. It's calculated by taking
## the square root of the average of the squared differences between the
## predicted values and the actual values. Hence, it provides a measure
## of deviation from observed values (on average). Lower RMSFE indicates
## better predicting performance.

np.sqrt(edf.apply(np.square).mean())

## We now want to perform a forecast analysis for the other variables:
## inflation rate and interest rate.

Yraw2 = df_cleaned['CPIAUCSL']
Xraw2 = df_cleaned[['INDPRO','TB3MS']]

## We need to establish the number of lags and leads
num_lags2 = 4
num_leads2 = 1
X2 = pd.DataFrame()

## We want a loop in which we insert the lagged values for Y and X.

col2 = 'CPIAUSCSL'
for lag in range(0,num_lags2+1):
  X2[f'{col2}_lag{lag}'] = Yraw2.shift(lag)

for lag in range(0,num_lags2+1):
  X2[f'{col2}_lag{lag}'] = Xraw2[col].shift(lag)

X2
## X2 has 5 Columns: CPIAUSCSL lagged by p going from 0 to 4 (1 year)
## We add a column of ones in order to have the intercept

X2.insert (0, 'Ones', np.ones(len(X2)))
X2
