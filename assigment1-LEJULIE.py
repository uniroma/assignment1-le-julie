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

df = pd.read_csv("/Users/admin/Downloads/current (1).csv")
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
    
e_y, e_cpi, e_r = [], [], []
T_1, T_2, T_3 = [], [], []
Yhat, CPIhat, rhat = [], [], []
Y_actual, CPI_actual, r_actual  = [], [], []

indpro = e_y, Yhat, Y_actual, T_1
inflation = e_cpi, CPIhat, CPI_actual, T_2
intrates = e_r, rhat, r_actual, T_3

##We now use a for loop in order to predict the path of industrial production, CPI and the interest rate of Treasury Bill      
for t, v, s in ('INDPRO', ['CPIAUCSL', 'TB3MS'], indpro), ('CPIAUCSL', ['TB3MS', 'INDPRO'], inflation), ('TB3MS', ['CPIAUCSL', 'INDPRO'], intrates):
    t0 = pd.Timestamp('12/1/1999')
    for j in range(0, 281):
        t0 = t0 + pd.DateOffset(months=1)
        print(f'Using data up to {t0}')
        ehat, yh, ya = calculate_forecast(df_cleaned, p = 4, H = [1,4,8], end_date = t0 ,target = t , xvars = v)
        s[0].append(ehat.flatten())
        s[1].append(yh.flatten())
        s[2].append(ya.flatten())
        s[3].append(t0)

## dividing the series related to r by 100, as the data was unnecessarily multiplied by 100 by the calculate_forecast function

for i in (e_r, r_actual, rhat):
    i[:] = [x / 100 for x in i]
    
edf, yhdf, yadf, T1df = pd.DataFrame(e_y), pd.DataFrame(Yhat), pd.DataFrame(Y_actual), pd.DataFrame(T_1)
ecpidf, cpihdf, cpiadf, T2df = pd.DataFrame(e_cpi), pd.DataFrame(CPIhat), pd.DataFrame(CPI_actual), pd.DataFrame(T_2)
erdf, rhdf, radf, T3df = pd.DataFrame(e_r), pd.DataFrame(rhat), pd.DataFrame(r_actual), pd.DataFrame(T_3)

## Plotting the data:

dataframes = [yhdf, yadf, cpihdf, cpiadf, rhdf, radf]
titles = ['Industrial Production', 'Inflation (CPI)', '3-month Treasury Bill rate']
h_values = [1, 4, 8]
dates = pd.to_datetime(df_cleaned['sasdate'][489:770], format='%m/%d/%Y')

figures = []
axes = []

for i in range(len(dataframes) // 2):
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    fig.suptitle(titles[i])

    for j, h in enumerate(h_values):
        axs[j].plot(dates, dataframes[i * 2 + 1][j], color='blue', label=f'Actual data')
        axs[j].plot(dates, dataframes[i * 2][j], color='red', label=(f'Prediction for t+{h}'))
        axs[j].set_title(f'h={h}')
        axs[j].legend(loc='lower right')

    figures.append(fig)
    axes.append(axs)

for fig in figures:
    fig.tight_layout()  
    plt.show()

## Calculate the RMSFE
## We use the Root Mean Squared Forecast Error to evaluate the accuracy
## and reliability of our forecasting model. It's calculated by taking
## the square root of the average of the squared differences between the
## predicted values and the actual values. Hence, it provides a measure
## of deviation from observed values (on average). Lower RMSFE indicates
## better predicting performance.

## Computing the RMSFE

for variable, errors in ('industrial prodcution', edf), ('inflation', ecpidf), ('interest rates', erdf):
    rmsfe = np.sqrt(errors.apply(np.square).mean())
    print(f'The RMSFEs of the forecasts of {variable} are: {rmsfe[0]:.6f} for h = 1, {rmsfe[1]:.6f} for h = 4, {rmsfe[2]:.6f} for h = 8')

## Computing the MSFE
    rmsfe = np.sqrt(errors.apply(np.square).mean())

for variable, errors in ('industrial prodcution', edf), ('inflation', ecpidf), ('interest rates', erdf):
    msfe = (errors.apply(np.square).mean())
    print(f'The MSFEs of the forecasts of {variable} are: {msfe[0]:.6f} for h = 1, {msfe[1]:.6f} for h = 4, {msfe[2]:.6f} for h = 8')
    
##The RMSFE for industrial production is 1.26 when h=1, 1.25 for h=4
##and 1.24 for h=8.
##The RMSFE for inflation is 0.30 for h=1, 0.32 when h=4 
##and  0.31 for h=8.
##The RMSFE for interest rate is 0.21 for h=1, 0.24 for h=4. and 0.20
##for h=8. So for all the 3 varibles the results suggest that
##the models are adequate in predicting the path of the variables. What we
##can notice is that the lowest RSFME for industrial production
##and the interest rate is for h=8. Actually,these values are not 
##so different from the respective RMSFE when we use h=1. 
##As for inflation, looking at the graph and also at our results,
##we can deduce that the model has a better 
##performance for h=1.
