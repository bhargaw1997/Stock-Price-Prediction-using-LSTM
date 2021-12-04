# Databricks notebook source
# Installing all the dependencies
%pip install pandas_datareader
%pip install keras
%pip install tensorflow-cpu==2.4.*

# COMMAND ----------

# Importing all the libraries
import pandas 
import numpy 

# Importing seaborn, matplot
import matplotlib.pyplot as plot
import seaborn
 
# Setting the all plot background
seaborn.set_style('darkgrid')

# Setting the graph style 
plot.style.use("ggplot")

# Adding the matplot graphs as inline to avoid using the plot.show()
%matplotlib inline


# COMMAND ----------

# Importing the Pandas DataReader
from pandas_datareader import DataReader

# Importing the datetime,timeDelta: for manipulating the Date
from datetime import datetime,timedelta

# COMMAND ----------

# All the companies ticker symbol used for predicting the closing price
companiesTicker= ['NFLX', 'FB', 'BTCS', 'ETHE','TSLA','WMT','AMD', 'DIS']

# Set the endTime,StartTime
endTime = datetime.now()
startTime = datetime(endTime.year - 5, endTime.month, endTime.day)


# Get the real time value of all the companies stocks
for stock in companiesTicker:   
  
    # Create a global dataframe using the company ticker symbol
    globals()[stock] = DataReader(stock, 'yahoo', startTime, endTime)
    
    # Removing the NAN values 
    globals()[stock] = globals()[stock].dropna()

# COMMAND ----------

# Create a list of all stock data
# Each object has all the respective stock data
organizationsList = [NFLX, FB, BTCS, ETHE, TSLA, WMT, AMD, DIS]

# Organizations Name
organizationNames = ["NETFLIX", "FACEBOOK", "BITCOIN", "ETHEREUM", "TESLA", "WALMART","AMD", "DISNEY"]

# Convert tuple to dictionary using organization name as key
for organization, orgName in zip(organizationsList, organizationNames):
    organization["organizationNames"] = orgName

# Append all Dataframes
pandasDataFrame = pandas.concat(organizationsList, axis=0)

# Show the last 5000 rows
pandasDataFrame.tail(5000)

# COMMAND ----------

# Displaying sample set of data 
pandasDataFrame.loc[pandasDataFrame['organizationNames']== "NETFLIX"].display()

# COMMAND ----------

# Sample dataframe coloumns
pandasDataFrame.loc[pandasDataFrame['organizationNames']== "NETFLIX"].info()

# COMMAND ----------

# Plot the historical daily closing price of each organization

# Set the Height and width of each plot
plot.figure(figsize=(20, 10))

# Set the top,bottom of each plot
plot.subplots_adjust(top=2, bottom=1.5)

# Each organization
for index, organization in enumerate(organizationsList, 1):
  
    # Set the Subplot grid as 4*2
    plot.subplot(4, 2, index)
    
    # Plot for closing prices
    organization['Adj Close'].plot(xlabel='Price',ylabel='Date',linewidth=2,title=f"Closing Price of {organizationNames[index - 1]}")
    
plot.tight_layout()

# COMMAND ----------

# Now let's plot the total volume of stock being traded each day
plot.figure(figsize=(20, 20))
plot.subplots_adjust(top=2, bottom=1.5)

for index, organization in enumerate(organizationsList, 1):
    plot.subplot(4, 2, index)
    organization['Volume'].plot(xlabel='Volume',ylabel='Date',title=f"Volume for {organizationNames[index - 1]}",linewidth=2,color='green')
    
plot.tight_layout()

# COMMAND ----------

# Moving average stock price per days
movingAvgPerDays = [10, 20, 50]

# Create a key in dictionary for each moving average
for day in movingAvgPerDays:
    for organization in organizationsList:
        organization[f"Moving average for {day} days"] = organization['Adj Close'].rolling(day).mean()

# COMMAND ----------

# Create the Grid for all subplots
eachPlot, axis = plot.subplots(nrows=4, ncols=2)

# Set the width,height
eachPlot.set_figheight(15)
eachPlot.set_figwidth(15)

# Plotting Each Stock price based on moving Averages
index =0
for subplotRow in range(0,4):
  for subPlotColoumn in range(0,2):
    organizationsList[index][['Adj Close', 'Moving average for 10 days', 'Moving average for 20 days', 'Moving average for 50 days']].plot(ax=axis[subplotRow,subPlotColoumn],linewidth=2)
    axis[subplotRow,subPlotColoumn].set_title(organizationNames[index])
    index+=1

eachPlot.tight_layout()

# COMMAND ----------

# Create a key in dictionary for each daily return
for organization in organizationsList:
    organization['Daily Return'] = organization['Adj Close'].pct_change()

# Create the Grid for all subplots
eachPlot, axis = plot.subplots(nrows=2, ncols=4)

# Set the width,height
eachPlot.set_figheight(9)
eachPlot.set_figwidth(20)

# Plotting Each Stock price based on daily return
index =0
for subplotRow in range(0,2):
  for subPlotColoumn in range(0,4):
    organizationsList[index][['Daily Return']].plot(ax=axis[subplotRow,subPlotColoumn], legend=True, linestyle='--', marker='o', markerfacecolor = 'red', color = 'yellow')
    axis[subplotRow,subPlotColoumn].set_title(organizationNames[index])
    index+=1

eachPlot.tight_layout()

# COMMAND ----------

# Todo Plot the graph datewise
# Set the height and width 
plot.figure(figsize=(15, 15))


for index, organization in enumerate(organizationsList, 1):
    plot.subplot(4, 2, index)
    seaborn.histplot(organization['Daily Return'], bins=100, color='#4333FF')
    
    plot.ylabel('Daily Return')
    plot.title(f'{organizationNames[index - 1]}')

plot.tight_layout()

# COMMAND ----------

# Fetch all the closing price of all organization
closingDataFrame = DataReader(companiesTicker, 'yahoo', startTime, endTime)['Adj Close']

closingDataFrame = closingDataFrame.dropna()
# Display the top 5 values
closingDataFrame.head() 

# COMMAND ----------

# Create the new percent change dataframe
percentChangeDataframe = closingDataFrame.pct_change()
percentChangeDataframe = percentChangeDataframe.dropna()
percentChangeDataframe.head()

# COMMAND ----------

# Create a pair grid to show the percentageChange Data 
returnFig = seaborn.PairGrid(percentChangeDataframe.dropna())

#specifying the upper triangle by calling map_upper.
returnFig.map_upper(plot.scatter, color='purple')

# defining the lower triangle for the figure reference, including the plot type (kde) or the color map (BluePurple)
returnFig.map_lower(seaborn.kdeplot, cmap='cool_d')

# Diagonal plots of series of histogram plots of daily returns  
returnFig.map_diag(plot.hist, bins=30)

# COMMAND ----------

# Create a pair grid for closing price Dataframe
returnsFig = seaborn.PairGrid(closingDataFrame)

#specifying the upper triangle by using map_upper.
returnsFig.map_upper(plot.scatter,color='purple')

# defining the lower triangle for the figure reference, including the plot type (kde) or the color map (BluePurple)
returnsFig.map_lower(seaborn.kdeplot,cmap='cool_d')

# A series of diagonal plots of daily return histogram  
returnsFig.map_diag(plot.hist,bins=30)

# COMMAND ----------

# Show heatmap for analysing quick correlation between daily returns
seaborn.heatmap(percentChangeDataframe.corr(), annot=True, cmap='summer')

# COMMAND ----------

seaborn.heatmap(closingDataFrame.corr(), annot=True, cmap='summer')

# COMMAND ----------

# defininging a new DataFrame as a cleaned version from percentChangeDataframe DataFrame
percentChangeDataframe = percentChangeDataframe.dropna()

area = numpy.pi * 20

plot.figure(figsize=(10, 7))
plot.scatter(percentChangeDataframe.mean(), percentChangeDataframe.std(), s=area)
plot.xlabel('Expected return')
plot.ylabel('Risk')

for label, eachMean, eachSTD in zip(percentChangeDataframe.columns, percentChangeDataframe.mean(), percentChangeDataframe.std()):
    plot.annotate(label, xy=(eachMean, eachSTD), xytext=(50, 50), textcoords='offset points', ha='right', va='bottom', 
                 arrowprops=dict(arrowstyle='-', color='green', connectionstyle='arc3,rad=-0.3'))

# COMMAND ----------

# Getting Walmart stock price quote from Yahoo API
WMTDataframe = DataReader('WMT', data_source='yahoo', start='2011-01-01', end=datetime.now())

# Display the dataframe
WMTDataframe

# COMMAND ----------

WMTDataframe['Close'].plot(xlabel='Date',ylabel='Close Price USD ($)',title='Close Price History',linewidth=2,color='green')
# Set the canvas size for plot
plot.rcParams["figure.figsize"] = (30,5)
plot.show()

# COMMAND ----------

# Creating a new dataFrame which contains only the 'Close' column 
data = WMTDataframe.filter(['Close'])
# Converting the created dataframe to a usable numpy array
dataSetArray = data.values
# Getting the number of rows from the model to train the model
trainingDataLen = int(numpy.ceil( len(dataSetArray) * .95 ))

trainingDataLen

# COMMAND ----------

# Scaling the data to according to our usage
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaledData = scaler.fit_transform(dataSetArray)

scaledData

# COMMAND ----------

# Gather all the training data
trainingDataSet = scaledData[0:int(trainingDataLen), :]

# Split the data into xTrainingDataSet and yTrainingDataSet data sets
xTrainingDataSet = []
yTrainingDataSet = []

for i in range(120, len(trainingDataSet)):
    xTrainingDataSet.append(trainingDataSet[i-120:i, 0])
    yTrainingDataSet.append(trainingDataSet[i, 0])
    if i<= 121:
        print(xTrainingDataSet)
        print(yTrainingDataSet)
        print()
        
# Convert the xTrainingDataSet and yTrainingDataSet to numpy arrays 
xTrainingDataSet, yTrainingDataSet = numpy.array(xTrainingDataSet), numpy.array(yTrainingDataSet)

# Reshape the data
xTrainingDataSet = numpy.reshape(xTrainingDataSet, (xTrainingDataSet.shape[0], xTrainingDataSet.shape[1], 1))


# COMMAND ----------

xTrainingDataSet.shape

# COMMAND ----------



import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential


# Building the LSTM model using LSTM built in function
model = Sequential([
  
  LSTM(64, return_sequences=True, input_shape= (xTrainingDataSet.shape[1], 1)),
  LSTM(32),
  Dense(25),
  Dense(1)
  
])

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# training the model using xTrainingDataSet and yTrainingDataSet
model.fit(xTrainingDataSet, yTrainingDataSet, batch_size=1, epochs=1)

# COMMAND ----------

# Creating the testing data set using acquired data
# Creating a new array which contains scaled values
testData = scaledData[trainingDataLen - 120: , :]
# Creating the data sets as two different data sets xTestDataSet and yTestDataSet
xTestDataSet = []
yTestDataSet = dataSetArray[trainingDataLen:, :]
for i in range(120, len(testData)):
    xTestDataSet.append(testData[i-120:i, 0])
    
# Converting the acquired data to a numpy array for our calculation
xTestDataSet = numpy.array(xTestDataSet)

# Reshaping the data according to our requirement
xTestDataSet = numpy.reshape(xTestDataSet, (xTestDataSet.shape[0], xTestDataSet.shape[1], 1 ))

# the predicted price values of the models 
predictions = model.predict(xTestDataSet)
predictions = scaler.inverse_transform(predictions)

# root mean squared error values of collected data
rmse = numpy.sqrt(numpy.mean(((predictions - yTestDataSet) ** 2)))
rmse

# COMMAND ----------

# Plotting the resulting data
trainDataset = data[:trainingDataLen]
valid = data[trainingDataLen:]
valid['Predictions'] = predictions
# Visualize the data
plot.figure(figsize=(16,6))
plot.title('Model')
plot.xlabel('Date', fontsize=18)
plot.ylabel('Close Price USD ($)', fontsize=18)
plot.plot(trainDataset['Close'])
plot.plot(valid[['Close', 'Predictions']])
plot.legend(['trainDataset', 'Val', 'Predictions'], loc='lower right')
plot.show()

# COMMAND ----------

# Display the i.e final result of prediction
valid

# COMMAND ----------

# Set up our time horizon for future prediction
timeHorizonInDays = 365

# Delta
delta = 1/timeHorizonInDays


# Mean is the mu for monte carlo analysis
mu = percentChangeDataframe['WMT'].mean()

# Sigma is the standard deviation in monte carlo analysis
sigma = percentChangeDataframe['WMT'].std()

# COMMAND ----------

WMTDataFrame = DataReader('WMT', data_source='yahoo', start='2011-01-01', end=datetime.now())
WMTDataFrame.filter(['Close']).tail(1)['Close']

# COMMAND ----------

def performMonteCarloSimulation(intialPrice,timeHorizonInDays,mu,sigma):
    
    # Define a price array
    priceAfterGivenDays = numpy.zeros(timeHorizonInDays)
    priceAfterGivenDays[0] = intialPrice
    
    # monteCarloEstimate 
    monteCarloEstimate = numpy.zeros(timeHorizonInDays)

    # variation of price also known as drift
    variationOfPrice = numpy.zeros(timeHorizonInDays)
    
    # Iterate over the given time horizon
    for eachDay in range(1,timeHorizonInDays):
        
        # Perfome the monte carlo analysis using numpy normal API
        monteCarloEstimate[eachDay] = numpy.random.normal(loc=mu * delta, scale=sigma * numpy.sqrt(delta))
        
        # Calculate variationOfPrice
        variationOfPrice[eachDay] = mu * delta

        # Calculate price using previous estimates
        priceAfterGivenDays[eachDay] = priceAfterGivenDays[eachDay-1] + (priceAfterGivenDays[eachDay-1] * (variationOfPrice[eachDay] + monteCarloEstimate[eachDay]))
        
    return priceAfterGivenDays

# Get the current price from the dataframe
intialPrice = WMTDataFrame.filter(['Close']).tail(1)['Close'][0]

# Set the canvas size for plot
plot.rcParams["figure.figsize"] = (20,10)

# convert the numpy array to dataFrame to display the graph
futurePrice = pandas.DataFrame(performMonteCarloSimulation(intialPrice, timeHorizonInDays, mu, sigma))
futurePrice.plot(xlabel='Days',ylabel='Price',linestyle='--', marker='o', markerfacecolor = 'red', color = 'yellow',title='Monte Carlo Analysis for WMT')
plot.show()
