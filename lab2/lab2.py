import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

#TODO:
#Plot Data
#look for coorilations
#clean the data
#split data test / train
#scaling feature
#create pipeline
#cross validation
#grid search to find best model
# evaluate model on test set
    
# Load the data
lifeIndex_db = pd.read_csv("betterLifeIndex.csv", thousands=',')

# Prepare the data
#X = np.c_[housing_db["median_income"]]
#y = np.c_[housing_db["median_house_value"]]

# Select a linear model

# Train the model

#prediction

# Visualize the data
#plt.plot( X , house_value_predict, color="r")
#plt.text(0, 650000, "medium income vs median house value", color='b')
#plt.text(0, 600000, "medium income vs predicted median house value", color='r')

#plt.show()
