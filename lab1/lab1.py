import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model


#def prepare_country_stats(oecd_bli, gdp_per_capita):
#    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
#    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
#    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
#    gdp_per_capita.set_index("Country", inplace=True)
#    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
#                                  left_index=True, right_index=True)
#    full_country_stats.sort_values(by="GDP per capita", inplace=True)
#    remove_indices = [0, 1, 6, 8, 33, 34, 35]
#    keep_indices = list(set(range(36)) - set(remove_indices))
#    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


#def prep_housing_data(housing_data)
#    housing_data.sort_values(by="median_income", inplace=True)
    
# Load the data
housing_db = pd.read_csv("../datasets/housing/housing.csv", thousands=',')

# Prepare the data
X = np.c_[housing_db["median_income"]]
y = np.c_[housing_db["median_house_value"]]

# Visualize the data
housing_db.plot(kind='scatter', x="median_income", y='median_house_value')
plt.show()

# Select a linear model
model = sklearn.linear_model.LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
X_new = [[22587]]  # Cyprus' GDP per capita
print(model.predict(X_new)) # outputs [[ 5.96242338]]
