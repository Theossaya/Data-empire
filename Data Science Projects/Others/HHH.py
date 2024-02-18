# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 23:27:43 2024

@author: Blue
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Load the dataset
data = pd.read_excel('Htwo Project.xlsx')

# Define input features
# Assuming each stream has flowrates named like 'Stream1_CO', 'Stream1_H2O', etc.
stream_features = ['Stream{}_{}'.format(i, param) for i in range(1, 9) for param in ['CO', 'H2O', 'CO2', 'H2']]
other_features = ['Catalyst type', 'Operating temperature (Co)']  # Adjust names as per your dataset

# Combine all features
input_features = stream_features + other_features

# Extracting the input and output data
X = data[input_features]
y = data['CO conversion (%)']  # Adjust the column name as per your dataset

# Preprocessing
# One-hot encode categorical data and normalize numerical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), stream_features),
        ('cat', OneHotEncoder(), ['Catalyst type'])
    ])

X_processed = preprocessor.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# Assuming 'X_train', 'X_test', 'y_train', and 'y_test' are already preprocessed and split datasets

# Define the ANN model
model = Sequential()

# Input layer and first hidden layer with sigmoid activation
model.add(Dense(units=64, activation='sigmoid', input_dim=X_train.shape[1]))

# Second hidden layer with sigmoid activation
model.add(Dense(units=32, activation='sigmoid'))

# Output layer with no activation function since this is a regression problem
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=[early_stopping], verbose=1)

# Evaluate the model using the test data
loss = model.evaluate(X_test, y_test)

# Output the loss and training history
print(f"Test Loss: {loss}")
print(history.history)


from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Calculate R-squared (R^2)
r2 = r2_score(y_test, y_pred)

# Output the calculated metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R^2): {r2}")




A = data.drop('CO conversion (%)', axis=1)
y_true = data['CO conversion (%)'].values
y_pred = model.predict(X_processed)

# Assuming your dataset has been split into training and test sets, and your model has been trained.

# Obtain the true values for the training set
y_true_train = y_train  # This is provided during the training phase

# Predict on the training set
y_pred_train = model.predict(X_train)

# Obtain the true values for the test set
y_true_test = y_test  # This is set aside and not used during training

# Predict on the test set
y_pred_test = model.predict(X_test)


import matplotlib.pyplot as plt

# Replace 'y_true' with your actual CO conversion percentages
# and 'y_pred' with your model's predicted CO conversion percentages
# For example:
# y_true = [actual value1, actual value2, ..., actual valueN]
# y_pred = [predicted value1, predicted value2, ..., predicted valueN]

# Scatter plot for the training result
plt.figure(figsize=(10, 8))
plt.scatter(y_true_train, y_pred_train, color='gold', edgecolors='black', label='Training Data')
plt.plot([0, 100], [0, 100], 'k--')  # Dashed diagonal line
plt.xlabel('Experimental CO Conversion (%)')
plt.ylabel('Predicted CO Conversion (%)')
plt.title('Prediction Model Training Result')
plt.text(0.75, 0.55, f'R-squared: {r2:.3f}', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='right', verticalalignment='top')
plt.text(0.75, 0.45, f'MSE: {mse:.3f}', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='right', verticalalignment='top')
plt.text(0.75, 0.35, f'RMSE: {rmse:.3f}', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='right', verticalalignment='top')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot for the test result
plt.figure(figsize=(10, 8))
plt.scatter(y_true_test, y_pred_test, color='red', edgecolors='black', label='Test Data')
plt.plot([0, 100], [0, 100], 'k--')  # Dashed diagonal line
plt.xlabel('Experimental CO Conversion (%)')
plt.ylabel('Predicted CO Conversion (%)')
plt.title('Prediction Model Test Result')
plt.text(0.75, 0.55, f'R-squared: {r2:.3f}', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='right', verticalalignment='top')
plt.text(0.75, 0.45, f'MSE: {mse:.3f}', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='right', verticalalignment='top')
plt.text(0.75, 0.35, f'RMSE: {rmse:.3f}', transform=plt.gca().transAxes, fontsize=12, horizontalalignment='right', verticalalignment='top')
plt.legend()
plt.grid(True)
plt.show()


#Group I

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('Technical performances of group I catalysts.xlsx')

# Prepare the figure
plt.figure(figsize=(12, 8))

# Manual list of labels for the legend
# This should be in the same order as the catalysts are plotted
legend_labels = ['TiO2', 'CeO2', 'La2O3', 'Co3O4', 'ThO2', 'Y2O3', 'Al2O3', 'Fe2O3']  # Continue for all catalysts

# Assuming the first catalyst's data starts at the first column
start_col = 0

# Loop through the sets of columns for each catalyst, assuming 6 columns per catalyst
# and skipping one column each time for the space between catalysts
for i, label in zip(range(0, df.shape[1], 7), legend_labels):  # Using zip to iterate over indices and labels
    # Select columns for temperature and CO conversion
    temp_col = df.iloc[:, i]  # Operating temperature
    co_conv_col = df.iloc[:, i + 5]  # CO conversion
    
    # Create the plot for this catalyst
    plt.plot(temp_col, co_conv_col, label=label)

# Customizing the plot
plt.xlabel('Temperature (°C)')
plt.ylabel('Unit Energy Cost (USD/Kg of H2)')
plt.title('Group I (Supported-Pt)')

# Add the manual legend
plt.legend(legend_labels)
plt.grid(True)

# Show the plot
plt.show()

########################################

#Group II

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('Technical performances of group II catalysts.xlsx')

# Prepare the figure
plt.figure(figsize=(12, 8))

# Manual list of labels for the legend
# This should be in the same order as the catalysts are plotted
legend_labels = ['TiO2/Nd10', 'TiO2/Gd10', 'TiO2/Ho10','TiO2/Y10']  # Continue for all catalysts

# Assuming the first catalyst's data starts at the first column
start_col = 0

# Loop through the sets of columns for each catalyst, assuming 6 columns per catalyst
# and skipping one column each time for the space between catalysts
for i, label in zip(range(0, df.shape[1], 7), legend_labels):  # Using zip to iterate over indices and labels
    # Select columns for temperature and CO conversion
    temp_col = df.iloc[:, i]  # Operating temperature
    co_conv_col = df.iloc[:, i + 1]  # CO conversion
    
    # Create the plot for this catalyst
    plt.plot(temp_col, co_conv_col, label=label)

# Customizing the plot
plt.xlabel('Temperature (°C)')
plt.ylabel('CO Conversion (%)')
plt.title('Group II (TiO2 supported - metal promoted Pt) ')

# Add the manual legend
plt.legend(legend_labels)
plt.grid(True)

# Show the plot
plt.show()


#Group III

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('Technical performances of group III catalysts.xlsx')

# Prepare the figure
plt.figure(figsize=(12, 8))

# Manual list of labels for the legend
# This should be in the same order as the catalysts are plotted
legend_labels = ['Al2O3/Ni10', 'Al2O3/Ti10', 'Al2O3/Fe10', 'Al2O3/Cr10', 'Al2O3/Mn', 'Al2O3/Ho', 'Al2O3/Nd', 'Al2O3/Tm10', 'Al2O3/Sm10', 'Al2O3/Er10', 'Al2O3/Ce10' ]  # Continue for all catalysts

# Assuming the first catalyst's data starts at the first column
start_col = 0

# Loop through the sets of columns for each catalyst, assuming 6 columns per catalyst
# and skipping one column each time for the space between catalysts
for i, label in zip(range(0, df.shape[1], 7), legend_labels):  # Using zip to iterate over indices and labels
    # Select columns for temperature and CO conversion
    temp_col = df.iloc[:, i]  # Operating temperature
    co_conv_col = df.iloc[:, i + 1]  # CO conversion
    
    # Create the plot for this catalyst
    plt.plot(temp_col, co_conv_col, label=label)

# Customizing the plot
plt.xlabel('Temperature (°C)')
plt.ylabel('CO Conversion (%)')
plt.title('Group III (Al203 Supported - metal promoted-Pt)')

# Add the manual legend
plt.legend(legend_labels)
plt.grid(True)

# Show the plot
plt.show()


#Group IV

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('Technical performances of group IV catalysts.xlsx')

# Prepare the figure
plt.figure(figsize=(12, 8))

# Manual list of labels for the legend
# This should be in the same order as the catalysts are plotted
legend_labels = ['2wt%', '5wt%','10wt%', '20wt%']  # Continue for all catalysts

# Assuming the first catalyst's data starts at the first column
start_col = 0

# Loop through the sets of columns for each catalyst, assuming 6 columns per catalyst
# and skipping one column each time for the space between catalysts
for i, label in zip(range(0, df.shape[1], 7), legend_labels):  # Using zip to iterate over indices and labels
    # Select columns for temperature and CO conversion
    temp_col = df.iloc[:, i]  # Operating temperature
    co_conv_col = df.iloc[:, i + 1]  # CO conversion
    
    # Create the plot for this catalyst
    plt.plot(temp_col, co_conv_col, label=label)

# Customizing the plot
plt.xlabel('Temperature (°C)')
plt.ylabel('CO Conversion (%)')
plt.title('Group IV (TiO2 supported- Ce promoted-Pt')

# Add the manual legend
plt.legend(legend_labels)
plt.grid(True)

# Show the plot
plt.show()


#Group V

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('Technical performances of group V catalysts.xlsx')

# Prepare the figure
plt.figure(figsize=(12, 8))

# Manual list of labels for the legend
# This should be in the same order as the catalysts are plotted
legend_labels = ['5wt%', '10wt%', '20wt%']  # Continue for all catalysts

# Assuming the first catalyst's data starts at the first column
start_col = 0

# Loop through the sets of columns for each catalyst, assuming 6 columns per catalyst
# and skipping one column each time for the space between catalysts
for i, label in zip(range(0, df.shape[1], 7), legend_labels):  # Using zip to iterate over indices and labels
    # Select columns for temperature and CO conversion
    temp_col = df.iloc[:, i]  # Operating temperature
    co_conv_col = df.iloc[:, i + 1]  # CO conversion
    
    # Create the plot for this catalyst
    plt.plot(temp_col, co_conv_col, label=label)

# Customizing the plot
plt.xlabel('Temperature (°C)')
plt.ylabel('CO Conversion (%)')
plt.title('Group V(Al2O3 supported- Co promoted-Pt)')

# Add the manual legend
plt.legend(legend_labels)
plt.grid(True)

# Show the plot
plt.show()

#################




import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('Technical performances of group I catalysts - Copy.xlsx')

# Prepare the figure
plt.figure(figsize=(12, 8))

# Manual list of labels for the legend
# This should be in the same order as the catalysts are plotted
legend_labels = ['TiO2', 'CeO2', 'La2O3', 'Co3O4', 'ThO2', 'Y2O3', 'Al2O3', 'Fe2O3']  # Continue for all catalysts

# Assuming the first catalyst's data starts at the first column
start_col = 0

# Loop through the sets of columns for each catalyst, assuming 6 columns per catalyst
# and skipping one column each time for the space between catalysts
for i, label in zip(range(0, df.shape[1], 7), legend_labels):  # Using zip to iterate over indices and labels
    # Select columns for temperature and CO conversion
    temp_col = df.iloc[:, i]  # Operating temperature
    co_uec_col = df.iloc[:, i + 1]  # CO conversion
    
    # Create the plot for this catalyst
    plt.plot(temp_col, co_uec_col, label=label)

# Customizing the plot
plt.xlabel('Temperature (°C)')
plt.ylabel('Unit Energy Cost (USD/Kg of H2)')
plt.title('Group I Pt supported ')

# Add the manual legend
plt.legend(legend_labels)
plt.grid(True)

# Show the plot
plt.show()





import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('Technical performances of group II catalysts - Copy.xlsx')

# Prepare the figure
plt.figure(figsize=(12, 8))

# Manual list of labels for the legend
# This should be in the same order as the catalysts are plotted
legend_labels = ['TiO2/Nd10', 'TiO2/Gd10', 'TiO2/Ho10','TiO2/Y10']  # Continue for all catalysts

# Assuming the first catalyst's data starts at the first column
start_col = 0

# Loop through the sets of columns for each catalyst, assuming 6 columns per catalyst
# and skipping one column each time for the space between catalysts
for i, label in zip(range(0, df.shape[1], 7), legend_labels):  # Using zip to iterate over indices and labels
    # Select columns for temperature and CO conversion
    temp_col = df.iloc[:, i]  # Operating temperature
    co_uec_col = df.iloc[:, i + 1]  # CO conversion
    
    # Create the plot for this catalyst
    plt.plot(temp_col, co_uec_col, label=label)

# Customizing the plot
plt.xlabel('Temperature (°C)')
plt.ylabel('Unit Energy Cost (USD/Kg of H2)')
plt.title('Group II (TiO2 supported - metal promoted Pt) ')

# Add the manual legend
plt.legend(legend_labels)
plt.grid(True)

# Show the plot
plt.show()


####################

#Group III

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('Technical performances of group III catalysts - Copy.xlsx')

# Prepare the figure
plt.figure(figsize=(12, 8))

# Manual list of labels for the legend
# This should be in the same order as the catalysts are plotted
legend_labels = ['Al2O3/Ni10', 'Al2O3/Ti10', 'Al2O3/Fe10', 'Al2O3/Cr10', 'Al2O3/Mn', 'Al2O3/Ho', 'Al2O3/Nd', 'Al2O3/Tm10', 'Al2O3/Sm10', 'Al2O3/Er10', 'Al2O3/Ce10' ]  # Continue for all catalysts

# Assuming the first catalyst's data starts at the first column
start_col = 0

# Loop through the sets of columns for each catalyst, assuming 6 columns per catalyst
# and skipping one column each time for the space between catalysts
for i, label in zip(range(0, df.shape[1], 7), legend_labels):  # Using zip to iterate over indices and labels
    # Select columns for temperature and CO conversion
    temp_col = df.iloc[:, i]  # Operating temperature
    co_uec_col = df.iloc[:, i + 1]  # CO conversion
    
    # Create the plot for this catalyst
    plt.plot(temp_col, co_uec_col, label=label)

# Customizing the plot
plt.xlabel('Temperature (°C)')
plt.ylabel('Unit Energy Cost (USD/Kg of H2)')
plt.title('Group III (Al203 Supported - metal promoted-Pt)')

# Add the manual legend
plt.legend(legend_labels)
plt.grid(True)

# Show the plot
plt.show()


#Group IV

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('Technical performances of group IV catalysts - Copy.xlsx')

# Prepare the figure
plt.figure(figsize=(12, 8))

# Manual list of labels for the legend
# This should be in the same order as the catalysts are plotted
legend_labels = ['2wt%', '5wt%','10wt%', '20wt%']  # Continue for all catalysts

# Assuming the first catalyst's data starts at the first column
start_col = 0

# Loop through the sets of columns for each catalyst, assuming 6 columns per catalyst
# and skipping one column each time for the space between catalysts
for i, label in zip(range(0, df.shape[1], 7), legend_labels):  # Using zip to iterate over indices and labels
    # Select columns for temperature and CO conversion
    temp_col = df.iloc[:, i]  # Operating temperature
    co_uec_col = df.iloc[:, i + 1]  # CO conversion
    
    # Create the plot for this catalyst
    plt.plot(temp_col, co_uec_col, label=label)

# Customizing the plot
plt.xlabel('Temperature (°C)')
plt.ylabel('Unit Energy Cost (USD/Kg of H2)')
plt.title('Group IV (TiO2 supported- Ce promoted-Pt')

# Add the manual legend
plt.legend(legend_labels)
plt.grid(True)

# Show the plot
plt.show()


#Group V

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel('Technical performances of group V catalysts - Copy.xlsx')

# Prepare the figure
plt.figure(figsize=(12, 8))

# Manual list of labels for the legend
# This should be in the same order as the catalysts are plotted
legend_labels = ['5wt%', '10wt%', '20wt%']  # Continue for all catalysts

# Assuming the first catalyst's data starts at the first column
start_col = 0

# Loop through the sets of columns for each catalyst, assuming 6 columns per catalyst
# and skipping one column each time for the space between catalysts
for i, label in zip(range(0, df.shape[1], 7), legend_labels):  # Using zip to iterate over indices and labels
    # Select columns for temperature and CO conversion
    temp_col = df.iloc[:, i]  # Operating temperature
    co_uec_col = df.iloc[:, i + 1]  # CO conversion
    
    # Create the plot for this catalyst
    plt.plot(temp_col, co_uec_col, label=label)

# Customizing the plot
plt.xlabel('Temperature (°C)')
plt.ylabel('Unit Energy Cost (USD/Kg of H2)')
plt.title('Group V(Al2O3 supported- Co promoted-Pt)')

# Add the manual legend
plt.legend(legend_labels)
plt.grid(True)

# Show the plot
plt.show()

####################################


catalysts_per_group = {
    'Group I': ['TiO2', 'CeO2', 'La2O3', 'Co3O4', 'ThO2', 'Y2O3', 'Al2O3', 'Fe2O3'],
    'Group II': ['TiO2/Nd10', 'TiO2/Gd10', 'TiO2/Ho10','TiO2/Y10'],
    'Group III': ['Al2O3/Ni10', 'Al2O3/Ti10', 'Al2O3/Fe10', 'Al2O3/Cr10', 'Al2O3/Mn', 'Al2O3/Ho', 'Al2O3/Nd', 'Al2O3/Tm10', 'Al2O3/Sm10', 'Al2O3/Er10', 'Al2O3/Ce10' ],
    'Group IV': ['2wt%', '5wt%','10wt%', '20wt%'],
    'Group V': ['5wt%', '10wt%', '20wt%'],
    # ... until Group V
}


import pandas as pd
import matplotlib.pyplot as plt

# Dictionary of catalyst names for each group
catalysts_per_group = {
    'Group I': ['TiO2', 'CeO2', 'La2O3', 'Co3O4', 'ThO2', 'Y2O3', 'Al2O3', 'Fe2O3'],
    'Group II': ['TiO2/Nd10', 'TiO2/Gd10', 'TiO2/Ho10','TiO2/Y10'],
    'Group III': ['Al2O3/Ni10', 'Al2O3/Ti10', 'Al2O3/Fe10', 'Al2O3/Cr10', 'Al2O3/Mn', 'Al2O3/Ho', 'Al2O3/Nd', 'Al2O3/Tm10', 'Al2O3/Sm10', 'Al2O3/Er10', 'Al2O3/Ce10' ],
    'Group IV': ['2wt%', '5wt%','10wt%', '20wt%'],
    'Group V': ['5wt%', '10wt%', '20wt%'],
    # ... until Group V
}


# Define the line styles for each group
line_styles = {
    'Group I': 'solid',
    'Group II': 'dotted',
    'Group III': 'dashed',
    'Group IV': 'dashdot',
    'Group V': (0, (3, 5, 1, 5))  # Custom dash pattern
}

# Assuming your file paths are based on the group names
file_paths = {
    'Group I': 'Technical performances of group I catalysts.xlsx',
    'Group II': 'Technical performances of group II catalysts.xlsx',
    'Group III': 'Technical performances of group III catalysts.xlsx',
    'Group IV': 'Technical performances of group IV catalysts.xlsx',
    'Group V': 'Technical performances of group V catalysts.xlsx'
}

# Load and plot each group's UEC trend
plt.figure(figsize=(14, 7))
for group_name, catalyst_list in catalysts_per_group.items():
    # Construct the file path for this group's dataset
    file_path = file_paths[group_name]  # Get the file path from the dictionary

    # Load the dataset for this group
    df = pd.read_excel(file_path)  # Use the file path here

    # Loop through the catalysts in this group
    for i, catalyst_name in enumerate(catalyst_list):
        # The columns for temperature and UEC for each catalyst
        temperature_col = df.iloc[:, i * 7]  # Assuming temperature is the first column in each catalyst's set
        uec_col = df.iloc[:, i * 7 + 5]  # Assuming UEC is the sixth column in each catalyst's set

        # Plot the UEC trend for this catalyst
        plt.plot(temperature_col, uec_col, label=f'{group_name} - {catalyst_name}',
                 linestyle=line_styles[group_name])

# Customizing the plot
plt.xlabel('Temperature (°C)')
plt.ylabel('UEC (USD/Kg of H2)')
plt.title('UEC Trend of All Catalyst Groups')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

paths = [ 'Technical performances of group I catalysts.xlsx',
 'Technical performances of group II catalysts.xlsx',
 'Technical performances of group III catalysts.xlsx',
 'Technical performances of group IV catalysts.xlsx',
'Technical performances of group V catalysts.xlsx'] 




import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the group files and the catalysts of interest
catalyst_info = {
    'Pt/Co(20 wt %)/Al2O3': ('C:/Users/Blue/Music/Technical performances of group V catalysts.xlsx', 14),  # Replace with the actual path and correct index
    'Pt/Co(10 wt %)/Al2O3': ('C:/Users/Blue/Music/Technical performances of group V catalysts.xlsx', 7), # Replace with the actual path and correct index
    'Pt/Ce(5 wt %)/TiO2': ('C:/Users/Blue/Music/Technical performances of group IV catalysts.xlsx', 7),    # Replace with the actual path and correct index
}

# Prepare the figure with 3 subplots
fig, axes = plt.subplots(3, 1, figsize=(14, 21))

# Loop over the catalysts and plot their data
for catalyst, (file_path, start_col) in catalyst_info.items():
    # Load the dataset for this catalyst
    df = pd.read_excel(file_path)

    # Assuming the first column is the temperature column for each catalyst's data
    temperature = df.iloc[:, start_col]
    hydrogen_amount = df.iloc[:, start_col + 4]  # Assuming this is the hydrogen production column
    energy_consumption = df.iloc[:, start_col + 2]  # Assuming this is the energy consumption column
    uec = df.iloc[:, start_col + 5]  # Assuming this is the UEC column

    # Plot hydrogen amount
    axes[0].plot(temperature, hydrogen_amount, label=catalyst)
    axes[0].set_title('Hydrogen Amount')
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Hydrogen production (kg/hr)')

    # Plot energy consumption
    axes[1].plot(temperature, energy_consumption, label=catalyst)
    axes[1].set_title('Energy Consumption')
    axes[1].set_xlabel('Temperature (°C)')
    axes[1].set_ylabel('Energy consumption (GJ/hr)')

    # Plot unit energy cost
    axes[2].plot(temperature, uec, label=catalyst)
    axes[2].set_title('Unit Energy Cost')
    axes[2].set_xlabel('Temperature (°C)')
    axes[2].set_ylabel('UEC (USD/Kg of H2)')

# Add legends
for ax in axes:
    ax.legend()
    ax.grid(True)

# Show the plot
plt.tight_layout()
plt.show()












import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming you have loaded the datasets into DataFrames named df_group_i to df_group_v
# and that each DataFrame contains a 'UEC (USD/Kg of H2)' column for UEC values.

# Create a list of UEC values for the catalysts, one from each group
uec_values = [
    df_group_i['UEC (USD/Kg of H2)'].min(),  # Min UEC for Group I
    df_group_ii['UEC (USD/Kg of H2)'].min(), # Min UEC for Group II
    df_group_iii['UEC (USD/Kg of H2)'].min(), # Min UEC for Group III
    df_group_iv['UEC (USD/Kg of H2)'].min(), # Min UEC for Group IV
    df_group_v['UEC (USD/Kg of H2)'].min()   # Min UEC for Group V
]

# The number of variables we're plotting.
num_vars = len(uec_values)

# Compute angle each bar is centered on:
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

# The plot is made circular, so we need to "complete the loop" and append the start to the end.
uec_values += uec_values[:1]
angles += angles[:1]

# Draw the plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, uec_values, color='red', alpha=0.25)
ax.plot(angles, uec_values, color='red', linewidth=2)  # Adjust the color to your liking

# Label each axis with the name of the group
group_names = ['Group I', 'Group II', 'Group III', 'Group IV', 'Group V']
ax.set_xticks(angles[:-1])
ax.set_xticklabels(group_names)

# Ensure that the radial labels go from 0 to 100, and set a custom formatter
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))

# Add a title and show the plot
plt.title('Holistic Representation of UEC Effectiveness for Hydrogen Production')
plt.show()






import plotly.graph_objs as go

# Assuming 'uec_values' is a list of UEC values for the catalysts from each group
uec_values = [
    df_group_i['UEC (USD/Kg of H2)'].min(),  # Min UEC for Group I
    df_group_ii['UEC (USD/Kg of H2)'].min(), # Min UEC for Group II
    df_group_iii['UEC (USD/Kg of H2)'].min(), # Min UEC for Group III
    df_group_iv['UEC (USD/Kg of H2)'].min(), # Min UEC for Group IV
    df_group_v['UEC (USD/Kg of H2)'].min()   # Min UEC for Group V
]
catalyst_labels = ['Group I', 'Group II', 'Group III', 'Group IV', 'Group V']

# Create the radar chart
data = [
    go.Scatterpolar(
        r=uec_values + uec_values[:1],  # complete the loop
        theta=catalyst_labels + [catalyst_labels[0]],
        fill='toself',
        name='UEC Effectiveness'
    )
]

layout = go.Layout(
    title='UEC Effectiveness for Hydrogen Production',
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, max(uec_values)]
        )
    ),
    showlegend=False
)

fig = go.Figure(data=data, layout=layout)

# ... (the rest of your plotly code)

# Instead of fig.show(), save the figure to an HTML file
fig.write_html('C:/Users/Blue/Music/Plots/uec_effectiveness_chart.html', auto_open=True)


















import pandas as pd
import matplotlib.pyplot as plt

# Define the path to the excel files for each catalyst
file_path = 'path_to_your_excel_file.xlsx'  # Assuming all catalysts are in the same file

# Load the excel file
df = pd.read_excel(file_path)

# Define the starting column index for each catalyst, assuming there is a space after every 6 columns of data
catalyst_columns = {
    'Pt/Co(20 wt %)/Al2O3': 0,  # The first catalyst starts at column 0
    'Pt/Co(10 wt %)/Al2O3': 7,  # The second catalyst starts at column 7 (6 data columns + 1 space column)
    'Pt/Ce(5 wt %)/TiO2': 14,   # The third catalyst starts at column 14 (2*6 data columns + 2 space columns)
}

# Define the figure and axes for subplots
fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

# Define colors for each plot for visual distinction
colors = ['blue', 'orange', 'green']

# Loop through the catalysts and plot each economic metric
for i, (catalyst, start_col) in enumerate(catalyst_columns.items()):
    temperature = df.iloc[:, start_col]
    hydrogen_amount = df.iloc[:, start_col + 4]
    energy_consumption = df.iloc[:, start_col + 2]
    uec = df.iloc[:, start_col + 5]

    # Hydrogen amount vs Temperature
    axes[0].plot(temperature, hydrogen_amount, label=catalyst, color=colors[i])
    axes[0].set_ylabel('Hydrogen amount (kg of H2/hr)')
    axes[0].set_title('Hydrogen Amount vs Temperature')

    # Energy consumption vs Temperature
    axes[1].plot(temperature, energy_consumption, label=catalyst, color=colors[i])
    axes[1].set_ylabel('Energy consumption (GJ/hr)')
    axes[1].set_title('Energy Consumption vs Temperature')

    # UEC vs Temperature
    axes[2].plot(temperature, uec, label=catalyst, color=colors[i])
    axes[2].set_xlabel('Temperature (℃)')
    axes[2].set_ylabel('UEC (USD/Kg of H2)')
    axes[2].set_title('UEC vs Temperature')

# Add legends and grid to each subplot
for ax in axes:
    ax.legend()
    ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.show()











import pandas as pd
import matplotlib.pyplot as plt

# Define a function to extract relevant data for a given catalyst
def get_catalyst_data(df, catalyst_index):
    # Operating temperature is assumed to be the first column in each catalyst's set
    temperature = df.iloc[:, catalyst_index * 7].values
    hydrogen_production = df.iloc[:, catalyst_index * 7 + 4].values  # Hydrogen production column
    energy_consumption = df.iloc[:, catalyst_index * 7 + 2].values  # Energy consumption column
    uec = df.iloc[:, catalyst_index * 7 + 5].values  # Unit energy cost column
    return temperature, hydrogen_production, energy_consumption, uec

# Define your catalysts and their group datasets
catalysts_info = {
    'Pt/Co(20wt%)/Al2O3': 'path_to_group_iv_dataset.xlsx',  # Replace with actual paths
    'Pt/Co(10wt%)/Al2O3': 'path_to_group_iv_dataset.xlsx',
    'Pt/Ce(5wt%)/TiO2': 'path_to_group_i_dataset.xlsx'
}

# Prepare subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

for i, (catalyst_name, file_path) in enumerate(catalysts_info.items()):
    # Load the dataset
    df = pd.read_excel(file_path)
    
    # Assuming the last catalyst in the dataset is the one we want to plot
    temperature, hydrogen_production, energy_consumption, uec = get_catalyst_data(df, -1)
    
    # Plot (a) Hydrogen amount
    axs[0].plot(temperature, hydrogen_production, label=catalyst_name)
    axs[0].set_title('Hydrogen Amount')
    axs[0].set_xlabel('Temperature (°C)')
    axs[0].set_ylabel('Hydrogen amount (kg of H2/hr)')
    
    # Plot (b) Energy consumption
    axs[1].plot(temperature, energy_consumption, label=catalyst_name)
    axs[1].set_title('Energy Consumption')
    axs[1].set_xlabel('Temperature (°C)')
    axs[1].set_ylabel('Energy consumption (GJ/hr)')
    
    # Plot (c) Unit energy cost
    axs[2].plot(temperature, uec, label=catalyst_name)
    axs[2].set_title('Unit Energy Cost')
    axs[2].set_xlabel('Temperature (°C)')
    axs[2].set_ylabel('UEC (USD/Kg of H2)')

# Add legends
for ax in axs:
    ax.legend()
    ax.grid(True)

# Show the plot
plt.show()







import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
htwo_project_df = pd.read_excel('Htwo Project.xlsx')  # Replace with your actual file path

# Define the calculation for overall CO conversion
def calculate_overall_conversion(row):
    total_co_in = row['Stream1_CO']  # CO mole flow in Stream #1
    total_co_out = row['Stream8_H2']  # H2 mole flow in Stream #8
    return (total_co_in - total_co_out) / total_co_in * 100

# Apply the calculation to each row
# Note: Stream names are placeholders; replace with your actual stream column names
htwo_project_df['Overall CO Conversion (%)'] = htwo_project_df.apply(calculate_overall_conversion, axis=1)

# Now let's plot the overall CO conversion for each catalyst type
plt.figure(figsize=(15, 10))

# Assuming 'Catalyst type' is a column in your DataFrame that lists the catalysts
unique_catalysts = htwo_project_df['Catalyst type'].unique()

for catalyst in unique_catalysts:
    subset = htwo_project_df[htwo_project_df['Catalyst type'] == catalyst]
    plt.plot(subset['Operating temperature (Co)'], subset['Overall CO Conversion (%)'], label=catalyst)

plt.xlabel('Operating Temperature (°C)')
plt.ylabel('Overall CO Conversion (%)')
plt.title('Estimated Overall CO Conversion of Selected Catalysts')
plt.legend()
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset here
htwo_project_df = pd.read_excel('Htwo Project.xlsx')  # Replace with your actual file path



# Placeholder for the actual catalyst group arrays
catalysts_group_i = ['Pt/TiO2', 'Pt/CeO2', 'Pt/La2O3', 'Pt/Co3O4', 'Pt/ThO2', 'Pt/Y2O3', 'Pt/Al2O3', 'Pt/Fe2O3'] # Replace with the actual catalysts for Group I
catalysts_group_ii = ['Pt/TiO2/Nd10', 'Pt/TiO2/Gd10', 'Pt/TiO2/Ho10','Pt/TiO2/Y10']
catalysts_group_iii =  ['Pt/Al2O3/Ni10', 'Pt/Al2O3/Ti10', 'Pt/Al2O3/Fe10', 'Pt/Al2O3/Cr10', 'Pt/Al2O3/Mn', 'Pt/Al2O3/Ho', 'Pt/Al2O3/Nd', 'Pt/Al2O3/Tm10', 'Pt/Al2O3/Sm10', 'Pt/Al2O3/Er10', 'Pt/Al2O3/Ce10' ]
catalysts_group_iv = ['Pt/TiO2/Ce2', 'Pt/TiO2/Ce5','Pt/TiO2/Ce10', 'Pt/TiO2/Ce20']
catalysts_group_v = ['Pt/Al2O3/Co5', 'Pt/Al2O3/Co10', 'Pt/Al2O3/Co20']


  # Replace with actual catalysts for Group II
# ... Do this for each group

# Define the calculation for overall CO conversion
def calculate_overall_conversion(row):
    total_co_in = row['Stream1_CO']  # Placeholder for the actual column name
    total_co_out = row['Stream8_H2']  # Placeholder for the actual column name
    return (total_co_in - total_co_out) / total_co_in * 100

# Apply the calculation to each row in the DataFrame
# Note: Column names 'Stream1_CO' and 'Stream8_H2' are placeholders
htwo_project_df['Overall CO Conversion (%)'] = htwo_project_df.apply(calculate_overall_conversion, axis=1)

# Define a function to plot the CO conversion for a group
def plot_conversion_for_group(group_catalysts, group_name):
    plt.figure(figsize=(15, 10))
    
    for catalyst in group_catalysts:
        # Filter the DataFrame for the current catalyst
        subset = htwo_project_df[htwo_project_df['Catalyst type'] == catalyst]
        plt.plot(subset['Operating temperature (Co)'], subset['Overall CO Conversion (%)'], label=catalyst)
    
    plt.xlabel('Operating Temperature (°C)')
    plt.ylabel('Overall CO Conversion (%)')
    plt.title(f'Estimated Overall CO Conversion for {group_name}')
    plt.legend()
    plt.show()

# Now you can call this function for each group
plot_conversion_for_group(catalysts_group_i, 'Group I')
plot_conversion_for_group(catalysts_group_ii, 'Group II')
plot_conversion_for_group(catalysts_group_ii, 'Group III')
plot_conversion_for_group(catalysts_group_ii, 'Group IV')
plot_conversion_for_group(catalysts_group_ii, 'Group V')

# ... Do this for each group



# ... [Your previous code for loading the dataset and defining the calculation]

def plot_conversion_for_group(group_catalysts, group_name, htwo_project_df):
    plt.figure(figsize=(15, 10))
    
    for catalyst in group_catalysts:
        subset = htwo_project_df[htwo_project_df['Catalyst type'] == catalyst]
        
        # Debug: Check the subset
        print(f"Subset for {catalyst} in {group_name}:")
        print(subset[['Operating temperature (Co)', 'Overall CO Conversion (%)']].head())
        
        plt.plot(subset['Operating temperature (Co)'], subset['Overall CO Conversion (%)'], label=catalyst)
    
    plt.xlabel('Operating Temperature (°C)')
    plt.ylabel('Overall CO Conversion (%)')
    plt.title(f'Estimated Overall CO Conversion for {group_name}')
    plt.legend()
    plt.show()

# Now you can call this function for each group, make sure to pass the DataFrame as well
plot_conversion_for_group(catalysts_group_i, 'Group I', htwo_project_df)
plot_conversion_for_group(catalysts_group_ii, 'Group II', htwo_project_df)
plot_conversion_for_group(catalysts_group_ii, 'Group III', htwo_project_df)
plot_conversion_for_group(catalysts_group_ii, 'Group IV', htwo_project_df)
plot_conversion_for_group(catalysts_group_ii, 'Group V', htwo_project_df)

# ... Do this for each group



























































import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Load your Group IV dataset
df_group_iv = pd.read_excel('Technical performances of group IV catalysts.xlsx')  # Replace with the actual path to your Excel file

# Let's assume the last catalyst "Pt/TiO2/Ce20" is in the last 6 columns of the dataset
# Extracting the relevant columns for the "Pt/TiO2/Ce20" catalyst
df_catalyst = df_group_iv.iloc[:, -6:]  # Adjust the index as per the actual location

# Splitting the data into components for the stacked area plot
temperature = df_catalyst['Operating temperature (℃).3']
# energy_components = df_catalyst[['Reactant heating', 'Regeneration heat', 'PSA Electricity', 'COSORB Electricity', 'Cooling water']]  # Replace with actual column names
total_cost = df_catalyst['Total energy price (USD/hr).3']
uec = df_catalyst['UEC (USD/Kg of H2).3']

plt.figure(figsize=(12, 6))
plt.stackplot(temperature, total_cost.T, labels=['Total Energy Cost'])  # You might need to normalize the costs
plt.xlabel('Temperature (°C)')
plt.ylabel('Cost Contribution (%)')
plt.title('Cost Contribution of Utilizing Pt/Ce(20wt%)/TiO2')
plt.show()
# =============================================================================
# # Plot (b) Cost contribution of utilizing Pt/Ce(20wt%)/TiO2
# # Assuming 'total_cost' is normalized to 100% across the components
# plt.figure(figsize=(12, 6))
# plt.stackplot(temperature, total_cost.T, labels=['Total Energy Cost'])  # You might need to normalize the costs
# plt.xlabel('Temperature (°C)')
# plt.ylabel('Cost Contribution (%)')
# plt.title('Cost Contribution of Utilizing Pt/Ce(20wt%)/TiO2')
# plt.show()
# 
# # Plot (c) Average energy cost according to CO conversion
# # For this plot, you need to calculate the average cost per CO conversion range
# # Let's assume df_catalyst has a 'CO conversion (%)' column
# co_conversion_ranges = pd.cut(df_catalyst['CO conversion (%)'], bins=5)  # Dividing CO conversion into ranges
# average_cost_per_conversion = df_catalyst.groupby(co_conversion_ranges)['Total energy price (USD/hr)'].mean()
# 
# plt.figure(figsize=(12, 6))
# plt.plot(average_cost_per_conversion.index, average_cost_per_conversion.values, marker='o')
# plt.xlabel('CO Conversion (%)')
# plt.ylabel('Average Energy Cost (USD)')
# plt.title('Average Energy Cost According to CO Conversion')
# plt.show()
# 
# # Plot (d) Average energy cost according to hydrogen amount
# # Similar approach as plot (c), but group by 'Hydrogen production (kg/hr)' instead
# hydrogen_production_ranges = pd.cut(df_catalyst['Hydrogen production (kg/hr)'], bins=5)  # Dividing hydrogen production into ranges
# average_cost_per
# =============================================================================







# =============================================================================
# 
# 
# import pandas as pd
# 
# # Load each dataset separately
# df_group_i = pd.read_excel('Technical performances of group I catalysts.xlsx')
# df_group_ii = pd.read_excel('Technical performances of group II catalysts.xlsx')
# df_group_iii = pd.read_excel('Technical performances of group III catalysts.xlsx')
# df_group_iv = pd.read_excel('Technical performances of group IV catalysts.xlsx')
# df_group_v = pd.read_excel('Technical performances of group V catalysts.xlsx')
# 
# # Add a 'Group' column to each DataFrame
# df_group_i['Group'] = 'Group I'
# df_group_ii['Group'] = 'Group II'
# df_group_iii['Group'] = 'Group III'
# df_group_iv['Group'] = 'Group IV'
# df_group_v['Group'] = 'Group V'
# 
# # Concatenate the DataFrames into one
# df_all_groups = pd.concat([df_group_i, df_group_ii, df_group_iii, df_group_iv, df_group_v], ignore_index=True)
# 
# 
# 
# import pandas as pd
# import matplotlib.pyplot as plt
# 
# # Placeholder for the DataFrame that contains all groups data
# # You'll need to concatenate the individual group datasets into one DataFrame
# # Add a 'Group' column in each dataset before concatenation to identify the group
# # df_all_groups = pd.concat([df_group_i, df_group_ii, df_group_iii, df_group_iv, df_group_v])
# 
# # Prepare the figure
# plt.figure(figsize=(14, 7))
# 
# # Define the line styles for each group
# line_styles = {
#     'Group I': 'solid',
#     'Group II': 'dotted',
#     'Group III': 'dashed',
#     'Group IV': 'dashdot',
#     'Group V': (0, (3, 5, 1, 5))  # Custom dash pattern
# }
# 
# # Define the colors for each group if desired
# group_colors = {
#     'Group I': 'blue',
#     'Group II': 'red',
#     'Group III': 'green',
#     'Group IV': 'yellow',
#     'Group V': 'purple'
# }
# 
# # Assuming 'group_name' is the name of the column that identifies the group each row belongs to
# # And assuming 'catalyst_name' is the name of the column that identifies the catalyst
# for group_name, group_df in df_all_groups.groupby('group_name'):
#     for catalyst_name in group_df['catalyst_name'].unique():
#         # Extract the data for this group and catalyst
#         catalyst_data = group_df[group_df['catalyst_name'] == catalyst_name]
#         
#         # Plot the UEC trend for this catalyst
#         plt.plot(catalyst_data['Operating temperature (℃)'], catalyst_data['UEC (USD/Kg of H2)'],
#                  label=f'{group_name} - {catalyst_name}',
#                  linestyle=line_styles[group_name],
#                  color=group_colors[group_name])
# 
# # Customizing the plot
# plt.xlabel('Temperature (°C)')
# plt.ylabel('UEC (USD/Kg of H2)')
# plt.title('UEC Trend of All Catalyst Groups')
# plt.legend()
# plt.grid(True)
# 
# # Show the plot
# plt.show()
# =============================================================================


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Define the dataset based on the image provided
data = {
    'Load of swinger (N)': [60, 77, 80, 81],
    'Speed of the swing': ['Faster', 'Average', 'Slower', 'Slowest'],
    'Output (V)': [4.56, 3.72, 2.94, 1.56],
    'Status of load': ['Glow', 'Glow', 'Lightly dimmed', 'Dimmed']
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

# Define marker styles based on speed
speed_markers = {
    'Faster': 'o',  # Circle
    'Average': 's',  # Square
    'Slower': 'D',  # Diamond
    'Slowest': '^'  # Triangle
}

# Choose a color map
cmap = plt.get_cmap('viridis')

# Normalize the Load of swinger (N) for the color mapping
norm = plt.Normalize(df['Load of swinger (N)'].min(), df['Load of swinger (N)'].max())

# Create a scatter plot
fig, ax = plt.subplots(figsize=(10, 6))
for i, row in df.iterrows():
    ax.scatter(row['Load of swinger (N)'], row['Output (V)'], 
                s=row['Output (V)']*20,  # Marker size proportional to Output (V)
                c=[cmap(norm(row['Load of swinger (N)']))],  # Color based on load
                marker=speed_markers[row['Speed of the swing']],  # Marker style based on speed
                label=f"{row['Speed of the swing']} - {row['Status of load']}" if i == 0 else "",
                alpha=0.6)

# Add annotations for status of load
for i, row in df.iterrows():
    ax.text(row['Load of swinger (N)'], row['Output (V)'],
             f"{row['Status of load']}",
             fontsize=9,
             ha='center', va='bottom')

# Create color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Load of swinger (N)')

# Adding titles and labels
ax.set_title('Load of Swinger vs. Output Voltage', fontsize=14)
ax.set_xlabel('Load of Swinger (N)', fontsize=12)
ax.set_ylabel('Output Voltage (V)', fontsize=12)

# Add a legend outside of the plot
ax.legend(title="Speed & Status", loc='lower left', bbox_to_anchor=(1.2, 0.5))

# Add a grid for better readability
ax.grid(True)

# Show the plot with a tight layout
# ... [previous code]

# Sort the DataFrame by 'Load of swinger (N)' to ensure the line connects points in the right order
df_sorted = df.sort_values('Load of swinger (N)')

# Plot a line connecting the sorted points
plt.plot(df_sorted['Load of swinger (N)'], df_sorted['Output (V)'], marker='', color='gray', linewidth=2, linestyle='-', label='Trend')

# ... [rest of the plotting code]

plt.tight_layout()
plt.show()












import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load the dataset (make sure to replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('Book2.csv')

# Set the style for seaborn plots
sns.set_style("whitegrid")

# Create a pairplot to visualize the relationships between all pairs of variables
sns.pairplot(df, kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}})

# Show the pairplot
plt.show()

# Now, let's create individual plots for each pair of variables.
# Plotting No. of oscillations vs Voltage
plt.figure(figsize=(10, 6))
sns.regplot(x='NO. OF OSCILLATIONS', y='CURRENT (AMPS)', data=df, color='blue', line_kws={'color': 'red'})
plt.title('Number of Oscillations vs CURRENT')
plt.xlabel('Number of Oscillations')
plt.ylabel('CURRENT (AMPS)')
plt.show()

# Plotting Swing angle (degree) vs Voltage
plt.figure(figsize=(10, 6))
sns.regplot(x='SWING ANGLES (DEGREE)', y='CURRENT (AMPS)', data=df, color='green', line_kws={'color': 'red'})
plt.title('Swing Angle vs AMPS')
plt.xlabel('Swing Angle (degree)')
plt.ylabel('CURRENT (AMPS)')
plt.show()

# Note: You need to have seaborn and pandas installed in your environment
# You can install them using pip:
# pip install seaborn pandas





import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator

# Since I cannot access the actual data, I will simulate a dataset based on the structure provided in the image
# For a real dataset, replace this with: df = pd.read_csv('path_to_your_data.csv')

# Simulated data based on the structure of the provided dataset
df = pd.read_csv('Book2.csv')


# Setting a style for seaborn
sns.set(style="whitegrid", palette="muted")

# Create the plot with matplotlib and seaborn
plt.figure(figsize=(14, 8))

# Create a scatter plot of Swing Angle vs Voltage with a regression line
sns.regplot(x='SWING ANGLES (DEGREE)', y='CURRENT (AMPS)', data=df, scatter_kws={'s': 100, 'alpha': 0.6}, line_kws={'color': 'red', 'lw': 2})

# Enhance the plot to make it more detailed and professional
plt.title('Swing Angle vs Current', fontsize=18, fontweight='bold')
plt.xlabel('Swing Angles (Degree)', fontsize=14)
plt.ylabel('CURRENT (AMPS)', fontsize=14)

# Improve the x and y axis ticks
ax = plt.gca()  # Get the current Axes instance on the current figure
ax.xaxis.set_minor_locator(AutoMinorLocator())  # Set minor ticks locator
ax.yaxis.set_minor_locator(AutoMinorLocator())

# Customize grid
ax.grid(which='major', color='dimgrey', linewidth=0.8)
ax.grid(which='minor', color='lightgrey', linestyle=':', linewidth=0.5)

# Set the limit for better representation if needed
plt.xlim([0, max(df['SWING ANGLES (DEGREE)']) + 10])
plt.ylim([0, max(df['CURRENT (AMPS)']) + 1])

# Add a legend with a shadow for better visual effect
plt.legend(['Regression Line', 'Data Points'], loc='upper left', shadow=True)

# Remove the top and right spines to make the plot cleaner
sns.despine()

# Show the plot
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Simulating the dataset based on the provided structure
# Replace this with your actual data loading method

# Create a DataFrame
df = pd.read_csv('Book2.csv')

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Create a figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 16), sharex=True)

# Plot No. of oscillations vs Voltage
sns.lineplot(ax=axes[0], x='NO. OF OSCILLATIONS', y='CURRENT (AMPS)', data=df, marker='o', color='blue', label='Current (V)')
axes[0].set_title('No. of oscillations vs Current', fontsize=16)
axes[0].set_xlabel('No. of oscillations', fontsize=14)
axes[0].set_ylabel('CURRENT (AMPS)', fontsize=14)
axes[0].legend()
axes[0].grid(True)

# Create a polar plot for Swing angle vs Voltage
angles = np.deg2rad(df['SWING ANGLES (DEGREE)'])  # Convert angles to radians for the polar plot
axes[1] = plt.subplot(2, 1, 2, polar=True)
axes[1].plot(angles, df['CURRENT (AMPS)'], label='CURRENT (AMPS)', color='red')  # Polar plot
axes[1].set_theta_zero_location('N')  # Set 0 degrees to the top
axes[1].set_theta_direction(-1)  # Set the direction of degrees clockwise
axes[1].set_title('SWING ANGLES (DEGREE) vs CURRENT (AMPS)', va='bottom', fontsize=16)
axes[1].set_xlabel('SWING ANGLES (DEGREE)', fontsize=14)
axes[1].set_ylabel('CURRENT (AMPS)', fontsize=14, labelpad=30)
axes[1].set_rlabel_position(0)  # Set the position of the radial labels
axes[1].set_rticks(np.linspace(df['CURRENT (AMPS)'].min(), df['CURRENT (AMPS)'].max(), 5))  # Set the radial ticks
axes[1].grid(True)
axes[1].legend(loc='upper right')

# Tight layout to adjust subplots to fit into figure area.
plt.tight_layout()

# Display the plot
plt.show()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns
import numpy as np

# Assuming you have loaded your dataset into a Pandas DataFrame named df
# df = pd.read_csv('your_dataset.csv')  # Make sure to load your actual data here

# Simulated data based on the provided image structure
df = pd.read_csv('Book2.csv')


# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with color mapping based on voltage
sc = ax.scatter(df['SWING ANGLES (DEGREE)'], df['NO. OF OSCILLATIONS'], df['VOLTAGE (VOLTS)'],
                c=df['VOLTAGE (VOLTS)'], cmap='coolwarm', s=50, edgecolor='k')

# Line plot to connect the points
for i in range(len(df['NO. OF OSCILLATIONS']) - 1):
    ax.plot(df['SWING ANGLES (DEGREE)'][i:i+2], df['NO. OF OSCILLATIONS'][i:i+2], df['VOLTAGE (VOLTS)'][i:i+2], color='gray')

# Color bar indicating voltage magnitude
cbar = plt.colorbar(sc)
cbar.set_label('Voltage (Volts)')

# Labels and title
ax.set_xlabel('Swing Angle (Degrees)')
ax.set_ylabel('Number of Oscillations')
ax.set_zlabel('Voltage (Volts)')
ax.set_title('3D Visualization of Oscillation Data')

# Pairplot to show pairwise relationships
sns.pairplot(df, kind='reg')

# Show the plots
plt.show()




import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('mode.csv')

# Assuming 'RSM TSOME Yield' is your actual values and 'GWO TSOME Yield' are the predicted values
y_true = df['TSOME Yield']
y_pred = df['RSM TSOME YIELD']

# Calculate R2, RMSE, and MAE
r2 = r2_score(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r = np.corrcoef(y_true, y_pred)[0, 1]
n = len(y_true)
sep = np.sqrt(np.sum((y_true - y_pred) ** 2) / n)
aad = np.mean(np.abs(y_true - y_pred))


# Calculate R, SEP, and AAD manually
# Placeholder for R, SEP, AAD calculations - Use appropriate formulas

# Visualization
metrics = ['R2', 'RMSE', 'MAE','R', 'SEP', 'AAD' ]  # Add 'R', 'SEP', 'AAD' after calculation
values = [r2, rmse, mae, r, sep, aad]  # Add calculated R, SEP, AAD values

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'green', 'red'])  # Use different colors as needed
plt.title('Comparison of Metrics for TSOME Yield')
plt.ylabel('Metric Values')
plt.show()

print(f'R: {r}')
print(f'SEP: {sep}')
print(f'AAD: {aad}')










import numpy as np
import matplotlib.pyplot as plt

# Assuming y_true is your true TSOME Yield values
# And y_pred_gwo and y_pred_rsm are your GWO TSOME Yield and RSM TSOME Yield values respectively
y_true = df['TSOME Yield']
y_pred_gwo = df['GWO TSOME YIELD']
y_pred_rsm = df['RSM TSOME YIELD']

def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # Pearson correlation coefficient (R)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    # Standard Error of Prediction (SEP)
    sep = np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))
    # Average Absolute Deviation (AAD)
    aad = np.mean(np.abs(y_true - y_pred))
    return r, r2, rmse, sep,   mae, aad

# Calculate metrics for both GWO and RSM predictions
r_gwo, r2_gwo, rmse_gwo, sep_gwo, mae_gwo, aad_gwo = calculate_metrics(y_true, y_pred_gwo)
r_rsm, r2_rsm, rmse_rsm, sep_rsm, mae_rsm, aad_rsm = calculate_metrics(y_true, y_pred_rsm)

# Data for plotting
metrics = ['R', 'R2', 'RMSE', 'SEP', 'MAE', 'AAD']
gwo_values = [r_gwo, r2_gwo, rmse_gwo, sep_gwo, mae_gwo, aad_gwo]
rsm_values = [r_rsm, r2_rsm, rmse_rsm, sep_rsm, mae_rsm, aad_rsm]

# Plotting
x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, gwo_values, width, label= 'RSM TSOME Yield')
rects2 = ax.bar(x + width/2, rsm_values, width, label='GWO TSOME Yield')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Values')
ax.set_title('Comparison of GWO and RSM TSOME Yield Predictions')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()



import matplotlib.pyplot as plt
import numpy as np

# Assuming df is your DataFrame with the dataset
# Replace 'actual_yield', 'gwo_yield', 'rsm_yield' with your actual column names
actual_yield = df['TSOME Yield']
gwo_yield = df['GWO TSOME YIELD']
rsm_yield = df['RSM TSOME YIELD']

# Function to plot scatter plot with a trend line
def plot_scatter_with_trend(x, y, color, label, title, xlabel, ylabel):
    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color=color, alpha=0.5, label=label)

    # Fit a simple linear regression line to the data
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")  # Dotted line

    # Plot details
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot GWO TSOME Yield vs. Actual TSOME Yield with trend line
plot_scatter_with_trend(
    gwo_yield, actual_yield, 'orange', 'GWO TSOME Yield',
    'GWO TSOME Yield vs. Actual TSOME Yield', 'Predicted TSOME Yield (%)',
    'Actual TSOME Yield (%)'
)

# Plot RSM TSOME Yield vs. Actual TSOME Yield with trend line
plot_scatter_with_trend(
    rsm_yield, actual_yield, 'blue', 'RSM TSOME Yield',
    'RSM TSOME Yield vs. Actual TSOME Yield', 'Predicted TSOME Yield (%)',
    'Actual TSOME Yield (%)'
)







import pandas as pd

# Assuming you have calculated the following metrics
r_gwo, r2_gwo, rmse_gwo, sep_gwo, mae_gwo, aad_gwo  = calculate_metrics(y_true, y_pred_gwo)
r_rsm, r2_rsm, rmse_rsm, sep_rsm, mae_rsm, aad_rsm = calculate_metrics(y_true, y_pred_rsm)

# Create a dictionary with the metrics
metrics_dict = {
    'Metric': ['R', 'R2', 'RMSE', 'SEP', 'MAE', 'AAD'],
    'GWO TSOME Yield': [r_gwo, r2_gwo, rmse_gwo, sep_gwo, mae_gwo, aad_gwo ],
    'RSM TSOME Yield': [r_rsm, r2_rsm, rmse_rsm, sep_rsm, mae_rsm, aad_rsm]
}

# Create a DataFrame
metrics_df = pd.DataFrame(metrics_dict)

# Set the 'Metric' column as the index
metrics_df.set_index('Metric', inplace=True)

# Display the table
print(metrics_df)

# Assuming metrics_df is the DataFrame containing your statistical metrics

# Specify the filename you wish to save as
excel_filename = 'statistical_metrics.xlsx'

# Save the DataFrame to an Excel file
metrics_df.to_excel(excel_filename)

print(f'Statistical metrics saved to {excel_filename}')





import numpy as np

# Assuming y_true and y_pred are your true and predicted values respectively
# y_true = df['RSM TSOME Yield']
# y_pred = df['GWO TSOME Yield']

# Calculate Pearson correlation coefficient (R)
r = np.corrcoef(y_true, y_pred)[0, 1]

# Calculate Standard Error of Prediction (SEP)
n = len(y_true)
sep = np.sqrt(np.sum((y_true - y_pred) ** 2) / n)

# Calculate Average Absolute Deviation (AAD)
aad = np.mean(np.abs(y_true - y_pred))

print(f'R: {r}')
print(f'SEP: {sep}')
print(f'AAD: {aad}')


