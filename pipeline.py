# %% [markdown]
# ## Step 4: Pipeline functions

# %%
# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# College Completion Dataset

# Read in data
college = pd.read_csv('https://github.com/UVADS/DS-3021/blob/main/data/cc_institution_details.csv')

# Drop ID variables
def drop_cols(data):
    return data.drop(columns = ['chronname', 'site', 'nicknames'])

# Drop NAs
def drop_missing_college(data):
    data = data.dropna()
    return data

# Change columns to category type
def change_to_category(data):
    cat_cols = ['control', 'city', 'state', 'level']
    for col in cat_cols:
        data[col] = data[col].astype('category')
    return data


# Collapse factor levels
def collapse_factors_college(data):
    for entry in data.index:
        val = data.loc[entry, 'control']
        if val == 'Public':
            data.loc[entry, 'control'] = 'Public'
        else:
            data.loc[entry, 'control'] = 'Private'
    return data

# One-hot encoding
def one_hot_college(data):
    cat_cols = ['control', 'city', 'state', 'level']
    data = pd.get_dummies(data, columns = cat_cols)
    return data

# Create target variable
def create_target_college(data, threshold = 0.5):
    data['target_completion'] = 0
    for i in data.index:
        if data.loc[i, 'fte_percentile'] >= threshold:
            data.loc[i, 'target_completion'] = 1
    prevalence = data['target_completion'].mean()
    return data, prevalence

# Scale numeric data
def scale_numeric_college(data):
    numeric_cols = data.select_dtypes(include = ['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'target_completion']
    scaler = StandardScaler()
    scaled_vals = scaler.fit_transform(data[numeric_cols])
    data[numeric_cols] = scaled_vals
    return data

# Split data into train, tune, test
def split_data_college(data):
    X = data.drop(columns = ['target_completion'])
    y = data['target_completion']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size = 0.3, stratify = y, random_state = 1)
    
    X_tune, X_test, y_tune, y_test = train_test_split(
        X_temp, y_temp, test_size = 0.5, stratify = y_temp, random_state = 1)
    
    return X_train, X_tune, X_test, y_train, y_tune, y_test




# %%

# Job Placement Dataset
