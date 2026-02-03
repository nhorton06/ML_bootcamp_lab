# %% [markdown]
# ## Step 4: Pipeline functions

# %%
# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



# -----------------------------------------------------------------------------------------------------------------



# College Completion Dataset

# Read in data
college = pd.read_csv('https://github.com/UVADS/DS-3021/blob/main/data/cc_institution_details.csv')

# Drop ID variables and unnecessary columns (and maybe missing values)
def drop_cols(data, drop_cols = [], drop_missing = False):
    for col in drop_cols:
        data = data.drop(columns = [col])
    if drop_missing:
        data = data.dropna()
    return data

# Change to category type
def type_cat_college(data, cat_cols = []):
    for col in cat_cols:
        data[col] = data[col].astype('category')
    return data

# One-hot encoding
def onehot_encode_college(data, cat_cols = []):
    return pd.get_dummies(data, columns = cat_cols)

# Create target variable and calculate prevalence
def create_target_college(data, source_col, target_name, threshold = 0.5):
    data[target_name] = (data[source_col] >= threshold).astype(int)
    prevalence = data[target_name].mean()
    return data, prevalence

# Normalize/scale numeric data
def scale_numeric_college(data, target_var, numeric_cols = None):
    if numeric_cols == None:
        numeric_cols = data.select_dtypes(include = ['int64', 'float64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != target_var]
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

# Split for train, tune, test
def split_college(data, target_var, test_size = 0.2, tune_size = 0.25, stratify = True):
    stratify_col = data[target_var] if stratify else None
    train, temp = train_test_split(data, test_size = test_size, random_state = 42, stratify = stratify_col)
    stratify_temp = temp[target_var] if stratify else None
    tune, test = train_test_split(temp, test_size = 0.5, random_state = 42, stratify = stratify_temp)

    print(f'Training set shape: {train.shape}')
    print(f'Tuning set shape: {tune.shape}')
    print(f'Testing set shape: {test.shape}')
    return train, tune, test



# -----------------------------------------------------------------------------------------------------------------



# %%

# Job Placement Dataset
