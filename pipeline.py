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
def drop_cols_college(data, drop_cols = [], drop_missing = False):
    '''Drop identifier variables and unnecessary columns (and optionally missing values since there are some in this dataset)'''
    if drop_cols:
        data = data.drop(columns = drop_cols)
    if drop_missing:
        data = data.dropna()
    return data

# Change to category type
def type_cat_college(data, cat_cols = []):
    '''Change given to columns to type category'''
    for col in cat_cols:
        data[col] = data[col].astype('category')
    return data

# One-hot encoding
def onehot_encode_college(data, cat_cols = []):
    '''One-hot encode specified categorical columns'''
    return pd.get_dummies(data, columns = cat_cols)

# Create target variable and calculate prevalence
def create_target_college(data, source_col, target_name, threshold = 0.5):
    '''Create binary target variable and calculate prevalence, threshold can be changed to what user sees fit'''
    data[target_name] = (data[source_col] >= threshold).astype(int) # defined target_name for new target variable should be used for any further reference of target_var in further functions
    prevalence = data[target_name].mean()
    return data, prevalence

# Normalize/scale numeric data
def scale_numeric_college(data, target_var, numeric_cols = None):
    '''Scale numeric columns using StandardScaler, but excludes specified target variable'''
    if numeric_cols == None:
        numeric_cols = data.select_dtypes(include = ['int64', 'float64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != target_var]
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

# Split for train, tune, test
def split_college(data, target_var, test_size = 0.2, tune_size = 0.25, stratify = True):  # test sizes and tune sizes are adjustable
    '''Splits data into train, tune, and test sets, with stratification of target variable on by default'''
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

# Read in data
job = pd.read_csv('https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv')

# Drop ID variables and unnecessary columns
def drop_cols_job(data, drop_cols = []):
    '''Drop identifier variables and unnecessary columns'''
    if drop_cols:
        data = data.drop(columns = drop_cols)
    return data

# Change to category type
def type_cat_job(data, cat_cols = []):
    '''Change given to columns to type category'''
    for col in cat_cols:
        data[col] = data[col].astype('category')
    return data

# One-hot encoding
def onehot_encode_job(data, cat_cols = []):
    '''One-hot encode specified categorical columns'''
    return pd.get_dummies(data, columns = cat_cols)

# Adjust target variable (already exists as placed/not placed, but can be made to 1/0 for easier computing) and calculate prevalence
def adjust_target_job(data, source_col, success): # source_col should be 'status' for this dataset in particular, positive should be 'Placed'
    '''Adjusts target variable ('status' in this case) to be binary to 1 and 0 given the success string'''
    data[source_col] = data[source_col].apply(lambda x: 1 if x == success else 0)
    prevalence = data[source_col].mean()
    return data, prevalence

# Normalize/scale numeric data
def scale_numeric_job(data, target_var, numeric_cols = None):
    '''Scale numeric columns using StandardScaler, but excludes specified target variable'''
    if numeric_cols == None:
        numeric_cols = data.select_dtypes(include = ['int64', 'float64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != target_var]
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

# Split for train, tune, test
def split_job(data, target_var, test_size = 0.2, tune_size = 0.25, stratify = True): # target_var should be 'status' for this dataset, test sizes and tune sizes are adjustable
    '''Splits data into train, tune, and test sets, with stratification of target variable on by default'''
    stratify_col = data[target_var] if stratify else None
    train, temp = train_test_split(data, test_size = test_size, random_state = 42, stratify = stratify_col)
    stratify_temp = temp[target_var] if stratify else None
    tune, test = train_test_split(temp, test_size = 0.5, random_state = 42, stratify = stratify_temp)

    print(f'Training set shape: {train.shape}')
    print(f'Tuning set shape: {tune.shape}')
    print(f'Testing set shape: {test.shape}')
    return train, tune, test