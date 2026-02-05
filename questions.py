# %% [markdown]
# # Machine Learning Bootcamp Lab

# %% [markdown]
# ## Step 1: Potential Dataset Questions

# - **College Completion Dataset**: What institutional characteristics (like region, cost, etc.) are most predictive of higher college completion rates?
# - **Job Placement Dataset**: How do factors like degree, specialization, and gender influence job placement rates after graduation?


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------



# %% [markdown]
# ## Step 2: Data Preparation

# ### Independent Business Metrics
# - College Completion Dataset: Completion rate of degree/program (will track whether any changes make any difference to actual percentage of students that complete their program)
# - Job Placement Dataset: Proportion of placement (will track whether any changes make any difference to how many students are getting placed in jobs)

# ### Data Preparation

# #### College Completion Data

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
# Read in data
college = pd.read_csv('https://raw.githubusercontent.com/UVADS/DS-3021/main/data/cc_institution_details.csv')
print(college.info())

# %%
# Drop unnecessary columns
college = college.drop(columns = ['chronname', 'site', 'nicknames'])

# %%
# Deal with missing values (these columns have a TON of missing values, so dropping them seems better than imputing or anything else)
college = college.drop(columns = [
    'vsa_year', 'vsa_grad_after4_first', 'vsa_grad_elsewhere_after4_first', 
    'vsa_enroll_after4_first', 'vsa_enroll_elsewhere_after4_first', 'vsa_grad_after6_first',
    'vsa_grad_elsewhere_after6_first', 'vsa_enroll_after6_first', 'vsa_enroll_elsewhere_after6_first',
    'vsa_grad_after4_transfer', 'vsa_grad_elsewhere_after4_transfer', 'vsa_enroll_after4_transfer',
    'vsa_enroll_elsewhere_after4_transfer', 'vsa_grad_after6_transfer', 'vsa_grad_elsewhere_after6_transfer',
    'vsa_enroll_after6_transfer', 'vsa_enroll_elsewhere_after6_transfer', 'med_sat_value', 'med_sat_percentile',
    'endow_value', 'endow_percentile'])

# %%
# Type corrections from string to category
cat_cols = ['control', 'city', 'state', 'level']
for col in cat_cols:
    college[col] = college[col].astype('category')

# %%
# Collapse factor levels (get rid of Non-profit for profit private schools, just group them both into private schools for simplicity and making it binary)
college['control'] = college['control'].apply(
    lambda x: 'Public' if x == 'Public' else 'Private')

# %%
# One-hot encoding categorical variables for model
cat_cols = ['control', 'city', 'state', 'level']
college = pd.get_dummies(college, columns = cat_cols)

# %%
# Create target variable with 0.5 as cutoff (can easily be changed, just set 0.5 as a default), 1 if true, 0 if not
college['target_completion'] = (college['fte_percentile'] >= 0.5).astype(int)

# %%
# Calculate prevalence
prevalence = college['target_completion'].mean()

# %%
# Scale numeric data to mean 0 and std dev 1 to ensure equal influence across numeric features
numeric_cols = college.select_dtypes(include = ['int64', 'float64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'target_completion']
scaler = StandardScaler()
college[numeric_cols] = scaler.fit_transform(college[numeric_cols])

# %%
# Train, tune, test split (train 60%, tune 20%, test 20% with given parameters)
train_data, test_data = train_test_split(college, test_size = 0.2, random_state = 42)
train_data, tune_data = train_test_split(train_data, test_size = 0.25, random_state = 42)



# -------------------------------------------------------------------------------------------------



# %%[markdown]
# #### Job Placement Data

# Libraries already imported above

# %%
# Read in data
job_placement = pd.read_csv('https://raw.githubusercontent.com/DG1606/CMS-R-2020/master/Placement_Data_Full_Class.csv')

# %%
# Drop unnecessary columns (sl_no is just an index label, not necessary)
job_placement = job_placement.drop(columns = ['sl_no'])

# %%
# Fill missing values for salary column (missing values represent not placed)
job_placement['salary'] = job_placement['salary'].fillna(0)

# %%
# Type corrections (string to category)
cat_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
for col in cat_cols:
    job_placement[col] = job_placement[col].astype('category')

# %%
# One-hot encoding categorical variables for model
cat_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
job_placement = pd.get_dummies(job_placement, columns = cat_cols)

# I find that this dataset doesn't really require collapsing factor
# levels since the categories are already pretty distinct.

# %%
# Normalize continuous variables to mean 0 and std dev 1
numeric_cols = job_placement.select_dtypes(include = ['int64', 'float64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'status']
scaler = StandardScaler()
job_placement[numeric_cols] = scaler.fit_transform(job_placement[numeric_cols])

# Adjust target variable column to be 0/1 for easier computation for model
job_placement['status'] = job_placement['status'].apply(lambda x: 1 if x == 'Placed' else 0)

# %%
# Calculate prevalence
prevalence = job_placement['status'].mean()

# %%
# Train, tune, test split (again train 60%, tune 20%, test 20%)
train_data, test_data = train_test_split(job_placement, test_size = 0.2, random_state = 42)
train_data, tune_data = train_test_split(train_data, test_size = 0.25, random_state = 42)



# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


# %% [markdown]
# ## Step 3: Instincts on Data

# My instincts tell me that both datasets will have a decent amount of missing values due to the nature of
# the education system and overall data collection of students. I also feel like there will be trends
# specifically within region types as well as control types (public vs private schools) for the completion data.
# For the job placement dataset, I feel like education level and work experience will be the main variables that
# will impact the job placement proportions, and the specific degree type/field will also probably have a decent
# impact on said proportion. 
#
# I'm a bit worried about the missing values I mentioned and how that might skew the data if the missing data
# is following a pattern and isn't just missing at random. I'm also worried about the size of the job placement
# dataset as it might not be large enough to really train a model on or anything like that since it only has a bit
# over 200 rows in it.