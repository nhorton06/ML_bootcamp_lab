# %% [markdown]
# # Machine Learning Bootcamp Lab

# %% [markdown]
# ## Step 1: Potential Dataset Questions

# - **College Completion Dataset**: What institutional characteristics (like region, cost, etc.) are most predictive of higher college completion rates?
# - **Job Placement Dataset**: How do factors like degree, specialization, and gender influence job placement rates after graduation?






# %% [markdown]
# ## Step 2: Data Preparation

# ### Independent Business Metrics
# - College Completion Dataset: Completion rate of degree/program (will track whether any changes make any difference to actual percentage of students that complete their program)
# - Job Placement Dataset: Proportion of placement (will track whether any changes make any difference to how many students are getting placed in jobs)

# ### Data Preparation !!! Is this supposed to be descriptive or actually coding the data prep?

# #### College Completion Data
# **Type corrections**:
# - Categorical variables (level, control, state, probably not city since too many) to category
# - All continuous variables already are floats in dataset, so no change needed
#
# **Collapse factor levels**:
# - Potentially group states into different regions rather than all 50 individual states
# - Probably can just simplify control down to public and private
#
# **One-hot encoding**:
# - All categorical types to one-hot encoded columns (like just have "Public" and 0 or 1 to indicate)
#
# **Normalize continuous variables**:
# - Use StandardScaler to standardize continuous variables
#
# **Drop unnecessary variables**:
# - Drop identifier columns as they won't help in model training, and
# probably city as well due to the high amount of different cities
#
# **Create target variable (if needed)**:
# - Create binary target variable based on whether completion rate is above or below 
# a certain completion rate (subject to change)
#
# **Calculate prevalence of target variable**:
# - Just use .mean() of target variable to get prevalence
#
# **Create necessary data partitions (train, tune, test)**:
# - Use train_test_split from sklearn to create train, tune, and test sets
#

# #### Job Placement Data
# **Type corrections**:
# 
#
# **Collapse factor levels**:
#
# **One-hot encoding**:
#
# **Normalize continuous variables**:
#
# **Drop unnecessary variables**:
#
# **Create target variable (if needed)**:
#
# **Calculate prevalence of target variable**:
#
# **Create necessary data partitions (train, tune, test)**:
#






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