# Notebook objective:
In this notebook, I investigate how much of a model's performance comes from the algorithm itself and how much comes from feature representation. To that end, we will compare three different approaches for training the model:
 - **Baseline**: Without any feature processing.
 - **Simple Feature Engineering**: Applying scaling or basic transformations.
 - **Advanced Feature Engineering**: Computing intelligent features.

To ensure a fair comparison, we will use the same model structure throughout; thus, any performance changes will be solely due to the input features.

# Problem Framing

Our objective is to predict house prices using 80 different features. This is a regression problem, and we will use a Ridge Regression model.

As in the competition, we will evaluate the model performance using the Root Mean Squared Error (RMSE):
$$
RMSE = \sqrt{\frac{1}{n} \sum_{i = 1}^{n} (y_i - \tilde{y}_{i})^2}
$$

We will use this same loss function to train and evaluate the model.
Since RMSE is very sensitive to outliers, we will pay special attention to handling extreme values to minimize their negative impact on the competition score.

# Baseline: Without any feature processing.


```python
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

# set all seeds to 42
np.random.seed(42)
random.seed(42)

# load data
data = pd.read_csv('train.csv')

print("Shape of the data: ",data.shape)

# split features and target
X = data.drop('SalePrice',axis = 1)
y = data['SalePrice']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

# specify between continuous and discrete columns
cont_cols = X_train.select_dtypes(include=['number']).columns.tolist() # continuous (or at least numerical)
disc_cols = X_train.select_dtypes(include=['object', 'category','str']).columns.tolist() # discrete 

# define the transformation of the discrete data
discrete_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # for NaN values
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
# define the transformation of the continuous data
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', continuous_transformer, cont_cols),
        ('cat', discrete_transformer, disc_cols)
    ])

# create the model
model = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])

# train the model
model.fit(X_train,y_train)

# make predictions
y_preds = model.predict(X_test)

# compare real data with predictions
rmse = root_mean_squared_error(y_true=y_test,y_pred = y_preds)

# plot the differences
plt.figure(figsize = (8,4))
plt.scatter(y_test,y_preds,alpha=0.5)
plt.plot([0,np.max([*y_preds,*y_test])], [0,np.max([*y_preds,*y_test])], 'k-')
plt.title(f"Difference between predictions and real house prices. $RMSE = {rmse:.2f}$")
plt.grid()
plt.show()
```

    Shape of the data:  (1460, 81)
    


    
![png](code_files/code_3_1.png)
    


Let's evaluate the model on the test data to submit to the Kaggle competition.


```python
data_test = pd.read_csv("test.csv")
test_id = data_test['Id']

# make predictions
predictions = model.predict(data_test)

df_final = pd.DataFrame({
    'Id': test_id,
    'SalePrice': predictions
})
df_final.to_csv('baseline_model.csv', index=False)
```

The score obtained from this model is **0.32534**, which is lower than 91% of the teams. This is a poor result, we can improve it.

# Simple Feature Engineering: Applying scaling or basic transformations.


## Data Study


```python
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv")
```

First, lets try to find some outliers in the data


```python
plt.figure(figsize=(5,5))
plt.scatter(data['GrLivArea'],data['SalePrice'])
plt.grid()
plt.xlabel("Ground Living Area")
plt.ylabel("Sale Price")
plt.show()
```


    
![png](code_files/code_11_0.png)
    


We can observe two data points with high living area but relatively low prices. We will remove these outliers from the dataset to minimize the impact of extreme values on the model.

Now, let's examine the continuous data by plotting all numeric columns in histograms.


```python
data.drop("Id",axis = 1).hist(bins = 30,figsize=(15,12))
plt.show()
```


    
![png](code_files/code_14_0.png)
    



```python
continuous_columns = ['MSSubClass','LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtUnfSF','BsmtFinSF2','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']
data[continuous_columns].hist(bins = 30,figsize=(12,8))
plt.show()
```


    
![png](code_files/code_15_0.png)
    


As we can see, these distributions are highly skewed. We need to will apply the Yeo-Johnson transformation to approximate the data to a normal distribution as much as possible.


Now, we will examine the count variables, such as the number of bathrooms. We will apply a RobustScaler to these features to ensure that the scaling is not distorted by potential outliers.


```python
from sklearn.preprocessing import RobustScaler

rs = RobustScaler()
discrete_numerical_columns = ['OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','YrSold']

data_transformed_df = pd.DataFrame(rs.fit_transform(data[discrete_numerical_columns]), columns=discrete_numerical_columns)

data_transformed_df.hist(bins=30, figsize=(12, 8))
plt.show()
```


    
![png](code_files/code_18_0.png)
    


Finally, we will perform cyclical encoding for the 'month sold' by mapping it to a 2D vector $(x, y)$ using sine and cosine transformations. This ensures that the circular nature of the months is preserved, making December and January numerically close.

Now lets study the non numerical columns


```python
total_cols = set(data.columns)
numerical_cols = set(data.select_dtypes(include = ['number']).columns)
non_numerical_columns = list(total_cols - numerical_cols)
```

We have two types of categorical variables: ordinal and nominal. The key difference lies in the order: ordinal variables have a natural, meaningful ranking (such as 'Low', 'Medium', 'High'), whereas nominal variables represent distinct categories without any inherent order (such as 'Color' or 'City')

Inside the ordinal columns we have a group that share a common quality scale (*Ex, Gd, TA, Fa, Po*) This columns are:
* `ExterQual`, `ExterCond`, `BsmtQual`, `BsmtCond`, `HeatingQC`, `KitchenQual`, `FireplaceQu`, `GarageQual`, `GarageCond`, `PoolQC`.

The rest of nominal features can have the values:

| Feature | Values (Ordered Hierarchy) |
| :--- | :--- |
| **PavedDrive** | `N` , `P` , `Y` |
| **BsmtExposure** | `No`, `Mn` , `Av`, `Gd` |
| **LandSlope** | `Gtl`, `Mod` , `Sev` |
| **BsmtFinType 1/2**| `Unf`, `LwQ`, `Rec`, `BLQ`, `ALQ`, `GLQ` |
| **Utilities** | `NoSeWa`, `AllPub` |
| **LotShape** | `IR3`, `IR2`, `IR1`, `Reg` |
| **GarageFinish** | `Unf`, `RFn`, `Fin` |
| **Fence** | `MnWw`, `GdWo`, `MnPrv`, `GdPrv` |
| **Electrical** | `Mix`, `FuseP`, `FuseF`, `FuseA`, `SBrkr` |
| **Functional** | `Sev`, `Maj2`, `Maj1`, `Mod`, `Min2`, `Min1`, `Typ` |


The remaining non-numerical columns are considered nominal. Since they do not have an intrinsic order, they will be treated using encoding techniques such as *One-Hot Encoding*.

## Model with Simple Feature Engineering 


```python
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,PowerTransformer,RobustScaler,FunctionTransformer,StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from typing import List

# set all seeds to 42
np.random.seed(42)
random.seed(42)

data = pd.read_csv('train.csv').drop(columns=['Id'])

# remove outliers
data = data[(data['GrLivArea'] < 4000) | (data['SalePrice'] > 300000)]

print("Shape of the data: ",data.shape)

# split
X = data.drop('SalePrice',axis = 1)
y = data['SalePrice']

#  apply log to the y data
y = np.log(y)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
```

    Shape of the data:  (1458, 80)
    


```python
# arrays with the different columns we have 
columns_with_continuous_values = ['MSSubClass','LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtUnfSF','BsmtFinSF2',
                                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF',
                                 'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']

columns_with_numerical_counts = ['OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFullBath','BsmtHalfBath','FullBath',
                                 'HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','YrSold']

mo_sold_column = ['MoSold']

columns_with_discrete_quality = ['GarageCond', 'GarageQual', 'ExterQual', 'ExterCond', 'KitchenQual', 
                                    'HeatingQC', 'FireplaceQu', 'BsmtQual', 'BsmtCond','PoolQC'
]

categorical_columns = ['RoofMatl', 'Neighborhood', 'Alley', 'BldgType', 'LotConfig', 'GarageType', 'SaleType', 'Street', 
                       'MSZoning', 'Heating', 'Exterior1st', 'LandContour', 'MasVnrType', 'Exterior2nd', 'SaleCondition', 
                       'HouseStyle', 'CentralAir', 'RoofStyle', 'Foundation', 'Condition1', 'MiscFeature', 'Condition2']
 
 
# create the transformation for the continuous data
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # for NaN values, put the most frequent
    ('transformer', PowerTransformer(method='yeo-johnson')), # transform the data to make it more gaussian
    ('scaler', RobustScaler()) # scale the data
])

# create the transformation for the numerical counts
numerical_count_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant',fill_value = 0)), # fill the NaN values with zero
    ('scaler',RobustScaler()) # scale the data
])

# create the transformation for the month sold column
def transform_cycle(X, period = 12):
    sin_month = np.sin(2*np.pi * X / period)
    cos_month = np.cos(2*np.pi * X / period)
    return np.column_stack([sin_month,cos_month])
mo_sold_transformer = FunctionTransformer(transform_cycle)

# create the transformation for the discrete condition data
def to_upper_case(X):
    return pd.DataFrame(X).map(lambda x: x.upper() if isinstance(x,str) else x)  

def create_pipeline_discrete(categories: List, n_columns: int) -> Pipeline:
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('to_upper', FunctionTransformer(to_upper_case)),
        ('ordinal_map', OrdinalEncoder(categories=[categories] * n_columns)), 
        ('scaler', StandardScaler())
    ])

discrete_related_categories = {
    "quality": ['MISSING', 'PO', 'FA', 'TA', 'GD', 'EX'],
    "GarageFinish": ['MISSING', 'UNF', 'RFN', 'FIN'],
    "PavedDrive": ['MISSING', 'N', 'P', 'Y'],
    "Electrical": ['MISSING', 'MIX', 'FUSEP', 'FUSEF', 'FUSEA', 'SBRKR'],
    "BsmtFinType": ['MISSING', 'UNF', 'LWQ', 'REC', 'BLQ', 'ALQ', 'GLQ'],
    "BsmtExposure": ['MISSING', 'NO', 'MN', 'AV', 'GD'],
    "LotShape": ['MISSING', 'IR3', 'IR2', 'IR1', 'REG'],
    "LandSlope": ['MISSING', 'SEV', 'MOD', 'GTL'],
    "Utilities": ['MISSING', 'NOSEWA', 'ALLPUB'],
    "Functional": ['MISSING', 'SEV', 'MAJ2', 'MAJ1', 'MOD', 'MIN2', 'MIN1', 'TYP'],
    "Fence": ['MISSING', 'MNWW', 'GDWO', 'MNPRV', 'GDPRV']
}

# create the transformation for the category data
category_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # fill NaN with 'missing'
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ('scaler',StandardScaler(with_mean=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cont', continuous_transformer, columns_with_continuous_values),
        ('count', numerical_count_transformer, columns_with_numerical_counts),
        ('mo_sold', FunctionTransformer(transform_cycle), mo_sold_column),
        ('qual', create_pipeline_discrete(discrete_related_categories['quality'], len(columns_with_discrete_quality)), columns_with_discrete_quality),
        
        ('gar_fin', create_pipeline_discrete(discrete_related_categories['GarageFinish'], 1), ['GarageFinish']),
        ('paved', create_pipeline_discrete(discrete_related_categories['PavedDrive'], 1), ['PavedDrive']),
        ('elec', create_pipeline_discrete(discrete_related_categories['Electrical'], 1), ['Electrical']),
        ('bsmt_fin', create_pipeline_discrete(discrete_related_categories['BsmtFinType'], 2), ['BsmtFinType1', 'BsmtFinType2']),
        
        ('bsmt_exp', create_pipeline_discrete(discrete_related_categories['BsmtExposure'], 1), ['BsmtExposure']),
        ('shape', create_pipeline_discrete(discrete_related_categories['LotShape'], 1), ['LotShape']),
        ('slope', create_pipeline_discrete(discrete_related_categories['LandSlope'], 1), ['LandSlope']),
        ('util', create_pipeline_discrete(discrete_related_categories['Utilities'], 1), ['Utilities']),
        ('func', create_pipeline_discrete(discrete_related_categories['Functional'], 1), ['Functional']),
        ('fence', create_pipeline_discrete(discrete_related_categories['Fence'], 1), ['Fence']),
        
        ('cat', category_transformer, categorical_columns)
    ],
    remainder='drop'
)

# create the model
model = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))
])
```


```python
# train the model
model.fit(X_train,y_train)

# make predictions
y_preds = model.predict(X_test)

y_preds = [np.exp(v) for v in y_preds]
y_test = [np.exp(v) for v in y_test]

# compare real data with predictions
rmse = root_mean_squared_error(y_true=y_test,y_pred = y_preds)

# plot the differences
plt.figure(figsize = (12,6))
plt.scatter(y_test,y_preds,alpha=0.5)
plt.plot([0,np.max([*y_preds,*y_test])], [0,np.max([*y_preds,*y_test])], 'k-')
plt.title(f"Difference between predictions and real house prices. $RMSE = {rmse:.2f}$")
plt.xlabel("Real Price")
plt.ylabel("Predicted Price")
plt.grid()
plt.show()
```


    
![png](code_files/code_27_0.png)
    


Again evaluating the model in kaggle.


```python
data_test = pd.read_csv("test.csv")
test_id = data_test['Id']

# make predictions
predictions = np.exp(model.predict(data_test))


df_final = pd.DataFrame({
    'Id': test_id,
    'SalePrice': predictions
})
df_final.to_csv('model_simple_featuring.csv', index=False)
```

The score obtained now is **0.1366**, which is lower than 49% of the teams. We have made some improvements, but we can keep pushing.

#  Advanced Feature Engineering: Computing intelligent features


First, we observe that several continuous variables have a high concentration of zero values (sparsity). Therefore, it would be beneficial to create a binary indicator for each of these features to determine whether the value is zero or not.

We will also introduce non-linear relationships into the model by calculating the powers of specific columns, which can help capture more complex patterns in the data.

Regarding temporal data, we will shift our focus from absolute years (Built/Sold) to relative age. Since the price is more likely influenced by how old the property is rather than the specific year it was constructed, we will compute the house age and the time since the last remodel.

Furthermore, we will engineer several new ratios to capture deeper insights:

* **% Living Area on First Floor**: 
    $$\text{LAIFF} = \frac{\text{1stFlrSF}}{\text{GrLivArea}}$$
* **Ground Living Area per Room**:
    $$\text{GLAPR} = \frac{\text{GrLivArea}}{\text{TotRmsAbvGrd}}$$
* **% Garage vs. Ground Living Area**:
    $$\text{GGLA} = \frac{\text{GarageArea}}{\text{GrLivArea}}$$
* **Bathrooms per Bedroom**:
    $$\text{BPB} = \frac{\text{FullBath}}{\text{BedroomAbvGr}}$$
* **Living Area per Bedroom**:
    $$\text{LAPB} = \frac{\text{GrLivArea}}{\text{BedroomAbvGr}}$$
* **2nd Floor to 1st Floor Ratio**:
    $$\text{SFR} = \frac{\text{2ndFlrSF}}{\text{1stFlrSF}}$$

During these calculations, we must ensure robust handling of potential *division by zero* or cases where the denominator is near zero to avoid numerical instability.

Also, instead of using a simple Ridge regression, we are now implementing *RidgeCV* within a pipeline. This allows us to find the optimal regularization strength ($\alpha$).

We will define a search space of 100 alpha values ranging from $10^{-3}$ to $10^{3}$ and to ensure a reliable estimation of the model's performance, we will use a 5-fold Cross-Validation strategy with shuffling.


```python
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import RidgeCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,PowerTransformer,RobustScaler,FunctionTransformer,StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from typing import List

# set all seeds to 42
np.random.seed(42)
random.seed(42)

data = pd.read_csv('train.csv').drop(columns=['Id'])

# lets remove the outliers from the data 
data = data[(data['GrLivArea'] < 4000) | (data['SalePrice'] > 300000)]

# arrays splitting the data we have
columns_with_continuous_values = ['MSSubClass','LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtUnfSF','BsmtFinSF2',
                                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','GarageArea','WoodDeckSF',
                                 'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal']

columns_with_numerical_counts = ['OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFullBath','BsmtHalfBath','FullBath',
                                 'HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','YrSold']

mo_sold_column = ['MoSold']

columns_with_discrete_quality = ['GarageCond', 'GarageQual', 'ExterQual', 'ExterCond', 'KitchenQual', 
                                    'HeatingQC', 'FireplaceQu', 'BsmtQual', 'BsmtCond','PoolQC']

categorical_columns = ['RoofMatl', 'Neighborhood', 'Alley', 'BldgType', 'LotConfig', 'GarageType', 'SaleType', 'Street', 
                       'MSZoning', 'Heating', 'Exterior1st', 'LandContour', 'MasVnrType', 'Exterior2nd', 'SaleCondition', 
                       'HouseStyle', 'CentralAir', 'RoofStyle', 'Foundation', 'Condition1', 'MiscFeature', 'Condition2']


print("Shape of the data: ",data.shape)
```

    Shape of the data:  (1458, 80)
    

Let's compute the new features for the model


```python
def advanced_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # first the binary feature
    for continuous_column in columns_with_continuous_values:
        df[f'{continuous_column}_bin'] = df[continuous_column] > 0

    # now add some non linearity
    columns_to_square = ['1stFlrSF','2ndFlrSF','GrLivArea']
    for col in columns_to_square:
        df[f'{col}_squared'] = df[col] * df[col]

    # compute the different ages
    df['houseAge'] = df['YrSold'] - df['YearBuilt']
    df['ageSinceRemodel'] = df['YearRemodAdd'] - df['YearBuilt']

    # compute the relations 
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['TotalBath'] = df['BsmtFullBath'] + 0.5*df['BsmtHalfBath'] + df['FullBath'] + 0.5*df['HalfBath']
    df['TotalQual'] = df['OverallQual'] * df['GrLivArea']

    threshold = 1 # anything lower is considered zero
    df['LAIFF'] = np.where(df['GrLivArea'] >= threshold, df['1stFlrSF'] / df['GrLivArea'], np.nan)
    df['GLAPR'] = np.where(df['TotRmsAbvGrd'] >= threshold, df['GrLivArea'] / df['TotRmsAbvGrd'], np.nan)
    df['GINGLR'] = np.where(df['GarageArea'] >= threshold, df['GrLivArea'] / df['GarageArea'], np.nan)
    df['BPB'] = np.where(df['BedroomAbvGr'] >= threshold, df['TotalBath'] / df['BedroomAbvGr'], np.nan)
    df['GLAB'] = np.where(df['BedroomAbvGr'] >= threshold, df['GrLivArea'] / df['BedroomAbvGr'], np.nan)
    df['SQ2FPQF1F'] = np.where(df['1stFlrSF'] >= threshold, df['2ndFlrSF'] / df['1stFlrSF'], np.nan)

    return df
```


```python
# compute new features
data = advanced_feature_engineering(data)

# split the data
X = data.drop('SalePrice',axis = 1)
y = np.log(data['SalePrice']) # apply log

# load the test set
test_df = pd.read_csv('test.csv')
test_id = test_df['Id'] 
data_test = test_df.drop(columns=['Id'])
data_test = advanced_feature_engineering(data_test)
```

Add these new features to the transformations


```python
columns_binary = [f'{continuous_column}_bin'for continuous_column in columns_with_continuous_values]
columns_with_continuous_values = columns_with_continuous_values + ['1stFlrSF_squared','2ndFlrSF_squared','GrLivArea_squared','LAIFF','GLAPR','GINGLR','BPB','SQ2FPQF1F','TotalSF','TotalBath','TotalQual','GLAB']
columns_with_numerical_counts = columns_with_numerical_counts + ['houseAge','ageSinceRemodel']

```


```python
# transformations for binary columns
binary_transformer = Pipeline(steps=[
    ('to_int', FunctionTransformer(lambda x: x.astype(int))),
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', StandardScaler()) 
])

# create the transformation for the continuous data
continuous_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # for NaN values, put the most frequent
    ('transformer', PowerTransformer(method='yeo-johnson')), # transform the data to make it more gaussian
    ('scaler', RobustScaler()) # scale the data
])

# create the transformation for the numerical counts
numerical_count_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant',fill_value = 0)), # fill the NaN values with zero
    ('scaler',RobustScaler()) # scale the data
])

# create the transformation for the month sold column
def transform_cycle(X, period = 12):
    sin_month = np.sin(2*np.pi * X / period)
    cos_month = np.cos(2*np.pi * X / period)
    return np.column_stack([sin_month,cos_month])
mo_sold_transformer = FunctionTransformer(transform_cycle)

# create the transformation for the discrete condition data
def to_upper_case(X):
    return pd.DataFrame(X).map(lambda x: x.upper() if isinstance(x,str) else x)  

def create_pipeline_discrete(categories: List, n_columns: int) -> Pipeline:
    return Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('to_upper', FunctionTransformer(to_upper_case)),
        ('ordinal_map', OrdinalEncoder(categories=[categories] * n_columns)), 
        ('scaler', StandardScaler())
    ])

discrete_related_categories = {
    "quality": ['MISSING', 'PO', 'FA', 'TA', 'GD', 'EX'],
    "GarageFinish": ['MISSING', 'UNF', 'RFN', 'FIN'],
    "PavedDrive": ['MISSING', 'N', 'P', 'Y'],
    "Electrical": ['MISSING', 'MIX', 'FUSEP', 'FUSEF', 'FUSEA', 'SBRKR'],
    "BsmtFinType": ['MISSING', 'UNF', 'LWQ', 'REC', 'BLQ', 'ALQ', 'GLQ'],
    "BsmtExposure": ['MISSING', 'NO', 'MN', 'AV', 'GD'],
    "LotShape": ['MISSING', 'IR3', 'IR2', 'IR1', 'REG'],
    "LandSlope": ['MISSING', 'SEV', 'MOD', 'GTL'],
    "Utilities": ['MISSING', 'NOSEWA', 'ALLPUB'],
    "Functional": ['MISSING', 'SEV', 'MAJ2', 'MAJ1', 'MOD', 'MIN2', 'MIN1', 'TYP'],
    "Fence": ['MISSING', 'MNWW', 'GDWO', 'MNPRV', 'GDPRV']
}

# create the transformation for the category data
category_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # fill NaN with 'missing'
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ('scaler',StandardScaler(with_mean=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('bin',binary_transformer,columns_binary),
        ('cont', continuous_transformer, columns_with_continuous_values),
        ('count', numerical_count_transformer, columns_with_numerical_counts),
        ('mo_sold', FunctionTransformer(transform_cycle), mo_sold_column),
        ('qual', create_pipeline_discrete(discrete_related_categories['quality'], len(columns_with_discrete_quality)), columns_with_discrete_quality),
        
        ('gar_fin', create_pipeline_discrete(discrete_related_categories['GarageFinish'], 1), ['GarageFinish']),
        ('paved', create_pipeline_discrete(discrete_related_categories['PavedDrive'], 1), ['PavedDrive']),
        ('elec', create_pipeline_discrete(discrete_related_categories['Electrical'], 1), ['Electrical']),
        ('bsmt_fin', create_pipeline_discrete(discrete_related_categories['BsmtFinType'], 2), ['BsmtFinType1', 'BsmtFinType2']),
        
        ('bsmt_exp', create_pipeline_discrete(discrete_related_categories['BsmtExposure'], 1), ['BsmtExposure']),
        ('shape', create_pipeline_discrete(discrete_related_categories['LotShape'], 1), ['LotShape']),
        ('slope', create_pipeline_discrete(discrete_related_categories['LandSlope'], 1), ['LandSlope']),
        ('util', create_pipeline_discrete(discrete_related_categories['Utilities'], 1), ['Utilities']),
        ('func', create_pipeline_discrete(discrete_related_categories['Functional'], 1), ['Functional']),
        ('fence', create_pipeline_discrete(discrete_related_categories['Fence'], 1), ['Fence']),
        
        ('cat', category_transformer, categorical_columns)
    ],
    remainder='drop'
)


alphas = np.logspace(-3, 3, 100)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# create the model
model = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ("ridge", RidgeCV(alphas=alphas, cv=kf, scoring="neg_root_mean_squared_error"))
])
```


```python
model.fit(X, y)
```




<style>#sk-container-id-8 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;
}

#sk-container-id-8.light {
  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: black;
  --sklearn-color-background: white;
  --sklearn-color-border-box: black;
  --sklearn-color-icon: #696969;
}

#sk-container-id-8.dark {
  --sklearn-color-text-on-default-background: white;
  --sklearn-color-background: #111;
  --sklearn-color-border-box: white;
  --sklearn-color-icon: #878787;
}

#sk-container-id-8 {
  color: var(--sklearn-color-text);
}

#sk-container-id-8 pre {
  padding: 0;
}

#sk-container-id-8 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-8 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-8 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-8 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-8 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-8 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-8 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-8 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-8 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-8 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-8 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-8 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-8 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: center;
  justify-content: center;
  gap: 0.5em;
}

#sk-container-id-8 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-8 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-8 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-8 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-8 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-8 div.sk-label label.sk-toggleable__label,
#sk-container-id-8 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-8 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-8 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-8 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  line-height: 1.2em;
}

#sk-container-id-8 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-8 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-8 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-8 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-8 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-3) 1pt solid;
  color: var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3) 1pt solid;
  color: var(--sklearn-color-fitted-level-3);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-unfitted-level-0);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  border: var(--sklearn-color-fitted-level-0) 1pt solid;
  color: var(--sklearn-color-fitted-level-0);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-8 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-unfitted-level-0);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-8 a.estimator_doc_link.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-8 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-8 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table {
    font-family: monospace;
}

.estimator-table summary {
    padding: .5rem;
    cursor: pointer;
}

.estimator-table summary::marker {
    font-size: 0.7rem;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
    margin-top: 0;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

/*
    `table td`is set in notebook with right text-align.
    We need to overwrite it.
*/
.estimator-table table td.param {
    text-align: left;
    position: relative;
    padding: 0;
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left !important;
}

.user-set td.value {
    color:rgb(255, 94, 0);
    background-color: transparent;
}

.default td {
    color: black;
    text-align: left !important;
}

.user-set td i,
.default td i {
    color: black;
}

/*
    Styles for parameter documentation links
    We need styling for visited so jupyter doesn't overwrite it
*/
a.param-doc-link,
a.param-doc-link:link,
a.param-doc-link:visited {
    text-decoration: underline dashed;
    text-underline-offset: .3em;
    color: inherit;
    display: block;
    padding: .5em;
}

/* "hack" to make the entire area of the cell containing the link clickable */
a.param-doc-link::before {
    position: absolute;
    content: "";
    inset: 0;
}

.param-doc-description {
    display: none;
    position: absolute;
    z-index: 9999;
    left: 0;
    padding: .5ex;
    margin-left: 1.5em;
    color: var(--sklearn-color-text);
    box-shadow: .3em .3em .4em #999;
    width: max-content;
    text-align: left;
    max-height: 10em;
    overflow-y: auto;

    /* unfitted */
    background: var(--sklearn-color-unfitted-level-0);
    border: thin solid var(--sklearn-color-unfitted-level-3);
}

/* Fitted state for parameter tooltips */
.fitted .param-doc-description {
    /* fitted */
    background: var(--sklearn-color-fitted-level-0);
    border: thin solid var(--sklearn-color-fitted-level-3);
}

.param-doc-link:hover .param-doc-description {
    display: block;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-8" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;preprocessor&#x27;,
                 ColumnTransformer(transformers=[(&#x27;bin&#x27;,
                                                  Pipeline(steps=[(&#x27;to_int&#x27;,
                                                                   FunctionTransformer(func=&lt;function &lt;lambda&gt; at 0x000001FBB3F07920&gt;)),
                                                                  (&#x27;imputer&#x27;,
                                                                   SimpleImputer(fill_value=0,
                                                                                 strategy=&#x27;constant&#x27;)),
                                                                  (&#x27;scaler&#x27;,
                                                                   StandardScaler())]),
                                                  [&#x27;MSSubClass_bin&#x27;,
                                                   &#x27;LotFrontage_bin&#x27;,
                                                   &#x27;LotArea_bin&#x27;,
                                                   &#x27;MasVnrArea_bin&#x27;,
                                                   &#x27;BsmtFinSF1_bin&#x27;,
                                                   &#x27;BsmtUnfSF_bi...
       7.05480231e+01, 8.11130831e+01, 9.32603347e+01, 1.07226722e+02,
       1.23284674e+02, 1.41747416e+02, 1.62975083e+02, 1.87381742e+02,
       2.15443469e+02, 2.47707636e+02, 2.84803587e+02, 3.27454916e+02,
       3.76493581e+02, 4.32876128e+02, 4.97702356e+02, 5.72236766e+02,
       6.57933225e+02, 7.56463328e+02, 8.69749003e+02, 1.00000000e+03]),
                         cv=KFold(n_splits=5, random_state=42, shuffle=True),
                         scoring=&#x27;neg_root_mean_squared_error&#x27;))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-526" type="checkbox" ><label for="sk-estimator-id-526" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>Pipeline</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('steps',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.pipeline.Pipeline.html#:~:text=steps,-list%20of%20tuples">
            steps
            <span class="param-doc-description">steps: list of tuples<br><br>List of (name of step, estimator) tuples that are to be chained in<br>sequential order. To be compatible with the scikit-learn API, all steps<br>must define `fit`. All non-last steps must also define `transform`. See<br>:ref:`Combining Estimators <combining_estimators>` for more details.</span>
        </a>
    </td>
            <td class="value">[(&#x27;preprocessor&#x27;, ...), (&#x27;ridge&#x27;, ...)]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transform_input',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.pipeline.Pipeline.html#:~:text=transform_input,-list%20of%20str%2C%20default%3DNone">
            transform_input
            <span class="param-doc-description">transform_input: list of str, default=None<br><br>The names of the :term:`metadata` parameters that should be transformed by the<br>pipeline before passing it to the step consuming it.<br><br>This enables transforming some input arguments to ``fit`` (other than ``X``)<br>to be transformed by the steps of the pipeline up to the step which requires<br>them. Requirement is defined via :ref:`metadata routing <metadata_routing>`.<br>For instance, this can be used to pass a validation set through the pipeline.<br><br>You can only set this if metadata routing is enabled, which you<br>can enable using ``sklearn.set_config(enable_metadata_routing=True)``.<br><br>.. versionadded:: 1.6</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('memory',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.pipeline.Pipeline.html#:~:text=memory,-str%20or%20object%20with%20the%20joblib.Memory%20interface%2C%20default%3DNone">
            memory
            <span class="param-doc-description">memory: str or object with the joblib.Memory interface, default=None<br><br>Used to cache the fitted transformers of the pipeline. The last step<br>will never be cached, even if it is a transformer. By default, no<br>caching is performed. If a string is given, it is the path to the<br>caching directory. Enabling caching triggers a clone of the transformers<br>before fitting. Therefore, the transformer instance given to the<br>pipeline cannot be inspected directly. Use the attribute ``named_steps``<br>or ``steps`` to inspect estimators within the pipeline. Caching the<br>transformers is advantageous when fitting is time consuming. See<br>:ref:`sphx_glr_auto_examples_neighbors_plot_caching_nearest_neighbors.py`<br>for an example on how to enable caching.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.pipeline.Pipeline.html#:~:text=verbose,-bool%2C%20default%3DFalse">
            verbose
            <span class="param-doc-description">verbose: bool, default=False<br><br>If True, the time elapsed while fitting each step will be printed as it<br>is completed.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-serial"><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-527" type="checkbox" ><label for="sk-estimator-id-527" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>preprocessor: ColumnTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.compose.ColumnTransformer.html">?<span>Documentation for preprocessor: ColumnTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transformers',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=transformers,-list%20of%20tuples">
            transformers
            <span class="param-doc-description">transformers: list of tuples<br><br>List of (name, transformer, columns) tuples specifying the<br>transformer objects to be applied to subsets of the data.<br><br>name : str<br>    Like in Pipeline and FeatureUnion, this allows the transformer and<br>    its parameters to be set using ``set_params`` and searched in grid<br>    search.<br>transformer : {'drop', 'passthrough'} or estimator<br>    Estimator must support :term:`fit` and :term:`transform`.<br>    Special-cased strings 'drop' and 'passthrough' are accepted as<br>    well, to indicate to drop the columns or to pass them through<br>    untransformed, respectively.<br>columns :  str, array-like of str, int, array-like of int,                 array-like of bool, slice or callable<br>    Indexes the data on its second axis. Integers are interpreted as<br>    positional columns, while strings can reference DataFrame columns<br>    by name.  A scalar string or int should be used where<br>    ``transformer`` expects X to be a 1d array-like (vector),<br>    otherwise a 2d array will be passed to the transformer.<br>    A callable is passed the input data `X` and can return any of the<br>    above. To select multiple columns by name or dtype, you can use<br>    :obj:`make_column_selector`.</span>
        </a>
    </td>
            <td class="value">[(&#x27;bin&#x27;, ...), (&#x27;cont&#x27;, ...), ...]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('remainder',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=remainder,-%7B%27drop%27%2C%20%27passthrough%27%7D%20or%20estimator%2C%20default%3D%27drop%27">
            remainder
            <span class="param-doc-description">remainder: {'drop', 'passthrough'} or estimator, default='drop'<br><br>By default, only the specified columns in `transformers` are<br>transformed and combined in the output, and the non-specified<br>columns are dropped. (default of ``'drop'``).<br>By specifying ``remainder='passthrough'``, all remaining columns that<br>were not specified in `transformers`, but present in the data passed<br>to `fit` will be automatically passed through. This subset of columns<br>is concatenated with the output of the transformers. For dataframes,<br>extra columns not seen during `fit` will be excluded from the output<br>of `transform`.<br>By setting ``remainder`` to be an estimator, the remaining<br>non-specified columns will use the ``remainder`` estimator. The<br>estimator must support :term:`fit` and :term:`transform`.<br>Note that using this feature requires that the DataFrame columns<br>input at :term:`fit` and :term:`transform` have identical order.</span>
        </a>
    </td>
            <td class="value">&#x27;drop&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('sparse_threshold',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=sparse_threshold,-float%2C%20default%3D0.3">
            sparse_threshold
            <span class="param-doc-description">sparse_threshold: float, default=0.3<br><br>If the output of the different transformers contains sparse matrices,<br>these will be stacked as a sparse matrix if the overall density is<br>lower than this value. Use ``sparse_threshold=0`` to always return<br>dense.  When the transformed output consists of all dense data, the<br>stacked result will be dense, and this keyword will be ignored.</span>
        </a>
    </td>
            <td class="value">0.3</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('n_jobs',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=n_jobs,-int%2C%20default%3DNone">
            n_jobs
            <span class="param-doc-description">n_jobs: int, default=None<br><br>Number of jobs to run in parallel.<br>``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.<br>``-1`` means using all processors. See :term:`Glossary <n_jobs>`<br>for more details.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('transformer_weights',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=transformer_weights,-dict%2C%20default%3DNone">
            transformer_weights
            <span class="param-doc-description">transformer_weights: dict, default=None<br><br>Multiplicative weights for features per transformer. The output of the<br>transformer is multiplied by these weights. Keys are transformer names,<br>values the weights.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=verbose,-bool%2C%20default%3DFalse">
            verbose
            <span class="param-doc-description">verbose: bool, default=False<br><br>If True, the time elapsed while fitting each transformer will be<br>printed as it is completed.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('verbose_feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=verbose_feature_names_out,-bool%2C%20str%20or%20Callable%5B%5Bstr%2C%20str%5D%2C%20str%5D%2C%20default%3DTrue">
            verbose_feature_names_out
            <span class="param-doc-description">verbose_feature_names_out: bool, str or Callable[[str, str], str], default=True<br><br>- If True, :meth:`ColumnTransformer.get_feature_names_out` will prefix<br>  all feature names with the name of the transformer that generated that<br>  feature. It is equivalent to setting<br>  `verbose_feature_names_out="{transformer_name}__{feature_name}"`.<br>- If False, :meth:`ColumnTransformer.get_feature_names_out` will not<br>  prefix any feature names and will error if feature names are not<br>  unique.<br>- If ``Callable[[str, str], str]``,<br>  :meth:`ColumnTransformer.get_feature_names_out` will rename all the features<br>  using the name of the transformer. The first argument of the callable is the<br>  transformer name and the second argument is the feature name. The returned<br>  string will be the new feature name.<br>- If ``str``, it must be a string ready for formatting. The given string will<br>  be formatted using two field names: ``transformer_name`` and ``feature_name``.<br>  e.g. ``"{feature_name}__{transformer_name}"``. See :meth:`str.format` method<br>  from the standard library for more info.<br><br>.. versionadded:: 1.0<br><br>.. versionchanged:: 1.6<br>    `verbose_feature_names_out` can be a callable or a string to be formatted.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('force_int_remainder_cols',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.compose.ColumnTransformer.html#:~:text=force_int_remainder_cols,-bool%2C%20default%3DFalse">
            force_int_remainder_cols
            <span class="param-doc-description">force_int_remainder_cols: bool, default=False<br><br>This parameter has no effect.<br><br>.. note::<br>    If you do not access the list of columns for the remainder columns<br>    in the `transformers_` fitted attribute, you do not need to set<br>    this parameter.<br><br>.. versionadded:: 1.5<br><br>.. versionchanged:: 1.7<br>   The default value for `force_int_remainder_cols` will change from<br>   `True` to `False` in version 1.7.<br><br>.. deprecated:: 1.7<br>   `force_int_remainder_cols` is deprecated and will be removed in 1.9.</span>
        </a>
    </td>
            <td class="value">&#x27;deprecated&#x27;</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-528" type="checkbox" ><label for="sk-estimator-id-528" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>bin</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__bin__"><pre>[&#x27;MSSubClass_bin&#x27;, &#x27;LotFrontage_bin&#x27;, &#x27;LotArea_bin&#x27;, &#x27;MasVnrArea_bin&#x27;, &#x27;BsmtFinSF1_bin&#x27;, &#x27;BsmtUnfSF_bin&#x27;, &#x27;BsmtFinSF2_bin&#x27;, &#x27;TotalBsmtSF_bin&#x27;, &#x27;1stFlrSF_bin&#x27;, &#x27;2ndFlrSF_bin&#x27;, &#x27;LowQualFinSF_bin&#x27;, &#x27;GrLivArea_bin&#x27;, &#x27;GarageArea_bin&#x27;, &#x27;WoodDeckSF_bin&#x27;, &#x27;OpenPorchSF_bin&#x27;, &#x27;EnclosedPorch_bin&#x27;, &#x27;3SsnPorch_bin&#x27;, &#x27;ScreenPorch_bin&#x27;, &#x27;PoolArea_bin&#x27;, &#x27;MiscVal_bin&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-529" type="checkbox" ><label for="sk-estimator-id-529" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>&lt;lambda&gt;</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__bin__to_int__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=func,-callable%2C%20default%3DNone">
            func
            <span class="param-doc-description">func: callable, default=None<br><br>The callable to use for the transformation. This will be passed<br>the same arguments as transform, with args and kwargs forwarded.<br>If func is None, then func will be the identity function.</span>
        </a>
    </td>
            <td class="value">&lt;function &lt;la...001FBB3F07920&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inverse_func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inverse_func,-callable%2C%20default%3DNone">
            inverse_func
            <span class="param-doc-description">inverse_func: callable, default=None<br><br>The callable to use for the inverse transformation. This will be<br>passed the same arguments as inverse transform, with args and<br>kwargs forwarded. If inverse_func is None, then inverse_func<br>will be the identity function.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=validate,-bool%2C%20default%3DFalse">
            validate
            <span class="param-doc-description">validate: bool, default=False<br><br>Indicate that the input X array should be checked before calling<br>``func``. The possibilities are:<br><br>- If False, there is no input validation.<br>- If True, then X will be converted to a 2-dimensional NumPy array or<br>  sparse matrix. If the conversion is not possible an exception is<br>  raised.<br><br>.. versionchanged:: 0.22<br>   The default of ``validate`` changed from True to False.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('accept_sparse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=accept_sparse,-bool%2C%20default%3DFalse">
            accept_sparse
            <span class="param-doc-description">accept_sparse: bool, default=False<br><br>Indicate that func accepts a sparse matrix as input. If validate is<br>False, this has no effect. Otherwise, if accept_sparse is false,<br>sparse matrix inputs will cause an exception to be raised.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('check_inverse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=check_inverse,-bool%2C%20default%3DTrue">
            check_inverse
            <span class="param-doc-description">check_inverse: bool, default=True<br><br>Whether to check that or ``func`` followed by ``inverse_func`` leads to<br>the original inputs. It can be used for a sanity check, raising a<br>warning when the condition is not fulfilled.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=feature_names_out,-callable%2C%20%27one-to-one%27%20or%20None%2C%20default%3DNone">
            feature_names_out
            <span class="param-doc-description">feature_names_out: callable, 'one-to-one' or None, default=None<br><br>Determines the list of feature names that will be returned by the<br>`get_feature_names_out` method. If it is 'one-to-one', then the output<br>feature names will be equal to the input feature names. If it is a<br>callable, then it must take two positional arguments: this<br>`FunctionTransformer` (`self`) and an array-like of input feature names<br>(`input_features`). It must return an array-like of output feature<br>names. The `get_feature_names_out` method is only defined if<br>`feature_names_out` is not None.<br><br>See ``get_feature_names_out`` for more details.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=kw_args,-dict%2C%20default%3DNone">
            kw_args
            <span class="param-doc-description">kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inv_kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inv_kw_args,-dict%2C%20default%3DNone">
            inv_kw_args
            <span class="param-doc-description">inv_kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to inverse_func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-530" type="checkbox" ><label for="sk-estimator-id-530" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__bin__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-531" type="checkbox" ><label for="sk-estimator-id-531" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__bin__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If False, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_mean,-bool%2C%20default%3DTrue">
            with_mean
            <span class="param-doc-description">with_mean: bool, default=True<br><br>If True, center the data before scaling.<br>This does not work (and will raise an exception) when attempted on<br>sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_std,-bool%2C%20default%3DTrue">
            with_std
            <span class="param-doc-description">with_std: bool, default=True<br><br>If True, scale the data to unit variance (or equivalently,<br>unit standard deviation).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-532" type="checkbox" ><label for="sk-estimator-id-532" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>cont</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__cont__"><pre>[&#x27;MSSubClass&#x27;, &#x27;LotFrontage&#x27;, &#x27;LotArea&#x27;, &#x27;MasVnrArea&#x27;, &#x27;BsmtFinSF1&#x27;, &#x27;BsmtUnfSF&#x27;, &#x27;BsmtFinSF2&#x27;, &#x27;TotalBsmtSF&#x27;, &#x27;1stFlrSF&#x27;, &#x27;2ndFlrSF&#x27;, &#x27;LowQualFinSF&#x27;, &#x27;GrLivArea&#x27;, &#x27;GarageArea&#x27;, &#x27;WoodDeckSF&#x27;, &#x27;OpenPorchSF&#x27;, &#x27;EnclosedPorch&#x27;, &#x27;3SsnPorch&#x27;, &#x27;ScreenPorch&#x27;, &#x27;PoolArea&#x27;, &#x27;MiscVal&#x27;, &#x27;1stFlrSF_squared&#x27;, &#x27;2ndFlrSF_squared&#x27;, &#x27;GrLivArea_squared&#x27;, &#x27;LAIFF&#x27;, &#x27;GLAPR&#x27;, &#x27;GINGLR&#x27;, &#x27;BPB&#x27;, &#x27;SQ2FPQF1F&#x27;, &#x27;TotalSF&#x27;, &#x27;TotalBath&#x27;, &#x27;TotalQual&#x27;, &#x27;GLAB&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-533" type="checkbox" ><label for="sk-estimator-id-533" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__cont__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;most_frequent&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-534" type="checkbox" ><label for="sk-estimator-id-534" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>PowerTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.PowerTransformer.html">?<span>Documentation for PowerTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__cont__transformer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('method',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.PowerTransformer.html#:~:text=method,-%7B%27yeo-johnson%27%2C%20%27box-cox%27%7D%2C%20default%3D%27yeo-johnson%27">
            method
            <span class="param-doc-description">method: {'yeo-johnson', 'box-cox'}, default='yeo-johnson'<br><br>The power transform method. Available methods are:<br><br>- 'yeo-johnson' [1]_, works with positive and negative values<br>- 'box-cox' [2]_, only works with strictly positive values</span>
        </a>
    </td>
            <td class="value">&#x27;yeo-johnson&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('standardize',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.PowerTransformer.html#:~:text=standardize,-bool%2C%20default%3DTrue">
            standardize
            <span class="param-doc-description">standardize: bool, default=True<br><br>Set to True to apply zero-mean, unit-variance normalization to the<br>transformed output.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.PowerTransformer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>Set to False to perform inplace computation during transformation.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-535" type="checkbox" ><label for="sk-estimator-id-535" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RobustScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.RobustScaler.html">?<span>Documentation for RobustScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__cont__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_centering',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.RobustScaler.html#:~:text=with_centering,-bool%2C%20default%3DTrue">
            with_centering
            <span class="param-doc-description">with_centering: bool, default=True<br><br>If `True`, center the data before scaling.<br>This will cause :meth:`transform` to raise an exception when attempted<br>on sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_scaling',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.RobustScaler.html#:~:text=with_scaling,-bool%2C%20default%3DTrue">
            with_scaling
            <span class="param-doc-description">with_scaling: bool, default=True<br><br>If `True`, scale the data to interquartile range.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('quantile_range',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.RobustScaler.html#:~:text=quantile_range,-tuple%20%28q_min%2C%20q_max%29%2C%200.0%20%3C%20q_min%20%3C%20q_max%20%3C%20100.0%2C%20%20%20%20%20%20%20%20%20default%3D%2825.0%2C%2075.0%29">
            quantile_range
            <span class="param-doc-description">quantile_range: tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0,         default=(25.0, 75.0)<br><br>Quantile range used to calculate `scale_`. By default this is equal to<br>the IQR, i.e., `q_min` is the first quantile and `q_max` is the third<br>quantile.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">(25.0, ...)</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.RobustScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If `False`, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('unit_variance',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.RobustScaler.html#:~:text=unit_variance,-bool%2C%20default%3DFalse">
            unit_variance
            <span class="param-doc-description">unit_variance: bool, default=False<br><br>If `True`, scale data so that normally distributed features have a<br>variance of 1. In general, if the difference between the x-values of<br>`q_max` and `q_min` for a standard normal distribution is greater<br>than 1, the dataset will be scaled down. If less than 1, the dataset<br>will be scaled up.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-536" type="checkbox" ><label for="sk-estimator-id-536" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>count</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__count__"><pre>[&#x27;OverallQual&#x27;, &#x27;OverallCond&#x27;, &#x27;YearBuilt&#x27;, &#x27;YearRemodAdd&#x27;, &#x27;BsmtFullBath&#x27;, &#x27;BsmtHalfBath&#x27;, &#x27;FullBath&#x27;, &#x27;HalfBath&#x27;, &#x27;BedroomAbvGr&#x27;, &#x27;KitchenAbvGr&#x27;, &#x27;TotRmsAbvGrd&#x27;, &#x27;Fireplaces&#x27;, &#x27;GarageYrBlt&#x27;, &#x27;GarageCars&#x27;, &#x27;YrSold&#x27;, &#x27;houseAge&#x27;, &#x27;ageSinceRemodel&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-537" type="checkbox" ><label for="sk-estimator-id-537" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__count__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-538" type="checkbox" ><label for="sk-estimator-id-538" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RobustScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.RobustScaler.html">?<span>Documentation for RobustScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__count__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_centering',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.RobustScaler.html#:~:text=with_centering,-bool%2C%20default%3DTrue">
            with_centering
            <span class="param-doc-description">with_centering: bool, default=True<br><br>If `True`, center the data before scaling.<br>This will cause :meth:`transform` to raise an exception when attempted<br>on sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_scaling',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.RobustScaler.html#:~:text=with_scaling,-bool%2C%20default%3DTrue">
            with_scaling
            <span class="param-doc-description">with_scaling: bool, default=True<br><br>If `True`, scale the data to interquartile range.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('quantile_range',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.RobustScaler.html#:~:text=quantile_range,-tuple%20%28q_min%2C%20q_max%29%2C%200.0%20%3C%20q_min%20%3C%20q_max%20%3C%20100.0%2C%20%20%20%20%20%20%20%20%20default%3D%2825.0%2C%2075.0%29">
            quantile_range
            <span class="param-doc-description">quantile_range: tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0,         default=(25.0, 75.0)<br><br>Quantile range used to calculate `scale_`. By default this is equal to<br>the IQR, i.e., `q_min` is the first quantile and `q_max` is the third<br>quantile.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">(25.0, ...)</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.RobustScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If `False`, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('unit_variance',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.RobustScaler.html#:~:text=unit_variance,-bool%2C%20default%3DFalse">
            unit_variance
            <span class="param-doc-description">unit_variance: bool, default=False<br><br>If `True`, scale data so that normally distributed features have a<br>variance of 1. In general, if the difference between the x-values of<br>`q_max` and `q_min` for a standard normal distribution is greater<br>than 1, the dataset will be scaled down. If less than 1, the dataset<br>will be scaled up.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-539" type="checkbox" ><label for="sk-estimator-id-539" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>mo_sold</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__mo_sold__"><pre>[&#x27;MoSold&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-540" type="checkbox" ><label for="sk-estimator-id-540" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>transform_cycle</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__mo_sold__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=func,-callable%2C%20default%3DNone">
            func
            <span class="param-doc-description">func: callable, default=None<br><br>The callable to use for the transformation. This will be passed<br>the same arguments as transform, with args and kwargs forwarded.<br>If func is None, then func will be the identity function.</span>
        </a>
    </td>
            <td class="value">&lt;function tra...001FBB3F07420&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inverse_func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inverse_func,-callable%2C%20default%3DNone">
            inverse_func
            <span class="param-doc-description">inverse_func: callable, default=None<br><br>The callable to use for the inverse transformation. This will be<br>passed the same arguments as inverse transform, with args and<br>kwargs forwarded. If inverse_func is None, then inverse_func<br>will be the identity function.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=validate,-bool%2C%20default%3DFalse">
            validate
            <span class="param-doc-description">validate: bool, default=False<br><br>Indicate that the input X array should be checked before calling<br>``func``. The possibilities are:<br><br>- If False, there is no input validation.<br>- If True, then X will be converted to a 2-dimensional NumPy array or<br>  sparse matrix. If the conversion is not possible an exception is<br>  raised.<br><br>.. versionchanged:: 0.22<br>   The default of ``validate`` changed from True to False.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('accept_sparse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=accept_sparse,-bool%2C%20default%3DFalse">
            accept_sparse
            <span class="param-doc-description">accept_sparse: bool, default=False<br><br>Indicate that func accepts a sparse matrix as input. If validate is<br>False, this has no effect. Otherwise, if accept_sparse is false,<br>sparse matrix inputs will cause an exception to be raised.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('check_inverse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=check_inverse,-bool%2C%20default%3DTrue">
            check_inverse
            <span class="param-doc-description">check_inverse: bool, default=True<br><br>Whether to check that or ``func`` followed by ``inverse_func`` leads to<br>the original inputs. It can be used for a sanity check, raising a<br>warning when the condition is not fulfilled.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=feature_names_out,-callable%2C%20%27one-to-one%27%20or%20None%2C%20default%3DNone">
            feature_names_out
            <span class="param-doc-description">feature_names_out: callable, 'one-to-one' or None, default=None<br><br>Determines the list of feature names that will be returned by the<br>`get_feature_names_out` method. If it is 'one-to-one', then the output<br>feature names will be equal to the input feature names. If it is a<br>callable, then it must take two positional arguments: this<br>`FunctionTransformer` (`self`) and an array-like of input feature names<br>(`input_features`). It must return an array-like of output feature<br>names. The `get_feature_names_out` method is only defined if<br>`feature_names_out` is not None.<br><br>See ``get_feature_names_out`` for more details.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=kw_args,-dict%2C%20default%3DNone">
            kw_args
            <span class="param-doc-description">kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inv_kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inv_kw_args,-dict%2C%20default%3DNone">
            inv_kw_args
            <span class="param-doc-description">inv_kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to inverse_func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-541" type="checkbox" ><label for="sk-estimator-id-541" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>qual</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__qual__"><pre>[&#x27;GarageCond&#x27;, &#x27;GarageQual&#x27;, &#x27;ExterQual&#x27;, &#x27;ExterCond&#x27;, &#x27;KitchenQual&#x27;, &#x27;HeatingQC&#x27;, &#x27;FireplaceQu&#x27;, &#x27;BsmtQual&#x27;, &#x27;BsmtCond&#x27;, &#x27;PoolQC&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-542" type="checkbox" ><label for="sk-estimator-id-542" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__qual__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">&#x27;MISSING&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-543" type="checkbox" ><label for="sk-estimator-id-543" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>to_upper_case</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__qual__to_upper__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=func,-callable%2C%20default%3DNone">
            func
            <span class="param-doc-description">func: callable, default=None<br><br>The callable to use for the transformation. This will be passed<br>the same arguments as transform, with args and kwargs forwarded.<br>If func is None, then func will be the identity function.</span>
        </a>
    </td>
            <td class="value">&lt;function to_...001FBB3F077E0&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inverse_func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inverse_func,-callable%2C%20default%3DNone">
            inverse_func
            <span class="param-doc-description">inverse_func: callable, default=None<br><br>The callable to use for the inverse transformation. This will be<br>passed the same arguments as inverse transform, with args and<br>kwargs forwarded. If inverse_func is None, then inverse_func<br>will be the identity function.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=validate,-bool%2C%20default%3DFalse">
            validate
            <span class="param-doc-description">validate: bool, default=False<br><br>Indicate that the input X array should be checked before calling<br>``func``. The possibilities are:<br><br>- If False, there is no input validation.<br>- If True, then X will be converted to a 2-dimensional NumPy array or<br>  sparse matrix. If the conversion is not possible an exception is<br>  raised.<br><br>.. versionchanged:: 0.22<br>   The default of ``validate`` changed from True to False.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('accept_sparse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=accept_sparse,-bool%2C%20default%3DFalse">
            accept_sparse
            <span class="param-doc-description">accept_sparse: bool, default=False<br><br>Indicate that func accepts a sparse matrix as input. If validate is<br>False, this has no effect. Otherwise, if accept_sparse is false,<br>sparse matrix inputs will cause an exception to be raised.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('check_inverse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=check_inverse,-bool%2C%20default%3DTrue">
            check_inverse
            <span class="param-doc-description">check_inverse: bool, default=True<br><br>Whether to check that or ``func`` followed by ``inverse_func`` leads to<br>the original inputs. It can be used for a sanity check, raising a<br>warning when the condition is not fulfilled.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=feature_names_out,-callable%2C%20%27one-to-one%27%20or%20None%2C%20default%3DNone">
            feature_names_out
            <span class="param-doc-description">feature_names_out: callable, 'one-to-one' or None, default=None<br><br>Determines the list of feature names that will be returned by the<br>`get_feature_names_out` method. If it is 'one-to-one', then the output<br>feature names will be equal to the input feature names. If it is a<br>callable, then it must take two positional arguments: this<br>`FunctionTransformer` (`self`) and an array-like of input feature names<br>(`input_features`). It must return an array-like of output feature<br>names. The `get_feature_names_out` method is only defined if<br>`feature_names_out` is not None.<br><br>See ``get_feature_names_out`` for more details.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=kw_args,-dict%2C%20default%3DNone">
            kw_args
            <span class="param-doc-description">kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inv_kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inv_kw_args,-dict%2C%20default%3DNone">
            inv_kw_args
            <span class="param-doc-description">inv_kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to inverse_func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-544" type="checkbox" ><label for="sk-estimator-id-544" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OrdinalEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__qual__ordinal_map__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=categories,-%27auto%27%20or%20a%20list%20of%20array-like%2C%20default%3D%27auto%27">
            categories
            <span class="param-doc-description">categories: 'auto' or a list of array-like, default='auto'<br><br>Categories (unique values) per feature:<br><br>- 'auto' : Determine categories automatically from the training data.<br>- list : ``categories[i]`` holds the categories expected in the ith<br>  column. The passed categories should not mix strings and numeric<br>  values, and should be sorted in case of numeric values.<br><br>The used categories can be found in the ``categories_`` attribute.</span>
        </a>
    </td>
            <td class="value">[[&#x27;MISSING&#x27;, &#x27;PO&#x27;, ...], [&#x27;MISSING&#x27;, &#x27;PO&#x27;, ...], ...]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=dtype,-number%20type%2C%20default%3Dnp.float64">
            dtype
            <span class="param-doc-description">dtype: number type, default=np.float64<br><br>Desired dtype of output.</span>
        </a>
    </td>
            <td class="value">&lt;class &#x27;numpy.float64&#x27;&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('handle_unknown',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=handle_unknown,-%7B%27error%27%2C%20%27use_encoded_value%27%7D%2C%20default%3D%27error%27">
            handle_unknown
            <span class="param-doc-description">handle_unknown: {'error', 'use_encoded_value'}, default='error'<br><br>When set to 'error' an error will be raised in case an unknown<br>categorical feature is present during transform. When set to<br>'use_encoded_value', the encoded value of unknown categories will be<br>set to the value given for the parameter `unknown_value`. In<br>:meth:`inverse_transform`, an unknown category will be denoted as None.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">&#x27;error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('unknown_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=unknown_value,-int%20or%20np.nan%2C%20default%3DNone">
            unknown_value
            <span class="param-doc-description">unknown_value: int or np.nan, default=None<br><br>When the parameter handle_unknown is set to 'use_encoded_value', this<br>parameter is required and will set the encoded value of unknown<br>categories. It has to be distinct from the values used to encode any of<br>the categories in `fit`. If set to np.nan, the `dtype` parameter must<br>be a float dtype.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoded_missing_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=encoded_missing_value,-int%20or%20np.nan%2C%20default%3Dnp.nan">
            encoded_missing_value
            <span class="param-doc-description">encoded_missing_value: int or np.nan, default=np.nan<br><br>Encoded value of missing categories. If set to `np.nan`, then the `dtype`<br>parameter must be a float dtype.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_frequency',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=min_frequency,-int%20or%20float%2C%20default%3DNone">
            min_frequency
            <span class="param-doc-description">min_frequency: int or float, default=None<br><br>Specifies the minimum frequency below which a category will be<br>considered infrequent.<br><br>- If `int`, categories with a smaller cardinality will be considered<br>  infrequent.<br><br>- If `float`, categories with a smaller cardinality than<br>  `min_frequency * n_samples`  will be considered infrequent.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=max_categories,-int%2C%20default%3DNone">
            max_categories
            <span class="param-doc-description">max_categories: int, default=None<br><br>Specifies an upper limit to the number of output categories for each input<br>feature when considering infrequent categories. If there are infrequent<br>categories, `max_categories` includes the category representing the<br>infrequent categories along with the frequent categories. If `None`,<br>there is no limit to the number of output features.<br><br>`max_categories` do **not** take into account missing or unknown<br>categories. Setting `unknown_value` or `encoded_missing_value` to an<br>integer will increase the number of unique integer codes by one each.<br>This can result in up to `max_categories + 2` integer codes.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-545" type="checkbox" ><label for="sk-estimator-id-545" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__qual__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If False, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_mean,-bool%2C%20default%3DTrue">
            with_mean
            <span class="param-doc-description">with_mean: bool, default=True<br><br>If True, center the data before scaling.<br>This does not work (and will raise an exception) when attempted on<br>sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_std,-bool%2C%20default%3DTrue">
            with_std
            <span class="param-doc-description">with_std: bool, default=True<br><br>If True, scale the data to unit variance (or equivalently,<br>unit standard deviation).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-546" type="checkbox" ><label for="sk-estimator-id-546" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>gar_fin</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__gar_fin__"><pre>[&#x27;GarageFinish&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-547" type="checkbox" ><label for="sk-estimator-id-547" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__gar_fin__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">&#x27;MISSING&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-548" type="checkbox" ><label for="sk-estimator-id-548" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>to_upper_case</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__gar_fin__to_upper__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=func,-callable%2C%20default%3DNone">
            func
            <span class="param-doc-description">func: callable, default=None<br><br>The callable to use for the transformation. This will be passed<br>the same arguments as transform, with args and kwargs forwarded.<br>If func is None, then func will be the identity function.</span>
        </a>
    </td>
            <td class="value">&lt;function to_...001FBB3F077E0&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inverse_func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inverse_func,-callable%2C%20default%3DNone">
            inverse_func
            <span class="param-doc-description">inverse_func: callable, default=None<br><br>The callable to use for the inverse transformation. This will be<br>passed the same arguments as inverse transform, with args and<br>kwargs forwarded. If inverse_func is None, then inverse_func<br>will be the identity function.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=validate,-bool%2C%20default%3DFalse">
            validate
            <span class="param-doc-description">validate: bool, default=False<br><br>Indicate that the input X array should be checked before calling<br>``func``. The possibilities are:<br><br>- If False, there is no input validation.<br>- If True, then X will be converted to a 2-dimensional NumPy array or<br>  sparse matrix. If the conversion is not possible an exception is<br>  raised.<br><br>.. versionchanged:: 0.22<br>   The default of ``validate`` changed from True to False.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('accept_sparse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=accept_sparse,-bool%2C%20default%3DFalse">
            accept_sparse
            <span class="param-doc-description">accept_sparse: bool, default=False<br><br>Indicate that func accepts a sparse matrix as input. If validate is<br>False, this has no effect. Otherwise, if accept_sparse is false,<br>sparse matrix inputs will cause an exception to be raised.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('check_inverse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=check_inverse,-bool%2C%20default%3DTrue">
            check_inverse
            <span class="param-doc-description">check_inverse: bool, default=True<br><br>Whether to check that or ``func`` followed by ``inverse_func`` leads to<br>the original inputs. It can be used for a sanity check, raising a<br>warning when the condition is not fulfilled.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=feature_names_out,-callable%2C%20%27one-to-one%27%20or%20None%2C%20default%3DNone">
            feature_names_out
            <span class="param-doc-description">feature_names_out: callable, 'one-to-one' or None, default=None<br><br>Determines the list of feature names that will be returned by the<br>`get_feature_names_out` method. If it is 'one-to-one', then the output<br>feature names will be equal to the input feature names. If it is a<br>callable, then it must take two positional arguments: this<br>`FunctionTransformer` (`self`) and an array-like of input feature names<br>(`input_features`). It must return an array-like of output feature<br>names. The `get_feature_names_out` method is only defined if<br>`feature_names_out` is not None.<br><br>See ``get_feature_names_out`` for more details.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=kw_args,-dict%2C%20default%3DNone">
            kw_args
            <span class="param-doc-description">kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inv_kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inv_kw_args,-dict%2C%20default%3DNone">
            inv_kw_args
            <span class="param-doc-description">inv_kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to inverse_func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-549" type="checkbox" ><label for="sk-estimator-id-549" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OrdinalEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__gar_fin__ordinal_map__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=categories,-%27auto%27%20or%20a%20list%20of%20array-like%2C%20default%3D%27auto%27">
            categories
            <span class="param-doc-description">categories: 'auto' or a list of array-like, default='auto'<br><br>Categories (unique values) per feature:<br><br>- 'auto' : Determine categories automatically from the training data.<br>- list : ``categories[i]`` holds the categories expected in the ith<br>  column. The passed categories should not mix strings and numeric<br>  values, and should be sorted in case of numeric values.<br><br>The used categories can be found in the ``categories_`` attribute.</span>
        </a>
    </td>
            <td class="value">[[&#x27;MISSING&#x27;, &#x27;UNF&#x27;, ...]]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=dtype,-number%20type%2C%20default%3Dnp.float64">
            dtype
            <span class="param-doc-description">dtype: number type, default=np.float64<br><br>Desired dtype of output.</span>
        </a>
    </td>
            <td class="value">&lt;class &#x27;numpy.float64&#x27;&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('handle_unknown',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=handle_unknown,-%7B%27error%27%2C%20%27use_encoded_value%27%7D%2C%20default%3D%27error%27">
            handle_unknown
            <span class="param-doc-description">handle_unknown: {'error', 'use_encoded_value'}, default='error'<br><br>When set to 'error' an error will be raised in case an unknown<br>categorical feature is present during transform. When set to<br>'use_encoded_value', the encoded value of unknown categories will be<br>set to the value given for the parameter `unknown_value`. In<br>:meth:`inverse_transform`, an unknown category will be denoted as None.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">&#x27;error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('unknown_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=unknown_value,-int%20or%20np.nan%2C%20default%3DNone">
            unknown_value
            <span class="param-doc-description">unknown_value: int or np.nan, default=None<br><br>When the parameter handle_unknown is set to 'use_encoded_value', this<br>parameter is required and will set the encoded value of unknown<br>categories. It has to be distinct from the values used to encode any of<br>the categories in `fit`. If set to np.nan, the `dtype` parameter must<br>be a float dtype.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoded_missing_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=encoded_missing_value,-int%20or%20np.nan%2C%20default%3Dnp.nan">
            encoded_missing_value
            <span class="param-doc-description">encoded_missing_value: int or np.nan, default=np.nan<br><br>Encoded value of missing categories. If set to `np.nan`, then the `dtype`<br>parameter must be a float dtype.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_frequency',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=min_frequency,-int%20or%20float%2C%20default%3DNone">
            min_frequency
            <span class="param-doc-description">min_frequency: int or float, default=None<br><br>Specifies the minimum frequency below which a category will be<br>considered infrequent.<br><br>- If `int`, categories with a smaller cardinality will be considered<br>  infrequent.<br><br>- If `float`, categories with a smaller cardinality than<br>  `min_frequency * n_samples`  will be considered infrequent.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=max_categories,-int%2C%20default%3DNone">
            max_categories
            <span class="param-doc-description">max_categories: int, default=None<br><br>Specifies an upper limit to the number of output categories for each input<br>feature when considering infrequent categories. If there are infrequent<br>categories, `max_categories` includes the category representing the<br>infrequent categories along with the frequent categories. If `None`,<br>there is no limit to the number of output features.<br><br>`max_categories` do **not** take into account missing or unknown<br>categories. Setting `unknown_value` or `encoded_missing_value` to an<br>integer will increase the number of unique integer codes by one each.<br>This can result in up to `max_categories + 2` integer codes.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-550" type="checkbox" ><label for="sk-estimator-id-550" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__gar_fin__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If False, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_mean,-bool%2C%20default%3DTrue">
            with_mean
            <span class="param-doc-description">with_mean: bool, default=True<br><br>If True, center the data before scaling.<br>This does not work (and will raise an exception) when attempted on<br>sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_std,-bool%2C%20default%3DTrue">
            with_std
            <span class="param-doc-description">with_std: bool, default=True<br><br>If True, scale the data to unit variance (or equivalently,<br>unit standard deviation).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-551" type="checkbox" ><label for="sk-estimator-id-551" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>paved</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__paved__"><pre>[&#x27;PavedDrive&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-552" type="checkbox" ><label for="sk-estimator-id-552" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__paved__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">&#x27;MISSING&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-553" type="checkbox" ><label for="sk-estimator-id-553" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>to_upper_case</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__paved__to_upper__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=func,-callable%2C%20default%3DNone">
            func
            <span class="param-doc-description">func: callable, default=None<br><br>The callable to use for the transformation. This will be passed<br>the same arguments as transform, with args and kwargs forwarded.<br>If func is None, then func will be the identity function.</span>
        </a>
    </td>
            <td class="value">&lt;function to_...001FBB3F077E0&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inverse_func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inverse_func,-callable%2C%20default%3DNone">
            inverse_func
            <span class="param-doc-description">inverse_func: callable, default=None<br><br>The callable to use for the inverse transformation. This will be<br>passed the same arguments as inverse transform, with args and<br>kwargs forwarded. If inverse_func is None, then inverse_func<br>will be the identity function.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=validate,-bool%2C%20default%3DFalse">
            validate
            <span class="param-doc-description">validate: bool, default=False<br><br>Indicate that the input X array should be checked before calling<br>``func``. The possibilities are:<br><br>- If False, there is no input validation.<br>- If True, then X will be converted to a 2-dimensional NumPy array or<br>  sparse matrix. If the conversion is not possible an exception is<br>  raised.<br><br>.. versionchanged:: 0.22<br>   The default of ``validate`` changed from True to False.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('accept_sparse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=accept_sparse,-bool%2C%20default%3DFalse">
            accept_sparse
            <span class="param-doc-description">accept_sparse: bool, default=False<br><br>Indicate that func accepts a sparse matrix as input. If validate is<br>False, this has no effect. Otherwise, if accept_sparse is false,<br>sparse matrix inputs will cause an exception to be raised.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('check_inverse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=check_inverse,-bool%2C%20default%3DTrue">
            check_inverse
            <span class="param-doc-description">check_inverse: bool, default=True<br><br>Whether to check that or ``func`` followed by ``inverse_func`` leads to<br>the original inputs. It can be used for a sanity check, raising a<br>warning when the condition is not fulfilled.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=feature_names_out,-callable%2C%20%27one-to-one%27%20or%20None%2C%20default%3DNone">
            feature_names_out
            <span class="param-doc-description">feature_names_out: callable, 'one-to-one' or None, default=None<br><br>Determines the list of feature names that will be returned by the<br>`get_feature_names_out` method. If it is 'one-to-one', then the output<br>feature names will be equal to the input feature names. If it is a<br>callable, then it must take two positional arguments: this<br>`FunctionTransformer` (`self`) and an array-like of input feature names<br>(`input_features`). It must return an array-like of output feature<br>names. The `get_feature_names_out` method is only defined if<br>`feature_names_out` is not None.<br><br>See ``get_feature_names_out`` for more details.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=kw_args,-dict%2C%20default%3DNone">
            kw_args
            <span class="param-doc-description">kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inv_kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inv_kw_args,-dict%2C%20default%3DNone">
            inv_kw_args
            <span class="param-doc-description">inv_kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to inverse_func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-554" type="checkbox" ><label for="sk-estimator-id-554" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OrdinalEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__paved__ordinal_map__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=categories,-%27auto%27%20or%20a%20list%20of%20array-like%2C%20default%3D%27auto%27">
            categories
            <span class="param-doc-description">categories: 'auto' or a list of array-like, default='auto'<br><br>Categories (unique values) per feature:<br><br>- 'auto' : Determine categories automatically from the training data.<br>- list : ``categories[i]`` holds the categories expected in the ith<br>  column. The passed categories should not mix strings and numeric<br>  values, and should be sorted in case of numeric values.<br><br>The used categories can be found in the ``categories_`` attribute.</span>
        </a>
    </td>
            <td class="value">[[&#x27;MISSING&#x27;, &#x27;N&#x27;, ...]]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=dtype,-number%20type%2C%20default%3Dnp.float64">
            dtype
            <span class="param-doc-description">dtype: number type, default=np.float64<br><br>Desired dtype of output.</span>
        </a>
    </td>
            <td class="value">&lt;class &#x27;numpy.float64&#x27;&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('handle_unknown',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=handle_unknown,-%7B%27error%27%2C%20%27use_encoded_value%27%7D%2C%20default%3D%27error%27">
            handle_unknown
            <span class="param-doc-description">handle_unknown: {'error', 'use_encoded_value'}, default='error'<br><br>When set to 'error' an error will be raised in case an unknown<br>categorical feature is present during transform. When set to<br>'use_encoded_value', the encoded value of unknown categories will be<br>set to the value given for the parameter `unknown_value`. In<br>:meth:`inverse_transform`, an unknown category will be denoted as None.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">&#x27;error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('unknown_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=unknown_value,-int%20or%20np.nan%2C%20default%3DNone">
            unknown_value
            <span class="param-doc-description">unknown_value: int or np.nan, default=None<br><br>When the parameter handle_unknown is set to 'use_encoded_value', this<br>parameter is required and will set the encoded value of unknown<br>categories. It has to be distinct from the values used to encode any of<br>the categories in `fit`. If set to np.nan, the `dtype` parameter must<br>be a float dtype.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoded_missing_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=encoded_missing_value,-int%20or%20np.nan%2C%20default%3Dnp.nan">
            encoded_missing_value
            <span class="param-doc-description">encoded_missing_value: int or np.nan, default=np.nan<br><br>Encoded value of missing categories. If set to `np.nan`, then the `dtype`<br>parameter must be a float dtype.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_frequency',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=min_frequency,-int%20or%20float%2C%20default%3DNone">
            min_frequency
            <span class="param-doc-description">min_frequency: int or float, default=None<br><br>Specifies the minimum frequency below which a category will be<br>considered infrequent.<br><br>- If `int`, categories with a smaller cardinality will be considered<br>  infrequent.<br><br>- If `float`, categories with a smaller cardinality than<br>  `min_frequency * n_samples`  will be considered infrequent.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=max_categories,-int%2C%20default%3DNone">
            max_categories
            <span class="param-doc-description">max_categories: int, default=None<br><br>Specifies an upper limit to the number of output categories for each input<br>feature when considering infrequent categories. If there are infrequent<br>categories, `max_categories` includes the category representing the<br>infrequent categories along with the frequent categories. If `None`,<br>there is no limit to the number of output features.<br><br>`max_categories` do **not** take into account missing or unknown<br>categories. Setting `unknown_value` or `encoded_missing_value` to an<br>integer will increase the number of unique integer codes by one each.<br>This can result in up to `max_categories + 2` integer codes.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-555" type="checkbox" ><label for="sk-estimator-id-555" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__paved__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If False, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_mean,-bool%2C%20default%3DTrue">
            with_mean
            <span class="param-doc-description">with_mean: bool, default=True<br><br>If True, center the data before scaling.<br>This does not work (and will raise an exception) when attempted on<br>sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_std,-bool%2C%20default%3DTrue">
            with_std
            <span class="param-doc-description">with_std: bool, default=True<br><br>If True, scale the data to unit variance (or equivalently,<br>unit standard deviation).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-556" type="checkbox" ><label for="sk-estimator-id-556" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>elec</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__elec__"><pre>[&#x27;Electrical&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-557" type="checkbox" ><label for="sk-estimator-id-557" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__elec__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">&#x27;MISSING&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-558" type="checkbox" ><label for="sk-estimator-id-558" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>to_upper_case</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__elec__to_upper__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=func,-callable%2C%20default%3DNone">
            func
            <span class="param-doc-description">func: callable, default=None<br><br>The callable to use for the transformation. This will be passed<br>the same arguments as transform, with args and kwargs forwarded.<br>If func is None, then func will be the identity function.</span>
        </a>
    </td>
            <td class="value">&lt;function to_...001FBB3F077E0&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inverse_func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inverse_func,-callable%2C%20default%3DNone">
            inverse_func
            <span class="param-doc-description">inverse_func: callable, default=None<br><br>The callable to use for the inverse transformation. This will be<br>passed the same arguments as inverse transform, with args and<br>kwargs forwarded. If inverse_func is None, then inverse_func<br>will be the identity function.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=validate,-bool%2C%20default%3DFalse">
            validate
            <span class="param-doc-description">validate: bool, default=False<br><br>Indicate that the input X array should be checked before calling<br>``func``. The possibilities are:<br><br>- If False, there is no input validation.<br>- If True, then X will be converted to a 2-dimensional NumPy array or<br>  sparse matrix. If the conversion is not possible an exception is<br>  raised.<br><br>.. versionchanged:: 0.22<br>   The default of ``validate`` changed from True to False.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('accept_sparse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=accept_sparse,-bool%2C%20default%3DFalse">
            accept_sparse
            <span class="param-doc-description">accept_sparse: bool, default=False<br><br>Indicate that func accepts a sparse matrix as input. If validate is<br>False, this has no effect. Otherwise, if accept_sparse is false,<br>sparse matrix inputs will cause an exception to be raised.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('check_inverse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=check_inverse,-bool%2C%20default%3DTrue">
            check_inverse
            <span class="param-doc-description">check_inverse: bool, default=True<br><br>Whether to check that or ``func`` followed by ``inverse_func`` leads to<br>the original inputs. It can be used for a sanity check, raising a<br>warning when the condition is not fulfilled.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=feature_names_out,-callable%2C%20%27one-to-one%27%20or%20None%2C%20default%3DNone">
            feature_names_out
            <span class="param-doc-description">feature_names_out: callable, 'one-to-one' or None, default=None<br><br>Determines the list of feature names that will be returned by the<br>`get_feature_names_out` method. If it is 'one-to-one', then the output<br>feature names will be equal to the input feature names. If it is a<br>callable, then it must take two positional arguments: this<br>`FunctionTransformer` (`self`) and an array-like of input feature names<br>(`input_features`). It must return an array-like of output feature<br>names. The `get_feature_names_out` method is only defined if<br>`feature_names_out` is not None.<br><br>See ``get_feature_names_out`` for more details.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=kw_args,-dict%2C%20default%3DNone">
            kw_args
            <span class="param-doc-description">kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inv_kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inv_kw_args,-dict%2C%20default%3DNone">
            inv_kw_args
            <span class="param-doc-description">inv_kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to inverse_func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-559" type="checkbox" ><label for="sk-estimator-id-559" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OrdinalEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__elec__ordinal_map__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=categories,-%27auto%27%20or%20a%20list%20of%20array-like%2C%20default%3D%27auto%27">
            categories
            <span class="param-doc-description">categories: 'auto' or a list of array-like, default='auto'<br><br>Categories (unique values) per feature:<br><br>- 'auto' : Determine categories automatically from the training data.<br>- list : ``categories[i]`` holds the categories expected in the ith<br>  column. The passed categories should not mix strings and numeric<br>  values, and should be sorted in case of numeric values.<br><br>The used categories can be found in the ``categories_`` attribute.</span>
        </a>
    </td>
            <td class="value">[[&#x27;MISSING&#x27;, &#x27;MIX&#x27;, ...]]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=dtype,-number%20type%2C%20default%3Dnp.float64">
            dtype
            <span class="param-doc-description">dtype: number type, default=np.float64<br><br>Desired dtype of output.</span>
        </a>
    </td>
            <td class="value">&lt;class &#x27;numpy.float64&#x27;&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('handle_unknown',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=handle_unknown,-%7B%27error%27%2C%20%27use_encoded_value%27%7D%2C%20default%3D%27error%27">
            handle_unknown
            <span class="param-doc-description">handle_unknown: {'error', 'use_encoded_value'}, default='error'<br><br>When set to 'error' an error will be raised in case an unknown<br>categorical feature is present during transform. When set to<br>'use_encoded_value', the encoded value of unknown categories will be<br>set to the value given for the parameter `unknown_value`. In<br>:meth:`inverse_transform`, an unknown category will be denoted as None.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">&#x27;error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('unknown_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=unknown_value,-int%20or%20np.nan%2C%20default%3DNone">
            unknown_value
            <span class="param-doc-description">unknown_value: int or np.nan, default=None<br><br>When the parameter handle_unknown is set to 'use_encoded_value', this<br>parameter is required and will set the encoded value of unknown<br>categories. It has to be distinct from the values used to encode any of<br>the categories in `fit`. If set to np.nan, the `dtype` parameter must<br>be a float dtype.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoded_missing_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=encoded_missing_value,-int%20or%20np.nan%2C%20default%3Dnp.nan">
            encoded_missing_value
            <span class="param-doc-description">encoded_missing_value: int or np.nan, default=np.nan<br><br>Encoded value of missing categories. If set to `np.nan`, then the `dtype`<br>parameter must be a float dtype.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_frequency',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=min_frequency,-int%20or%20float%2C%20default%3DNone">
            min_frequency
            <span class="param-doc-description">min_frequency: int or float, default=None<br><br>Specifies the minimum frequency below which a category will be<br>considered infrequent.<br><br>- If `int`, categories with a smaller cardinality will be considered<br>  infrequent.<br><br>- If `float`, categories with a smaller cardinality than<br>  `min_frequency * n_samples`  will be considered infrequent.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=max_categories,-int%2C%20default%3DNone">
            max_categories
            <span class="param-doc-description">max_categories: int, default=None<br><br>Specifies an upper limit to the number of output categories for each input<br>feature when considering infrequent categories. If there are infrequent<br>categories, `max_categories` includes the category representing the<br>infrequent categories along with the frequent categories. If `None`,<br>there is no limit to the number of output features.<br><br>`max_categories` do **not** take into account missing or unknown<br>categories. Setting `unknown_value` or `encoded_missing_value` to an<br>integer will increase the number of unique integer codes by one each.<br>This can result in up to `max_categories + 2` integer codes.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-560" type="checkbox" ><label for="sk-estimator-id-560" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__elec__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If False, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_mean,-bool%2C%20default%3DTrue">
            with_mean
            <span class="param-doc-description">with_mean: bool, default=True<br><br>If True, center the data before scaling.<br>This does not work (and will raise an exception) when attempted on<br>sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_std,-bool%2C%20default%3DTrue">
            with_std
            <span class="param-doc-description">with_std: bool, default=True<br><br>If True, scale the data to unit variance (or equivalently,<br>unit standard deviation).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-561" type="checkbox" ><label for="sk-estimator-id-561" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>bsmt_fin</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__bsmt_fin__"><pre>[&#x27;BsmtFinType1&#x27;, &#x27;BsmtFinType2&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-562" type="checkbox" ><label for="sk-estimator-id-562" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__bsmt_fin__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">&#x27;MISSING&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-563" type="checkbox" ><label for="sk-estimator-id-563" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>to_upper_case</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__bsmt_fin__to_upper__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=func,-callable%2C%20default%3DNone">
            func
            <span class="param-doc-description">func: callable, default=None<br><br>The callable to use for the transformation. This will be passed<br>the same arguments as transform, with args and kwargs forwarded.<br>If func is None, then func will be the identity function.</span>
        </a>
    </td>
            <td class="value">&lt;function to_...001FBB3F077E0&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inverse_func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inverse_func,-callable%2C%20default%3DNone">
            inverse_func
            <span class="param-doc-description">inverse_func: callable, default=None<br><br>The callable to use for the inverse transformation. This will be<br>passed the same arguments as inverse transform, with args and<br>kwargs forwarded. If inverse_func is None, then inverse_func<br>will be the identity function.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=validate,-bool%2C%20default%3DFalse">
            validate
            <span class="param-doc-description">validate: bool, default=False<br><br>Indicate that the input X array should be checked before calling<br>``func``. The possibilities are:<br><br>- If False, there is no input validation.<br>- If True, then X will be converted to a 2-dimensional NumPy array or<br>  sparse matrix. If the conversion is not possible an exception is<br>  raised.<br><br>.. versionchanged:: 0.22<br>   The default of ``validate`` changed from True to False.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('accept_sparse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=accept_sparse,-bool%2C%20default%3DFalse">
            accept_sparse
            <span class="param-doc-description">accept_sparse: bool, default=False<br><br>Indicate that func accepts a sparse matrix as input. If validate is<br>False, this has no effect. Otherwise, if accept_sparse is false,<br>sparse matrix inputs will cause an exception to be raised.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('check_inverse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=check_inverse,-bool%2C%20default%3DTrue">
            check_inverse
            <span class="param-doc-description">check_inverse: bool, default=True<br><br>Whether to check that or ``func`` followed by ``inverse_func`` leads to<br>the original inputs. It can be used for a sanity check, raising a<br>warning when the condition is not fulfilled.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=feature_names_out,-callable%2C%20%27one-to-one%27%20or%20None%2C%20default%3DNone">
            feature_names_out
            <span class="param-doc-description">feature_names_out: callable, 'one-to-one' or None, default=None<br><br>Determines the list of feature names that will be returned by the<br>`get_feature_names_out` method. If it is 'one-to-one', then the output<br>feature names will be equal to the input feature names. If it is a<br>callable, then it must take two positional arguments: this<br>`FunctionTransformer` (`self`) and an array-like of input feature names<br>(`input_features`). It must return an array-like of output feature<br>names. The `get_feature_names_out` method is only defined if<br>`feature_names_out` is not None.<br><br>See ``get_feature_names_out`` for more details.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=kw_args,-dict%2C%20default%3DNone">
            kw_args
            <span class="param-doc-description">kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inv_kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inv_kw_args,-dict%2C%20default%3DNone">
            inv_kw_args
            <span class="param-doc-description">inv_kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to inverse_func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-564" type="checkbox" ><label for="sk-estimator-id-564" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OrdinalEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__bsmt_fin__ordinal_map__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=categories,-%27auto%27%20or%20a%20list%20of%20array-like%2C%20default%3D%27auto%27">
            categories
            <span class="param-doc-description">categories: 'auto' or a list of array-like, default='auto'<br><br>Categories (unique values) per feature:<br><br>- 'auto' : Determine categories automatically from the training data.<br>- list : ``categories[i]`` holds the categories expected in the ith<br>  column. The passed categories should not mix strings and numeric<br>  values, and should be sorted in case of numeric values.<br><br>The used categories can be found in the ``categories_`` attribute.</span>
        </a>
    </td>
            <td class="value">[[&#x27;MISSING&#x27;, &#x27;UNF&#x27;, ...], [&#x27;MISSING&#x27;, &#x27;UNF&#x27;, ...]]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=dtype,-number%20type%2C%20default%3Dnp.float64">
            dtype
            <span class="param-doc-description">dtype: number type, default=np.float64<br><br>Desired dtype of output.</span>
        </a>
    </td>
            <td class="value">&lt;class &#x27;numpy.float64&#x27;&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('handle_unknown',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=handle_unknown,-%7B%27error%27%2C%20%27use_encoded_value%27%7D%2C%20default%3D%27error%27">
            handle_unknown
            <span class="param-doc-description">handle_unknown: {'error', 'use_encoded_value'}, default='error'<br><br>When set to 'error' an error will be raised in case an unknown<br>categorical feature is present during transform. When set to<br>'use_encoded_value', the encoded value of unknown categories will be<br>set to the value given for the parameter `unknown_value`. In<br>:meth:`inverse_transform`, an unknown category will be denoted as None.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">&#x27;error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('unknown_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=unknown_value,-int%20or%20np.nan%2C%20default%3DNone">
            unknown_value
            <span class="param-doc-description">unknown_value: int or np.nan, default=None<br><br>When the parameter handle_unknown is set to 'use_encoded_value', this<br>parameter is required and will set the encoded value of unknown<br>categories. It has to be distinct from the values used to encode any of<br>the categories in `fit`. If set to np.nan, the `dtype` parameter must<br>be a float dtype.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoded_missing_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=encoded_missing_value,-int%20or%20np.nan%2C%20default%3Dnp.nan">
            encoded_missing_value
            <span class="param-doc-description">encoded_missing_value: int or np.nan, default=np.nan<br><br>Encoded value of missing categories. If set to `np.nan`, then the `dtype`<br>parameter must be a float dtype.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_frequency',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=min_frequency,-int%20or%20float%2C%20default%3DNone">
            min_frequency
            <span class="param-doc-description">min_frequency: int or float, default=None<br><br>Specifies the minimum frequency below which a category will be<br>considered infrequent.<br><br>- If `int`, categories with a smaller cardinality will be considered<br>  infrequent.<br><br>- If `float`, categories with a smaller cardinality than<br>  `min_frequency * n_samples`  will be considered infrequent.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=max_categories,-int%2C%20default%3DNone">
            max_categories
            <span class="param-doc-description">max_categories: int, default=None<br><br>Specifies an upper limit to the number of output categories for each input<br>feature when considering infrequent categories. If there are infrequent<br>categories, `max_categories` includes the category representing the<br>infrequent categories along with the frequent categories. If `None`,<br>there is no limit to the number of output features.<br><br>`max_categories` do **not** take into account missing or unknown<br>categories. Setting `unknown_value` or `encoded_missing_value` to an<br>integer will increase the number of unique integer codes by one each.<br>This can result in up to `max_categories + 2` integer codes.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-565" type="checkbox" ><label for="sk-estimator-id-565" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__bsmt_fin__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If False, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_mean,-bool%2C%20default%3DTrue">
            with_mean
            <span class="param-doc-description">with_mean: bool, default=True<br><br>If True, center the data before scaling.<br>This does not work (and will raise an exception) when attempted on<br>sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_std,-bool%2C%20default%3DTrue">
            with_std
            <span class="param-doc-description">with_std: bool, default=True<br><br>If True, scale the data to unit variance (or equivalently,<br>unit standard deviation).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-566" type="checkbox" ><label for="sk-estimator-id-566" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>bsmt_exp</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__bsmt_exp__"><pre>[&#x27;BsmtExposure&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-567" type="checkbox" ><label for="sk-estimator-id-567" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__bsmt_exp__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">&#x27;MISSING&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-568" type="checkbox" ><label for="sk-estimator-id-568" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>to_upper_case</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__bsmt_exp__to_upper__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=func,-callable%2C%20default%3DNone">
            func
            <span class="param-doc-description">func: callable, default=None<br><br>The callable to use for the transformation. This will be passed<br>the same arguments as transform, with args and kwargs forwarded.<br>If func is None, then func will be the identity function.</span>
        </a>
    </td>
            <td class="value">&lt;function to_...001FBB3F077E0&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inverse_func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inverse_func,-callable%2C%20default%3DNone">
            inverse_func
            <span class="param-doc-description">inverse_func: callable, default=None<br><br>The callable to use for the inverse transformation. This will be<br>passed the same arguments as inverse transform, with args and<br>kwargs forwarded. If inverse_func is None, then inverse_func<br>will be the identity function.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=validate,-bool%2C%20default%3DFalse">
            validate
            <span class="param-doc-description">validate: bool, default=False<br><br>Indicate that the input X array should be checked before calling<br>``func``. The possibilities are:<br><br>- If False, there is no input validation.<br>- If True, then X will be converted to a 2-dimensional NumPy array or<br>  sparse matrix. If the conversion is not possible an exception is<br>  raised.<br><br>.. versionchanged:: 0.22<br>   The default of ``validate`` changed from True to False.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('accept_sparse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=accept_sparse,-bool%2C%20default%3DFalse">
            accept_sparse
            <span class="param-doc-description">accept_sparse: bool, default=False<br><br>Indicate that func accepts a sparse matrix as input. If validate is<br>False, this has no effect. Otherwise, if accept_sparse is false,<br>sparse matrix inputs will cause an exception to be raised.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('check_inverse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=check_inverse,-bool%2C%20default%3DTrue">
            check_inverse
            <span class="param-doc-description">check_inverse: bool, default=True<br><br>Whether to check that or ``func`` followed by ``inverse_func`` leads to<br>the original inputs. It can be used for a sanity check, raising a<br>warning when the condition is not fulfilled.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=feature_names_out,-callable%2C%20%27one-to-one%27%20or%20None%2C%20default%3DNone">
            feature_names_out
            <span class="param-doc-description">feature_names_out: callable, 'one-to-one' or None, default=None<br><br>Determines the list of feature names that will be returned by the<br>`get_feature_names_out` method. If it is 'one-to-one', then the output<br>feature names will be equal to the input feature names. If it is a<br>callable, then it must take two positional arguments: this<br>`FunctionTransformer` (`self`) and an array-like of input feature names<br>(`input_features`). It must return an array-like of output feature<br>names. The `get_feature_names_out` method is only defined if<br>`feature_names_out` is not None.<br><br>See ``get_feature_names_out`` for more details.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=kw_args,-dict%2C%20default%3DNone">
            kw_args
            <span class="param-doc-description">kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inv_kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inv_kw_args,-dict%2C%20default%3DNone">
            inv_kw_args
            <span class="param-doc-description">inv_kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to inverse_func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-569" type="checkbox" ><label for="sk-estimator-id-569" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OrdinalEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__bsmt_exp__ordinal_map__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=categories,-%27auto%27%20or%20a%20list%20of%20array-like%2C%20default%3D%27auto%27">
            categories
            <span class="param-doc-description">categories: 'auto' or a list of array-like, default='auto'<br><br>Categories (unique values) per feature:<br><br>- 'auto' : Determine categories automatically from the training data.<br>- list : ``categories[i]`` holds the categories expected in the ith<br>  column. The passed categories should not mix strings and numeric<br>  values, and should be sorted in case of numeric values.<br><br>The used categories can be found in the ``categories_`` attribute.</span>
        </a>
    </td>
            <td class="value">[[&#x27;MISSING&#x27;, &#x27;NO&#x27;, ...]]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=dtype,-number%20type%2C%20default%3Dnp.float64">
            dtype
            <span class="param-doc-description">dtype: number type, default=np.float64<br><br>Desired dtype of output.</span>
        </a>
    </td>
            <td class="value">&lt;class &#x27;numpy.float64&#x27;&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('handle_unknown',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=handle_unknown,-%7B%27error%27%2C%20%27use_encoded_value%27%7D%2C%20default%3D%27error%27">
            handle_unknown
            <span class="param-doc-description">handle_unknown: {'error', 'use_encoded_value'}, default='error'<br><br>When set to 'error' an error will be raised in case an unknown<br>categorical feature is present during transform. When set to<br>'use_encoded_value', the encoded value of unknown categories will be<br>set to the value given for the parameter `unknown_value`. In<br>:meth:`inverse_transform`, an unknown category will be denoted as None.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">&#x27;error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('unknown_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=unknown_value,-int%20or%20np.nan%2C%20default%3DNone">
            unknown_value
            <span class="param-doc-description">unknown_value: int or np.nan, default=None<br><br>When the parameter handle_unknown is set to 'use_encoded_value', this<br>parameter is required and will set the encoded value of unknown<br>categories. It has to be distinct from the values used to encode any of<br>the categories in `fit`. If set to np.nan, the `dtype` parameter must<br>be a float dtype.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoded_missing_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=encoded_missing_value,-int%20or%20np.nan%2C%20default%3Dnp.nan">
            encoded_missing_value
            <span class="param-doc-description">encoded_missing_value: int or np.nan, default=np.nan<br><br>Encoded value of missing categories. If set to `np.nan`, then the `dtype`<br>parameter must be a float dtype.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_frequency',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=min_frequency,-int%20or%20float%2C%20default%3DNone">
            min_frequency
            <span class="param-doc-description">min_frequency: int or float, default=None<br><br>Specifies the minimum frequency below which a category will be<br>considered infrequent.<br><br>- If `int`, categories with a smaller cardinality will be considered<br>  infrequent.<br><br>- If `float`, categories with a smaller cardinality than<br>  `min_frequency * n_samples`  will be considered infrequent.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=max_categories,-int%2C%20default%3DNone">
            max_categories
            <span class="param-doc-description">max_categories: int, default=None<br><br>Specifies an upper limit to the number of output categories for each input<br>feature when considering infrequent categories. If there are infrequent<br>categories, `max_categories` includes the category representing the<br>infrequent categories along with the frequent categories. If `None`,<br>there is no limit to the number of output features.<br><br>`max_categories` do **not** take into account missing or unknown<br>categories. Setting `unknown_value` or `encoded_missing_value` to an<br>integer will increase the number of unique integer codes by one each.<br>This can result in up to `max_categories + 2` integer codes.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-570" type="checkbox" ><label for="sk-estimator-id-570" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__bsmt_exp__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If False, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_mean,-bool%2C%20default%3DTrue">
            with_mean
            <span class="param-doc-description">with_mean: bool, default=True<br><br>If True, center the data before scaling.<br>This does not work (and will raise an exception) when attempted on<br>sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_std,-bool%2C%20default%3DTrue">
            with_std
            <span class="param-doc-description">with_std: bool, default=True<br><br>If True, scale the data to unit variance (or equivalently,<br>unit standard deviation).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-571" type="checkbox" ><label for="sk-estimator-id-571" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>shape</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__shape__"><pre>[&#x27;LotShape&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-572" type="checkbox" ><label for="sk-estimator-id-572" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__shape__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">&#x27;MISSING&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-573" type="checkbox" ><label for="sk-estimator-id-573" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>to_upper_case</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__shape__to_upper__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=func,-callable%2C%20default%3DNone">
            func
            <span class="param-doc-description">func: callable, default=None<br><br>The callable to use for the transformation. This will be passed<br>the same arguments as transform, with args and kwargs forwarded.<br>If func is None, then func will be the identity function.</span>
        </a>
    </td>
            <td class="value">&lt;function to_...001FBB3F077E0&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inverse_func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inverse_func,-callable%2C%20default%3DNone">
            inverse_func
            <span class="param-doc-description">inverse_func: callable, default=None<br><br>The callable to use for the inverse transformation. This will be<br>passed the same arguments as inverse transform, with args and<br>kwargs forwarded. If inverse_func is None, then inverse_func<br>will be the identity function.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=validate,-bool%2C%20default%3DFalse">
            validate
            <span class="param-doc-description">validate: bool, default=False<br><br>Indicate that the input X array should be checked before calling<br>``func``. The possibilities are:<br><br>- If False, there is no input validation.<br>- If True, then X will be converted to a 2-dimensional NumPy array or<br>  sparse matrix. If the conversion is not possible an exception is<br>  raised.<br><br>.. versionchanged:: 0.22<br>   The default of ``validate`` changed from True to False.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('accept_sparse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=accept_sparse,-bool%2C%20default%3DFalse">
            accept_sparse
            <span class="param-doc-description">accept_sparse: bool, default=False<br><br>Indicate that func accepts a sparse matrix as input. If validate is<br>False, this has no effect. Otherwise, if accept_sparse is false,<br>sparse matrix inputs will cause an exception to be raised.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('check_inverse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=check_inverse,-bool%2C%20default%3DTrue">
            check_inverse
            <span class="param-doc-description">check_inverse: bool, default=True<br><br>Whether to check that or ``func`` followed by ``inverse_func`` leads to<br>the original inputs. It can be used for a sanity check, raising a<br>warning when the condition is not fulfilled.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=feature_names_out,-callable%2C%20%27one-to-one%27%20or%20None%2C%20default%3DNone">
            feature_names_out
            <span class="param-doc-description">feature_names_out: callable, 'one-to-one' or None, default=None<br><br>Determines the list of feature names that will be returned by the<br>`get_feature_names_out` method. If it is 'one-to-one', then the output<br>feature names will be equal to the input feature names. If it is a<br>callable, then it must take two positional arguments: this<br>`FunctionTransformer` (`self`) and an array-like of input feature names<br>(`input_features`). It must return an array-like of output feature<br>names. The `get_feature_names_out` method is only defined if<br>`feature_names_out` is not None.<br><br>See ``get_feature_names_out`` for more details.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=kw_args,-dict%2C%20default%3DNone">
            kw_args
            <span class="param-doc-description">kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inv_kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inv_kw_args,-dict%2C%20default%3DNone">
            inv_kw_args
            <span class="param-doc-description">inv_kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to inverse_func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-574" type="checkbox" ><label for="sk-estimator-id-574" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OrdinalEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__shape__ordinal_map__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=categories,-%27auto%27%20or%20a%20list%20of%20array-like%2C%20default%3D%27auto%27">
            categories
            <span class="param-doc-description">categories: 'auto' or a list of array-like, default='auto'<br><br>Categories (unique values) per feature:<br><br>- 'auto' : Determine categories automatically from the training data.<br>- list : ``categories[i]`` holds the categories expected in the ith<br>  column. The passed categories should not mix strings and numeric<br>  values, and should be sorted in case of numeric values.<br><br>The used categories can be found in the ``categories_`` attribute.</span>
        </a>
    </td>
            <td class="value">[[&#x27;MISSING&#x27;, &#x27;IR3&#x27;, ...]]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=dtype,-number%20type%2C%20default%3Dnp.float64">
            dtype
            <span class="param-doc-description">dtype: number type, default=np.float64<br><br>Desired dtype of output.</span>
        </a>
    </td>
            <td class="value">&lt;class &#x27;numpy.float64&#x27;&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('handle_unknown',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=handle_unknown,-%7B%27error%27%2C%20%27use_encoded_value%27%7D%2C%20default%3D%27error%27">
            handle_unknown
            <span class="param-doc-description">handle_unknown: {'error', 'use_encoded_value'}, default='error'<br><br>When set to 'error' an error will be raised in case an unknown<br>categorical feature is present during transform. When set to<br>'use_encoded_value', the encoded value of unknown categories will be<br>set to the value given for the parameter `unknown_value`. In<br>:meth:`inverse_transform`, an unknown category will be denoted as None.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">&#x27;error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('unknown_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=unknown_value,-int%20or%20np.nan%2C%20default%3DNone">
            unknown_value
            <span class="param-doc-description">unknown_value: int or np.nan, default=None<br><br>When the parameter handle_unknown is set to 'use_encoded_value', this<br>parameter is required and will set the encoded value of unknown<br>categories. It has to be distinct from the values used to encode any of<br>the categories in `fit`. If set to np.nan, the `dtype` parameter must<br>be a float dtype.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoded_missing_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=encoded_missing_value,-int%20or%20np.nan%2C%20default%3Dnp.nan">
            encoded_missing_value
            <span class="param-doc-description">encoded_missing_value: int or np.nan, default=np.nan<br><br>Encoded value of missing categories. If set to `np.nan`, then the `dtype`<br>parameter must be a float dtype.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_frequency',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=min_frequency,-int%20or%20float%2C%20default%3DNone">
            min_frequency
            <span class="param-doc-description">min_frequency: int or float, default=None<br><br>Specifies the minimum frequency below which a category will be<br>considered infrequent.<br><br>- If `int`, categories with a smaller cardinality will be considered<br>  infrequent.<br><br>- If `float`, categories with a smaller cardinality than<br>  `min_frequency * n_samples`  will be considered infrequent.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=max_categories,-int%2C%20default%3DNone">
            max_categories
            <span class="param-doc-description">max_categories: int, default=None<br><br>Specifies an upper limit to the number of output categories for each input<br>feature when considering infrequent categories. If there are infrequent<br>categories, `max_categories` includes the category representing the<br>infrequent categories along with the frequent categories. If `None`,<br>there is no limit to the number of output features.<br><br>`max_categories` do **not** take into account missing or unknown<br>categories. Setting `unknown_value` or `encoded_missing_value` to an<br>integer will increase the number of unique integer codes by one each.<br>This can result in up to `max_categories + 2` integer codes.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-575" type="checkbox" ><label for="sk-estimator-id-575" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__shape__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If False, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_mean,-bool%2C%20default%3DTrue">
            with_mean
            <span class="param-doc-description">with_mean: bool, default=True<br><br>If True, center the data before scaling.<br>This does not work (and will raise an exception) when attempted on<br>sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_std,-bool%2C%20default%3DTrue">
            with_std
            <span class="param-doc-description">with_std: bool, default=True<br><br>If True, scale the data to unit variance (or equivalently,<br>unit standard deviation).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-576" type="checkbox" ><label for="sk-estimator-id-576" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>slope</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__slope__"><pre>[&#x27;LandSlope&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-577" type="checkbox" ><label for="sk-estimator-id-577" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__slope__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">&#x27;MISSING&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-578" type="checkbox" ><label for="sk-estimator-id-578" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>to_upper_case</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__slope__to_upper__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=func,-callable%2C%20default%3DNone">
            func
            <span class="param-doc-description">func: callable, default=None<br><br>The callable to use for the transformation. This will be passed<br>the same arguments as transform, with args and kwargs forwarded.<br>If func is None, then func will be the identity function.</span>
        </a>
    </td>
            <td class="value">&lt;function to_...001FBB3F077E0&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inverse_func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inverse_func,-callable%2C%20default%3DNone">
            inverse_func
            <span class="param-doc-description">inverse_func: callable, default=None<br><br>The callable to use for the inverse transformation. This will be<br>passed the same arguments as inverse transform, with args and<br>kwargs forwarded. If inverse_func is None, then inverse_func<br>will be the identity function.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=validate,-bool%2C%20default%3DFalse">
            validate
            <span class="param-doc-description">validate: bool, default=False<br><br>Indicate that the input X array should be checked before calling<br>``func``. The possibilities are:<br><br>- If False, there is no input validation.<br>- If True, then X will be converted to a 2-dimensional NumPy array or<br>  sparse matrix. If the conversion is not possible an exception is<br>  raised.<br><br>.. versionchanged:: 0.22<br>   The default of ``validate`` changed from True to False.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('accept_sparse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=accept_sparse,-bool%2C%20default%3DFalse">
            accept_sparse
            <span class="param-doc-description">accept_sparse: bool, default=False<br><br>Indicate that func accepts a sparse matrix as input. If validate is<br>False, this has no effect. Otherwise, if accept_sparse is false,<br>sparse matrix inputs will cause an exception to be raised.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('check_inverse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=check_inverse,-bool%2C%20default%3DTrue">
            check_inverse
            <span class="param-doc-description">check_inverse: bool, default=True<br><br>Whether to check that or ``func`` followed by ``inverse_func`` leads to<br>the original inputs. It can be used for a sanity check, raising a<br>warning when the condition is not fulfilled.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=feature_names_out,-callable%2C%20%27one-to-one%27%20or%20None%2C%20default%3DNone">
            feature_names_out
            <span class="param-doc-description">feature_names_out: callable, 'one-to-one' or None, default=None<br><br>Determines the list of feature names that will be returned by the<br>`get_feature_names_out` method. If it is 'one-to-one', then the output<br>feature names will be equal to the input feature names. If it is a<br>callable, then it must take two positional arguments: this<br>`FunctionTransformer` (`self`) and an array-like of input feature names<br>(`input_features`). It must return an array-like of output feature<br>names. The `get_feature_names_out` method is only defined if<br>`feature_names_out` is not None.<br><br>See ``get_feature_names_out`` for more details.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=kw_args,-dict%2C%20default%3DNone">
            kw_args
            <span class="param-doc-description">kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inv_kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inv_kw_args,-dict%2C%20default%3DNone">
            inv_kw_args
            <span class="param-doc-description">inv_kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to inverse_func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-579" type="checkbox" ><label for="sk-estimator-id-579" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OrdinalEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__slope__ordinal_map__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=categories,-%27auto%27%20or%20a%20list%20of%20array-like%2C%20default%3D%27auto%27">
            categories
            <span class="param-doc-description">categories: 'auto' or a list of array-like, default='auto'<br><br>Categories (unique values) per feature:<br><br>- 'auto' : Determine categories automatically from the training data.<br>- list : ``categories[i]`` holds the categories expected in the ith<br>  column. The passed categories should not mix strings and numeric<br>  values, and should be sorted in case of numeric values.<br><br>The used categories can be found in the ``categories_`` attribute.</span>
        </a>
    </td>
            <td class="value">[[&#x27;MISSING&#x27;, &#x27;SEV&#x27;, ...]]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=dtype,-number%20type%2C%20default%3Dnp.float64">
            dtype
            <span class="param-doc-description">dtype: number type, default=np.float64<br><br>Desired dtype of output.</span>
        </a>
    </td>
            <td class="value">&lt;class &#x27;numpy.float64&#x27;&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('handle_unknown',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=handle_unknown,-%7B%27error%27%2C%20%27use_encoded_value%27%7D%2C%20default%3D%27error%27">
            handle_unknown
            <span class="param-doc-description">handle_unknown: {'error', 'use_encoded_value'}, default='error'<br><br>When set to 'error' an error will be raised in case an unknown<br>categorical feature is present during transform. When set to<br>'use_encoded_value', the encoded value of unknown categories will be<br>set to the value given for the parameter `unknown_value`. In<br>:meth:`inverse_transform`, an unknown category will be denoted as None.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">&#x27;error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('unknown_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=unknown_value,-int%20or%20np.nan%2C%20default%3DNone">
            unknown_value
            <span class="param-doc-description">unknown_value: int or np.nan, default=None<br><br>When the parameter handle_unknown is set to 'use_encoded_value', this<br>parameter is required and will set the encoded value of unknown<br>categories. It has to be distinct from the values used to encode any of<br>the categories in `fit`. If set to np.nan, the `dtype` parameter must<br>be a float dtype.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoded_missing_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=encoded_missing_value,-int%20or%20np.nan%2C%20default%3Dnp.nan">
            encoded_missing_value
            <span class="param-doc-description">encoded_missing_value: int or np.nan, default=np.nan<br><br>Encoded value of missing categories. If set to `np.nan`, then the `dtype`<br>parameter must be a float dtype.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_frequency',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=min_frequency,-int%20or%20float%2C%20default%3DNone">
            min_frequency
            <span class="param-doc-description">min_frequency: int or float, default=None<br><br>Specifies the minimum frequency below which a category will be<br>considered infrequent.<br><br>- If `int`, categories with a smaller cardinality will be considered<br>  infrequent.<br><br>- If `float`, categories with a smaller cardinality than<br>  `min_frequency * n_samples`  will be considered infrequent.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=max_categories,-int%2C%20default%3DNone">
            max_categories
            <span class="param-doc-description">max_categories: int, default=None<br><br>Specifies an upper limit to the number of output categories for each input<br>feature when considering infrequent categories. If there are infrequent<br>categories, `max_categories` includes the category representing the<br>infrequent categories along with the frequent categories. If `None`,<br>there is no limit to the number of output features.<br><br>`max_categories` do **not** take into account missing or unknown<br>categories. Setting `unknown_value` or `encoded_missing_value` to an<br>integer will increase the number of unique integer codes by one each.<br>This can result in up to `max_categories + 2` integer codes.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-580" type="checkbox" ><label for="sk-estimator-id-580" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__slope__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If False, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_mean,-bool%2C%20default%3DTrue">
            with_mean
            <span class="param-doc-description">with_mean: bool, default=True<br><br>If True, center the data before scaling.<br>This does not work (and will raise an exception) when attempted on<br>sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_std,-bool%2C%20default%3DTrue">
            with_std
            <span class="param-doc-description">with_std: bool, default=True<br><br>If True, scale the data to unit variance (or equivalently,<br>unit standard deviation).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-581" type="checkbox" ><label for="sk-estimator-id-581" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>util</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__util__"><pre>[&#x27;Utilities&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-582" type="checkbox" ><label for="sk-estimator-id-582" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__util__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">&#x27;MISSING&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-583" type="checkbox" ><label for="sk-estimator-id-583" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>to_upper_case</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__util__to_upper__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=func,-callable%2C%20default%3DNone">
            func
            <span class="param-doc-description">func: callable, default=None<br><br>The callable to use for the transformation. This will be passed<br>the same arguments as transform, with args and kwargs forwarded.<br>If func is None, then func will be the identity function.</span>
        </a>
    </td>
            <td class="value">&lt;function to_...001FBB3F077E0&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inverse_func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inverse_func,-callable%2C%20default%3DNone">
            inverse_func
            <span class="param-doc-description">inverse_func: callable, default=None<br><br>The callable to use for the inverse transformation. This will be<br>passed the same arguments as inverse transform, with args and<br>kwargs forwarded. If inverse_func is None, then inverse_func<br>will be the identity function.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=validate,-bool%2C%20default%3DFalse">
            validate
            <span class="param-doc-description">validate: bool, default=False<br><br>Indicate that the input X array should be checked before calling<br>``func``. The possibilities are:<br><br>- If False, there is no input validation.<br>- If True, then X will be converted to a 2-dimensional NumPy array or<br>  sparse matrix. If the conversion is not possible an exception is<br>  raised.<br><br>.. versionchanged:: 0.22<br>   The default of ``validate`` changed from True to False.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('accept_sparse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=accept_sparse,-bool%2C%20default%3DFalse">
            accept_sparse
            <span class="param-doc-description">accept_sparse: bool, default=False<br><br>Indicate that func accepts a sparse matrix as input. If validate is<br>False, this has no effect. Otherwise, if accept_sparse is false,<br>sparse matrix inputs will cause an exception to be raised.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('check_inverse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=check_inverse,-bool%2C%20default%3DTrue">
            check_inverse
            <span class="param-doc-description">check_inverse: bool, default=True<br><br>Whether to check that or ``func`` followed by ``inverse_func`` leads to<br>the original inputs. It can be used for a sanity check, raising a<br>warning when the condition is not fulfilled.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=feature_names_out,-callable%2C%20%27one-to-one%27%20or%20None%2C%20default%3DNone">
            feature_names_out
            <span class="param-doc-description">feature_names_out: callable, 'one-to-one' or None, default=None<br><br>Determines the list of feature names that will be returned by the<br>`get_feature_names_out` method. If it is 'one-to-one', then the output<br>feature names will be equal to the input feature names. If it is a<br>callable, then it must take two positional arguments: this<br>`FunctionTransformer` (`self`) and an array-like of input feature names<br>(`input_features`). It must return an array-like of output feature<br>names. The `get_feature_names_out` method is only defined if<br>`feature_names_out` is not None.<br><br>See ``get_feature_names_out`` for more details.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=kw_args,-dict%2C%20default%3DNone">
            kw_args
            <span class="param-doc-description">kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inv_kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inv_kw_args,-dict%2C%20default%3DNone">
            inv_kw_args
            <span class="param-doc-description">inv_kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to inverse_func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-584" type="checkbox" ><label for="sk-estimator-id-584" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OrdinalEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__util__ordinal_map__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=categories,-%27auto%27%20or%20a%20list%20of%20array-like%2C%20default%3D%27auto%27">
            categories
            <span class="param-doc-description">categories: 'auto' or a list of array-like, default='auto'<br><br>Categories (unique values) per feature:<br><br>- 'auto' : Determine categories automatically from the training data.<br>- list : ``categories[i]`` holds the categories expected in the ith<br>  column. The passed categories should not mix strings and numeric<br>  values, and should be sorted in case of numeric values.<br><br>The used categories can be found in the ``categories_`` attribute.</span>
        </a>
    </td>
            <td class="value">[[&#x27;MISSING&#x27;, &#x27;NOSEWA&#x27;, ...]]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=dtype,-number%20type%2C%20default%3Dnp.float64">
            dtype
            <span class="param-doc-description">dtype: number type, default=np.float64<br><br>Desired dtype of output.</span>
        </a>
    </td>
            <td class="value">&lt;class &#x27;numpy.float64&#x27;&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('handle_unknown',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=handle_unknown,-%7B%27error%27%2C%20%27use_encoded_value%27%7D%2C%20default%3D%27error%27">
            handle_unknown
            <span class="param-doc-description">handle_unknown: {'error', 'use_encoded_value'}, default='error'<br><br>When set to 'error' an error will be raised in case an unknown<br>categorical feature is present during transform. When set to<br>'use_encoded_value', the encoded value of unknown categories will be<br>set to the value given for the parameter `unknown_value`. In<br>:meth:`inverse_transform`, an unknown category will be denoted as None.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">&#x27;error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('unknown_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=unknown_value,-int%20or%20np.nan%2C%20default%3DNone">
            unknown_value
            <span class="param-doc-description">unknown_value: int or np.nan, default=None<br><br>When the parameter handle_unknown is set to 'use_encoded_value', this<br>parameter is required and will set the encoded value of unknown<br>categories. It has to be distinct from the values used to encode any of<br>the categories in `fit`. If set to np.nan, the `dtype` parameter must<br>be a float dtype.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoded_missing_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=encoded_missing_value,-int%20or%20np.nan%2C%20default%3Dnp.nan">
            encoded_missing_value
            <span class="param-doc-description">encoded_missing_value: int or np.nan, default=np.nan<br><br>Encoded value of missing categories. If set to `np.nan`, then the `dtype`<br>parameter must be a float dtype.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_frequency',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=min_frequency,-int%20or%20float%2C%20default%3DNone">
            min_frequency
            <span class="param-doc-description">min_frequency: int or float, default=None<br><br>Specifies the minimum frequency below which a category will be<br>considered infrequent.<br><br>- If `int`, categories with a smaller cardinality will be considered<br>  infrequent.<br><br>- If `float`, categories with a smaller cardinality than<br>  `min_frequency * n_samples`  will be considered infrequent.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=max_categories,-int%2C%20default%3DNone">
            max_categories
            <span class="param-doc-description">max_categories: int, default=None<br><br>Specifies an upper limit to the number of output categories for each input<br>feature when considering infrequent categories. If there are infrequent<br>categories, `max_categories` includes the category representing the<br>infrequent categories along with the frequent categories. If `None`,<br>there is no limit to the number of output features.<br><br>`max_categories` do **not** take into account missing or unknown<br>categories. Setting `unknown_value` or `encoded_missing_value` to an<br>integer will increase the number of unique integer codes by one each.<br>This can result in up to `max_categories + 2` integer codes.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-585" type="checkbox" ><label for="sk-estimator-id-585" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__util__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If False, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_mean,-bool%2C%20default%3DTrue">
            with_mean
            <span class="param-doc-description">with_mean: bool, default=True<br><br>If True, center the data before scaling.<br>This does not work (and will raise an exception) when attempted on<br>sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_std,-bool%2C%20default%3DTrue">
            with_std
            <span class="param-doc-description">with_std: bool, default=True<br><br>If True, scale the data to unit variance (or equivalently,<br>unit standard deviation).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-586" type="checkbox" ><label for="sk-estimator-id-586" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>func</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__func__"><pre>[&#x27;Functional&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-587" type="checkbox" ><label for="sk-estimator-id-587" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__func__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">&#x27;MISSING&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-588" type="checkbox" ><label for="sk-estimator-id-588" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>to_upper_case</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__func__to_upper__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=func,-callable%2C%20default%3DNone">
            func
            <span class="param-doc-description">func: callable, default=None<br><br>The callable to use for the transformation. This will be passed<br>the same arguments as transform, with args and kwargs forwarded.<br>If func is None, then func will be the identity function.</span>
        </a>
    </td>
            <td class="value">&lt;function to_...001FBB3F077E0&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inverse_func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inverse_func,-callable%2C%20default%3DNone">
            inverse_func
            <span class="param-doc-description">inverse_func: callable, default=None<br><br>The callable to use for the inverse transformation. This will be<br>passed the same arguments as inverse transform, with args and<br>kwargs forwarded. If inverse_func is None, then inverse_func<br>will be the identity function.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=validate,-bool%2C%20default%3DFalse">
            validate
            <span class="param-doc-description">validate: bool, default=False<br><br>Indicate that the input X array should be checked before calling<br>``func``. The possibilities are:<br><br>- If False, there is no input validation.<br>- If True, then X will be converted to a 2-dimensional NumPy array or<br>  sparse matrix. If the conversion is not possible an exception is<br>  raised.<br><br>.. versionchanged:: 0.22<br>   The default of ``validate`` changed from True to False.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('accept_sparse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=accept_sparse,-bool%2C%20default%3DFalse">
            accept_sparse
            <span class="param-doc-description">accept_sparse: bool, default=False<br><br>Indicate that func accepts a sparse matrix as input. If validate is<br>False, this has no effect. Otherwise, if accept_sparse is false,<br>sparse matrix inputs will cause an exception to be raised.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('check_inverse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=check_inverse,-bool%2C%20default%3DTrue">
            check_inverse
            <span class="param-doc-description">check_inverse: bool, default=True<br><br>Whether to check that or ``func`` followed by ``inverse_func`` leads to<br>the original inputs. It can be used for a sanity check, raising a<br>warning when the condition is not fulfilled.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=feature_names_out,-callable%2C%20%27one-to-one%27%20or%20None%2C%20default%3DNone">
            feature_names_out
            <span class="param-doc-description">feature_names_out: callable, 'one-to-one' or None, default=None<br><br>Determines the list of feature names that will be returned by the<br>`get_feature_names_out` method. If it is 'one-to-one', then the output<br>feature names will be equal to the input feature names. If it is a<br>callable, then it must take two positional arguments: this<br>`FunctionTransformer` (`self`) and an array-like of input feature names<br>(`input_features`). It must return an array-like of output feature<br>names. The `get_feature_names_out` method is only defined if<br>`feature_names_out` is not None.<br><br>See ``get_feature_names_out`` for more details.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=kw_args,-dict%2C%20default%3DNone">
            kw_args
            <span class="param-doc-description">kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inv_kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inv_kw_args,-dict%2C%20default%3DNone">
            inv_kw_args
            <span class="param-doc-description">inv_kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to inverse_func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-589" type="checkbox" ><label for="sk-estimator-id-589" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OrdinalEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__func__ordinal_map__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=categories,-%27auto%27%20or%20a%20list%20of%20array-like%2C%20default%3D%27auto%27">
            categories
            <span class="param-doc-description">categories: 'auto' or a list of array-like, default='auto'<br><br>Categories (unique values) per feature:<br><br>- 'auto' : Determine categories automatically from the training data.<br>- list : ``categories[i]`` holds the categories expected in the ith<br>  column. The passed categories should not mix strings and numeric<br>  values, and should be sorted in case of numeric values.<br><br>The used categories can be found in the ``categories_`` attribute.</span>
        </a>
    </td>
            <td class="value">[[&#x27;MISSING&#x27;, &#x27;SEV&#x27;, ...]]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=dtype,-number%20type%2C%20default%3Dnp.float64">
            dtype
            <span class="param-doc-description">dtype: number type, default=np.float64<br><br>Desired dtype of output.</span>
        </a>
    </td>
            <td class="value">&lt;class &#x27;numpy.float64&#x27;&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('handle_unknown',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=handle_unknown,-%7B%27error%27%2C%20%27use_encoded_value%27%7D%2C%20default%3D%27error%27">
            handle_unknown
            <span class="param-doc-description">handle_unknown: {'error', 'use_encoded_value'}, default='error'<br><br>When set to 'error' an error will be raised in case an unknown<br>categorical feature is present during transform. When set to<br>'use_encoded_value', the encoded value of unknown categories will be<br>set to the value given for the parameter `unknown_value`. In<br>:meth:`inverse_transform`, an unknown category will be denoted as None.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">&#x27;error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('unknown_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=unknown_value,-int%20or%20np.nan%2C%20default%3DNone">
            unknown_value
            <span class="param-doc-description">unknown_value: int or np.nan, default=None<br><br>When the parameter handle_unknown is set to 'use_encoded_value', this<br>parameter is required and will set the encoded value of unknown<br>categories. It has to be distinct from the values used to encode any of<br>the categories in `fit`. If set to np.nan, the `dtype` parameter must<br>be a float dtype.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoded_missing_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=encoded_missing_value,-int%20or%20np.nan%2C%20default%3Dnp.nan">
            encoded_missing_value
            <span class="param-doc-description">encoded_missing_value: int or np.nan, default=np.nan<br><br>Encoded value of missing categories. If set to `np.nan`, then the `dtype`<br>parameter must be a float dtype.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_frequency',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=min_frequency,-int%20or%20float%2C%20default%3DNone">
            min_frequency
            <span class="param-doc-description">min_frequency: int or float, default=None<br><br>Specifies the minimum frequency below which a category will be<br>considered infrequent.<br><br>- If `int`, categories with a smaller cardinality will be considered<br>  infrequent.<br><br>- If `float`, categories with a smaller cardinality than<br>  `min_frequency * n_samples`  will be considered infrequent.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=max_categories,-int%2C%20default%3DNone">
            max_categories
            <span class="param-doc-description">max_categories: int, default=None<br><br>Specifies an upper limit to the number of output categories for each input<br>feature when considering infrequent categories. If there are infrequent<br>categories, `max_categories` includes the category representing the<br>infrequent categories along with the frequent categories. If `None`,<br>there is no limit to the number of output features.<br><br>`max_categories` do **not** take into account missing or unknown<br>categories. Setting `unknown_value` or `encoded_missing_value` to an<br>integer will increase the number of unique integer codes by one each.<br>This can result in up to `max_categories + 2` integer codes.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-590" type="checkbox" ><label for="sk-estimator-id-590" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__func__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If False, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_mean,-bool%2C%20default%3DTrue">
            with_mean
            <span class="param-doc-description">with_mean: bool, default=True<br><br>If True, center the data before scaling.<br>This does not work (and will raise an exception) when attempted on<br>sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_std,-bool%2C%20default%3DTrue">
            with_std
            <span class="param-doc-description">with_std: bool, default=True<br><br>If True, scale the data to unit variance (or equivalently,<br>unit standard deviation).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-591" type="checkbox" ><label for="sk-estimator-id-591" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>fence</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__fence__"><pre>[&#x27;Fence&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-592" type="checkbox" ><label for="sk-estimator-id-592" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__fence__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">&#x27;MISSING&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-593" type="checkbox" ><label for="sk-estimator-id-593" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>to_upper_case</div><div class="caption">FunctionTransformer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html">?<span>Documentation for FunctionTransformer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__fence__to_upper__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=func,-callable%2C%20default%3DNone">
            func
            <span class="param-doc-description">func: callable, default=None<br><br>The callable to use for the transformation. This will be passed<br>the same arguments as transform, with args and kwargs forwarded.<br>If func is None, then func will be the identity function.</span>
        </a>
    </td>
            <td class="value">&lt;function to_...001FBB3F077E0&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inverse_func',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inverse_func,-callable%2C%20default%3DNone">
            inverse_func
            <span class="param-doc-description">inverse_func: callable, default=None<br><br>The callable to use for the inverse transformation. This will be<br>passed the same arguments as inverse transform, with args and<br>kwargs forwarded. If inverse_func is None, then inverse_func<br>will be the identity function.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('validate',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=validate,-bool%2C%20default%3DFalse">
            validate
            <span class="param-doc-description">validate: bool, default=False<br><br>Indicate that the input X array should be checked before calling<br>``func``. The possibilities are:<br><br>- If False, there is no input validation.<br>- If True, then X will be converted to a 2-dimensional NumPy array or<br>  sparse matrix. If the conversion is not possible an exception is<br>  raised.<br><br>.. versionchanged:: 0.22<br>   The default of ``validate`` changed from True to False.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('accept_sparse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=accept_sparse,-bool%2C%20default%3DFalse">
            accept_sparse
            <span class="param-doc-description">accept_sparse: bool, default=False<br><br>Indicate that func accepts a sparse matrix as input. If validate is<br>False, this has no effect. Otherwise, if accept_sparse is false,<br>sparse matrix inputs will cause an exception to be raised.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('check_inverse',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=check_inverse,-bool%2C%20default%3DTrue">
            check_inverse
            <span class="param-doc-description">check_inverse: bool, default=True<br><br>Whether to check that or ``func`` followed by ``inverse_func`` leads to<br>the original inputs. It can be used for a sanity check, raising a<br>warning when the condition is not fulfilled.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_names_out',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=feature_names_out,-callable%2C%20%27one-to-one%27%20or%20None%2C%20default%3DNone">
            feature_names_out
            <span class="param-doc-description">feature_names_out: callable, 'one-to-one' or None, default=None<br><br>Determines the list of feature names that will be returned by the<br>`get_feature_names_out` method. If it is 'one-to-one', then the output<br>feature names will be equal to the input feature names. If it is a<br>callable, then it must take two positional arguments: this<br>`FunctionTransformer` (`self`) and an array-like of input feature names<br>(`input_features`). It must return an array-like of output feature<br>names. The `get_feature_names_out` method is only defined if<br>`feature_names_out` is not None.<br><br>See ``get_feature_names_out`` for more details.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=kw_args,-dict%2C%20default%3DNone">
            kw_args
            <span class="param-doc-description">kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('inv_kw_args',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.FunctionTransformer.html#:~:text=inv_kw_args,-dict%2C%20default%3DNone">
            inv_kw_args
            <span class="param-doc-description">inv_kw_args: dict, default=None<br><br>Dictionary of additional keyword arguments to pass to inverse_func.<br><br>.. versionadded:: 0.18</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-594" type="checkbox" ><label for="sk-estimator-id-594" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OrdinalEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html">?<span>Documentation for OrdinalEncoder</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__fence__ordinal_map__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=categories,-%27auto%27%20or%20a%20list%20of%20array-like%2C%20default%3D%27auto%27">
            categories
            <span class="param-doc-description">categories: 'auto' or a list of array-like, default='auto'<br><br>Categories (unique values) per feature:<br><br>- 'auto' : Determine categories automatically from the training data.<br>- list : ``categories[i]`` holds the categories expected in the ith<br>  column. The passed categories should not mix strings and numeric<br>  values, and should be sorted in case of numeric values.<br><br>The used categories can be found in the ``categories_`` attribute.</span>
        </a>
    </td>
            <td class="value">[[&#x27;MISSING&#x27;, &#x27;MNWW&#x27;, ...]]</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=dtype,-number%20type%2C%20default%3Dnp.float64">
            dtype
            <span class="param-doc-description">dtype: number type, default=np.float64<br><br>Desired dtype of output.</span>
        </a>
    </td>
            <td class="value">&lt;class &#x27;numpy.float64&#x27;&gt;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('handle_unknown',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=handle_unknown,-%7B%27error%27%2C%20%27use_encoded_value%27%7D%2C%20default%3D%27error%27">
            handle_unknown
            <span class="param-doc-description">handle_unknown: {'error', 'use_encoded_value'}, default='error'<br><br>When set to 'error' an error will be raised in case an unknown<br>categorical feature is present during transform. When set to<br>'use_encoded_value', the encoded value of unknown categories will be<br>set to the value given for the parameter `unknown_value`. In<br>:meth:`inverse_transform`, an unknown category will be denoted as None.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">&#x27;error&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('unknown_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=unknown_value,-int%20or%20np.nan%2C%20default%3DNone">
            unknown_value
            <span class="param-doc-description">unknown_value: int or np.nan, default=None<br><br>When the parameter handle_unknown is set to 'use_encoded_value', this<br>parameter is required and will set the encoded value of unknown<br>categories. It has to be distinct from the values used to encode any of<br>the categories in `fit`. If set to np.nan, the `dtype` parameter must<br>be a float dtype.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoded_missing_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=encoded_missing_value,-int%20or%20np.nan%2C%20default%3Dnp.nan">
            encoded_missing_value
            <span class="param-doc-description">encoded_missing_value: int or np.nan, default=np.nan<br><br>Encoded value of missing categories. If set to `np.nan`, then the `dtype`<br>parameter must be a float dtype.<br><br>.. versionadded:: 1.1</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_frequency',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=min_frequency,-int%20or%20float%2C%20default%3DNone">
            min_frequency
            <span class="param-doc-description">min_frequency: int or float, default=None<br><br>Specifies the minimum frequency below which a category will be<br>considered infrequent.<br><br>- If `int`, categories with a smaller cardinality will be considered<br>  infrequent.<br><br>- If `float`, categories with a smaller cardinality than<br>  `min_frequency * n_samples`  will be considered infrequent.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OrdinalEncoder.html#:~:text=max_categories,-int%2C%20default%3DNone">
            max_categories
            <span class="param-doc-description">max_categories: int, default=None<br><br>Specifies an upper limit to the number of output categories for each input<br>feature when considering infrequent categories. If there are infrequent<br>categories, `max_categories` includes the category representing the<br>infrequent categories along with the frequent categories. If `None`,<br>there is no limit to the number of output features.<br><br>`max_categories` do **not** take into account missing or unknown<br>categories. Setting `unknown_value` or `encoded_missing_value` to an<br>integer will increase the number of unique integer codes by one each.<br>This can result in up to `max_categories + 2` integer codes.<br><br>.. versionadded:: 1.3<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-595" type="checkbox" ><label for="sk-estimator-id-595" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__fence__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If False, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_mean,-bool%2C%20default%3DTrue">
            with_mean
            <span class="param-doc-description">with_mean: bool, default=True<br><br>If True, center the data before scaling.<br>This does not work (and will raise an exception) when attempted on<br>sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_std,-bool%2C%20default%3DTrue">
            with_std
            <span class="param-doc-description">with_std: bool, default=True<br><br>If True, scale the data to unit variance (or equivalently,<br>unit standard deviation).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-596" type="checkbox" ><label for="sk-estimator-id-596" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>cat</div></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__cat__"><pre>[&#x27;RoofMatl&#x27;, &#x27;Neighborhood&#x27;, &#x27;Alley&#x27;, &#x27;BldgType&#x27;, &#x27;LotConfig&#x27;, &#x27;GarageType&#x27;, &#x27;SaleType&#x27;, &#x27;Street&#x27;, &#x27;MSZoning&#x27;, &#x27;Heating&#x27;, &#x27;Exterior1st&#x27;, &#x27;LandContour&#x27;, &#x27;MasVnrType&#x27;, &#x27;Exterior2nd&#x27;, &#x27;SaleCondition&#x27;, &#x27;HouseStyle&#x27;, &#x27;CentralAir&#x27;, &#x27;RoofStyle&#x27;, &#x27;Foundation&#x27;, &#x27;Condition1&#x27;, &#x27;MiscFeature&#x27;, &#x27;Condition2&#x27;]</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-597" type="checkbox" ><label for="sk-estimator-id-597" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>SimpleImputer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html">?<span>Documentation for SimpleImputer</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__cat__imputer__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('missing_values',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=missing_values,-int%2C%20float%2C%20str%2C%20np.nan%2C%20None%20or%20pandas.NA%2C%20default%3Dnp.nan">
            missing_values
            <span class="param-doc-description">missing_values: int, float, str, np.nan, None or pandas.NA, default=np.nan<br><br>The placeholder for the missing values. All occurrences of<br>`missing_values` will be imputed. For pandas' dataframes with<br>nullable integer dtypes with missing values, `missing_values`<br>can be set to either `np.nan` or `pd.NA`.</span>
        </a>
    </td>
            <td class="value">nan</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strategy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=strategy,-str%20or%20Callable%2C%20default%3D%27mean%27">
            strategy
            <span class="param-doc-description">strategy: str or Callable, default='mean'<br><br>The imputation strategy.<br><br>- If "mean", then replace missing values using the mean along<br>  each column. Can only be used with numeric data.<br>- If "median", then replace missing values using the median along<br>  each column. Can only be used with numeric data.<br>- If "most_frequent", then replace missing using the most frequent<br>  value along each column. Can be used with strings or numeric data.<br>  If there is more than one such value, only the smallest is returned.<br>- If "constant", then replace missing values with fill_value. Can be<br>  used with strings or numeric data.<br>- If an instance of Callable, then replace missing values using the<br>  scalar statistic returned by running the callable over a dense 1d<br>  array containing non-missing values of each column.<br><br>.. versionadded:: 0.20<br>   strategy="constant" for fixed value imputation.<br><br>.. versionadded:: 1.5<br>   strategy=callable for custom value imputation.</span>
        </a>
    </td>
            <td class="value">&#x27;constant&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fill_value',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=fill_value,-str%20or%20numerical%20value%2C%20default%3DNone">
            fill_value
            <span class="param-doc-description">fill_value: str or numerical value, default=None<br><br>When strategy == "constant", `fill_value` is used to replace all<br>occurrences of missing_values. For string or object data types,<br>`fill_value` must be a string.<br>If `None`, `fill_value` will be 0 when imputing numerical<br>data and "missing_value" for strings or object data types.</span>
        </a>
    </td>
            <td class="value">&#x27;missing&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If True, a copy of X will be created. If False, imputation will<br>be done in-place whenever possible. Note that, in the following cases,<br>a new copy will always be made, even if `copy=False`:<br><br>- If `X` is not an array of floating values;<br>- If `X` is encoded as a CSR matrix;<br>- If `add_indicator=True`.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('add_indicator',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=add_indicator,-bool%2C%20default%3DFalse">
            add_indicator
            <span class="param-doc-description">add_indicator: bool, default=False<br><br>If True, a :class:`MissingIndicator` transform will stack onto output<br>of the imputer's transform. This allows a predictive estimator<br>to account for missingness despite imputation. If a feature has no<br>missing values at fit/train time, the feature won't appear on<br>the missing indicator even if there are missing values at<br>transform/test time.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('keep_empty_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.impute.SimpleImputer.html#:~:text=keep_empty_features,-bool%2C%20default%3DFalse">
            keep_empty_features
            <span class="param-doc-description">keep_empty_features: bool, default=False<br><br>If True, features that consist exclusively of missing values when<br>`fit` is called are returned in results when `transform` is called.<br>The imputed value is always `0` except when `strategy="constant"`<br>in which case `fill_value` will be used instead.<br><br>.. versionadded:: 1.2</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-598" type="checkbox" ><label for="sk-estimator-id-598" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>OneHotEncoder</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OneHotEncoder.html">?<span>Documentation for OneHotEncoder</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__cat__encoder__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=categories,-%27auto%27%20or%20a%20list%20of%20array-like%2C%20default%3D%27auto%27">
            categories
            <span class="param-doc-description">categories: 'auto' or a list of array-like, default='auto'<br><br>Categories (unique values) per feature:<br><br>- 'auto' : Determine categories automatically from the training data.<br>- list : ``categories[i]`` holds the categories expected in the ith<br>  column. The passed categories should not mix strings and numeric<br>  values within a single feature, and should be sorted in case of<br>  numeric values.<br><br>The used categories can be found in the ``categories_`` attribute.<br><br>.. versionadded:: 0.20</span>
        </a>
    </td>
            <td class="value">&#x27;auto&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('drop',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=drop,-%7B%27first%27%2C%20%27if_binary%27%7D%20or%20an%20array-like%20of%20shape%20%28n_features%2C%29%2C%20%20%20%20%20%20%20%20%20%20%20%20%20default%3DNone">
            drop
            <span class="param-doc-description">drop: {'first', 'if_binary'} or an array-like of shape (n_features,),             default=None<br><br>Specifies a methodology to use to drop one of the categories per<br>feature. This is useful in situations where perfectly collinear<br>features cause problems, such as when feeding the resulting data<br>into an unregularized linear regression model.<br><br>However, dropping one category breaks the symmetry of the original<br>representation and can therefore induce a bias in downstream models,<br>for instance for penalized linear classification or regression models.<br><br>- None : retain all features (the default).<br>- 'first' : drop the first category in each feature. If only one<br>  category is present, the feature will be dropped entirely.<br>- 'if_binary' : drop the first category in each feature with two<br>  categories. Features with 1 or more than 2 categories are<br>  left intact.<br>- array : ``drop[i]`` is the category in feature ``X[:, i]`` that<br>  should be dropped.<br><br>When `max_categories` or `min_frequency` is configured to group<br>infrequent categories, the dropping behavior is handled after the<br>grouping.<br><br>.. versionadded:: 0.21<br>   The parameter `drop` was added in 0.21.<br><br>.. versionchanged:: 0.23<br>   The option `drop='if_binary'` was added in 0.23.<br><br>.. versionchanged:: 1.1<br>    Support for dropping infrequent categories.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('sparse_output',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=sparse_output,-bool%2C%20default%3DTrue">
            sparse_output
            <span class="param-doc-description">sparse_output: bool, default=True<br><br>When ``True``, it returns a :class:`scipy.sparse.csr_matrix`,<br>i.e. a sparse matrix in "Compressed Sparse Row" (CSR) format.<br><br>.. versionadded:: 1.2<br>   `sparse` was renamed to `sparse_output`</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=dtype,-number%20type%2C%20default%3Dnp.float64">
            dtype
            <span class="param-doc-description">dtype: number type, default=np.float64<br><br>Desired dtype of output.</span>
        </a>
    </td>
            <td class="value">&lt;class &#x27;numpy.float64&#x27;&gt;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('handle_unknown',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=handle_unknown,-%7B%27error%27%2C%20%27ignore%27%2C%20%27infrequent_if_exist%27%2C%20%27warn%27%7D%2C%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20default%3D%27error%27">
            handle_unknown
            <span class="param-doc-description">handle_unknown: {'error', 'ignore', 'infrequent_if_exist', 'warn'},                      default='error'<br><br>Specifies the way unknown categories are handled during :meth:`transform`.<br><br>- 'error' : Raise an error if an unknown category is present during transform.<br>- 'ignore' : When an unknown category is encountered during<br>  transform, the resulting one-hot encoded columns for this feature<br>  will be all zeros. In the inverse transform, an unknown category<br>  will be denoted as None.<br>- 'infrequent_if_exist' : When an unknown category is encountered<br>  during transform, the resulting one-hot encoded columns for this<br>  feature will map to the infrequent category if it exists. The<br>  infrequent category will be mapped to the last position in the<br>  encoding. During inverse transform, an unknown category will be<br>  mapped to the category denoted `'infrequent'` if it exists. If the<br>  `'infrequent'` category does not exist, then :meth:`transform` and<br>  :meth:`inverse_transform` will handle an unknown category as with<br>  `handle_unknown='ignore'`. Infrequent categories exist based on<br>  `min_frequency` and `max_categories`. Read more in the<br>  :ref:`User Guide <encoder_infrequent_categories>`.<br>- 'warn' : When an unknown category is encountered during transform<br>  a warning is issued, and the encoding then proceeds as described for<br>  `handle_unknown="infrequent_if_exist"`.<br><br>.. versionchanged:: 1.1<br>    `'infrequent_if_exist'` was added to automatically handle unknown<br>    categories and infrequent categories.<br><br>.. versionadded:: 1.6<br>   The option `"warn"` was added in 1.6.</span>
        </a>
    </td>
            <td class="value">&#x27;ignore&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_frequency',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=min_frequency,-int%20or%20float%2C%20default%3DNone">
            min_frequency
            <span class="param-doc-description">min_frequency: int or float, default=None<br><br>Specifies the minimum frequency below which a category will be<br>considered infrequent.<br><br>- If `int`, categories with a smaller cardinality will be considered<br>  infrequent.<br><br>- If `float`, categories with a smaller cardinality than<br>  `min_frequency * n_samples`  will be considered infrequent.<br><br>.. versionadded:: 1.1<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_categories',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=max_categories,-int%2C%20default%3DNone">
            max_categories
            <span class="param-doc-description">max_categories: int, default=None<br><br>Specifies an upper limit to the number of output features for each input<br>feature when considering infrequent categories. If there are infrequent<br>categories, `max_categories` includes the category representing the<br>infrequent categories along with the frequent categories. If `None`,<br>there is no limit to the number of output features.<br><br>.. versionadded:: 1.1<br>    Read more in the :ref:`User Guide <encoder_infrequent_categories>`.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('feature_name_combiner',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.OneHotEncoder.html#:~:text=feature_name_combiner,-%22concat%22%20or%20callable%2C%20default%3D%22concat%22">
            feature_name_combiner
            <span class="param-doc-description">feature_name_combiner: "concat" or callable, default="concat"<br><br>Callable with signature `def callable(input_feature, category)` that returns a<br>string. This is used to create feature names to be returned by<br>:meth:`get_feature_names_out`.<br><br>`"concat"` concatenates encoded feature name and category with<br>`feature + "_" + str(category)`.E.g. feature X with values 1, 6, 7 create<br>feature names `X_1, X_6, X_7`.<br><br>.. versionadded:: 1.3</span>
        </a>
    </td>
            <td class="value">&#x27;concat&#x27;</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-599" type="checkbox" ><label for="sk-estimator-id-599" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>StandardScaler</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html">?<span>Documentation for StandardScaler</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="preprocessor__cat__scaler__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('copy',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=copy,-bool%2C%20default%3DTrue">
            copy
            <span class="param-doc-description">copy: bool, default=True<br><br>If False, try to avoid a copy and do inplace scaling instead.<br>This is not guaranteed to always work inplace; e.g. if the data is<br>not a NumPy array or scipy.sparse CSR matrix, a copy may still be<br>returned.</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_mean',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_mean,-bool%2C%20default%3DTrue">
            with_mean
            <span class="param-doc-description">with_mean: bool, default=True<br><br>If True, center the data before scaling.<br>This does not work (and will raise an exception) when attempted on<br>sparse matrices, because centering them entails building a dense<br>matrix which in common use cases is likely to be too large to fit in<br>memory.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('with_std',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.preprocessing.StandardScaler.html#:~:text=with_std,-bool%2C%20default%3DTrue">
            with_std
            <span class="param-doc-description">with_std: bool, default=True<br><br>If True, scale the data to unit variance (or equivalently,<br>unit standard deviation).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div></div></div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-600" type="checkbox" ><label for="sk-estimator-id-600" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>RidgeCV</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.RidgeCV.html">?<span>Documentation for RidgeCV</span></a></div></label><div class="sk-toggleable__content fitted" data-param-prefix="ridge__">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('alphas',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.RidgeCV.html#:~:text=alphas,-array-like%20of%20shape%20%28n_alphas%2C%29%2C%20default%3D%280.1%2C%201.0%2C%2010.0%29">
            alphas
            <span class="param-doc-description">alphas: array-like of shape (n_alphas,), default=(0.1, 1.0, 10.0)<br><br>Array of alpha values to try.<br>Regularization strength; must be a positive float. Regularization<br>improves the conditioning of the problem and reduces the variance of<br>the estimates. Larger values specify stronger regularization.<br>Alpha corresponds to ``1 / (2C)`` in other linear models such as<br>:class:`~sklearn.linear_model.LogisticRegression` or<br>:class:`~sklearn.svm.LinearSVC`.<br>If using Leave-One-Out cross-validation, alphas must be strictly positive.</span>
        </a>
    </td>
            <td class="value">array([1.0000...00000000e+03])</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('fit_intercept',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.RidgeCV.html#:~:text=fit_intercept,-bool%2C%20default%3DTrue">
            fit_intercept
            <span class="param-doc-description">fit_intercept: bool, default=True<br><br>Whether to calculate the intercept for this model. If set<br>to false, no intercept will be used in calculations<br>(i.e. data is expected to be centered).</span>
        </a>
    </td>
            <td class="value">True</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('scoring',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.RidgeCV.html#:~:text=scoring,-str%2C%20callable%2C%20default%3DNone">
            scoring
            <span class="param-doc-description">scoring: str, callable, default=None<br><br>The scoring method to use for cross-validation. Options:<br><br>- str: see :ref:`scoring_string_names` for options.<br>- callable: a scorer callable object (e.g., function) with signature<br>  ``scorer(estimator, X, y)``. See :ref:`scoring_callable` for details.<br>- `None`: negative :ref:`mean squared error <mean_squared_error>` if cv is<br>  None (i.e. when using leave-one-out cross-validation), or<br>  :ref:`coefficient of determination <r2_score>` (:math:`R^2`) otherwise.</span>
        </a>
    </td>
            <td class="value">&#x27;neg_root_mean_squared_error&#x27;</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('cv',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.RidgeCV.html#:~:text=cv,-int%2C%20cross-validation%20generator%20or%20an%20iterable%2C%20default%3DNone">
            cv
            <span class="param-doc-description">cv: int, cross-validation generator or an iterable, default=None<br><br>Determines the cross-validation splitting strategy.<br>Possible inputs for cv are:<br><br>- None, to use the efficient Leave-One-Out cross-validation<br>- integer, to specify the number of folds.<br>- :term:`CV splitter`,<br>- An iterable yielding (train, test) splits as arrays of indices.<br><br>For integer/None inputs, if ``y`` is binary or multiclass,<br>:class:`~sklearn.model_selection.StratifiedKFold` is used, else,<br>:class:`~sklearn.model_selection.KFold` is used.<br><br>Refer :ref:`User Guide <cross_validation>` for the various<br>cross-validation strategies that can be used here.</span>
        </a>
    </td>
            <td class="value">KFold(n_split... shuffle=True)</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('gcv_mode',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.RidgeCV.html#:~:text=gcv_mode,-%7B%27auto%27%2C%20%27svd%27%2C%20%27eigen%27%7D%2C%20default%3D%27auto%27">
            gcv_mode
            <span class="param-doc-description">gcv_mode: {'auto', 'svd', 'eigen'}, default='auto'<br><br>Flag indicating which strategy to use when performing<br>Leave-One-Out Cross-Validation. Options are::<br><br>    'auto' : use 'svd' if n_samples > n_features, otherwise use 'eigen'<br>    'svd' : force use of singular value decomposition of X when X is<br>        dense, eigenvalue decomposition of X^T.X when X is sparse.<br>    'eigen' : force computation via eigendecomposition of X.X^T<br><br>The 'auto' mode is the default and is intended to pick the cheaper<br>option of the two depending on the shape of the training data.</span>
        </a>
    </td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('store_cv_results',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.RidgeCV.html#:~:text=store_cv_results,-bool%2C%20default%3DFalse">
            store_cv_results
            <span class="param-doc-description">store_cv_results: bool, default=False<br><br>Flag indicating if the cross-validation values corresponding to<br>each alpha should be stored in the ``cv_results_`` attribute (see<br>below). This flag is only compatible with ``cv=None`` (i.e. using<br>Leave-One-Out Cross-Validation).<br><br>.. versionchanged:: 1.5<br>    Parameter name changed from `store_cv_values` to `store_cv_results`.</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('alpha_per_target',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">
        <a class="param-doc-link"
            rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.RidgeCV.html#:~:text=alpha_per_target,-bool%2C%20default%3DFalse">
            alpha_per_target
            <span class="param-doc-description">alpha_per_target: bool, default=False<br><br>Flag indicating whether to optimize the alpha value (picked from the<br>`alphas` parameter list) for each target separately (for multi-output<br>settings: multiple prediction targets). When set to `True`, after<br>fitting, the `alpha_` attribute will contain a value for each target.<br>When set to `False`, a single alpha is used for all targets.<br><br>.. versionadded:: 0.24</span>
        </a>
    </td>
            <td class="value">False</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.copy-paste-icon').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling
        .textContent.trim().split(' ')[0];
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});


/**
 * Adapted from Skrub
 * https://github.com/skrub-data/skrub/blob/403466d1d5d4dc76a7ef569b3f8228db59a31dc3/skrub/_reporting/_data/templates/report.js#L789
 * @returns "light" or "dark"
 */
function detectTheme(element) {
    const body = document.querySelector('body');

    // Check VSCode theme
    const themeKindAttr = body.getAttribute('data-vscode-theme-kind');
    const themeNameAttr = body.getAttribute('data-vscode-theme-name');

    if (themeKindAttr && themeNameAttr) {
        const themeKind = themeKindAttr.toLowerCase();
        const themeName = themeNameAttr.toLowerCase();

        if (themeKind.includes("dark") || themeName.includes("dark")) {
            return "dark";
        }
        if (themeKind.includes("light") || themeName.includes("light")) {
            return "light";
        }
    }

    // Check Jupyter theme
    if (body.getAttribute('data-jp-theme-light') === 'false') {
        return 'dark';
    } else if (body.getAttribute('data-jp-theme-light') === 'true') {
        return 'light';
    }

    // Guess based on a parent element's color
    const color = window.getComputedStyle(element.parentNode, null).getPropertyValue('color');
    const match = color.match(/^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*$/i);
    if (match) {
        const [r, g, b] = [
            parseFloat(match[1]),
            parseFloat(match[2]),
            parseFloat(match[3])
        ];

        // https://en.wikipedia.org/wiki/HSL_and_HSV#Lightness
        const luma = 0.299 * r + 0.587 * g + 0.114 * b;

        if (luma > 180) {
            // If the text is very bright we have a dark theme
            return 'dark';
        }
        if (luma < 75) {
            // If the text is very dark we have a light theme
            return 'light';
        }
        // Otherwise fall back to the next heuristic.
    }

    // Fallback to system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}


function forceTheme(elementId) {
    const estimatorElement = document.querySelector(`#${elementId}`);
    if (estimatorElement === null) {
        console.error(`Element with id ${elementId} not found.`);
    } else {
        const theme = detectTheme(estimatorElement);
        estimatorElement.classList.add(theme);
    }
}

forceTheme('sk-container-id-8');</script></body>



## Final predictions in the test set


```python
predictions = model.predict(data_test)

predictions = np.exp(predictions)

df_final = pd.DataFrame({
    'Id': test_id,
    'SalePrice': predictions
})

df_final.to_csv('model_feature_engineering.csv', index=False)
```

We have achieved a score of 0.12939! This is better than 66% of all teams, using only a simple linear model.

# Conclusion

Throughout the notebook, we have explored how data processing directly impacts machine learning model performance. The main takeaway is clear: **robust feature engineering is significantly more impactful than model complexity or selection alone.**

To demonstrate this, we maintained a consistent model structure (Ridge Regression) and observed dramatic improvements based solely on how we treated the input data:

Without any feature processing, we achieved an score of **0.32534**, which was poor and uncompetitive.
By identifying outliers, performing log transformations on the target variable to handle skewness, and creating "intelligent" new features, we were able to significantly refine the signal the model receives.
Through these data-driven refinements, we achieved a final score of **0.12939**, a massive improvement from our starting point.

While choosing the right algorithm is important, a system's "intelligence" ultimately relies on the quality and preparation of the input data. This project proves that time spent understanding and transforming variables yields a much higher return on accuracy than simple hyperparameter tuning.

| Approach | Feature Engineering Level | RMSE Score | Improvement |
| :--- | :--- | :--- | :--- |
| **Baseline** | None (Raw Data) | 0.32534 | - |
| **Simple Featurnig** | Scaling and basic transformations | 0.1366 | **+58.01%** |
| **Final Model** | Advanced (Outliers + Log Transform + New Features) | **0.12939** | **+5.27%** |
