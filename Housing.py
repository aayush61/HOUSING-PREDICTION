#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("hello world")


# In[2]:


import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[3]:


housing= load_housing_data()
housinf.head()


# In[4]:


housing= load_housing_data()
housing.head()


# In[5]:


import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH:


# In[6]:


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[7]:


housing= load_housing_data()
housing.head()


# In[11]:


import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# In[9]:


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[10]:


housing= load_housing_data()
housing.head()


# In[12]:


fetch_housing_data()


# In[13]:


housing= load_housing_data()
housing.head()


# In[14]:


housing.info()


# In[15]:


housing.describe()


# In[16]:


import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))


# In[17]:


plt.show()


# In[18]:


import numpy as np

def split_train_set(data, test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    train_indices=shuffled_indices(:test_set_size)
    test_indices=shuffled_indices(test_set_size:)
    return data.iloc[train_indices], data.iloc[test_indices]


# In[19]:


import numpy as np

def split_train_set(data, test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[20]:


train_set, test_set=split_train_data(housin,0.2)
print(len(train_set), "train+", len(test_set), "test")


# In[21]:


train_set, test_set=split_train_set(housing,0.2)
print(len(train_set), "train+", len(test_set), "test")


# In[22]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[23]:


housing.head()


# In[24]:


print(len(train_set))


# In[25]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[26]:


housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# In[27]:


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[28]:


housing["income_cat"].value_counts() / len(housing)


# In[29]:


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


# In[30]:


housing = strat_train_set.copy()


# In[31]:


housing.plot(kind="scatter", x="longitude", y="latitude")


# In[32]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)


# In[33]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()


# In[34]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4)


# In[35]:


housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()


# In[36]:


corr_matrix = housing.corr()


# In[37]:


corr_matrix["median_house_value"].sort_values(ascending=False)
median_house_value    1.000000


# In[ ]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[ ]:


housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)


# In[38]:


housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[39]:


corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)


# In[40]:


housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[41]:


from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")


# In[42]:


housing_num = housing.drop("ocean_proximity", axis=1)


# In[43]:


from sklearn.preprocessing import SimpleImputer

imputer = Imputer(strategy="median")


# In[44]:


from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")


# In[45]:


imputer.fit(housing_num)


# In[46]:


imputer.statistics_


# In[47]:


housing_num.median().values


# In[48]:


X = imputer.transform(housing_num)


# In[49]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# In[50]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded


# In[51]:


from sklearn.preprocessing import OneHotEncoder


# In[52]:


encoder = OneHotEncoder()


# In[53]:


housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))


# In[54]:


housing_cat_1hot


# In[73]:


housing_cat_1hot.toarray()


# In[74]:


from sklearn.preprocessing import LabelBinarizer


# In[75]:


encoder = LabelBinarizer()


# In[76]:


housing_cat_1hot = encoder.fit_transform(housing_cat)


# In[77]:


housing_cat_1hot


# In[56]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])


# In[57]:


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): 
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[58]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),


# In[59]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),


# In[60]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# In[61]:


import sklearn
sklearn.__version__


# In[62]:


from sklearn.impute import SimpleImputer


# In[63]:


from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# In[64]:


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', LabelBinarizer()),
    ])


# In[68]:


from sklearn.preprocessing import LabelBinarizer


# In[79]:


class CustomBinarizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None,**fit_params):
        return self
    def transform(self, X):
        return LabelBinarizer().fit(X).transform(X)


# In[80]:


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', CustomBinarizer()),
    ])


# In[81]:


from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# In[82]:


housing_prepared = full_pipeline.fit_transform(housing)


# In[83]:


housing_prepared


# In[84]:


housing_prepared.shape


# In[104]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# In[121]:


some_data = housing.iloc[:5]


# In[106]:


some_data.shape


# In[107]:


some_labels = housing_labels.iloc[:5]


# In[108]:


some_labels.shape


# In[109]:


some_data_prepared = full_pipeline.transform(some_data)


# In[110]:


some_data_prepared.shape


# In[111]:


print("Labels:", list(some_labels))


# In[126]:


print("Predictions:", lin_reg.predict(housing_prepared))


# In[113]:


features.shape


# In[114]:


np.reshape(some_data_prepared, (5,16))


# In[115]:


some_data_prepared.shape


# In[116]:


housing.shape


# In[117]:


housing_prepared.shape


# In[122]:


a = housing_prepared.iloc[:5]


# In[123]:


housing


# In[124]:


housing_labels


# In[125]:


housing_prepared


# In[127]:


print("Predictions:", lin_reg.predict(housing_prepared))


# In[128]:


from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)


# In[129]:


lin_mse = mean_squared_error(housing_labels, housing_predictions)


# In[130]:


lin_rmse = np.sqrt(lin_mse)


# In[131]:


lin_rmse


# In[132]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# In[133]:


housing_predictions = tree_reg.predict(housing_prepared)


# In[134]:


tree_mse = mean_squared_error(housing_labels, housing_predictions)


# In[135]:


tree_rmse = np.sqrt(tree_mse)


# In[136]:


tree_rmse


# In[137]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)


# In[139]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[140]:


display_scores(tree_rmse_scores)


# In[141]:


lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)


# In[142]:


lin_rmse_scores = np.sqrt(-lin_scores)


# In[143]:


display_scores(lin_rmse_scores)


# In[144]:


from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()


# In[145]:


forest_reg.fit(housing_prepared, housing_labels)


# In[146]:


[...]


# In[153]:


housing_prediction = housing_predictions = forest_reg.predict(housing_prepared)


# In[154]:


forest_mse = mean_squared_error(housing_labels, housing_predictions)


# In[155]:


forest_rmse = np.sqrt(forest_mse)


# In[156]:


forest_rmse


# In[157]:


lin_rmse


# In[159]:


display_scores(forest_rmse_scores)


# In[160]:


forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)


# In[161]:


forest_rmse_scores = np.sqrt(-forest_scores)


# In[162]:


display_scores(forest_rmse_scores)


# In[163]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)


# In[164]:


grid_search.best_params_


# In[165]:


grid_search.best_estimator_


# In[166]:


cvres = grid_search.cv_results_


# In[167]:


for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


# In[168]:


feature_importances = grid_search.best_estimator_.feature_importances_


# In[169]:


feature_importances


# In[170]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]


# In[171]:


cat_one_hot_attribs = list(encoder.classes_)


# In[172]:


attributes = num_attribs + extra_attribs + cat_one_hot_attribs


# In[173]:


sorted(zip(feature_importances, attributes), reverse=True)


# In[174]:


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[175]:


final_rmse


# In[ ]:




