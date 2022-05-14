#!/usr/bin/env python
# coding: utf-8

# # Predicting the type of the forest - comparison of performance of different ensemble machine learning algorithms

# Final project for Machine Learning II
# Ewelina Osowska

# In[8]:


# Loading libraries
import pandas as pd
import numpy as np
import time
import scipy.stats as stats
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
import sklearn.metrics as slm
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from xgboost.sklearn import XGBClassifier
from IPython.display import display_html
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV

# General settings
pd.set_option("display.max_columns",130)
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set(style="whitegrid")
np.random.seed(13579)

# Displaying results
# source: https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)


# ## Introduction <a class="anchor" id="introduction"></a>
# 
# 
# The main objective of this research is to predict the multilabel variable indicating cover of undisturbed forests based on cartographic variables. It will be obtained with the use of three machine learning methods: XGBoost, Light GBM and AdaBoost. Later these algorithms will be evaluated and compared in terms of performance and execution time.
# 
# 
# 
# ## Dataset description <a class="anchor" id="dataset-description"></a>
# 
# The dataset used in the analysis was sourced from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Covertype). It consists of only cartographic type data. Dependent variable was originally determined from US Forest Service (USFS) Region 2 Resource Information System (RIS) data. It is a multilabel (7 levels) variable which is defined based on 30 x 30 meter cell of the forest area. It assumes following cover forest type classes:
# * Spruce/Fir,
# * Lodgepole Pine,
# * Ponderosa Pine,
# * Cottonwood/Willow,
# * Aspen,
# * Douglas-fir,
# * Krummholz.
# 
# The dataset concerns four wilderness areas located in the Roosevelt National Forest (northern Colorado). These areas are a result of ecological processes rather than forest management practices. These areas include: Rawah Wilderness Area, Neota Wilderness Area, Comanche Peak Wilderness Area and Cache la Poudre Wilderness Area. 
# 
# Independent variables were obtained from US Geological Survey (USGS) and also USFS data. They describe elevation, aspect, slope, distances to water, roadways and wildfire ignition points, wilderness area designation and soil type designation. A more detailed description of the variables and dataset itself can be found on UCI website. Initial descriptive analysis of all variables can be found in the supplement.
# 
# ## Data preprocessing <a class="anchor" id="data-preprocessing"></a>
# 
# The data consists of 55 variables and 581012 observations. The initial analysis of dependent variable indicate that we deal with unbalanced sample. Missing data do not occur.

# In[9]:


mydata = pd.read_pickle("dataset/mydata_step1.p")


# In[10]:


mydata.shape


# In[12]:


mydata.Cover_Type.value_counts().sort_index()


# In[14]:


len(mydata.isnull().sum()[mydata.isnull().sum()>1])


# As an itinial step, we encode the target variable into dummy variables. Later, we divide the data into train and test samples using train_test_split function (70:30 ratio). Since there are two categorical variables in the data which are decoded into dummies, we remove reference levels in order to get rid off collinearity.

# In[15]:


labl_encode = preprocessing.LabelEncoder()
mydata["Cover_Type"] = labl_encode.fit_transform(mydata.Cover_Type.values)


# In[16]:


mydata.Cover_Type.value_counts().sort_index()


# In[17]:


target_var = "Cover_Type"


# In[18]:


target_names = ["class1", "class2", "class3", "class4", "class5", "class6", "class7"]


# In[19]:


train , test = train_test_split(mydata, test_size = 0.3, random_state=13579)

x_train = train.drop(target_var, axis=1)
y_train = train[target_var]

x_test = test.drop(target_var, axis = 1)
y_test = test[target_var]


# In[20]:


len(train) / len(mydata)


# In[21]:


len(test) / len(mydata)


# In[22]:


cols = list(mydata.columns)
cols.remove("Cover_Type")
cols.remove("Soil_Type40")
cols.remove("Wilderness_Area4")


# In[23]:


train_data = x_train.join(y_train)
y_train = train_data.Cover_Type
x_train = train_data.drop("Cover_Type", axis = 1)
x_train = x_train[cols]


# In[24]:


x_train.shape


# In[25]:


x_test.shape


# ## Modelling <a class="anchor" id="modelling"></a>
# 
# After data preprocessing, I move to the modelling process. I will build three models namely XGBoost, LightGBM and AdaBoost. All of these models are examples of ensemble algorithms and specifically boosting algorithms. Benchmark model will be XGBoost. 
# 
# At this stage models will be assessed based on chosen evaluation metrics calculated using train and test samples. I will also compare the calculation time in order to choose the most efficient model.

# ### XGBoost <a class="anchor" id="xgboost"></a>
#  
# First implemented algorithm will be Extreme Gradient Boosting (XGBoost) which is an example of ensemble model in a way it implements gradient boosted decision trees. In order to find the optimal hyperpatrameters, I used randomized search algorithm. In contrast to grid search algorithm, not all parameter values are tried out, but rather a fixed number of them. The number of parameter settings that are tried is given by parameter n_iter. Due to the fact that the algorithm is computationally intensive, I set the parameter to only 3 iterations, having in mind that this might influence the results. This approach is continued for all built models.
# 
# Tuning is conducted in the specific way. Firtly, I set initial values of hyperparameters and run the "zero" version of the model. Later on, I move to optimizing tree-specific and regularization parameters using random search. At the end, I decrease learning rate and in the same time increase number of trees. 
# 
# Below can be seen the general diagram of the model implementation.
# 
# ![title](xgboost.png)

# **Initial values of hyperparameters (will be tuned later)**:   
# learning_rate = 0.1  
# max_depth = 5    
# min_child_weight = 1 (a smaller value is chosen because of class imbalance and leaf nodes can have smaller size groups)    
# gamma = 0   
# subsample, colsample_bytree = 0.8  
# scale_pos_weight = 1 (because of class imbalance)  

# In[21]:


xgb0 = XGBClassifier(learning_rate = 0.01,
                     n_estimators = 100,
                     max_depth = 5,
                     min_child_weight = 1,
                     gamma = 0,
                     subsample = 0.8,
                     colsample_bytree = 0.8,
                     objective = 'multi:softprob',
                     nthread = 30,
                     njobs = -1,
                     scale_pos_weight = 1,
                     silent = True,
                     seed = 13579)


# In[23]:


start = time.time()
xgb0.fit(x_train, y_train)
end = time.time()
print(end - start)


# In[24]:


predicted_train_xgb0 = xgb0.predict(x_train)
predicted_test_xgb0 = xgb0.predict(x_test[cols])


# In[29]:


def model_evaluation(y_true, y_pred, model_type = "Model"):
    confusion = slm.confusion_matrix(y_true, y_pred)
    TP = confusion[1, 1] # True Positives
    TN = confusion[0, 0] # True Negatives
    FP = confusion[0, 1] # False Positives
    FN = confusion[1, 0] # False Negatives
    result_df = pd.DataFrame({model_type:
                                      [slm.accuracy_score(y_true, y_pred),
                                      slm.balanced_accuracy_score(y_true, y_pred),
                                      TP / float(TP + FP),
                                      TP / float(TP + FN),
                                      TN / float(TN + FP),
                                      (2*TP)/float(2*TP + FP + FN),
                                     ]},
                             index=["Accuracy",
                                    "Balanced accuracy",
                                    "Precision",
                                    "Sensitivity",
                                    "Specificity",
                                    "F1 score"])
    return result_df


# In[26]:


model0_train = model_evaluation(y_train, predicted_train_xgb0, 'XGBoost (train; version: 0)')
model0_test = model_evaluation(y_test, predicted_test_xgb0, 'XGBoost (test; version: 0)')
display_side_by_side(model0_train, model0_test)


# In[27]:


slm.confusion_matrix(y_train, predicted_train_xgb0)


# In[28]:


slm.confusion_matrix(y_test, predicted_test_xgb0)


# In[29]:


plt.figure(figsize=(11,5))
feat_imp = pd.Series(xgb0.get_booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')


# In[30]:


#saving model into pickle
pickle.dump(xgb0, open("xgb0.pickle", "wb"))


# **Parameters optimization - step 1**   
# Tuning tree-specific parameters: max_depth, min_child_weight and gamma.

# In[31]:


params_1 = {
 'max_depth': range(3, 10, 2),
 'min_child_weight': range(1, 6, 2)
}


# In[32]:


random_search_1 = RandomizedSearchCV(
    estimator = XGBClassifier(learning_rate = 0.01, 
                              n_estimators = 100, 
                              gamma = 0, 
                              subsample = 0.8, 
                              colsample_bytree = 0.8, 
                              objective = 'multi:softprob', 
                              nthread = 30, 
                              silent = True,
                              n_jobs = -1,
                              scale_pos_weight = 1, 
                              seed = 13579), 
    param_distributions = params_1, 
    n_jobs = -1,
    pre_dispatch = 1,
    n_iter = 3,
    cv = 3)


# In[33]:


start = time.time()
random_search_11 = random_search_1.fit(x_train, y_train)
end = time.time()
print(end - start)


# In[34]:


random_search_11.best_params_, random_search_1.best_score_


# In[35]:


pickle.dump(random_search_11, open("random_search_11.pickle", "wb"))
pickle.dump(random_search_1, open("random_search_1.pickle", "wb"))


# In[36]:


params_2 = {
 'gamma':[i/10.0 for i in range(0,5)]
}


# In[37]:


random_search_2 = RandomizedSearchCV(
    estimator = XGBClassifier(learning_rate = 0.01, 
                              n_estimators = 100, 
                              max_depth = 9, 
                              min_child_weight = 1, 
                              subsample = 0.8, 
                              colsample_bytree = 0.8, 
                              objective = 'multi:softprob', 
                              nthread = 30, 
                              n_jobs = -1,
                              silent = True,
                              scale_pos_weight = 1, 
                              seed = 13579), 
    param_distributions = params_2, 
    n_jobs = -1,
    n_iter = 3,
    cv = 3)


# In[38]:


start = time.time()
random_search_22 = random_search_2.fit(x_train, y_train)
end = time.time()
print(end - start)


# In[39]:


random_search_22.best_params_, random_search_22.best_score_


# In[40]:


#saving model into pickle
pickle.dump(random_search_22, open("random_search_22.pickle", "wb"))
pickle.dump(random_search_2, open("random_search_2.pickle", "wb"))


# In[41]:


params_3 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}


# In[42]:


random_search_3 = RandomizedSearchCV(
    estimator = XGBClassifier(learning_rate = 0.01, 
                              n_estimators = 100, 
                              max_depth = 9, 
                              min_child_weight = 1,  
                              gamma = 0.2,
                              objective = 'multi:softprob', 
                              nthread = 30, 
                              n_jobs = -1,
                              silent = True,
                              scale_pos_weight = 1, 
                              seed = 13579), 
    param_distributions = params_3, 
    n_jobs = -1,
    n_iter = 3,
    cv = 3)


# In[43]:


start = time.time()
random_search_33 = random_search_3.fit(x_train, y_train)
end = time.time()
print(end - start)


# In[44]:


random_search_33.best_params_, random_search_33.best_score_


# In[45]:


#saving model into pickle
pickle.dump(random_search_33, open("random_search_33.pickle", "wb"))
pickle.dump(random_search_3, open("random_search_3.pickle", "wb"))


# In[36]:


xgb1 = XGBClassifier(learning_rate = 0.01,
                     n_estimators = 100,
                     max_depth = 9,
                     min_child_weight = 1,
                     gamma = 0.2,
                     subsample = 0.8,
                     colsample_bytree = 0.85,
                     objective = 'multi:softprob',
                     nthread = 30, 
                     n_jobs = -1,
                     silent = True,
                     seed = 13579)


# In[37]:


start = time.time()
xgb11 = xgb1.fit(x_train, y_train)
end = time.time()
print(end - start)


# In[38]:


predicted_train_xgb1 = xgb11.predict(x_train)
predicted_test_xgb1 = xgb11.predict(x_test[cols])


# In[39]:


model1_train = model_evaluation(y_train, predicted_train_xgb1, 'XGBoost (train; version: 1)')
model1_test = model_evaluation(y_test, predicted_test_xgb1, 'XGBoost (test; version: 1)')
display_side_by_side(model1_train, model1_test)


# In[40]:


#saving model into pickle
pickle.dump(xgb1, open("xgb1.pickle", "wb"))
pickle.dump(xgb11, open("xgb11.pickle", "wb"))


# **Parameters optimization - step 2**   
# Tuning Regularization Parameters: alpha and lambda.

# In[41]:


params_4 = {
 'reg_alpha': [1e-5, 1e-2, 0.01, 0.05, 0.1, 1],
 'reg_lambda': [1e-5, 1e-2, 0.01, 0.05, 0.1, 1]
}


# In[42]:


random_search_4 = RandomizedSearchCV(
    estimator = XGBClassifier(learning_rate = 0.01, 
                              n_estimators = 100,
                              max_depth = 9,
                              min_child_weight = 1,
                              gamma = 0.2,
                              subsample = 0.8,
                              colsample_bytree = 0.85,
                              objective = 'multi:softprob',
                              nthread = 30, 
                              n_jobs = -1,
                              silent = True,
                              scale_pos_weight = 1, 
                              seed = 13579), 
    param_distributions = params_4, 
    n_jobs = -1,
    n_iter = 3,
    cv = 3)


# In[43]:


start = time.time()
random_search_44 = random_search_4.fit(x_train, y_train)
end = time.time()
print(end - start)


# In[44]:


random_search_44.best_params_, random_search_44.best_score_


# In[45]:


#saving model into pickle
pickle.dump(random_search_4, open("random_search_4.pickle", "wb"))
pickle.dump(random_search_44, open("random_search_44.pickle", "wb"))


# In[65]:


xgb2 = XGBClassifier(learning_rate = 0.01,
                     n_estimators = 100,
                     max_depth = 9,
                     min_child_weight = 1,
                     gamma = 0.2,
                     subsample = 0.8,
                     colsample_bytree = 0.85,
                     objective = 'multi:softprob',
                     reg_alpha = 0.01,
                     reg_lambda = 0.01,
                     nthread = 30, 
                     n_jobs = -1,
                     silent = True,
                     seed = 13579)


# In[66]:


start = time.time()
xgb22 = xgb2.fit(x_train, y_train)
end = time.time()
print(end - start)


# In[67]:


predicted_train_xgb2 = xgb22.predict(x_train)
predicted_test_xgb2 = xgb22.predict(x_test[cols])


# In[68]:


model2_train = model_evaluation(y_train, predicted_train_xgb2, 'XGBoost (train; version: 2)')
model2_test = model_evaluation(y_test, predicted_test_xgb2, 'XGBoost (test; version: 2)')
display_side_by_side(model2_train, model2_test)


# In[69]:


#saving model into pickle
pickle.dump(xgb2, open("xgb2.pickle", "wb"))
pickle.dump(xgb22, open("xgb22.pickle", "wb"))


# **Parameters optimization - step 3**   
# Reducing learning rate and increasing number of trees to estimate.

# In[31]:


xgb3 = XGBClassifier(learning_rate = 0.001,
                     n_estimators = 1000,
                     max_depth = 9,
                     min_child_weight = 1,
                     gamma = 0.2,
                     subsample = 0.8,
                     colsample_bytree = 0.85,
                     objective = 'multi:softprob',
                     reg_alpha = 0.01,
                     reg_lambda = 0.01,
                     nthread = 30, 
                     n_jobs = -1,
                     silent = True,
                     seed = 13579)


# In[32]:


start = time.time()
xgb33 = xgb3.fit(x_train, y_train)
end = time.time()
print(end - start)


# In[33]:


predicted_train_xgb3 = xgb3.predict(x_train)
predicted_test_xgb3 = xgb3.predict(x_test[cols])


# It resulted that the final set of hyperparameters resulted in the accuracy of 83.68% on the train set and 82.86% on the test set.

# In[34]:


model3_train = model_evaluation(y_train, predicted_train_xgb3, 'XGBoost (train; version: 3)')
model3_test = model_evaluation(y_test, predicted_test_xgb3, 'XGBoost (test; version: 3)')
display_side_by_side(model3_train, model3_test)


# In[35]:


slm.confusion_matrix(y_train, predicted_train_xgb3)


# In[36]:


slm.confusion_matrix(y_test, predicted_test_xgb3)


# In[ ]:


#saving model into pickle
pickle.dump(xgb3, open("xgb3.pickle", "wb"))
pickle.dump(xgb33, open("xgb33.pickle", "wb"))


# ### Light GBM <a class="anchor" id="lgbm"></a>
# 
# LightGBM is a gradient boosting framework that uses tree-based algorithm and follows leaf-wise approach (grows tree vertically) while other algorithms (like e.g. XGBoost) work in a level-wise approach (grows trees horizontally). This is the reason why Light GBM works more efficient when the dataset is large and takes lower memory to run. In order to tune hyperparameters, I also used random search. 
# 
# Important parameters which should be tuned in case of LightGBM are num_leaves (maximum tree leaves for base learners), min_child_samples (inimum number of data needed in a leaf) and max_depth. num_leaves is the main parameter to control the complexity of the tree model. min_data_in_leaf parameter indicate the minimum number of the records a leaf may have. 
# 
# Below can be seen the general diagram of the model implementation.
# 
# ![title](lightgbm.png)

# In[70]:


params_5 = {'num_leaves': range(15, 35, 1), 
            'min_child_samples': range(20, 40, 2), 
            'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
            'subsample': np.arange(0.5, 1, 0.01), 
            'colsample_bytree': np.arange(0.5, 1, 0.01),
            'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
            'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100],
            'learning_rate': [0.001, 0.003, 0.005, 0.01]
           }


# In[71]:


random_search_5 = RandomizedSearchCV(
    estimator=lgb.LGBMClassifier(max_depth=9, 
                                 random_state=13579, 
                                 silent=True, 
                                 metric='None', 
                                 n_jobs=-1, 
                                 nthread = 30,
                                 n_estimators=1000), 
    param_distributions=params_5, 
    n_iter = 3,
    n_jobs = -1,
    cv = 3,
    random_state=13579)

#saving into pickle
pickle.dump(random_search_5, open("random_search_5.pickle", "wb"))


# In[72]:


start = time.time()
random_search_55 = random_search_5.fit(x_train, y_train)
end = time.time()
print(end - start)

#saving into pickle
pickle.dump(random_search_55, open("random_search_55.pickle", "wb"))


# In[73]:


random_search_55 = pickle.load(open('random_search_55.pickle', 'rb'))
random_search_5 = pickle.load(open('random_search_5.pickle', 'rb'))


# In[74]:


random_search_55.best_params_, random_search_55.best_score_


# In[6]:


lgb1 = lgb.LGBMClassifier(max_depth = 9,
                          learning_rate = 0.01,
                          subsample = 0.72,
                          reg_lambda = 20,
                          reg_alpha = 0.1,
                          num_leaves = 33,
                          min_child_weight = 10,
                          min_child_samples = 22,
                          colsample_bytree = 0.77,
                          random_state=13579, 
                          silent=True, 
                          metric='None', 
                          n_jobs=-1, 
                          nthread = 30,
                          n_estimators=1000)


# In[26]:


start = time.time()
lgb11 = lgb1.fit(x_train, y_train)
end = time.time()
print(end - start)


# In[27]:


predicted_train_lgb1 = lgb11.predict(x_train)
predicted_test_lgb1 = lgb11.predict(x_test[cols])


# In[30]:


model1_train = model_evaluation(y_train, predicted_train_lgb1, 'Light GBM (train; version: 1)')
model1_test = model_evaluation(y_test, predicted_test_lgb1, 'Light GBM (test; version: 1)')
display_side_by_side(model1_train, model1_test)


# In[31]:


#saving model into pickle
pickle.dump(lgb1, open("lgb1.pickle", "wb"))
pickle.dump(lgb11, open("lgb11.pickle", "wb"))


# In[32]:


slm.confusion_matrix(y_train, predicted_train_lgb1)


# In[33]:


slm.confusion_matrix(y_test, predicted_test_lgb1)


# ### AdaBoost <a class="anchor" id="ada"></a>
# 
# The next implemented model is AdaBoost. The algorithm increases the accuracy by giving higher weight to the target that is misclassified by the model. At each iteration, AdaBoost changes the sample distribution by modifying the weights assigned to each of the instances. It increases the weights of the wrongly predicted classes and decreases the ones of the correctly predicted classes.
# 
# One of the most important parameters in tuning AdaBoost are max_depth (maximum depth of the tree), min_samples_split (minimum number of samples required to split an internal node), min_samples_leaf (minimum number of samples required to be at a leaf node) and max_features (number of features to consider when looking for the best split). These will be optimized using random search algorithm.
# 
# Below can be seen the general diagram of the model implementation.
# 
# ![title](adaboost.png)

# In[60]:


params_6 = {'base_estimator__max_depth': range(3,11,1),
            'base_estimator__min_samples_split': range(2,100,2),
            'base_estimator__min_samples_leaf': range(1,15,1),
            'base_estimator__max_features': range(2,20,1),
            'learning_rate': [0.001, 0.003, 0.005, 0.01]        
           }


# In[61]:


random_search_6 = RandomizedSearchCV(
    AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier()), 
    param_distributions = params_6, 
    n_jobs = -1,
    n_iter = 3,
    cv = 3)

#saving model into pickle
pickle.dump(random_search_6, open("random_search_6.pickle", "wb"))


# In[62]:


start = time.time()
random_search_66 = random_search_6.fit(x_train, y_train)
end = time.time()
print(end - start)

#saving model into pickle
pickle.dump(random_search_66, open("random_search_66.pickle", "wb"))


# In[63]:


random_search_66.best_params_, random_search_66.best_score_


# In[84]:


ada1 = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=9,
                                              min_samples_split=10,
                                              max_features=14,
                                              min_samples_leaf = 13),
        learning_rate = 0.01,
        n_estimators = 1000)

#saving model into pickle
pickle.dump(ada1, open("ada1.pickle", "wb"))


# In[85]:


start = time.time()
ada11 = ada1.fit(x_train, y_train)
end = time.time()
print(end - start)

#saving model into pickle
pickle.dump(ada11, open("ada11.pickle", "wb"))


# In[86]:


predicted_train_ada1 = ada11.predict(x_train)
predicted_test_ada1 = ada11.predict(x_test[cols])


# In[87]:


model1_train = model_evaluation(y_train, predicted_train_ada1, 'AdaBoost (train; version: 1)')
model1_test = model_evaluation(y_test, predicted_test_ada1, 'AdaBoost (test; version: 1)')
display_side_by_side(model1_train, model1_test)


# In[88]:


slm.confusion_matrix(y_train, predicted_train_ada1)


# In[89]:


slm.confusion_matrix(y_test, predicted_test_ada1)


# ### Models comparison and conclusions <a class="anchor" id="conclusions"></a>

# Below table concludes the comparison of three trained models including XGBoost, LightGBM and AdaBoost. It present the parameters optimized, values of accuracy and balanced accuracy on both training and test sample, training time and also parameter tuning time (in case of LightGBM and AdaBoost). It occured that the best in case of accuracy was XGBoost. Slightly worse result was obtained by LightGBM. AdaBoost occured to be the weakest model of all trained in this paper. Regarding training time, the advantage of the quickest training has LightGBM which together with his relatively high accuracy in this case is a winner.
# 
# ![title](conclusions.png)
