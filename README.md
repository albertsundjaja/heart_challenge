# Heart Disease Detection

## Project Overview

This is a classification challenge provided by DrivenData
https://www.drivendata.org/competitions/54/machine-learning-with-a-heart/

The project objective is to classify whether a patient has heart disease given the patient's various measurement

## Data Overview

The dataset is relatively small with only 180 rows and 14 columns for training data.

Overall, the data is already well structured, there is no tidiness issue. There is also no missing value and the measurements seems to be accurate, so there is no data quality issue as well.

## Data Preprocessing

As the data is already high quality, the only minor modification to do with regards to quality are the resulting pandas' dataframe encoding and data types.

### Label Encoding

the feature `thal` is of string datatype, as such we will encode it so that it has numerical representation for easier preprocessing

### One Hot Encoding

Categorical columns are `chest_pain_type, fasting_blood_sugar_gt_120_mg_per_dl, resting_ekg_results, sex, exercise_induced_angina, slope_of_peak_exercise_st_segment, num_major_vessels`, they will be transformed using One Hot Encoding. It is important to note that while `slope_of_peak_exercise_st_segment, num_major_vessels` are numerical in nature, it gives better performance when they are treated as categorical

### Min Max Scaling

Numerical columns are `resting_blood_pressure, serum_cholesterol_mg_per_dl, oldpeak_eq_st_depression, age, max_heart_rate_achieved`, they will be scaled using a `MinMaxScaler`

## Classifier

The classifier used in this project is `XGBoost` 
source can be found here https://github.com/dmlc/xgboost

the parameters for the model are as below
```
xg = xgb.XGBClassifier(n_estimators=100,
                        tree_method='gpu_hist',
                        n_jobs=4,
                        n_gpus=1,
                        max_depth=9,
                        seed=1,
                        learning_rate=0.05,
                        subsample=0.5,
                        colsample_bytree=0.5)
```

## Result

The model achieved a 5-fold cross validation log-loss of `0.4121` (average)

And, the prediction achieved a log-loss score of `0.35262`