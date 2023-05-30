# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
<p>Prediction task is to determine whether a person makes over 50K a year.</p>
<p>GradientBoostingClassifier was used with optimized hyperparameters on scikit-learn</p>
<p>Hyperparameters tuning was realized using GridSearchCV.</p>
<p>Optimal parameters used are:</p>

- learning_rate: 0.1
- max_depth: 5
- min_samples_split: 100
- n_estimators: 10

<p>Model is saved as a pickle file in the model folder. All training steps and metrics are logged in the file log file at the <em>/src/ml/log</em> directory.</p>

## Intended Use
<p>This model can be used to predict the salary level of an individual based off a handful of attributes. The usage is meant for students, academics or research purpose.</p>

## Training Data
<p>The Census Income Dataset was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income) as a csv file.</p>
<p>The original data set has 32.561 rows and 15 columns composed of the target label "salary", 8 categorical features and 6 numerical features.</p>
<p>Details on each of the features ae available at the UCI link above.</p>
<p>Target label "salary" has two classes ('<=50K', '>50K') and shows class imbalance with a ratio of circa 75% / 25%.</p>
<p>A simple data cleansing was performed on the original dataset to remove leading and trailing whitespaces. See <em>/eda/Census-EDA.ipynb</em> notebook for data exploration and cleansing step.</p>

<p>A 80-20 split was used to break this dataset into a train and test set. Stratification on target label "salary" was applied.</p>
<p>To use the data for training a One Hot Encoder was used on the categorical features and a label binarizer was used on the target.</p>

## Evaluation Data
<p>The 20% of the dataset was set aside for model evaluation. Transformation was applied on the categorical features and the target label respectively using the One Hot Encoder and label binarizer fitted on the train set.</p>

## Metrics
<p>The classification performance is evaluated using precision, recall and fbeta metrics.
The confusion matrix is also calculated.</p>

<p>The model achieves below scores using the test set:</p>

- precision:0.781
- recall:0.638
- fbeta:0.702
- Confusion matrix:\
[ 4258 | 269 ]\
[  544 | 957 ]

## Ethical Considerations
<p>The dataset should <strong>NOT</strong> be considered as a fair representation of the salary distribution and should not be used to assume salary level of certain population categories.</p>


## Caveats and Recommendations
<p>Extraction was done from the 1994 Census database. The dataset is a outdated sample, unbalanced and cannot adequately be used as a statistical representation of the population. It is recommended to use the dataset for training purpose on ML classification or related problems but not for full productions environments.</p>
