# Model Card


## Model Details
It's a compound of models which are
- "encoder": it's  a one-hot encoder for categorical columns
- "lb" it's a label binarize for target columns
- "model": Random forest model from sklearn library which is trained with the parameters exist in "conf/config.yaml" file.

## Intended Use
This model will be used  on census dataset to classify the salary if it more or less than 50k

## Training Data
It's 80 % of the whole data exists in data folder under the name "train-data.csv" and contains 32561 rows, beside contains 
6 numerical columns which are ['age', 'fnlgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
and 9 categorical columns which are ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']

## Evaluation Data
It's 20 % of the whole data exists in data folder under the name "test-data.csv" and contains 5603 rows, beside contains 
6 numerical columns which are ['age', 'fnlgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
and 9 categorical columns which are ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'salary']

## Metrics
The model was evaluated using recall, precision and f1-score which their values are 0.716, 0.563, 0.6301

## Ethical Considerations
- **Fairness and Bias**: As the model is intended to classify salary based on various features, it's crucial to ensure that the predictions are fair and not biased against any specific group.
- **Privacy Protection**: The model might potentially handle sensitive information related to individuals' income and demographic characteristics. It's important to take appropriate measures to protect user privacy and comply with relevant data protection regulations. 
- **Security**: The model might be vulnerable to adversarial attacks, which could be exploited by malicious agents to perform unauthorized operations or access private data. It's important to take appropriate measures to ensure model security and prevent such attacks.


## Caveats and Recommendations
It's recommended keeping all the folders and paths the same and change only in conf/config.yaml file with different parameters.
In addition to that, trying different models and hyperparameter tunning
