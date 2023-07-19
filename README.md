# Project
## Description
This is a project for income classification using Census data from [here](https://archive.ics.uci.edu/dataset/20/census+income) which contains 
32561 instances and 15 attributes; 6 numerical and 9 categorical attributes . The goal is to predict whether a person makes over 50K a year. The data is preprocessed and then splitted to train and test data. After that, the model is trained and evaluated using recall, precision and f1-score. Finally, the model is saved in a pickle file.

The purpose of this project is to build a pipeline using data version control (dvc) and aws s3 in addition to apllaying CI/CD using github actions, FastApi and heroku.

## Project Structure
```
├──Conf
│   ├── config.yaml
├── data
│   ├── census.csv.dvc
│   └── cleaned_data.csv.dvc
│   ├── test_data.csv.dvc
│   └── train_data.csv.dvc
├── model
│   ├── model.pkl
├── screenshots
│   ├── continous_integration.png
│   ├── continous_deployment.png
│   ├── live_get.png
│   ├── live_post.png
├── starter
│   ├── EDA
│   │   ├── EDA.ipynb
│   ├── ml
│   │   ├── model.py
│   │   ├── data.py
│   ├── dvc.lock
│   ├── dvc.yaml
│   ├── main.py
│   ├── prepare.py
│   ├── splitting_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
├── main.py
├── test_model_api.py
├── heroku-app-post.py
├── LICENSE
├── Procfile
├── README.md
├── requirements.txt
├── setup.py
├── snitycheck.py
├── model_card_template.md
```
## Installation
- Clone the repo
```
git clone https://github.com/Kyrillos-Botros/census-income-classification-using-dvc-github-actions-fastapi-heroku.git
```
- Install the dependencies
```
pip install -r requirements.txt
```
- Install aws cli from [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
- Configure aws cli, in terminal insert
```
aws configure
```
- Insert your AWS Access Key ID and AWS Secret Access Key
- Create a new IAM user with AmazonS3FullAccess permission
- Create S3 bucket

- Configure dvc, in terminal insert
```
git init
dvc init
dvc remote add -d <storage_name> s3://<your-bucket-name>
```
- insert your own storage name and your github repo url in conf/config.yaml file

- Rebuild the pipeline and Run dvc experiment
```
cd starter
python main.py
```
- Run the app
```
cd ..
uvicorn main:app --host 0.0.0.0 --port 5000
```
- Run the tests
```
pytest
```

## Dependencies
- jupyter==1.0.0
- jupyterlab==4.0.2
- numpy==1.24.4
- pandas==2.0.3
- scikit-learn==1.3.0
- pytest==7.4.0
- requests==2.31.0
- dvc==3.4.0
- dvclive==2.12.1
- fastapi==0.63.0
- uvicorn==0.22.0
- gunicorn==20.1.0




