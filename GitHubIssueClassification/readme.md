### Title
Tutorial: Use ML.NET in a multiclass classification scenario to classify GitHub issues

## URL
https://docs.microsoft.com/en-us/dotnet/machine-learning/tutorials/github-issue-classification

##### IDE
Visual Studio Code

##### Training Data

##### Test Data

##### Things to learn
1. Understand the problem
1. Select the appropriate machine learning algorithm
1. Prepare your data
1. Transform the data
1. Train the model
1. Evaluate the model
1. Predict with the trained model
1. Deploy and Predict with a loaded model

##### Versions
This tutorial and related sample are currently using ML.NET version 0.11, however I am using 1.0.0-preview

##### Workflow phases
1. Understand the problem
1. Prepare you data
    - Load the data
    - Extract features (Transform you data)
1. Build and train
    - Train the model
    - Evaluate the model
1. Deploy Model
    - User the Model to predict

##### Understand the problem
The problem can be broken into the following parts:
- the issue title text
- the issue description text
- an area value for the model training data
- a predicted area value that you can evaluate and then

##### Select the appropriate mahine learning algorithm
Given that GitHub issues can be labeled in several areas (Area), we want to predict the Area of a new GitHub issue. This makes the classification learning algorithm is best suited for this scenario. A classification leaning algorithm that users data to determine the category, type or class of an item or row of data. Classification learning algorithms are generally used in either a binary or multiclass fashion. For this problem, we are going to use a Multiclass classification learning algorithm, since the issue category prediction can be one of many categories.

##### Create console app
dotnet new console -n GitHubIssueClassification
New-Item -Type Directory Data
New-Item -Type Directory Models
dotnet add package Microsoft.ML --version 1.0.0-preview

##### Prepare my data
Download the [issues_train.tsv](https://raw.githubusercontent.com/dotnet/samples/master/machine-learning/tutorials/GitHubIssueClassification/Data/issues_train.tsv) and the [issues_test.tsv](https://raw.githubusercontent.com/dotnet/samples/master/machine-learning/tutorials/GitHubIssueClassification/Data/issues_test.tsv) data sets and save them to the Data folder previously created. The first dataset trains the machine learning model and the second can be used to evaluate how accurate your model is.

##### Load the data
As the input and output of Transforms, a DataView is the fundamental data pipeline type, comparable to IEnumerable for LINQ. In ML.NET, data is similar to a SQL view. It is lazy evaluated, schematized, and heterogenous. The object is the first part of the pipeline, and loads the data. For this tutorial, it loads a dataset with issue titles descriptions, and corresponding area GitHub label. The DataView is used to create and train the model. The loading of the data seems to resemble how EF works in that you define model classes and then load the data into memory into a context using those model.

##### Extract Features and transform the data
Its important that pre-processing and cleaning of the data happen before before a dataset can be used effectively for machine learning since raw data is often noisy and unreliable, and may be missing values. Using data without these modeling tasks can produce misleading results. ML.NET's transform pipeliens compose a cutom transforms set that is applied to your data before training or testing. The transforms' pupose is data featurization. Since machinge learning alogrithms understand featurized data, the next step is to transform our data from a textual format into a numeric vector which our ML algorithms can understand. 

Once the model is trained and evaluated, the values in the Label column are considered as correct values to be predicted. 

Featurizing assignes different numeric keys to different values in each of the columns and is used by the ML algorithm.

By default, a ML algorithm processes only features from the Features column.

##### Choosing a learning algorithm
The SdcaMultiClassTrainer is appended to the pipline and accepts the featurized Title and Description ( Features ) and the Label input parameters to learn from the historic data. You also need to map the label to the value to return to its original readable state.

##### Train the model
The model, TransformerChain<TLastTransformer>, is trained based on the dataset that has been loaded and transformed. Once the estimator has been defined, you train your model using the Fit while providing the already loaded training data. This method returns a model to use for predictions. It trains the pipeline and returns a Transformer based on the DataView passed in. The experiment is not executedc until the Fit() method runs. While the model is a transformer that operates on many rows of data, a need for predictions on individual examples is a common production scenarion.