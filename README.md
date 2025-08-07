# Supreme-Court-Judgement-Prediction-with-Natural-Language-Processing-Techniques

**Introduction:**

In this project, I use Natural Language Processing (NLP) techniques and machine learning to predict the results of US Supreme Court cases. Using a dataset of 3303 Supreme Court cases, I turn the facts of the case into numerical data using TF-IDF (Term Frequency - Inverse Document Frequency) vectorization. I then train and evaluate the accuracy of a Linear SVC (Linear Support Vector Classifier) classification model. My goal is to develop an accurate model and understand how to effectively apply the techniques of NLP to the field of law. 

**Parts of code:**

Dropping irrelevant columns:
df = df.drop("index", axis=1)
df = df.drop("docket", axis=1)
df = df.drop("name", axis=1)
df = df.drop("ID", axis=1)
df = df.drop("href", axis=1)
df = df.dropna(axis=0, subset=['first_party', 'second_party'])
Because I am using the facts of the case to predict the outcome, I am deleting all other columns from my dataset, as they are irrelevant.

Making all the words lowercase
def lowerCase(facts):
    return facts.lower()


**Fine-Tuning:**


**Results and conclusion:**
