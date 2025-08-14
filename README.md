## Introduction:

In this project, I use Natural Language Processing (NLP) techniques and machine learning to predict the outcomes of civil law cases. I turn the facts of the case into numerical data using TF-IDF (Term Frequency - Inverse Document Frequency) vectorization. Then, using 3302 US Supreme Court cases, I train a Linear SVC classification model to determine the result of a case, and how far from the decision boundary the datapoint is, and I train a Calibrated Classification Model to translate distances from the decision point into win probabilities. I then use the Calibrated Classification Model to predict the probability of a first party victory of the case inputted by the user. The applications of this technology are twofold. First, it helps law firms. By showing them the probability of winning a case, this technology can help law firms decide whether to take a case or not. If law firms know that the probability of winning a case is low, then they might decide that pursuing that case is not worth it, and if they know that the probability of winning a case is high, they might decide that a case is worth pursuing. Second, it helps potential plaintiffs in civil cases. Before they initiate a case, they can see the probability of victory. This can help them decide whether or not to pursue it. If the probability of winning a case is low, they may decide not to pursue it, which would reduce or eliminate costly legal expenditures.

## Parts of code:

### Dropping irrelevant columns: 

```
df = pd.read_csv("justice.csv")

print("Dropping irrelevant columns.....")
df = df.drop("index", axis=1)
df = df.drop("docket", axis=1)
df = df.drop("name", axis=1)
df = df.drop("ID", axis=1)
df = df.drop("href", axis=1)
df = df.drop("term", axis=1)
df = df.drop("first_party", axis=1)
df = df.drop("second_party", axis=1)
df = df.drop("facts_len", axis=1)
df = df.drop("majority_vote", axis=1)
df = df.drop("minority_vote", axis=1)
df = df.drop("decision_type", axis=1)
df = df.drop("disposition", axis=1)
df = df.drop("issue_area", axis=1)
```

Because I am using the facts of the case to predict the outcome, I am deleting all other columns from my dataset, as they are irrelevant.

### Inputted user case facts:
```
inputFacts = input("Enter the facts of your case: ")
new_row = pd.DataFrame({'facts': ["<p>"+inputFacts+"</p>"]})
df = pd.concat([df, new_row], ignore_index=True)
```

This code allows the user to input the facts of their case, which will be processed and converted to numerical data using NLP, so that its outcome may be predicted

### Making all the words lowercase:

```
def lowerCase(facts):
    return facts.lower()

df["facts"] = df["facts"].apply(lowerCase)
```
Suppose two words are the same, but the first letter of one is capitalized, while the first letter of the other isn’t. The meaning of these words are the same, but the computer interprets them as two different words. To rectify this, all words are converted to lowercase.

### Removing HTML tags from the facts:

```
def removeTags(facts):
    newFacts = facts[3:-4]
    return newFacts
df["facts"] = df["facts"].apply(removeTags)
```

The dataset is formatted in a strange manner. There are HTML tags at the beginning and end of each description in the facts column, such as <p> and </p>. This code simply removes those tags.

### Removing stop words:
```
print("Removing stop words.....")
stop_words = set(nltk.corpus.stopwords.words("english"))
def removeStopWords(facts):

    filteredList = []

    factsList = facts.split(" ")

    for word in factsList:

        if word.isalpha() and word.casefold() not in stop_words:
            filteredList.append(word)
    return filteredList
df["facts"] = df["facts"].apply(removeStopWords)
```
Stop words are words that are used as transitions or conjunctions, and do not add meaning to the text. Examples include “in,” “is,” and “an.” Because they lack importance for the purposes of predicting case outcomes, they would just introduce noise and dilute the importance of meaningful words. To rectify this problem, this code removes all stop words from the facts of each case.

### Stemming words:
```
print("Stemming words.....")
stemmer = nltk.stem.SnowballStemmer("english")

def stemmingWords(facts2):
    stemmed_words = [stemmer.stem(fact) for fact in facts2]
    formated = " ".join(stemmed_words)
    return formated

df["facts"] = df["facts"].apply(stemmingWords)
```
Words like “suing” and “sue” have the same basic meaning, but are conjugated differently. Stemming is a technique used to reduce all words to a common base form, which allows the computer to treat all related words as the same, focusing on their core meanings, instead of their conjugations.

###Quantifying outcomes:
```
print("Making outcome numerical.....")
def outcomeQuantifying(outcome):
    if outcome:
        return 0
    else:
        return 1
df["first_party_winner"] = df["first_party_winner"].apply(outcomeQuantifying)
```
The computer only understands numbers, not booleans like “True” or “False.” This function turns the True and False values in the first_party_winner column into 0’s and 1’s.

### Assigning a value to each word using TF-IDF vectorizer:
```
print("Assigning value to each word using TFIDF vectorizer.....")

trainingData = df.iloc[:-1]
testData = df.iloc[[-1]]
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2, max_df=0.75, stop_words="english")

trainMatrix = vectorizer.fit_transform(trainingData["facts"])
testMatrix = vectorizer.transform(testData["facts"])

train_dense = trainMatrix.toarray()
test_dense = testMatrix.toarray()
```
First, I separate the dataset into training data and testing data. I use the original dataset of Supreme Court cases as my training data, and I use the user-inputted dataset as my testing data. 

Then, I create a Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer. Computers only understand numbers, not text, so the TFIDF vectorizer turns the facts of each case into a vector. It goes through each word of each case and assigns a value of importance to each word, using two criteria. Firstly, it determines how often that word appears in the case (term frequency). If a word appears multiple times in a case, it is important to that case, but if a word just appears once in a case, it is probably insignificant. Secondly, it determines how rare the word is throughout the entire dataset (inverse document frequency). Some words are common throughout all cases, like “court” and “said.” If it is that common, it is probably not significant and not useful for distinguishing between them. But words that are case specific, like “anti-trust” or “first amendment” are probably more important, so they are given a higher score.

ngram_range=(1, 2) makes it so that the vectorizer assigns values to single words as well as pairs of words. This is because sometimes pairs of two words have specific meanings that aren’t captured by separating the two words. For instance, “first amendment” means something specific that isn’t reflected if “first” and “amendment” are recorded independently.

Next, I fit the vectorizer with the training data. When the vectorizer is fit, it finds all the words and pairs of words in the training data, and keeps the 10000 most frequent words and pairs. It then calculates the inverse document frequency of those words. 

The training data and testing data are then transformed into matrices, where each row is a case, and the columns are the 10000 words or pairs of words found when fitting the vectorizer. The values are the calculated TF-IDF values for the words in each case.

The matrices are then converted from sparse matrices into dense matrices. Sparse matrices are matrices where values are mostly zero. To save memory, the zero values are not stored. By default, the matrices formed from the training data and testing data will be sparse. Since the vectorizer keeps 10000 words, but each individual case only has a few of them, most of the values in each row will be 0. These matrices are converted to dense matrices, where all the values, including the zero values, are stored.

### Using SMOTE:
```
smote = SMOTE(random_state=39)
x_train, y_train = smote.fit_resample(train_dense, trainingData["first_party_winner"])
```
SMOTE stands for Synthetic Minority Over-sampling Technique. It is a technique used in unbalanced datasets, where one value is far more present than another. In this dataset, 2156 cases resulted in a first party win, while 1148 resulted in a first party loss. In these situations, when trying to predict if the first party won, machine learning models will often just select the majority case, instead of examining the facts of the case. SMOTE rectifies this by generating synthetic cases where the first party lost, by interpolating between existing cases where the first party lost. This ensures that the dataset is more balanced, so the machine learning model won’t just pick the majority case.

### Determining the win probability of the user’s case:
```
print("Implementing linear SVC.....")

model3 = LinearSVC(class_weight='balanced', max_iter=5000, C=10, loss="squared_hinge", dual=True, penalty="l2", tol=1e-3, intercept_scaling=5)

print("Determining win probability.....")
calibrated_svm = CalibratedClassifierCV(model3, cv=5)
calibrated_svm.fit(x_train, y_train)
probs = calibrated_svm.predict_proba(test_dense)
print("Win probability: "+str(probs[0,0]*100)+"%")
```
First, I create a Linear SVC model. This model predicts if the case will be won or lost, but does not give the probability of that outcome. However, this model has a feature where it gives the strength of its prediction. The Linear SVC model works by plotting all the data points on a multidimensional vector space, and drawing a decision boundary through the vector space. The points on one side of the decision boundary have a first party win, and the points on the other side of the line have a first party loss. The distance of the point from the decision boundary correlates to the strength of the prediction. The Calibrated Classifier takes the strength of the prediction, and converts it into a probability. I train the calibrated classifier and the Linear SVC model with the training data, and I use the models to predict the probability that the user inputted case will result in a first party victory. I then output that probability to the user.

## Evaluating accuracy:
In a previous iteration of this project, instead of using the dataset as training data and the user input as testing data, the dataset was split into training and testing data. This is included for the purposes of evaluating accuracy. After implementing SMOTE:

```
print("Dividing data into train data and testing data.....")

x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.22, random_state=25)
print("\n")

print("Implementing linear SVC.....")

model3 = LinearSVC(class_weight='balanced', max_iter=5000, C=10, loss="squared_hinge", dual=True, penalty="l2", tol=1e-3, intercept_scaling=5)

print("Determining win probability.....")

calibrated_svm = CalibratedClassifierCV(model3, cv=5)
calibrated_svm.fit(x_train, y_train)
probs = calibrated_svm.predict_proba(x_test)
win_probs = probs[:, 1]
pred_labels = (win_probs >= 0.5).astype(int)
acc = accuracy_score(y_test, pred_labels)
print("Accuracy:", acc)
```

#### Output:
```
Dividing data into train data and testing data.....
Implementing linear SVC.....
Determining win probability.....
Accuracy: 79.21940928270043%
```
22% of the data is used as testing data, and the rest is training data. Like in the final version, I train the Calibrated Classifier with the training data, the model predicts the chances of a first party victory in the testing data. However, I create an additional array that just records if each testing case has a first party win or not. For each case in the testing data, if the chances of victory are greater than 50 percent, I assign it a value of 1, and if the chances of victory are lower than 50 percent, I assign it a value of 0. I then compare the predicted outcome of the testing cases with the outcome recorded in the dataset, and output how accurate my model was at predicting this. After fine-tuning the model, it was able to achieve an accuracy of 79.22%.
