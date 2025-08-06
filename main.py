import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from imblearn.over_sampling import SMOTE

df = pd.read_csv("justice.csv")

print("Dropping irrelevant columns.....")
df = df.drop("index", axis=1)
df = df.drop("docket", axis=1)
df = df.drop("name", axis=1)
df = df.drop("ID", axis=1)
df = df.drop("href", axis=1)
df = df.dropna(axis=0, subset=['first_party', 'second_party'])


print("Making all words lowercase.....")
def lowerCase(facts):
    return facts.lower()

df["facts"] = df["facts"].apply(lowerCase)

print("Removing HTML tags from facts.....")
def removeTags(facts):
    newFacts = facts[3:-4]
    return newFacts
df["facts"] = df["facts"].apply(removeTags)

print("Removing punctuation and stop words.....")
stop_words = set(nltk.corpus.stopwords.words("english"))
def removeStopWords(facts):

    # empty list of filtered words
    filteredList = []

    # turning string of facts into list or words in facts
    factsList = facts.split(" ")

    # for each word in facts
    for word in factsList:
        word.strip(string.punctuation)
        # check that word isn't a stop word
        if word.isalpha() and word.casefold() not in stop_words:
            filteredList.append(word)
    return filteredList


df["facts"] = df["facts"].apply(removeStopWords)

print("Stemming words.....")
stemmer = nltk.stem.SnowballStemmer("english")

def stemmingWords(facts2):
    stemmed_words = [stemmer.stem(fact) for fact in facts2]
    formated = " ".join(stemmed_words)
    return formated

df["facts"] = df["facts"].apply(stemmingWords)

print("Making outcome numerical.....")
def outcomeQuantifying(outcome):
    if outcome:
        return 0
    else:
        return 1
df["first_party_winner"] = df["first_party_winner"].apply(outcomeQuantifying)

print("Assigning value to each word using TFIDF vectorizer.....")

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2, max_df=0.75, stop_words="english")
X = vectorizer.fit_transform(df["facts"])
X_dense = X.toarray()
smote = SMOTE(random_state=39)
X_resampled, y_resampled = smote.fit_resample(X_dense, df["first_party_winner"])

print("Dividing data into train data and testing data")

x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.22, random_state=25)
print("\n")

print("Implementing linear SVC.....")

model3 = LinearSVC(class_weight='balanced', max_iter=5000, C=10, loss="squared_hinge", dual=True, penalty="l2", tol=1e-3, intercept_scaling=5)
model3.fit(x_train, y_train)
y_pred3 = model3.predict(x_test)
percent3 = accuracy_score(y_test, y_pred3) * 100
print("Linear SVC score:", str(percent3)+"%")

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(x_train, y_train)
baseline = dummy.score(x_test, y_test)
print("Baseline accuracy:", str(baseline*100)+"%")