import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV

nltk.download("stopwords")

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

print("Dividing data into train data and testing data.....")

x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.22, random_state=25)

print("Implementing linear SVC.....")

model3 = LinearSVC(class_weight='balanced', max_iter=5000, C=10, loss="squared_hinge", dual=True, penalty="l2", tol=1e-3, intercept_scaling=5)

print("Determining win probability.....")

calibrated_svm = CalibratedClassifierCV(model3, cv=5)
calibrated_svm.fit(x_train, y_train)
probs = calibrated_svm.predict_proba(x_test)
win_probs = probs[:, 1]
pred_labels = (win_probs >= 0.5).astype(int)
acc = accuracy_score(y_test, pred_labels)
print("Accuracy: "+str(acc*100)+"%")
