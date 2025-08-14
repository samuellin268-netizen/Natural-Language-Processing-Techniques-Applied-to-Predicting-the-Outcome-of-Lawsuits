import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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
df = df.drop("term", axis=1)
df = df.drop("first_party", axis=1)
df = df.drop("second_party", axis=1)
df = df.drop("facts_len", axis=1)
df = df.drop("majority_vote", axis=1)
df = df.drop("minority_vote", axis=1)
df = df.drop("decision_type", axis=1)
df = df.drop("disposition", axis=1)
df = df.drop("issue_area", axis=1)

inputFacts = input("Enter the facts of your case: ")
new_row = pd.DataFrame({'facts': ["<p>"+inputFacts+"</p>"]})
df = pd.concat([df, new_row], ignore_index=True)

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

trainingData = df.iloc[:-1]
testData = df.iloc[[-1]]
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, min_df=2, max_df=0.75, stop_words="english")

trainMatrix = vectorizer.fit_transform(trainingData["facts"])
testMatrix = vectorizer.transform(testData["facts"])

train_dense = trainMatrix.toarray()
test_dense = testMatrix.toarray()

smote = SMOTE(random_state=39)
x_train, y_train = smote.fit_resample(train_dense, trainingData["first_party_winner"])

print("Implementing linear SVC.....")

model3 = LinearSVC(class_weight='balanced', max_iter=5000, C=10, loss="squared_hinge", dual=True, penalty="l2", tol=1e-3, intercept_scaling=5)

print("Determining win probability.....")
calibrated_svm = CalibratedClassifierCV(model3, cv=5)
calibrated_svm.fit(x_train, y_train)
probs = calibrated_svm.predict_proba(test_dense)
print("Win probability: "+str(probs[0,0]*100)+"%")
