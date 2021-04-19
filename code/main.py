import plotly
from plotly import graph_objs
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier as XGBoostClassifier
import random

from helper_functions import test_classifier, cv
from data_cleanup import CleanupData
from data_word2vec import Word2Vec
from data_model import ModelData


"""
DATA PREPROCESSING 
"""
print("Data preprocessing...")
td = ModelData()
td.initialize("data/data_mapbook.csv")
td.build_features()
td.cleanup(CleanupData())
td.tokenize()
td.stem()
td.build_wordlist()
word2vec = Word2Vec()
word2vec.load("./glove.twitter.27B.200d.txt")
td.build_final_model(word2vec)

data_model = td.data_model
# REMOVE CONTENT ID AS IT DOES NOT GIVE ANY INFORMATION
data_model.drop("original_id", axis=1, inplace=True)


"""
PLOT DATA DISTRIBUTION AFTER PREPROCESSING
Includes less data as some content is removed in the process
If we wanted to plot the same distribution as was shown in the report, we need to plot it right after td.initialize
"""
df = td.processed_data
negative = len(df[df["class"] == "Content Discussion"])
positive = len(df[df["class"] == "Greeting"])
neutral = len(df[df["class"] == "Logistics"])
forth = len(df[df["class"] == "Instruction Question"])
fifth = len(df[df["class"] == "Assignment Question"])
six = len(df[df["class"] == "General Comment"])
seven = len(df[df["class"] == "Incomplete/typo"])
eight = len(df[df["class"] == "Feedback"])
nine = len(df[df["class"] == "Discussion Wrap-up"])
ten = len(df[df["class"] == "Outside Material"])
eleven = len(df[df["class"] == "Opening Statement"])
twelve = len(df[df["class"] == "General Question"])
thirteen = len(df[df["class"] == "Content Question"])
fourteen = len(df[df["class"] == "Emoticon/Non-verbal"])
fifteen = len(df[df["class"] == "Assignment Instructions"])
sixteen = len(df[df["class"] == "Response"])
#, "General Question", "Content Question"
dist = [
    graph_objs.Bar(
        x=["Content Discussion", "Logistics", "Greeting", "Instruction Question", "Assignment Question", "General Comment", "Incomplete/typo", "Feedback", "Discussion Wrap-up", "Outside Material", "Opening Statement", "General Question", "Content Question", "Emoticon/Non-verbal", "Assignment Instructions", "Response"],
        y=[negative, neutral, positive, forth, fifth, six, seven, eight, ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen]
)]
plotly.offline.plot({"data": dist, "layout": graph_objs.Layout(title="Distribution in dataset")})


"""
BUILDING A MODEL AND TESTING THE PERFORMANCE
"""
print("Building models and testing the performance...")
seed = 42
random.seed(seed)

# NAIVE BAYES
X_train, X_test, y_train, y_test = train_test_split(data_model.iloc[:, 1:], data_model.iloc[:, 0], train_size=0.7, stratify=data_model.iloc[:, 0], random_state=seed)
X_train = X_train.fillna(0)
precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, BernoulliNB())
nb_acc = cv(BernoulliNB(),data_model.iloc[:, 1:], data_model.iloc[:, 0])

# RANDOM FOREST
X_train, X_test, y_train, y_test = train_test_split(data_model.iloc[:, 1:], data_model.iloc[:, 0], train_size=0.7, stratify=data_model.iloc[:, 0], random_state=seed)
X_train = X_train.fillna(0)
precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, RandomForestClassifier(n_estimators=403, n_jobs=-1, random_state=seed))
rf_acc = cv(RandomForestClassifier(n_estimators=403, n_jobs=-1, random_state=seed), data_model.iloc[:, 1:], data_model.iloc[:, 0])

# XGBOOST
#X_train, X_test, y_train, y_test = train_test_split(data_model.iloc[:, 1:], data_model.iloc[:, 0], train_size=0.7, stratify=data_model.iloc[:, 0], random_state=seed)
#precision, recall, accuracy, f1 = test_classifier(X_train, y_train, X_test, y_test, XGBoostClassifier(seed=seed))
#xgb_acc = cv(XGBoostClassifier(seed=seed),data_model.iloc[:, 1:], data_model.iloc[:, 0])
