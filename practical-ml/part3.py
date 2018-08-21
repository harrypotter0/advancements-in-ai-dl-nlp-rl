#import dependencies
import os

DATA_DIR ="./txt_sentoken"
classes = ['pos', 'neg']

#vars to store data
train_data = []
train_labels = []
test_data = []
test_labels = []

for c in classes:
    data_dir = os.path.join(DATA_DIR, c)
    for fname in os.listdir(data_dir):
        with open(os.path.join(data_dir, fname), 'r') as f:
            content = f.read()
            if fname.startswith('cv9'):
                test_data.append(content)
                test_labels.append(c)
            else:
                train_data.append(content)
                train_labels.append(c)

type(train_data)
# print(len(train_data), len(test_data))
# print(train_data[3])
# print(train_labels[3])

X = ["That was an awesome movie","I really appreciate your work in this movie"]
#import count vectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
data = vectorizer.fit_transform(X)
print(data)

#get the vocabulary
vectorizer.vocabulary_ # "I" was removed because it is one of the stopwords, that is, that doesnt contain important significance

#transform sparce matrix into an array
data.toarray()

#print feature names
vectorizer.get_feature_names()

# TF-IDF
#we will use tf-idf for our sentiment analysis task
from sklearn.feature_extraction.text import TfidfVectorizer
import random
random.shuffle(train_data)
random.shuffle(test_data)

vect = TfidfVectorizer(min_df=5,max_df = 0.8,sublinear_tf=True,use_idf=True)
train_data_processed = vect.fit_transform(train_data)
test_data_processed = vect.transform(test_data)

from sklearn.preprocessing import LabelEncoder
random.shuffle(train_labels)
random.shuffle(test_labels)

le = LabelEncoder()
train_labels_processed = le.fit_transform(train_labels)
test_labels_processed = le.transform(test_labels)

train_labels_processed[:33] 
le.classes_  #0 for neg and 1 for pos

# Train a model
from sklearn.svm import SVC
model = SVC(C=10, kernel="rbf", random_state=42)
#train
model.fit(train_data_processed, train_labels_processed)
model.score(test_data_processed, test_labels_processed)

x7 = ["that movie was awesome", "that movie was so bad", "i love watching that movie","that is painful"]
y7 = [1,0,1,0]
x7_p = vect.transform(x7)

print(y7)
model.predict(x7_p)

from sklearn.ensemble import RandomForestClassifier
m = RandomForestClassifier(n_estimators=500, random_state=42,max_depth=4)
m.fit(train_data_processed, train_labels_processed)
m.score(test_data_processed, test_labels_processed)

x = ["That movie was bad but, I loved it ","that movie was so bad","the sandwich is worth eating","that movie was awesome","i hate it"]
x_p = vect.transform(x) #perform vectorization

m.predict(x_p) #1 means positive
