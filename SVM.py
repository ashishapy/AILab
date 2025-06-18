from sklearn.datasets import fetch_20newsgroups

newsgroup_train = fetch_20newsgroups(subset='train', shuffle=True)
newsgroup_test = fetch_20newsgroups(subset='test', shuffle=True)

i = 0
for cat in newsgroup_train.target_names:
    i = i + 1
    print(str(i) + "- " + str(cat))

# Printing single ost
print("\n".join(newsgroup_train.data[5].split("\n")[:10]))

# Extracting features
from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()
newsgroup_train_counts = count_vector.fit_transform(newsgroup_train.data)

# Calculating TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
newsgroup_train_tfidf = tfidf_transformer.fit_transform(newsgroup_train_counts)

# Training Support Vector Machines
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
import numpy as np

SVM_Classifier = Pipeline([('vectorizer', CountVectorizer()),('tfidf_matrix', TfidfTransformer()),('svm_classifier', SGDClassifier(loss='hinge', penalty='12', max_iter=100, alpha=0.01, random_state=42))])

SVM_Classifier = SVM_Classifier.fit(newsgroup_train.data, newsgroup_train.target)
Predicted_SVM = SVM_Classifier.predict(newsgroup_test.data)
print(np.mean(Predicted_SVM == newsgroup_test.target))

singleInputResult = SVM_Classifier.predict(newsgroup_test.data[14:15])
print(singleInputResult)
print(newsgroup_train.data[14:15])
