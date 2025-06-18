# Getting the dataset
from sklearn.datasets import fetch_20newsgroups

from SVM import count_vector

# Getting the training and test data subsets
newsgroup_train = fetch_20newsgroups(subset='train', shuffle=True)
newsgroup_test = fetch_20newsgroups(subset='test', shuffle=True)

# Checking categories Names
i = 0
for cat in newsgroup_train.target_names:
    i = i + 1
    print(str(i) + ". " + str(cat))

# Printing a single ost
print("\n".join(newsgroup_train.data[5].split("\n")[:10]))

# Extracting feature
from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()
newsgroup_train_counts = count_vector.fit_transform(newsgroup_train.data)

# Calculating TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
newsgroup_train_tfidf = tfidf_transformer.fit_transform(newsgroup_train_counts)

# Training Naive Bayes
from sklearn.naive_bayes import MultinomialNB
