from sklearn.neighbors._nearest_centroid import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

def run(dataSet):
    X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target) 
    text_clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', NearestCentroid()),
                        ])
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    return(metrics.classification_report(y_test, predicted))