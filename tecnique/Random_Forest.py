from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split


def run(dataSet):
    X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target) 
    text_clf = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', RandomForestClassifier(n_estimators=100)),
                        ])
    text_clf.fit(X_train, y_train)
    predicted = text_clf.predict(X_test)
    return(metrics.classification_report(y_test, predicted))