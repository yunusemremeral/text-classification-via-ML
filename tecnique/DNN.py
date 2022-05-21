from keras.layers import  Dropout, Dense
from keras.models import Sequential
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

def TFIDF(X_train, X_test,MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with",str(np.array(X_train).shape[1]),"features")
    return (X_train,X_test)


def Build_Model_DNN_Text(shape, nClasses, dropout=0.5):
    """
    buildModel_DNN_Tex(shape, nClasses,dropout)
    Build Deep neural networks Model for text classification
    Shape is input feature space
    nClasses is number of classes
    """
    model = Sequential()
    node = 512 # number of nodes
    nLayers = 4 # number of  hidden layer

    model.add(Dense(node,input_dim=shape,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(node,input_dim=node,activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def run(dataSet):
    X_train, X_test, y_train, y_test = train_test_split(dataSet.data, dataSet.target) 

    X_train_tfidf,X_test_tfidf = TFIDF(X_train,X_test)


    model_DNN = Build_Model_DNN_Text(X_train_tfidf.shape[1], 20)
    model_DNN.summary()
    model_DNN.fit(X_train_tfidf, y_train,
                                validation_data=(X_test_tfidf, y_test),
                                epochs=10,
                                batch_size=128,
                                verbose=2)

    predicted = model_DNN.predict_classes(X_test_tfidf)

    return(metrics.classification_report(y_test, predicted))