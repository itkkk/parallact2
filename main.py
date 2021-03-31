import dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier  # best k = 3
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD, PCA

import tensorflow as tf
from tensorflow import keras


def execute_deep(x_train, x_test, y_train, y_test):
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)  # 0.25x0.8=0.2

    model = keras.Sequential([
        keras.layers.Dense(units=np.shape(x_train)[1], activation='relu'),
        keras.layers.Dense(units=(np.shape(x_train)[1]/2), activation='relu'),
        keras.layers.Dense(units=np.shape(y_train)[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        epochs=10,
        validation_data=(x_val, y_val),
        verbose=False
    )

    predictions = model.predict(x_test)

    discrete_pred = np.argmax(predictions, axis=1)
    discrete_y_test = np.argmax(y_test, axis=1)
    equality = tf.math.equal(discrete_pred, discrete_y_test)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    print(f"        train accuracy = {history.history['accuracy'][-1]}")
    print(f"        test accuracy = {accuracy}")


def execute_ml(x_train, x_test, y_train, y_test):
    # model = OneVsRestClassifier(tree.DecisionTreeClassifier()).fit(x_train, y_train)
    # model = OneVsRestClassifier(SVC(gamma=100, C=1000)).fit(x_train, y_train)
    # predictions_train = model.predict(x_train)
    # predictions_test = model.predict(x_test)

    pca = PCA(n_components=3)

    tree = OneVsRestClassifier(DecisionTreeClassifier())

    pipe = Pipeline(steps=[('pca', pca), ('tree', tree)])
    pipe.fit(x_train, y_train)
    predictions_train = pipe.predict(x_train)
    predictions_test = pipe.predict(x_test)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    print(f"        Train accuracy: {accuracy_score(y_train, predictions_train)}")
    print(f"        Test accuracy: {accuracy_score(y_test, predictions_test)}")


def bpi13():
    print("bpi13 ")
    bpi13_dataset = dataset.load_bpi13(3, verbose=True, save_to_disk=True, filename="bpi13")
    features, targets, features_name, targets_name = dataset.create_matrices(bpi13_dataset, save_to_disk=True,
                                                                             filename="bpi13")
    print("Dataset loaded and preprocessed")

    print("    bpi13 ml:")
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=3)
    execute_ml(x_train, x_test, y_train, y_test)

    print("    bpi13 mlp:")
    execute_deep(x_train, x_test, y_train, y_test)

    print("")


def bpi12():
    print("bpi12")
    bpi12_dataset = dataset.load_generic_dataset("datasets/vinc/bpi_12_w.csv", save_to_disk=True, verbose=True,
                                                 filename="bpi12")
    features, targets, features_name, targets_name = dataset.create_matrices(bpi12_dataset, save_to_disk=True,
                                                                             filename="bpi12")
    print("Dataset loaded and preprocessed")

    print("    bpi12 ml:")
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=3)
    execute_ml(x_train, x_test, y_train, y_test)

    print("    bpi12 mlp:")
    execute_deep(x_train, x_test, y_train, y_test)

    print("")


def helpdesk():
    print("helpdesk")
    helpdesk_dataset = dataset.load_generic_dataset("datasets/vinc/helpdesk.csv", save_to_disk=True, verbose=True,
                                                    filename="helpdesk")

    features, targets, features_name, targets_name = dataset.create_matrices(helpdesk_dataset, save_to_disk=True,
                                                                             filename="helpdesk")
    print("Dataset loaded and preprocessed")

    print("    helpdesk ml:")
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=3)
    execute_ml(x_train, x_test, y_train, y_test)

    print("    helpdesk mlp:")
    execute_deep(x_train, x_test, y_train, y_test)

    print("")


if __name__ == '__main__':
    bpi13()
    bpi12()
    helpdesk()
