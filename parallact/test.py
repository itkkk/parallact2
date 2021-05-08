import parallact
import numpy as np
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras


def execute_deep(x_train, x_test, y_train, y_test):
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)  # 0.25x0.8=0.2

    '''
    model = keras.Sequential([
        keras.layers.Dense(units=np.shape(x_train)[1], activation='relu'),
        keras.layers.Dense(units=(np.shape(x_train)[1]/2), activation='relu'),
        keras.layers.Dense(units=np.shape(y_train)[1], activation='softmax')
    ])
    '''

    model = keras.Sequential([
        keras.layers.InputLayer(np.shape(x_train)[1]),
        keras.layers.Dense(units=np.shape(x_train)[1]/2, activation='relu', kernel_initializer='glorot_uniform'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(units=np.shape(y_train)[1], activation='softmax', kernel_initializer='glorot_uniform')
    ])

    model.summary()

    #optimizer = keras.optimizers.Adam(learning_rate=0.005)
    optimizer = keras.optimizers.Adam(learning_rate=0.009, decay=0.0001)
    model.compile(optimizer=optimizer, loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    history = model.fit(
        x_train, y_train,
        epochs=100,
        validation_data=(x_val, y_val),
        verbose=True,
        batch_size=256,
    )

    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()

    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()

    predictions = model.predict(x_test)

    discrete_pred = np.argmax(predictions, axis=1)
    discrete_y_test = np.argmax(y_test, axis=1)
    equality = tf.math.equal(discrete_pred, discrete_y_test)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    print(f"        neural network dimensions: input layer: {np.shape(x_train)[1]}, hidden layer: {np.shape(x_train)[1]/2}, output layer: {np.shape(y_train)[1]}")
    print(f"        train accuracy = {history.history['accuracy'][-1]}")
    print(f"        test accuracy = {accuracy}")


def execute_lstm(x_train, x_test, y_train, y_test):
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=1)  # 0.25x0.8=0.2

    model = keras.Sequential([
        keras.layers.InputLayer(np.shape(x_train)[1:]),
        keras.layers.LSTM(100, return_sequences=True),
        keras.layers.BatchNormalization(),
        keras.layers.LSTM(100, return_sequences=True),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(np.shape(y_train)[1], activation='softmax')
    ])

    model.summary()

    optimizer = keras.optimizers.Adam(learning_rate=0.002)
    model.compile(optimizer=optimizer, loss=tf.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    history = model.fit(
        x_train, y_train,
        epochs=500,
        validation_data=(x_val, y_val),
        verbose=True,
        batch_size=256,
    )

    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()

    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()

    predictions = model.predict(x_test)

    discrete_pred = np.argmax(predictions, axis=1)
    discrete_y_test = np.argmax(y_test, axis=1)
    equality = tf.math.equal(discrete_pred, discrete_y_test)
    accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

    print(f"        neural network dimensions: input layer: {np.shape(x_train)[1]}, hidden layer: {np.shape(x_train)[1] / 2}, output layer: {np.shape(y_train)[1]}")
    print(f"        train accuracy = {history.history['accuracy'][-1]}")
    print(f"        test accuracy = {accuracy}")



    '''
    l1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(main_input) # the shared layer
    b1 = BatchNormalization()(l1)
    l2_1 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in activity prediction
    b2_1 = BatchNormalization()(l2_1)
    l2_2 = LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(b1) # the layer specialized in time prediction
    b2_2 = BatchNormalization()(l2_2)
    act_output = Dense(len(target_chars), activation='softmax', kernel_initializer='glorot_uniform', name='act_output')(b2_1)
    '''



def execute_ml(x_train, x_test, y_train, y_test):
    # model = OneVsRestClassifier(tree.DecisionTreeClassifier()).fit(x_train, y_train)
    # model = OneVsRestClassifier(SVC(gamma=100, C=1000)).fit(x_train, y_train)
    # predictions_train = model.predict(x_train)
    # predictions_test = model.predict(x_test)

    pca = PCA(n_components=3)

    tree = OneVsRestClassifier(DecisionTreeClassifier())

    pipe = Pipeline(steps=[('pca', pca), ('tree', tree)])
    pipe = Pipeline(steps=[('tree', tree)])
    pipe.fit(x_train, y_train)
    predictions_train = pipe.predict(x_train)
    predictions_test = pipe.predict(x_test)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
    print(f"        Train accuracy: {accuracy_score(y_train, predictions_train)}")
    print(f"        Test accuracy: {accuracy_score(y_test, predictions_test)}")


def bpi12():
    print("bpi12")
    bpi12_dataset = parallact.load_generic_dataset("../datasets/vinc/bpi_12_w.csv", save_to_disk=True, verbose=True,
                                                 filename="bpi12", time_format="%Y-%m-%d %H:%M:%S")
    features, targets, features_name, targets_name = parallact.create_matrices(bpi12_dataset, save_to_disk=True,
                                                                             filename="bpi12")

    #features = np.reshape(features, (-1, np.shape(features)[0], np.shape(features)[1]))
    #targets = np.reshape(targets, (-1, np.shape(targets)[0], np.shape(targets)[1]))

    print("Dataset loaded and preprocessed")

    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=3)

    # print("    bpi12 ml:")
    # execute_ml(x_train, x_test, y_train, y_test)

    print("    bpi12 mlp:")
    execute_deep(x_train, x_test, y_train, y_test)

    '''
    x_train = x_train.reshape(np.shape(x_train)[0], np.shape(x_train)[1], -1)
    x_test = x_test.reshape(np.shape(x_test)[0], np.shape(x_test)[1], -1)
    y_train = y_train.reshape(np.shape(y_train)[0], np.shape(y_train)[1], -1)
    y_test = y_test.reshape(np.shape(y_test)[0], np.shape(y_test)[1], -1)
    execute_lstm(x_train, x_test, y_train, y_test)
    '''
    print("")


if __name__ == '__main__':
    # bpi13()
    bpi12()
    # helpdesk()




