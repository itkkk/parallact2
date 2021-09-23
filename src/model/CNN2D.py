from src.dataset_preprocessing import Dataset

import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

from hyperopt import space_eval
from hyperopt import fmin, tpe, hp, Trials
from hyperopt import STATUS_OK

from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelBinarizer

import numpy as np
import pickle as pk

from time import perf_counter

DATASET_FOLDER_PATH = "datasets/"
RESULT_FOLDER_PATH = "results/"
MODEL_FOLDER_PATH = "models/"
LOG_FOLDER = "logs/"

iters = 0

tf.random.set_seed(0)
np.random.seed(0)


def optimize(dataset: Dataset):
    dataset_path = DATASET_FOLDER_PATH + dataset.value
    outfile = open(RESULT_FOLDER_PATH + dataset.value + "CNN2D/trials.txt", 'w')

    for fold in range(3):
        pickle_train = open(dataset_path + "kfoldcv_" + str(fold) + "_train_16x16_MI.pickle", "rb")
        trainX = pk.load(pickle_train)
        trainX = np.asarray(trainX)
        trainY = np.load(dataset_path + "kfoldcv_" + str(fold) + "_train_targets.npy")

        image_size = trainX.shape[1]
        trainX = np.reshape(trainX, [-1, image_size, image_size, 1])

        search_space = {
            'batch_size': hp.choice('batch_size', [128, 256]),
            'learning_rate_init': hp.loguniform('learning_rate_init', np.log(0.00001), np.log(0.01)),
            'convolution1': hp.choice('convolution1', [32, 64]),
            'convolution2': hp.choice('convolution2', [32, 64]),
            'kernel_size1':  hp.choice('kernel_size1', [(2, 2), (4, 4)]),
            'kernel_size2': hp.choice('kernel_size2', [(2, 2), (4, 4)]),
            'pool_size': hp.choice('pool_size', [(2, 2), (4, 4)])
        }

        def hyperopt_fcn(params):
            start_time = perf_counter()
            h, model = get_model(trainX, trainY, params, fold)
            scores = [h.history['val_loss'][epoch] for epoch in range(len(h.history['loss']))]
            score = min(scores)
            end_time = perf_counter()
            K.clear_session()
            time = end_time - start_time
            return {'loss': score, 'status': STATUS_OK, 'train_time': time, 'model': model}

        trials = Trials()
        best = fmin(hyperopt_fcn, search_space, algo=tpe.suggest, max_evals=25, trials=trials,
                    rstate=np.random.RandomState(fold))

        outfile.write("\nFold: %d" % fold)
        outfile.write("\nHyperopt trials")
        for trial in trials.trials:
            outfile.write("\ntid, loss, learning_rate, batch_size, cells, LSTMLayers, train_time")
            outfile.write("\n%d,%f,%f,%d,%d,%d,%d,%d,%d,%f" % (
                trial['tid'],
                trial['result']['loss'],
                trial['misc']['vals']['batch_size'][0],
                trial['misc']['vals']['learning_rate_init'][0],
                trial['misc']['vals']['convolution1'][0],
                trial['misc']['vals']['convolution2'][0],
                trial['misc']['vals']['kernel_size1'][0],
                trial['misc']['vals']['kernel_size2'][0],
                trial['misc']['vals']['pool_size'][0],
                trial['result']['train_time']
            ))

        outfile.write("\n\nBest parameters:")
        best_params = space_eval(search_space, best)
        print(best_params, file=outfile)

        model = getBestModelfromTrials(trials)
        model.save(MODEL_FOLDER_PATH + dataset.value + f"CNN2D/kfold_{fold}")


def get_model(X, y, param, fold):
    global iters

    model = keras.Sequential()
    reg = 0.0001
    input_shape = (X.shape[1], X.shape[1], 1)
    model.add(keras.layers.Conv2D(param['convolution1'], param['kernel_size1'], input_shape=input_shape, padding='same',
                                  kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.l2(reg)))

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=param['pool_size']))

    model.add(keras.layers.Conv2D(param['convolution2'], param['kernel_size2'], padding='same',
                                  kernel_initializer='glorot_uniform', kernel_regularizer=keras.regularizers.l2(reg)))

    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.GlobalMaxPooling2D())

    model.add(keras.layers.Dense(np.shape(y)[1], activation='softmax', name='act_output'))

    tf.keras.utils.plot_model(model, to_file="cnn2D.png", show_shapes=True, dpi=300)

    opt = keras.optimizers.Adam(lr=param['learning_rate_init'])
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    # model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=40)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto',
                                   min_delta=0.0001, cooldown=0, min_lr=0)

    log_dir = f"logs/fold_{fold}_run_{iters}/"
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    h = model.fit(X, y, epochs=500, verbose=False, validation_split=0.2,  callbacks=[  # tensorboard_callback,
        early_stopping, lr_reducer],batch_size=param['batch_size'])

    iters += 1
    return h, model


def getBestModelfromTrials(trials):
    valid_trial_list = [trial for trial in trials if STATUS_OK == trial['result']['status']]
    losses = [ float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argmin(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss]
    return best_trial_obj['result']['model']


def testModel(dataset: Dataset):
    dataset_path = DATASET_FOLDER_PATH + dataset.value

    outfile = open(RESULT_FOLDER_PATH + dataset.value + f"CNN2D/results.txt", 'w')
    # outfile2 = open(RESULT_FOLDER_PATH + dataset.value + "confusion_matrices.txt", 'w')

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    aucs = []
    auprcs = []

    outfile.write("fold, accuracy, precision, recall, f1-score, AUC, AUPRC\n")

    for fold in range(3):
        model = keras.models.load_model(MODEL_FOLDER_PATH + dataset.value + f"CNN2D/kfold_{fold}")


        pickle_test = open(dataset_path + "kfoldcv_" + str(fold) + "_test_16x16_MI.pickle", "rb")
        testX = pk.load(pickle_test)
        testX = np.asarray(testX)
        testY = np.load(dataset_path + "kfoldcv_" + str(fold) + "_test_targets.npy")

        image_size = testX.shape[1]
        testX = np.reshape(testX, [-1, image_size, image_size, 1])


        predictions = model.predict(testX)

        discrete_pred = np.argmax(predictions, axis=1)
        discrete_y_test = np.argmax(testY, axis=1)

        equality = tf.math.equal(discrete_pred, discrete_y_test)
        accuracy = tf.math.reduce_mean(tf.cast(equality, tf.float32))

        print(f"fold {fold} test accuracy = {accuracy}")

        # cm = confusion_matrix(discrete_y_test, discrete_pred)

        cl = classification_report(discrete_y_test, discrete_pred, digits=3)

        # print(cm)
        print(cl)

        cl = classification_report(discrete_y_test, discrete_pred, digits=3, output_dict=True)

        auc = multiclass_roc_auc_score(discrete_y_test, discrete_pred, average="macro")
        auprc = multiclass_pr_auc_score(discrete_y_test, discrete_pred, average="macro")

        accuracies.append((cl['accuracy']))
        precisions.append(cl['macro avg']['precision'])
        recalls.append(cl['macro avg']['recall'])
        f1_scores.append(cl['macro avg']['f1-score'])
        aucs.append(auc)
        auprcs.append(auprc)

        outfile.write(f"{fold},{cl['accuracy']} ,{cl['macro avg']['precision']}, {cl['macro avg']['recall']}, {cl['macro avg']['f1-score']}, "
                      f"{auc}, {auprc}\n")

        # outfile2.write(f"Fold {fold}>>>>>>>>>>>>>>")
        # outfile2.write(cm)

    avg_accuracies = sum (accuracies) / len(accuracies)
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1_score = sum(f1_scores) / len(f1_scores)
    avg_AUC = sum(aucs) / len(aucs)
    avg_AUPR = sum(auprcs) / len(auprcs)

    outfile.write(
        f"avg, {avg_accuracies}, {avg_precision}, {avg_recall}, {avg_f1_score}, {avg_AUC}, {avg_AUPR}")


def multiclass_roc_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def multiclass_pr_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return average_precision_score(y_test, y_pred, average=average)
