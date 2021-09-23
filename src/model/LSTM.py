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

    outfile = open(RESULT_FOLDER_PATH + dataset.value + "trials.txt", 'w')

    for fold in range(3):
        global iters
        iters = 0

        trainX = np.load(dataset_path + "kfoldcv_" + str(fold) + "_train_LSTMfeatures.npy")
        trainY = np.load(dataset_path + "kfoldcv_" + str(fold) + "_train_targets.npy")

        # Xtrain, Xval, ytrain, yval = model_selection.train_test_split(trainX, trainY)

        search_space = {
            'batch_size': hp.choice('batch_size', [128, 256]),
            'learning_rate_init': hp.loguniform('learning_rate_init', np.log(0.00001), np.log(0.01)),
            'cells': hp.choice('cells', np.arange(50, 201, 50)),
            'LSTMLayers': hp.choice('LSTMLayers', [1, 2])
        }

        def hyperopt_fcn(params):
            #global iters

            start_time = perf_counter()
            h, model = get_model(trainX, trainY, params, fold)
            scores = [h.history['val_loss'][epoch] for epoch in range(len(h.history['loss']))]
            score = min(scores)
            end_time = perf_counter()
            # test_acc = test_fcn(model, Xval, yval)
            K.clear_session()
            time = end_time - start_time
            # model.save(MODEL_FOLDER_PATH + dataset.value + f"kfold_{fold}_{iter}")
            # iter += 1
            return {'loss': score, 'status': STATUS_OK, 'train_time': time, 'model': model}

        '''
        def test_fcn(model, features, labels):
            test_accuracy = model.evaluate(features, labels)
            return test_accuracy
        '''

        trials = Trials()

        best = fmin(hyperopt_fcn, search_space, algo=tpe.suggest, max_evals=25, trials=trials, rstate= np.random.RandomState(fold))

        outfile.write("\nFold: %d" % fold)
        outfile.write("\nHyperopt trials")
        outfile.write("\ntid, loss, learning_rate, batch_size, cells, LSTMLayers, train_time")
        for trial in trials.trials:
            outfile.write("\n%d,%f,%f,%d,%d,%d,%f" % (
                trial['tid'],
                trial['result']['loss'],
                trial['misc']['vals']['learning_rate_init'][0],
                trial['misc']['vals']['batch_size'][0],
                trial['misc']['vals']['cells'][0],
                trial['misc']['vals']['LSTMLayers'][0],
                trial['result']['train_time']
            ))

        outfile.write("\n\nBest parameters:")
        best_params = space_eval(search_space, best)
        print(best_params, file=outfile)

        model = getBestModelfromTrials(trials)
        model.save(MODEL_FOLDER_PATH + dataset.value + f"kfold_{fold}")


def get_model(X, y, param, fold):
    global iters

    inputs = keras.Input(shape=np.shape(X)[1:])
    if (param["LSTMLayers"]) == 2:
        lstm1 = keras.layers.LSTM(param["cells"], return_sequences=True)
        x = lstm1(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LSTM(param["cells"], return_sequences=False)(x)
    else:
        x = keras.layers.LSTM(param["cells"], return_sequences=False)(inputs)
    x = keras.layers.BatchNormalization()(x)

    outputs = keras.layers.Dense(np.shape(y)[1], activation='softmax')(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="PM-LSTM")
    tf.keras.utils.plot_model(model, to_file="lstm.png", show_shapes=True, dpi=300)

    """
    model = keras.Sequential([
        keras.layers.InputLayer(np.shape(X)[1:]),
        # keras.layers.LSTM(param["cells"], return_sequences=True, kernel_initializer='glorot_uniform', dropout=param["dropout"]),
        # keras.layers.LSTM(param["cells"], return_sequences=True),
        # keras.layers.BatchNormalization(),
        # keras.layers.LSTM(param["cells"], return_sequences=False, kernel_initializer='glorot_uniform', dropout=param["dropout"]),
        keras.layers.LSTM(param["cells"], return_sequences=False, dropout=param["dropout"]),
        keras.layers.BatchNormalization(),
        # keras.layers.Dense(np.shape(y_train)[1], activation='softmax'kernel_initializer='glorot_uniform')
        keras.layers.Dense(np.shape(y)[1], activation='softmax')  # , kernel_initializer='glorot_uniform')
    ])
    """

    opt = keras.optimizers.Adam(lr=param['learning_rate_init'])
    # opt = keras.optimizers.Nadam(lr=param['learning_rate_init'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])
    # model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=40)
    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    log_dir = f"logs/fold_{fold}_run_{iters}/" #+ datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    h = model.fit(X, y, epochs=500, verbose=False, validation_split=0.2,  callbacks=[#tensorboard_callback,
                                                                                     early_stopping, lr_reducer],
                  batch_size=param['batch_size'])

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

    outfile = open(RESULT_FOLDER_PATH + dataset.value + f"results.txt", 'w')
    # outfile2 = open(RESULT_FOLDER_PATH + dataset.value + "confusion_matrices.txt", 'w')

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    aucs = []
    auprcs = []

    outfile.write("fold, accuracy, precision, recall, f1-score, AUC, AUPRC\n")

    for fold in range(3):
        model = keras.models.load_model(MODEL_FOLDER_PATH + dataset.value + f"kfold_{fold}")

        testX = np.load(dataset_path + "kfoldcv_" + str(fold) + "_test_LSTMfeatures.npy")
        testY = np.load(dataset_path + "kfoldcv_" + str(fold) + "_test_targets.npy")

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
