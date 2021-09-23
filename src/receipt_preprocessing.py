import pandas as pd
import numpy as np
from bpi12_preprocessing import chain_dataset, create_dictionaries, save_numpy_matrices, saveLSTMMatrices, createImages

pd.set_option('display.max_columns', None)

DATASET_FOLDER_PATH = "datasets/"
TIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f"

def preprocess(dataset, window_size=3):
    dataset_path = DATASET_FOLDER_PATH + dataset.value

    print("The selected dataset is: " + dataset.value)
    print("The dataset path is: " + dataset_path)

    for fold in range(3):
        print(f"preprocessing fold {fold + 1}", end='')
        train_path = dataset_path + "kfoldcv_" + str(fold) + "_train.csv"
        test_path = dataset_path + "kfoldcv_" + str(fold) + "_test.csv"

        train = pd.read_csv(train_path, header=None)
        test = pd.read_csv(test_path, header=None)

        # assegno il nome alle colonne
        columns = ["CaseID", "Activity", "Resource", "Timestamp", "Channel", "Department", "Group", "Org_group", "Responsible"]
        train.columns = columns
        test.columns = columns

        train, avg_length = chain_dataset(train, window_size)
        test, _ = chain_dataset(test, window_size)
        print(".", end='')

        train_caseID_matrix, train_features_matrix, train_time_matrix, train_resource_matrix, train_target_matrix = create_dictionaries(
            train)
        test_caseID_matrix, test_features_matrix, test_time_matrix, test_resource_matrix, test_target_matrix = create_dictionaries(
            test)
        print(".", end='')

        # riordino le colonne del test set
        train_features_matrix, test_features_matrix = reorder_and_match_columns(train_features_matrix, test_features_matrix)
        train_target_matrix, test_target_matrix = reorder_and_match_columns(train_target_matrix, test_target_matrix)

        print(".", end='')

        train_features, train_targets = save_numpy_matrices(train_caseID_matrix, train_features_matrix,
                                                            train_time_matrix, train_resource_matrix,
                                                            train_target_matrix,
                                                            dataset_path + "kfoldcv_" + str(fold) + "_train")
        test_features, test_targets = save_numpy_matrices(test_caseID_matrix, test_features_matrix,
                                                          test_time_matrix, test_resource_matrix, test_target_matrix,
                                                          dataset_path + "kfoldcv_" + str(fold) + "_test")
        print(".", end='')

        saveLSTMMatrices(train_features, avg_length, dataset_path + "kfoldcv_" + str(fold) + "_train")
        saveLSTMMatrices(test_features, avg_length, dataset_path + "kfoldcv_" + str(fold) + "_test")

        print(".", end='')

        # creo le immagini
        images_train_features = train_features[:, 1:]
        images_test_features = test_features[:, 1:]
        classes = train_target_matrix.shape[1]
        images_train_targets = np.argmax(train_targets, axis=1)
        images_test_targets = np.argmax(test_targets, axis=1)
        createImages(DATASET_FOLDER_PATH + dataset.value, images_train_features, images_train_targets, images_test_features, images_test_targets, classes, "kfoldcv_" + str(fold))

        print(" Done!")


def reorder_and_match_columns(first_matrix, second_matrix):
    train_columns = first_matrix.columns.tolist()
    test_columns = second_matrix.columns.tolist()

    in_first = set(train_columns)
    in_second = set(test_columns)

    in_train_but_not_in_test = in_first - in_second
    in_test_but_not_in_train = in_second - in_first

    for column in in_train_but_not_in_test:
        second_matrix[column] = 0

    for column in in_test_but_not_in_train:
        first_matrix[column] = 0

    train_columns = first_matrix.columns.tolist()
    second_matrix = second_matrix[train_columns]

    return first_matrix, second_matrix
