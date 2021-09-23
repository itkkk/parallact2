import pandas as pd
import numpy as np
from datetime import datetime
import statistics

from bpi12_preprocessing import create_dictionaries, save_numpy_matrices, saveLSTMMatrices, createImages
from receipt_preprocessing import reorder_and_match_columns

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

        train = pd.read_csv(train_path, header=None, encoding='mbcs')
        test = pd.read_csv(test_path, header=None, encoding='mbcs')

        # assegno il nome alle colonne
        columns = ["CaseID", "Activity", "Timestamp", "Monthly_cost", "Credit_score", "First_withdrawal_amount", "Offered_amount", "Number_of_terms", "Action", "Resource"]
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
        createImages(DATASET_FOLDER_PATH + dataset.value, images_train_features, images_train_targets,
                     images_test_features, images_test_targets, classes, "kfoldcv_" + str(fold))

        print(" Done!")


def chain_dataset(dataset, window_size):
    # Raggruppo le righe con lo stesso caseID, le altre colonne verranno raggruppate in liste
    # Es:  CaseID: 1-364285768, Activity: ['Accepted - In Progress', ..., 'Completed - Closed'],
    # Timestamp: ['2010-03-31T15:59:42+01:00', ... '2012-05-11T00:26:15+01:00'].
    # Il tutto sarà un nuovo dataframe che avrà come row index CaseID.
    grouped = dataset.groupby('CaseID').agg(lambda x: list(x))

    trace_lengths = []
    dataset_dicts = []
    # Vado a creare i dizionari da inserire in bpi13_dicts, uso iterrows per iterare sulle righe del dataframe.
    # iterrows restituisce due valori: l' indice della riga (nel nostro caso il caseID) e la riga stessa
    for caseID, row in grouped.iterrows():
        timestamp_list = row[1]
        activity_list = row[0]
        trace_lengths.append(activity_list.__len__())
        resource_list = row[2]

        # calcolo la differenza in minuti tra ogni timestamp e il timestamp della prima activity del trace,
        # successivamente normalizzo il tutto dividendo per la differenza massima nel trace
        timestamp_diff_list = (list(int(round((datetime.strptime(timestamp, TIME_FORMAT) - datetime.strptime(
            timestamp_list[0], TIME_FORMAT)).total_seconds() / 60)) for timestamp in timestamp_list))

        timestamp_diff_list[:] = [x / max(timestamp_diff_list) if max(timestamp_diff_list) > 0 else 0 for x in
                                  timestamp_diff_list]

        # calcolo la differenza in minuti tra ogni timestamp e il timestamp dell'activity precedente del trace,
        # successivamente normalizzo il tutto dividendo per la differenza massima nel trace
        # timestamp_local_diff_list è inizializzato con 0 cioè la differenza tra la prima activity e se stessa
        timestamp_local_diff_list = [0]
        for i in range(1, len(timestamp_list)):
            timestamp_local_diff_list.append(
                int(round((datetime.strptime(timestamp_list[i], TIME_FORMAT) - datetime.strptime(timestamp_list[i - 1],
                                                                                                 TIME_FORMAT)).total_seconds() / 60)))

        timestamp_local_diff_list[:] = [x / max(timestamp_local_diff_list) if max(timestamp_local_diff_list) > 0 else 0
                                        for x in timestamp_local_diff_list]

        # Divido la activity list in n liste, ognuna con una lunghezza di window_size.
        # Aggiungo alla lista bpi13_dicts un dizionario per ogni catena ottenuta dal trace. Il dizionario è composto da:
        # "caseID" => l' id del trace
        # "activity_chain": il trace pariale di dimensione window_size
        # "next_activity": la prossima attività da eseguire o "end" se il trace è finito
        # "global_timestamps" => differenza in minuti normalizzata tra ogni timestamp e il timestamp della prima
        #                        activity del trace
        # "local_timestamps" => differenza in minuti normalizzata tra ogni timestamp e il timestamp dell' activity
        #                       precedente nel trace

        for i in range(len(activity_list)):
            partial_trace = activity_list[i: (i + window_size)]

            # Considero solo i trace parziali con almeno 2 attività
            if partial_trace.__len__() > 1:
                dataset_dicts.append({
                    "caseID": caseID,
                    "activity_chain": partial_trace,
                    "resource_list": resource_list[i: (i + window_size)],
                    "next_activity": activity_list[(i + window_size)] if (
                                (i + window_size) < len(activity_list)) else "end",
                    "global_timestamps": timestamp_diff_list[i: (i + window_size)],
                    "local_timestamps": timestamp_local_diff_list[i: (i + window_size)]
                })

    # Creo il dataframe da restituire in output
    dataset_chained = pd.DataFrame(dataset_dicts)

    '''
    if verbose:
        print("Average trace length: " + str(statistics.mean(trace_lengths)))
        print("standard deviation trace length: " + str(statistics.stdev(trace_lengths)))
    '''

    # dataset_chained.to_csv(DATASET_FOLDER_PATH + "prova.csv", index=False, header=True)

    return dataset_chained, statistics.mean(trace_lengths)


