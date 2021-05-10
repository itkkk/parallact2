import pandas as pd
import numpy as np
import statistics
from datetime import datetime


def load_generic_dataset(path, window_size=3, verbose=False, save_to_disk=False, filename="", time_format=""):
    """
    This function loads a generic dataset, and exploits chain_dataset function in order to apply the sliding window
    to each trace.
    The dataset must have at least 3 columns in this order: CaseID, ActivityID, CompleteTimestamp

    :param path: the dataset path
    :param window_size: the sliding window size
    :param verbose: if True show average trace length and standard deviation. DEFAULT = False
    :param save_to_disk: if True the preprocessed dataset will be saved to disk. DEFAULT = False
    :param filename: the name of the file that will be written to disk
    :param time_format: the timestamp format, usually "%Y-%m-%d %H:%M:%S"
    :return: a pandas dataframe containing the preprocessed dataset.
             This dataframe contains 5 columns:
             "caseID": the case id
             "activity_chain": a partial trace with length window_size
             "next_activity": the activity executed after the last one in the partial trace
             "global_timestamps": the normalized difference between the timestamp of each activity in the partial trace
                                  and the timestamp of the first activity in the original trace
             "local_timestamps": the normalized difference between the timestamp of each activity in the partial trace
                                 and the timestamp of the previous one
    """
    dataset = pd.read_csv(path, index_col=0)
    pd.set_option('display.max_columns', None)

    return chain_dataset(dataset, window_size, verbose, save_to_disk, filename, time_format)


def chain_dataset(dataset, window_size, verbose, save_to_disk, filename, time_format):
    """
    This function is used to preprocess a dataset, applying the sliding window.
    NB: if you want to load a dataset which is stored in a three column csv, use "load_generic_dataset" function,
        otherwise if your file has a different structure write your own "load_dataset" in order to create
        a three columns pandas dataframe which has to be passed to this function.

    :param dataset: a pandas dataframe containg a dataset with at least 3 columns in this order:
                    CaseID, ActivityID, CompleteTimestamp
    :param window_size: the sliding window size
    :param verbose: if True show average trace length and standard deviation
    :param save_to_disk: if True the preprocessed dataset will be saved to disk.
    :param filename: the name of the file that will be written to disk
    :param time_format: the timestamp format, usually "%Y-%m-%d %H:%M:%S"
    :return: a pandas dataframe containing the preprocessed dataset.
             This dataframe contains 5 columns:
             "caseID": the case id
             "activity_chain": a partial trace with length window_size
             "next_activity": the activity executed after the last one in the partial trace
             "global_timestamps": the normalized difference between the timestamp of each activity in the partial trace
                                  and the timestamp of the first activity in the original trace
             "local_timestamps": the normalized difference between the timestamp of each activity in the partial trace
                                 and the timestamp of the previous one
    """
    # Raggruppo le righe con lo stesso caseID, i valori di timestamp e activity verranno raggruppati in liste
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

        # calcolo la differenza in minuti tra ogni timestamp e il timestamp della prima activity del trace,
        # successivamente normalizzo il tutto dividendo per la differenza massima nel trace
        timestamp_diff_list = (list(
            int(round((datetime.strptime(timestamp, time_format) - datetime.strptime(timestamp_list[0], time_format)).
                      total_seconds() / 60)) for timestamp in timestamp_list))

        timestamp_diff_list[:] = [x / max(timestamp_diff_list) if max(timestamp_diff_list) > 0 else 0
                                  for x in timestamp_diff_list]

        # calcolo la differenza in minuti tra ogni timestamp e il timestamp dell'activity precedente del trace,
        # successivamente normalizzo il tutto dividendo per la differenza massima nel trace
        # timestamp_local_diff_list è inizializzato con 0 cioè la differenza tra la prima activity e se stessa
        timestamp_local_diff_list = [0]
        for i in range(1, len(timestamp_list)):
            timestamp_local_diff_list.append(
                int(round((datetime.strptime(timestamp_list[i], time_format) -
                           datetime.strptime(timestamp_list[i-1], time_format)).total_seconds() / 60)))

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
                    "next_activity": activity_list[(i + window_size)] if ((i + window_size)
                                                                          < len(activity_list)) else "end",
                    "global_timestamps": timestamp_diff_list[i: (i + window_size)],
                    "local_timestamps": timestamp_local_diff_list[i: (i + window_size)]
                })

    # Creo il dataframe da restituire in output
    dataset_chained = pd.DataFrame(dataset_dicts)

    if verbose:
        print("Average trace length: " + str(statistics.mean(trace_lengths)))
        print("standard deviation trace length: " + str(statistics.stdev(trace_lengths)))

    if save_to_disk:
        dataset_chained.to_csv(fr"..\datasets\processed\{filename}_chained.csv", index=False, header=True)

    return dataset_chained


def create_matrices(dataset, save_to_disk=False, filename="", consider_time_features=True):
    """
    This function creates the parallact matrices (both features and target matrix).

    The features matrix takes into account the time relation between the activity in the chain.
    It has a column for each transition between activities at a given time moment t.
    i.e. given the two chains:
     - 'Accepted - In Progress', 'Accepted - In Progress', 'Queued - Awaiting Assignment' => 'Accepted - In Progress'
     - 'Accepted - In Progress', 'Queued - Awaiting Assignment', 'Accepted - In Progress' => 'Completed - Resolved'
     There will be generated 5 columns:
       - 1_Accepted_In_Progress => which represents the transition from t0 to t1 executing Accepted_In_Progress action
       - 2_Accepted-In Progress => which represents the transition from t1 to t2 executing Accepted_In_Progress action
       - 3_Queued_Awaiting_Assignment => which represents the transition from t2 to t3 executing
                                        Queued_Awaiting_Assignment action
       - 2_Queued_Awaiting Assignment => which represents the transition from t1 to t2 executing
                                        Queued_Awaiting_Assignment action
       - 3_Accepted_In Progress => which represents the transition from t2 to t3 executing Accepted_In Progress action
    Considering the column order just described the two feature matrix will have two rows:
      - 1 1 1 0 0
      - 1 0 0 1 1

    In a similar manner we compute the target matrix. It has a column for each of the possible next activity, and the
    value of the column is 1 if that activity is executed, 0 otherwise.
    If we consider the previous example chains we'll have two columns:
      - Accepted-InProgress
      - Completed-Resolved
    Given this column order the target matrix will have two rows:
     - 1 0
     - 0 1

    :param save_to_disk: if True both the features and the targets matrix will be stored into a csv
    :param dataset: a pandas dataframe contains three columns: caseID", "activity_chain", "next_activity".
    :param filename: the name of the file that will be written to disk
    :param consider_time_features: if True the features extracted from timestamp will be considered

    :return:
      - features: a numpy array representing the features matrix just described
      - targets: a numpy array representing the target matrix just described
      - list(features_matrix.columns): a list of all the features name
      - list(target_matrix.columns): a list of all the target name
    """

    # sfrutto una lista di dizionari per creare il dataframe contente la matrice delle feature.
    # ogni dizionario presente nella lista avrà come chiave il nome della colonna
    # (composto da posizione temporale + attività eseguita) e come valore 1.
    caseID_dict_list = []
    features_dict_list = []
    time_dict_list = []
    target_dict_list = []
    for _, row in dataset.iterrows():
        caseID_dict = {}
        feature_dict = {}
        time_dict = {}

        caseID_dict["caseID"] = row[0]

        for time_instant, activity in enumerate(row[1]):
            feature_dict[str(time_instant + 1) + "_" + str(activity).replace(" ", "")] = 1

        if consider_time_features:
            for time_instant, timestamp in enumerate(row[3]):
                time_dict["time_diff_" + str(time_instant + 1)] = timestamp
            for time_instant, timestamp in enumerate(row[4]):
                time_dict["time_local_diff_" + str(time_instant + 1)] = timestamp

        caseID_dict_list.append(caseID_dict)
        features_dict_list.append(feature_dict)
        time_dict_list.append(time_dict)
        target_dict_list.append({str(row[2]).replace(" ", ""): 1})

    caseID_matrix = pd.DataFrame(caseID_dict_list).fillna(int(0)).astype(int)
    features_matrix = pd.DataFrame(features_dict_list).fillna(int(0))  # .astype(int)
    time_matrix = pd.DataFrame(time_dict_list).fillna(int(0))

    final_matrix = caseID_matrix.join(features_matrix)
    final_matrix = final_matrix.join(time_matrix)
    features = final_matrix.to_numpy()

    target_matrix = pd.DataFrame(target_dict_list).fillna(0)  # .astype(int)
    targets = target_matrix.to_numpy()

    if save_to_disk:
        final_matrix.to_csv(fr"..\datasets\processed\{filename}_features_matrix.csv", index=False, header=True)
        target_matrix.to_csv(fr"..\datasets\processed\{filename}_targets_matrix.csv", index=False, header=True)

    return features, targets, list(features_matrix.columns), list(target_matrix.columns)


def createLSTMMatrices(dataset):
    LSTMFeaturesDict = {}

    max_size = 0
    for row in dataset:
        if row[0] in LSTMFeaturesDict:
            features_list = LSTMFeaturesDict[row[0]]
            features_list.append(list(row[1:]))
            LSTMFeaturesDict[row[0]] = features_list
            max_size = features_list.__len__() if max_size < features_list.__len__() else max_size
        else:
            features_list = [list(row[1:])]
            LSTMFeaturesDict[row[0]] = features_list
            max_size = features_list.__len__() if max_size < features_list.__len__() else max_size

    # dataset.shape[1]-1 perchè non considero la colonna caseID
    LSTMFeaturesMatrix = np.zeros((dataset.shape[0], max_size, dataset.shape[1] - 1))

    i = 0
    for key in LSTMFeaturesDict.keys():
        partialTraces = LSTMFeaturesDict[key]
        LSTMFeaturesRow = np.zeros((max_size, dataset.shape[1] - 1))
        for partialTrace in partialTraces:
            LSTMFeaturesRow = np.delete(LSTMFeaturesRow, 0, 0)
            LSTMFeaturesRow = np.vstack([LSTMFeaturesRow, np.zeros((dataset.shape[1] - 1))])
            LSTMFeaturesRow[max_size-1] = partialTrace
            LSTMFeaturesMatrix[i] = LSTMFeaturesRow
            i += 1

    return LSTMFeaturesMatrix
