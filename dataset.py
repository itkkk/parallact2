import pandas as pd
import statistics
from datetime import datetime


def load_generic_dataset(path, chain_len=3, verbose=False, save_to_disk=False, filename="", time_format=""):
    dataset = pd.read_csv(path, index_col=0)
    pd.set_option('display.max_columns', None)

    return chain_dataset(dataset, chain_len, verbose, save_to_disk, filename, time_format)


def load_bpi13(chain_len=3, verbose=False, save_to_disk=False, filename=""):
    """
    This function load and preprocess the csv file containing the bpi13 cases incidents dataset

    :param chain_len: the length of the generated activity chain
    :param verbose: if True the method will print some dataset statistics
    :param save_to_disk: if True the preprocessed dataset will be saved to disk
    :param filename: the name of the file that will be written to disk


    :return bpi13_chained: a pandas dataframe containing bpi13 dataset preprocessed in order to represent each trace as
        a chain of chain_len activities. The dataframe contains three columns: "caseID", "activity_chain",
        "next_activity".
    """
    bpi13 = pd.read_csv("datasets/VINST cases incidents.csv", encoding="ISO-8859-1", sep=";")
    pd.set_option('display.max_columns', None)

    # Creo la colonna activity unendo status e sub status
    bpi13["Activity"] = bpi13["Status"].astype(str) + " - " + bpi13["Sub Status"]

    # Seleziono le colonne che mi servono e le rinomino
    bpi13 = bpi13[['SR Number', 'Activity', 'Change Date+Time']]
    bpi13 = bpi13.rename(columns={'SR Number': 'CaseID', 'Change Date+Time': 'Timestamp'})

    return chain_dataset(bpi13, chain_len, verbose, save_to_disk, filename, time_format="%Y-%m-%dT%H:%M:%S%z")


def chain_dataset(dataset, chain_len, verbose, save_to_disk, filename, time_format):
    # Raggruppo le righe con lo stesso caseID, i valori di timestamp e activity verranno raggruppati in liste
    # Es:  CaseID: 1-364285768, Timestamp: ['2010-03-31T15:59:42+01:00', ... '2012-05-11T00:26:15+01:00'],
    # Activity: ['Accepted - In Progress', ..., 'Completed - Closed'].
    # Il tutto sar?? un nuovo dataframe che avr?? come row index CaseID.
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
        # timestamp_local_diff_list ?? inizializzato con 0 cio?? la differenza tra la prima activity e se stessa
        timestamp_local_diff_list = [0]
        for i in range(1, len(timestamp_list)):
            timestamp_local_diff_list.append(
                int(round((datetime.strptime(timestamp_list[i], time_format) -
                           datetime.strptime(timestamp_list[i-1], time_format)).total_seconds() / 60)))

        timestamp_local_diff_list[:] = [x / max(timestamp_local_diff_list) if max(timestamp_local_diff_list) > 0 else 0
                                        for x in timestamp_local_diff_list]

        # Divido la activity list in n liste, ognuna con una lunghezza di chain_len.
        # Aggiungo alla lista bpi13_dicts un dizionario per ogni catena ottenuta dal trace. Il dizionario ?? composto da:
        # "caseID" => l' id del trace
        # "activity_chain": activity_list[i * chain_len: (i + 1) * chain_len] => la catena di chain_len attivit??
        # "next_activity": activity_list[(i + 1) * chain_len] if (((i + 1) * chain_len)
        #       < len(activity_list)) else "end") => l' attivit?? successiva (se esiste),
        #       oppure "end" (se mi trovo a fine lista)
        # "timestamps" => differenza in minuti tra il timestamp del??le attivit?? della catena e il timestamp della prima
        #       attivit?? eseguita nel trace a cui quella catena appartiene
        for i in range((len(activity_list) + chain_len - 1) // chain_len):
            dataset_dicts.append({
                "caseID": caseID,
                "activity_chain": activity_list[i * chain_len: (i + 1) * chain_len],
                "next_activity": activity_list[(i + 1) * chain_len] if (((i + 1) * chain_len)
                                                                        < len(activity_list)) else "end",
                "global_timestamps": timestamp_diff_list[i * chain_len: (i + 1) * chain_len],
                "local_timestamps": timestamp_local_diff_list[i * chain_len: (i + 1) * chain_len]
            })

    # Creo il dataframe da restituire in output
    dataset_chained = pd.DataFrame(dataset_dicts)

    if verbose:
        print("Average trace length: " + str(statistics.mean(trace_lengths)))
        print("standard deviation trace length: " + str(statistics.stdev(trace_lengths)))

    if save_to_disk:
        dataset_chained.to_csv(fr"datasets\processed\{filename}_chained.csv", index=False, header=True)

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
    # ogni dizionario presente nella lista avr?? come chiave il nome della colonna
    # (composto da posizione temporale + attivit?? eseguita) e come valore 1.
    features_dict_list = []
    target_dict_list = []
    for _, row in dataset.iterrows():
        feature_dict = {}
        for time_instant, activity in enumerate(row[1]):
            feature_dict[str(time_instant + 1) + "_" + str(activity).replace(" ", "")] = 1

        if consider_time_features:

            for time_instant, timestamp in enumerate(row[3]):
                feature_dict["time_diff_" + str(time_instant + 1)] = timestamp

            for time_instant, timestamp in enumerate(row[4]):
                feature_dict["time_local_diff_" + str(time_instant + 1)] = timestamp

        features_dict_list.append(feature_dict)

        target_dict_list.append({str(row[2]).replace(" ", ""): 1})

    features_matrix = pd.DataFrame(features_dict_list).fillna(int(0))  # .astype(int)
    features = features_matrix.to_numpy()

    target_matrix = pd.DataFrame(target_dict_list).fillna(0)  # .astype(int)
    targets = target_matrix.to_numpy()

    if save_to_disk:
        features_matrix.to_csv(fr"datasets\processed\{filename}_features_matrix.csv", index=False, header=True)
        target_matrix.to_csv(fr"datasets\processed\{filename}_targets_matrix.csv", index=False, header=True)

    return features, targets, list(features_matrix.columns), list(target_matrix.columns)


