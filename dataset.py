import pandas as pd


def load_bpi13(chain_len=3):
    """
    This function load and preprocess the csv file containing the bpi13 cases incidents dataset

    :param chain_len: the length of the generated activity chain
    :return bpi13_chained: a pandas dataframe containing bpi13 dataset preprocessed in order to represent each trace as
        a chain of chain_len activities. The dataframe contains three columns: "caseID", "activity_chain",
        "next_activity".
    """
    bpi13 = pd.read_csv("datasets/VINST cases incidents.csv", encoding="ISO-8859-1", sep=";")
    pd.set_option('display.max_columns', None)

    # Creo la colonna activity unendo status e sub status
    bpi13["Activity"] = bpi13["Status"].astype(str) + " - " + bpi13["Sub Status"]

    # Seleziono le colonne che mi servono e le rinomino
    bpi13 = bpi13[['SR Number', 'Change Date+Time', 'Activity']]
    bpi13 = bpi13.rename(columns={'SR Number': 'CaseID', 'Change Date+Time': 'Timestamp'})

    # Raggruppo le righe con lo stesso caseID, i valori di timestamp e activity verranno raggruppati in liste
    # Es:  CaseID: 1-364285768, Timestamp: ['2010-03-31T15:59:42+01:00', ... '2012-05-11T00:26:15+01:00'],
    # Activity: ['Accepted - In Progress', ..., 'Completed - Closed'].
    # Il tutto sarà un nuovo dataframe che avrà come row index CaseID.
    grouped = bpi13.groupby('CaseID').agg(lambda x: list(x))

    bpi13_dicts = []
    # Vado a creare i dizionari da inserire in bpi13_dicts, uso iterrows per iterare sulle righe del dataframe.
    # iterrows restituisce due valori: l' indice della riga (nel nostro caso il caseID) e la riga stessa
    for caseID, row in grouped.iterrows():
        # timestamp_list = row[0]
        activity_list = row[1]

        # Divido la activity list in n liste, ognuna con una lunghezza di chain_len.
        # Aggiungo alla lista bpi13_dicts un dizionario per ogni catena ottenuta dal trace. Il dizionario è composto da:
        # "caseID" => l' id del trace
        # "activity_chain": activity_list[i * chain_len: (i + 1) * chain_len] => la catena di chain_len attività
        # "next_activity": activity_list[(i + 1) * chain_len] if (((i + 1) * chain_len)
        #       < len(activity_list)) else "end") => l' attività successiva (se esiste),
        #       oppure "end" (se mi trovo a fine lista)
        for i in range((len(activity_list) + chain_len - 1) // chain_len):
            bpi13_dicts.append({
                "caseID": caseID,
                "activity_chain": activity_list[i * chain_len: (i + 1) * chain_len],
                "next_activity": activity_list[(i + 1) * chain_len] if (((i + 1) * chain_len)
                                                                        < len(activity_list)) else "end"
            })

    # Creo il dataframe da restituire in output
    bpi13_chained = pd.DataFrame(bpi13_dicts)

    return bpi13_chained


def create_matrices(dataset):
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
                                        ueued_Awaiting_Assignment action
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

    :param dataset: a pandas dataframe contains three columns: caseID", "activity_chain", "next_activity".
    :return:
      - features: a numpy array representing the features matrix just described
      - targets: a numpy array representing the target matrix just described
      - list(features_matrix.columns): a list of all the features name
      - list(target_matrix.columns): a list of all the target name
    """

    # sfrutto una lista di dizionari per creare il dataframe contente la matrice delle feature.
    # ogni dizionario presente nella lista avrà come chiave il nome della colonna
    # (composto da posizione temporale + attività eseguita) e come valore 1.
    features_dict_list = []
    target_dict_list = []
    for _, row in dataset.iterrows():
        feature_dict = {}
        for time_instant, activity in enumerate(row[1]):
            feature_dict[str(time_instant+1) + "_" + activity.replace(" ", "_")] = 1
        features_dict_list.append(feature_dict)

        target_dict_list.append({row[2].replace(" ", "_"): 1})

    features_matrix = pd.DataFrame(features_dict_list).fillna(int(0)).astype(int)
    features = features_matrix.to_numpy()

    target_matrix = pd.DataFrame(target_dict_list).fillna(0).astype(int)
    targets = target_matrix.to_numpy()

    return features, targets, list(features_matrix.columns), list(target_matrix.columns)
