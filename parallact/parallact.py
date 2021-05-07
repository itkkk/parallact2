import pandas as pd
import statistics
from datetime import datetime


def load_generic_dataset(path, chain_len=3, verbose=False, save_to_disk=False, filename="", time_format=""):
    """
    :param path:
    :param chain_len:
    :param verbose:
    :param save_to_disk:
    :param filename:
    :param time_format:
    :return:
    """
    dataset = pd.read_csv(path, index_col=0)
    pd.set_option('display.max_columns', None)

    return chain_dataset(dataset, chain_len, verbose, save_to_disk, filename, time_format)


def chain_dataset(dataset, chain_len, verbose, save_to_disk, filename, time_format):
    # Raggruppo le righe con lo stesso caseID, i valori di timestamp e activity verranno raggruppati in liste
    # Es:  CaseID: 1-364285768, Timestamp: ['2010-03-31T15:59:42+01:00', ... '2012-05-11T00:26:15+01:00'],
    # Activity: ['Accepted - In Progress', ..., 'Completed - Closed'].
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


        """
        # Divido la activity list in n liste, ognuna con una lunghezza di chain_len.
        # Aggiungo alla lista bpi13_dicts un dizionario per ogni catena ottenuta dal trace. Il dizionario è composto da:
        # "caseID" => l' id del trace
        # "activity_chain": activity_list[i * chain_len: (i + 1) * chain_len] => la catena di chain_len attività
        # "next_activity": activity_list[(i + 1) * chain_len] if (((i + 1) * chain_len)
        #       < len(activity_list)) else "end") => l' attività successiva (se esiste),
        #       oppure "end" (se mi trovo a fine lista)
        # "timestamps" => differenza in minuti tra il timestamp delòle attività della catena e il timestamp della prima
        #       attività eseguita nel trace a cui quella catena appartiene
        for i in range((len(activity_list) + chain_len - 1) // chain_len):
            dataset_dicts.append({
                "caseID": caseID,
                "activity_chain": activity_list[i * chain_len: (i + 1) * chain_len],
                "next_activity": activity_list[(i + 1) * chain_len] if (((i + 1) * chain_len)
                                                                        < len(activity_list)) else "end",
                "global_timestamps": timestamp_diff_list[i * chain_len: (i + 1) * chain_len],
                "local_timestamps": timestamp_local_diff_list[i * chain_len: (i + 1) * chain_len]
            })
        """
        for i in range(len(activity_list)):
            dataset_dicts.append({
                "caseID": caseID,
                "activity_chain": activity_list[i: (i + chain_len)],
                "next_activity": activity_list[(i + chain_len)] if ((i + chain_len)
                                                                        < len(activity_list)) else "end",
                "global_timestamps": timestamp_diff_list[i: (i + chain_len)],
                "local_timestamps": timestamp_local_diff_list[i: (i + chain_len)]
            })

    # Creo il dataframe da restituire in output
    dataset_chained = pd.DataFrame(dataset_dicts)

    if verbose:
        print("Average trace length: " + str(statistics.mean(trace_lengths)))
        print("standard deviation trace length: " + str(statistics.stdev(trace_lengths)))

    if save_to_disk:
        dataset_chained.to_csv(fr"..\datasets\processed\{filename}_chained.csv", index=False, header=True)

    return dataset_chained

