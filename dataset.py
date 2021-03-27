import pandas as pd


def load_bpi13(chain_len=3):
    bpi13 = pd.read_csv("datasets/VINST cases incidents.csv", encoding="ISO-8859-1", sep=";")
    pd.set_option('display.max_columns', None)

    # Creo la colonna activity unendo status e sub status
    bpi13["Activity"] = bpi13["Status"].astype(str) + " - " + bpi13["Sub Status"]

    # Rinomino le colonne
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
