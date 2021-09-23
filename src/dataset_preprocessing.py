import src.bpi12_preprocessing as bpi12
import src.receipt_preprocessing as receipt
import src.bpi13_incidents_preprocessing as bpi13i
import src.bpi13_problems_preprocessing as bpi13p
import src.bpic2017_o_preprocessing as bpi17o
import src.bpic2020_preprocessing as bpi20

from enum import Enum

DATASET_FOLDER_PATH = "datasets/"


class Dataset(Enum):
    bpi12_all_complete = "bpi12_all_complete/"
    bpi12_work_all = "bpi12_work_all/"
    bpi12w_complete = "bpi12w_complete/"
    bpi13_incidents = "bpi13_incidents/"
    bpi13_problems = "bpi13_problems/"
    bpic2017_o = "bpic2017_o/"
    bpic2020 = "bpic2020/"
    receipt = "receipt/"
    bpi12_reduced = "bpi12_reduced/"


def preprocess(dataset: Dataset, window_size=3):
    print(dataset.value)

    if "bpi12" in dataset.name:
        bpi12.preprocess(dataset, window_size)
    elif "bpi13" in dataset.name:
        if dataset == Dataset.bpi13_incidents:
            bpi13i.preprocess(dataset, window_size)
        else:
            bpi13p.preprocess(dataset, window_size)
    elif "bpic2017" in dataset.name:
        bpi17o.preprocess(dataset, window_size)
    elif "bpic2020" in dataset.name:
        bpi20.preprocess(dataset, window_size)
    elif "receipt" in dataset.name:
        receipt.preprocess(dataset, window_size)