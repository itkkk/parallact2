from src.dataset_preprocessing import preprocess, Dataset
from src.model import CNN2D, LSTM, InceptionCNN

# Preprocess step
#preprocess(dataset=Dataset.bpic2020, window_size=3)

# Training step
CNN2D.optimize(dataset=Dataset.bpic2020)

# Test stage
CNN2D.testModel(dataset=Dataset.bpic2020)
