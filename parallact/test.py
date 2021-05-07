import parallact

bpi12_dataset = parallact.load_generic_dataset("../datasets/vinc/bpi_12_w.csv", save_to_disk=True, verbose=True,
                                                 filename="prova", time_format="%Y-%m-%d %H:%M:%S")