import dataset

if __name__ == '__main__':
    bpi13 = dataset.load_bpi13()
    features, targets, features_name, targets_name = dataset.create_matrices(bpi13)

    print(features.shape)
    print(targets.shape)
