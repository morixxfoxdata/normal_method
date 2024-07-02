import numpy as np


def load_data():
    file_path = "data/raw/HP_mosaic_random_size8x8_image64+10+500_alternate.npz"
    collection = "data/raw/HP+mosaic+rand_image64+10+500_size8x8_alternate_200x20020240618_collect.npz"
    using_data = np.load(file_path)
    collection_data = np.load(collection)
    X = using_data["arr_0"]
    y = collection_data["arr_0"]
    return X, y


def hadamard_total():
    X, y = load_data()
    X_hadamard = X[:128, :]
    y_hadamard = y[:128, :]
    # 取得アダマールパターンの偶数列のみを利用する
    y_hadamard = y_hadamard[:, ::2]
    # 色反転を取り出す
    X_hadamard_1 = X_hadamard[0::2, :]
    X_hadamard_2 = X_hadamard[1::2, :]
    y_hadamard_1 = y_hadamard[0::2, :]
    y_hadamard_2 = y_hadamard[1::2, :]
    # 取得アダマールパターン, 利用アダマールパターンを合わせる
    X_hadamard_total = X_hadamard_1 - X_hadamard_2
    y_hadamard_total = y_hadamard_1 - y_hadamard_2
    # yy_hadamard_t.shape: (64, 500)
    # xx_hadamard_t.shape: (64, 64)
    return X_hadamard_total, y_hadamard_total


def random_total():
    X, y = load_data()
    X_random = X[148:, :]
    y_random = y[148:, :]
    # 取得ランダムパターンをskipして500に縮小する
    y_random = y_random[:, ::2]
    # ランダムデータの色反転データをそれぞれ格納
    X_random_1 = X_random[0::2, :]
    X_random_2 = X_random[1::2, :]
    y_random_1 = y_random[0::2, :]
    y_random_2 = y_random[1::2, :]
    # 取得ランダムパターン, 利用ランダムパターンを合わせる
    X_random_total = X_random_1 - X_random_2
    y_random_total = y_random_1 - y_random_2
    # yy_random_t.shape: (500, 500)
    # xx_random_t.shape: (500, 64)
    return X_random_total, y_random_total


def mnist_total():
    X, y = load_data()
    X_mnist = X[128:148, :]
    y_mnist = y[128:148, :]

    y_mnist = y_mnist[:, ::2]
    # MNISTデータの色反転データをそれぞれ格納
    X_mnist_1 = X_mnist[0::2, :]
    X_mnist_2 = X_mnist[1::2, :]
    y_mnist_1 = y_mnist[0::2, :]
    y_mnist_2 = y_mnist[1::2, :]
    # 取得MNISTパターン, 利用MNISTパターンを合わせる
    X_mnist_total = X_mnist_1 - X_mnist_2
    y_mnist_total = y_mnist_1 - y_mnist_2
    return X_mnist_total, y_mnist_total


def merge_data():
    X, y = load_data()
    X_hadamard, y_hadamard = hadamard_total(X, y)
    X_mnist, y_mnist = mnist_total(X, y)
    X_random, y_random = random_total(X, y)
    X_merged = np.vstack((X_hadamard, X_mnist, X_random))
    y_merged = np.vstack((y_hadamard, y_mnist, y_random))
    return X_merged, y_merged
