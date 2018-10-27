import keras.backend as bk


def seu(x):
    return bk.sigmoid(x) * x


def make_lseu(c_val):
    def _lseu(x):
        x1 = bk.sigmoid(x) * x
        x2 = c_val + bk.log(1 + bk.relu(x - c_val))
        return bk.minimum(x1, x2)

    return _lseu


def lseu(x):
    return make_lseu(9)(x)
