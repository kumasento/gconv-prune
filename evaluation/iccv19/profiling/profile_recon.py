""" Profile the reconstruction procedure """

import time
import numpy as np
import scipy.optimize


def main():
    iters = 10
    N = int(1.2e6)
    H = 7
    W = 7

    print("==> Initialising matrices ...")
    X = np.zeros(H * W * H * W)
    Y = np.random.random((N, H * W)).astype("float32")
    Y_ = np.random.random((N, H * W)).astype("float32")

    f = lambda X: ((Y - np.dot(Y_, X.reshape(H * W, H * W))) ** 2).sum()

    print("==> Running optimisation ...")
    start = time.time()
    for _ in range(iters):
        scipy.optimize.minimize(f, X)
    end = time.time()
    print(
        "Total time: {:.2f}s   Time per iter: {:.2f} s".format(
            end - start, (end - start) / iters
        )
    )


if __name__ == "__main__":
    main()
