import multiprocessing as mp


def task(p):
    p = p
    return p


if __name__ == "__main__":
    params = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    with mp.Pool() as pool:
        results = pool.map(task, params)
    for p, result in zip(params, results):
        print(result)
        print(p)
