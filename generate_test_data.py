import numpy as np
# import pandas as pd

# def symmetrize(a):
#     return a + a.T - numpy.diag(a.diagonal())

def generate_nested_data(size=64, noise=0.05):
    """generate some data for testing the algorithm"""

    tiny_size = int(size / 16)
    quad_size = int(size / 4)
    half_size = int(size / 2)
    a = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                a[i, j] = 1.
            elif i > j:
                a[i, j] = a[j, i]
            else:
                if int(i / tiny_size) == int(j / tiny_size):
                    a[i, j] = np.random.normal(0.6, noise)
                elif int(i / quad_size) == int(j / quad_size):
                    a[i, j] = np.random.normal(0.5, noise)
                elif int(i / half_size) == int(j / half_size):
                    a[i, j] = np.random.normal(0.4, noise)
                else:
                    a[i, j] = np.random.normal(0.3, noise)

    # df = pd.DataFrame(a)
    # print('test key generated')
    # df.to_csv('test_key.csv')

    return a

# if __name__ == '__main__':
#     data = generate_test_data()
#     # file_name = sys.argv[1]
#     # data = [map(float, line.strip().split()) \
#     #        for line in open(file_name)]

#     c = ClusterMatrix(data)

#     (data, order) = c.deep_sort(
#         cooling_factor=0.96, verbose=True, finishing_criterion=3)

#     with open('result.txt', 'w') as result:
#         for n in order:
#             print(n, file=result)
