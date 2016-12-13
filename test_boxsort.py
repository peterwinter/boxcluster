import numpy as np
from .boxsort import BoxSort
from .fake_data import generate_nested_data
import random


def basic_block_array(shape=(10, 10)):
    a = np.ones(shape=shape)
    a[:5, :5] = 0
    a[5:, 5:] = 0
    return a


# def test_SortingAlgorithmReorder():
#     a = basic_block_array()
#     sa = SortingAlgorithm(a)

#     new_order = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9]
#     test_soln = a[:, new_order][new_order, :]
#     test_result = sa.reorder(order=new_order)

#     assert (test_soln == test_result).all()


# def test_SortingAlgorithm_SmartReorder():
#     a = basic_block_array()
#     sa = SortingAlgorithm(a)

#     new_order = [0, 1, 2, 3, 5, 4, 6, 7, 8, 9]
#     test_soln = a[:, new_order][new_order, :]

#     matrix_range = [4, 5]
#     order = [5, 4]

#     test_result = sa.smart_reorder(order=order, matrix_range=matrix_range)

#     assert (test_soln == test_result).all()


# def test_BoxSort():
#     test_solution = generate_nested_data(noise=0.00001)
#     n = len(test_solution)
#     order = np.arange(n)
#     random.shuffle(order)
#     test = test_solution[:, order][order, :]
#     sas = BoxSort(test)
#     ds_result, ds_order = sas()
#     sqerror_mat = (ds_result - test_solution)**2
#     sqerror = sqerror_mat.sum()
#     assert sqerror < 0.001


# def test_HierSort():
#     test_solution = generate_nested_data(noise=0.00001)

#     n = len(test_solution)
#     order = np.arange(n)
#     random.shuffle(order)
#     test = test_solution[:, order][order, :]
#     hcs = HierarchicalClustering(test)
#     ds_result, ds_order = hcs()

#     sqerror_mat = (ds_result - test_solution)**2
#     sqerror = sqerror_mat.sum()
#     assert sqerror < 0.001



# Test. determine_size()
# size >= 1
# size <= self.matrix_size
# size is integer... random sample 100 times.
