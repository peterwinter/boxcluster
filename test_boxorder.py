import numpy as np
from .boxorder import OrderedArray
from .fake_data import generate_nested_data

test_soln = generate_nested_data(noise=0.05)
blankmove = np.arange(len(test_soln))



def ordered_array_ok(oa):
    # all lengths must be same
    N = len(oa)
    assert len(oa) == len(oa.order) == len(oa.matrix)
    # order must contain all integers up to N - 1
    assert set(oa.order) == set(np.arange(N))


def init_ok(oa):
    N = len(oa)
    assert (oa.order == np.arange(N)).all()


def test_single_random_reorder():
    # random reorder move
    oa_solution = OrderedArray(matrix=test_soln)
    init_ok(oa_solution)
    oa2 = oa_solution.copy()

    # generate new order
    n = len(test_soln)
    new_move = np.arange(n)
    np.random.shuffle(new_move)
    inv_order = np.argsort(new_move)

    # reorder Ordered Array
    oa2.reorder(new_move)
    ordered_array_ok(oa2)

    # for a fresh array,
    # result the array's order should equal the new order
    assert (oa2.order == new_move).all()

    # check that inversion functions work
    assert (oa2.inverse_order() == inv_order).all()
    assert (oa2.origional_matrix() == oa_solution.matrix).all()

    # return order back to origional
    oa3 = oa2.copy()
    oa3.reorder(inv_order)
    # both the order and the matrix are back to normal
    assert (oa3.order == np.arange(n)).all()
    assert (oa3.matrix == oa_solution.matrix).all()


def test_blank_reorder():
    # blank move
    oa_solution = OrderedArray(matrix=test_soln)
    oa2 = oa_solution.copy()

    oa2.reorder(blankmove)

    # new order is same as orignional order is same as blankmove
    assert (oa2.order == blankmove).all()
    assert (oa2.order == oa_solution.order).all()
    # new matrix is same as orignional matrix
    assert (oa2.matrix == oa_solution.matrix).all()
    assert (oa2.origional_matrix() == oa_solution.matrix).all()


# set up
def swap_positions(a, pos1, pos2):
    a = a.copy()
    a[pos1], a[pos2] = a[pos2], a[pos1]
    return a


def test_double_swap():
    move1 = swap_positions(blankmove, 0, 10)
    move2 = swap_positions(blankmove, 10, 20)

    resulting_order = blankmove.copy()[move1][move2]
    # test multiple simple moves

    oa_solution = OrderedArray(matrix=test_soln)
    oa2 = oa_solution.copy()
    oa2.reorder(move1)
    assert (oa2.order == move1).all()
    # mplot(oa2.matrix)

    oa2.reorder(move2)
    assert (oa2.order == resulting_order).all()
    # mplot(oa2.matrix)

    inv = oa2.inverse_order()
    oa2.reorder(inv)
    # mplot(oa2.matrix)

    assert (oa2.order == blankmove).all()
    assert (oa2.matrix == oa_solution.matrix).all()


def test_multiple_random_reorders():
    # test multiple comples moves
    oa_solution = OrderedArray(matrix=test_soln)
    oa2 = oa_solution.copy()

    move1 = blankmove.copy()
    np.random.shuffle(move1)

    move2 = blankmove.copy()
    np.random.shuffle(move2)

    oa2.reorder(move1)
    assert (oa2.order == move1).all()
    # clmplot(oa2.matrix)

    resulting_order = blankmove[move1][move2]

    oa2.reorder(move2)
    assert (oa2.order == resulting_order).all()
    # mplot(oa2.matrix)

    inv = oa2.inverse_order()
    oa2.reorder(inv)
    # mplot(oa2.matrix)

    assert (oa2.order == np.arange(len(oa2))).all()
    assert (oa2.matrix == oa_solution.matrix).all()
