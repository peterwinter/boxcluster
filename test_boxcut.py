import pytest
from . import BoxCut
from . import fake_data


def check_box_integrity(boxlist):
    """ values must be ordered integers. no duplicates."""
    values = sorted(set(boxlist))
    assert len(values) == len(boxlist)
    for i, j in zip(values, boxlist):
        assert i == j
        assert type(i) == int


@pytest.fixture()
def boxcut1():
    test_matrix = fake_data.generate_nested_data()
    bc = BoxCut(test_matrix)
    bc()
    return bc


class TestBoxCutBasics:

    # potentially testable
    def test_propose_box_move(self, boxcut1):
        bc = boxcut1
        # print(boxcut1.boxes)
        # bc.test_propose_box_move(boxes)

    def test_probability_to_accept(self, boxcut1):
        """ p must be between 0 and 1 """
        bc = boxcut1
        temp1 = bc.temp
        p = bc._probability_to_accept(improvement=0)
        assert abs(p-1) < 0.0001
        p = bc._probability_to_accept(improvement=0.5)
        assert p >= 1
        # assert p == 1
        p = bc._probability_to_accept(improvement=-0.1*temp1)
        assert 0 < p < 1
        # lower temperature should have lower probability
        p = bc._probability_to_accept(improvement=-0.1)
        bc.temp = bc.temp * 0.5
        p2 = bc._probability_to_accept(improvement=-0.1)
        assert p2 < p

    def test_move_is_accepted(self, boxcut1):
        bc = boxcut1
        print('hey')

    def test_update_if_best(self, boxcut1):
        bc = boxcut1
        boxes = bc.current
        print(boxes)
        candidate = boxes.pop(0)
        # bc._update_if_best(self, candidate)

    # def _evaluate_box_fitness(self, boxes=None, matrix=None):

    # too extensive to test properly
    # def box_turn(self):
    # def _initialize_search(self, matrix):
    # def fit_boxes(self, matrix=None, debug=False):
    # def _iter_solve(self, matrix, debug=False):
    def test_pass(self):
        pass

class TestBoxCutSolution:
    # run debug. check properties in dataframe
    # def debug(self, matrix):

    def test_pass(self):
        pass
