import pytest
from .mod import Modules, BaseModules, SimpleModules

@pytest.fixture()
def basemod():
    bm = BaseModules()
    bm.nodes
    bm.mods
    bm.fitness = 5.0
    return


# this was mainly copied from test_boxlist. not properly adjusted
class aTestBaseModules:
    """ tests for basic properties of a BasicBoxList"""

    def test_getitem(self, baseboxlist):
        """should behave like list inside"""
        l = baseboxlist.boxes.copy()
        n = len(l)
        for i in range(n):
            assert baseboxlist[i] == l[i]
        for i, b in enumerate(baseboxlist):
            assert b == l[i]

    def test_setitem(self, baseboxlist):
        """should set items into list"""
        baseboxlist[0] = 2
        print(baseboxlist)
        assert baseboxlist[0] == 2

    def test_del(self, baseboxlist):
        """should delete item like list"""
        l = baseboxlist.boxes.copy()
        del baseboxlist[1]
        del l[1]
        assert baseboxlist[0] == l[0]
        assert baseboxlist[1] == l[1]

    def test_len(self, baseboxlist):
        """should say len of internal list"""
        l = baseboxlist.boxes
        assert len(l) == len(baseboxlist)

    def test_copy_identical_contents(self, baseboxlist):
        """should say len of internal list"""
        box1 = baseboxlist
        box2 = baseboxlist.copy()
        assert type(box1) == type(box2)
        for i, b in enumerate(box1):
            assert b == box2[i]
        assert box1.fitness == box2.fitness

    def test_copy_seperate_pointers(self, baseboxlist):
        """should say len of internal list"""
        box1 = baseboxlist
        box2 = baseboxlist.copy()
        del box1[0]
        assert len(box1) == len(box2) - 1
        box1.fitness += 1
        assert box1.fitness != box2.fitness

    # TODO: add in this method
    def test_insert(self, baseboxlist):
        baseboxlist

    # TODO: add in this method
    def test_items(self, baseboxlist):
        pass
