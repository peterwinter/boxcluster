import pytest
from .boxlist import BaseBoxList, BoxList

boxtypes = [
    BaseBoxList(boxes=[4, 8, 16], fitness=13.2),
    BoxList(boxes=[1, 4, 16], fitness=3.2)
    ]


# add parameters later
@pytest.fixture(scope='function',
                params=boxtypes,
                ids=['BaseBoxList', 'BoxList'])
def baseboxlist(request):
    # yield smtp  # provide the fixture value
    return request.param


# @pytest.mark.usefixtures("baseboxlist")
class TestBaseBoxListProperties:
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


class TestBaseBoxInteractions:
    """ test interactions between BasicBoxList """
    a = BaseBoxList([2, 7, 16], fitness=4.3)
    b = BaseBoxList([1, 8, 16], fitness=10.1)
    b2 = BaseBoxList([1, 8, 16], fitness=10.1)

    def test_lt(self):
        assert self.a < self.b
        assert not self.b < self.b2
        assert not self.b < self.a

    def test_le(self):
        assert self.a <= self.b
        assert self.b <= self.b2
        assert not self.b <= self.a

    def test_eq(self):
        assert not self.a == self.b
        assert self.b == self.b2
        assert not self.b == self.a

    def test_ne(self):
        assert self.a != self.b
        assert not self.b != self.b2
        assert self.b != self.a

    def test_gt(self):
        assert not self.a > self.b
        assert not self.b > self.b2
        assert self.b > self.a

    def test_ge(self):
        assert not self.a >= self.b
        assert self.b >= self.b2
        assert self.b >= self.a


@pytest.fixture()
def boxlist1(request):
    return BoxList(boxes=[1, 4, 5], fitness=10.5)


class TestBoxListMoves:
    def check_integrity(self, boxlist):
        """ values must be ordered integers. no duplicates."""
        values = sorted(set(boxlist))
        assert len(values) == len(boxlist)
        for i, j in zip(values, boxlist):
            assert i == j
            assert type(i) == int

    def boxes_equal(b1, b2):
        return set(b1) == set(b2)

    def test_determine_takeover_size(self, boxlist1):
        for i in range(1, 10):
            s = boxlist1._determine_takeover_size(limit=i)
            assert s >= 1
            assert s <= i

    def test_right_join(self, boxlist1):
        b = boxlist1.copy()
        # middle. shift first box over by 1
        out1 = b.right_join(box_pos=0, size=1)
        self.check_integrity(out1)
        assert set(out1) == set([2, 4, 5])
        # middle. shift first box over by 2
        out2 = b.right_join(box_pos=0, size=2)
        self.check_integrity(out2)
        assert set(out2) == set([3, 4, 5])
        # edgecase. box1 and 2 completely merge.
        out3 = b.right_join(box_pos=0, size=3)
        self.check_integrity(out3)
        assert set(out3) == set([4, 5])

    def test_nonsense_joins(self, boxlist1):
        """ """
        b = boxlist1.copy()
        # b = BoxList(boxes=[1, 4, 5], fitness=10.5)
        assert b.left_join(box_pos=0) is None
        assert b.right_join(box_pos=2) is None

    def test_left_join(self, boxlist1):
        b = boxlist1.copy()
        # edgecase. eliminate lowest box
        out0 = b.left_join(box_pos=1, size=1)
        self.check_integrity(out0)
        assert set(out0) == set([4, 5])
        # midcase. move middle 1
        out1 = b.left_join(box_pos=2, size=1)
        self.check_integrity(out1)
        assert set(out1) == set([1, 3, 5])
        # midcase. move middle 2
        out2 = b.left_join(box_pos=2, size=2)
        self.check_integrity(out2)
        assert set(out2) == set([1, 2, 5])
        # edgecase. eliminate middle box
        out3 = b.left_join(box_pos=2, size=3)
        self.check_integrity(out3)
        assert set(out3) == set([1, 5])

    def test_split(self, boxlist1):
        b = boxlist1.copy()
        # invalid. box is already size 1
        out0 = b.split(box_pos=0)
        assert out0 is None
        # midbox split.
        out1 = b.split(box_pos=1, size=1)
        print(out1)
        self.check_integrity(out1)
        assert set(out1) == set([1, 3, 4, 5])
        # midbox split.
        out2 = b.split(box_pos=1, size=2)
        print(out2)
        self.check_integrity(out2)
        assert set(out2) == set([1, 2, 4, 5])
        # invalid. box is already size 1
        out3 = b.split(box_pos=2)
        assert out3 is None

# TODO: add this stuff in.
# def to_matrix(self)
# def from_matrix(cls, matrix)
