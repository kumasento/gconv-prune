import torch

from gumi.group_utils import get_group_allocation


def test_get_group_allocation_all_ones():
    mask = torch.ones((2, 2), dtype=torch.bool)
    gaf, gac = get_group_allocation(mask, G=1)

    assert gaf is not None
    assert gac is not None
    assert (gaf == 1).all()
    assert (gac == 1).all()

    # Cannot split into valid groups in this case.
    gaf, gac = get_group_allocation(mask, G=2)
    assert gaf is None
    assert gac is None


def test_get_group_allocation_block_diagonal():
    mask = torch.ones((4, 4), dtype=torch.bool)
    mask[2:, :2] = 0
    mask[:2, 2:] = 0

    gaf, gac = get_group_allocation(mask, G=2)
    assert gaf is not None
    assert gac is not None
    assert (gaf[:2] == 1).all()
    assert (gaf[2:] == 2).all()
    assert (gac[:2] == 1).all()
    assert (gac[2:] == 2).all()

    # anti-diagonal
    mask = torch.ones((4, 4), dtype=torch.bool)
    mask[:2, :2] = 0
    mask[2:, 2:] = 0

    gaf, gac = get_group_allocation(mask, G=2)
    assert gaf is not None
    assert gac is not None
    assert (gaf[:2] == 2).all()
    assert (gaf[2:] == 1).all()
    assert (gac[:2] == 1).all()
    assert (gac[2:] == 2).all()


def test_get_group_allocation_scattered():
    mask = torch.tensor(
        [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0],], dtype=torch.bool
    )

    gaf, gac = get_group_allocation(mask, G=2)
    assert gaf is not None
    assert gac is not None
    assert (gaf == [2, 1, 2, 1]).all()
    assert (gac == [1, 2, 1, 2]).all()
