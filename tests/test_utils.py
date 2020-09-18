import unittest

import numpy as np
from manifoldmodeling.utils import create_grids


class TestUtils(unittest.TestCase):
    def test_create_grids(self):
        n_dim = 3
        n_grids = 13
        range_ = np.array(
            [
                [-1.0, 1.0],
                [-2.3, 2.3],
                [-3.4, 3.4]
            ]
        )
        list_include_min_max = [True, False]
        list_equal_step = [True, False]
        for include_min_max in list_include_min_max:
            for equal_step in list_equal_step:
                grids, step = create_grids(n_dim=n_dim, n_grids=10, range_=range_,
                                           include_min_max=include_min_max,
                                           equal_step=equal_step,
                                           return_step=True)
                for i, minmax in enumerate(range_):
                    if include_min_max and not equal_step:
                        self.assertEqual(grids[:, i].min(), minmax[0])
                        self.assertEqual(grids[:, i].max(), minmax[1])
                    else:
                        self.assertTrue(grids[:, i].min() >= minmax[0])
                        self.assertTrue(grids[:, i].max() <= minmax[1])


if __name__ == "__main__":
    unittest.main()
