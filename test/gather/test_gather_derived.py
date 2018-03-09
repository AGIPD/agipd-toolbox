import numpy as np
import os
import sys
import unittest

try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
SRC_PATH = os.path.join(BASE_PATH, "src")
GATHER_PATH = os.path.join(SRC_PATH, "gather")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

if GATHER_PATH not in sys.path:
    sys.path.insert(0, GATHER_PATH)

import utils  # noqa

from gather_pcdrs import AgipdGatherPcdrs  # noqa E402
from gather_drscs import AgipdGatherDrscs  # noqa E402


#
# PCDRS
#
class GatherPcdrsTests(unittest.TestCase):
    # per test
    def setUp(self):

        class Gather(AgipdGatherPcdrs):
            def __init__(self):
                self.n_runs = 8

                self.n_rows = 128
                self.n_cols = 512

        self.gather_obj = Gather()

    def test_pos_indices_all0(self):

        run_idx = 0
        # reference values
        ref_row = [7, 15, 23, 31, 39, 47, 55, 63,
                   64, 72, 80, 88, 96, 104, 112, 120]
        ref_col = slice(None)

        # get result calculated by set_pos_indices
        res = self.gather_obj.set_pos_indices(run_idx)[0]

        # compare result to reference
        self.assertTrue(np.all(res[0] == ref_row))
        self.assertTrue(np.all(res[1] == ref_col))

    def test_pos_indices_all1(self):

        run_idx = 1
        # reference values
        ref_row = [6, 14, 22, 30, 38, 46, 54, 62,
                   65, 73, 81, 89, 97, 105, 113, 121]
        ref_col = slice(None)

        # get result calculated by set_pos_indices
        res = self.gather_obj.set_pos_indices(run_idx)[0]

        # compare result to reference
        self.assertTrue(np.all(res[0] == ref_row))
        self.assertTrue(np.all(res[1] == ref_col))


#
# DRSCS
#
class GatherDrscsTest(AgipdGatherDrscs):
    def __init__(self, asic):
        self.asic = asic

        self.n_runs = 4

        self.n_rows_total = 128
        self.n_cols_total = 512
        self.asic_size = 64

        if self.asic is None:
            self.n_rows = self.n_rows_total
            self.n_cols = self.n_cols_total

            self.a_row_start = 0
            self.a_row_stop = self.n_rows
            self.a_col_start = 0
            self.a_col_stop = self.n_cols

        else:
            self.n_rows = self.asic_size
            self.n_cols = self.asic_size

            asic_order = utils.get_asic_order()
            mapped_asic = utils.calculate_mapped_asic(asic=asic,
                                                      asic_order=asic_order)
            (self.a_row_start,
             self.a_row_stop,
             self.a_col_start,
             self.a_col_stop) = utils.determine_asic_border(
                mapped_asic=mapped_asic,
                asic_size=self.asic_size,
                verbose=False
            )


class GatherDrscsTests(unittest.TestCase):

    def test_pos_indices_all0(self):
        self.gather_obj = GatherDrscsTest(asic=None)

        run_idx = 0
        # reference values
        ref_row_top = slice(0, 64)
        ref_col_top = [3, 7, 11, 15, 19, 23, 27, 31,
                       35, 39, 43, 47, 51, 55, 59, 63,
                       67, 71, 75, 79, 83, 87, 91, 95,
                       99, 103, 107, 111, 115, 119, 123, 127,
                       131, 135, 139, 143, 147, 151, 155, 159,
                       163, 167, 171, 175, 179, 183, 187, 191,
                       195, 199, 203, 207, 211, 215, 219, 223,
                       227, 231, 235, 239, 243, 247, 251, 255,
                       259, 263, 267, 271, 275, 279, 283, 287,
                       291, 295, 299, 303, 307, 311, 315, 319,
                       323, 327, 331, 335, 339, 343, 347, 351,
                       355, 359, 363, 367, 371, 375, 379, 383,
                       387, 391, 395, 399, 403, 407, 411, 415,
                       419, 423, 427, 431, 435, 439, 443, 447,
                       451, 455, 459, 463, 467, 471, 475, 479,
                       483, 487, 491, 495, 499, 503, 507, 511]
        ref_row_bottom = slice(64, 128)
        ref_col_bottom = [0, 4, 8, 12, 16, 20, 24, 28,
                          32, 36, 40, 44, 48, 52, 56, 60,
                          64, 68, 72, 76, 80, 84, 88, 92,
                          96, 100, 104, 108, 112, 116, 120, 124,
                          128, 132, 136, 140, 144, 148, 152, 156,
                          160, 164, 168, 172, 176, 180, 184, 188,
                          192, 196, 200, 204, 208, 212, 216, 220,
                          224, 228, 232, 236, 240, 244, 248, 252,
                          256, 260, 264, 268, 272, 276, 280, 284,
                          288, 292, 296, 300, 304, 308, 312, 316,
                          320, 324, 328, 332, 336, 340, 344, 348,
                          352, 356, 360, 364, 368, 372, 376, 380,
                          384, 388, 392, 396, 400, 404, 408, 412,
                          416, 420, 424, 428, 432, 436, 440, 444,
                          448, 452, 456, 460, 464, 468, 472, 476,
                          480, 484, 488, 492, 496, 500, 504, 508]

        # get result calculated by set_pos_indices
        res = self.gather_obj.set_pos_indices(run_idx)

        # compare result to reference
        self.assertTrue(np.all(res[0][0] == ref_row_top))
        self.assertTrue(np.all(res[0][1] == ref_col_top))
        self.assertTrue(np.all(res[1][0] == ref_row_bottom))
        self.assertTrue(np.all(res[1][1] == ref_col_bottom))

    def test_pos_indices_all1(self):
        self.gather_obj = GatherDrscsTest(asic=None)

        run_idx = 1
        # reference values
        ref_row_top = slice(0, 64)
        ref_col_top = [2, 6, 10, 14, 18, 22, 26, 30,
                       34, 38, 42, 46, 50, 54, 58, 62,
                       66, 70, 74, 78, 82, 86, 90, 94,
                       98, 102, 106, 110, 114, 118, 122, 126,
                       130, 134, 138, 142, 146, 150, 154, 158,
                       162, 166, 170, 174, 178, 182, 186, 190,
                       194, 198, 202, 206, 210, 214, 218, 222,
                       226, 230, 234, 238, 242, 246, 250, 254,
                       258, 262, 266, 270, 274, 278, 282, 286,
                       290, 294, 298, 302, 306, 310, 314, 318,
                       322, 326, 330, 334, 338, 342, 346, 350,
                       354, 358, 362, 366, 370, 374, 378, 382,
                       386, 390, 394, 398, 402, 406, 410, 414,
                       418, 422, 426, 430, 434, 438, 442, 446,
                       450, 454, 458, 462, 466, 470, 474, 478,
                       482, 486, 490, 494, 498, 502, 506, 510]
        ref_row_bottom = slice(64, 128)
        ref_col_bottom = [1, 5, 9, 13, 17, 21, 25, 29,
                          33, 37, 41, 45, 49, 53, 57, 61,
                          65, 69, 73, 77, 81, 85, 89, 93,
                          97, 101, 105, 109, 113, 117, 121, 125,
                          129, 133, 137, 141, 145, 149, 153, 157,
                          161, 165, 169, 173, 177, 181, 185, 189,
                          193, 197, 201, 205, 209, 213, 217, 221,
                          225, 229, 233, 237, 241, 245, 249, 253,
                          257, 261, 265, 269, 273, 277, 281, 285,
                          289, 293, 297, 301, 305, 309, 313, 317,
                          321, 325, 329, 333, 337, 341, 345, 349,
                          353, 357, 361, 365, 369, 373, 377, 381,
                          385, 389, 393, 397, 401, 405, 409, 413,
                          417, 421, 425, 429, 433, 437, 441, 445,
                          449, 453, 457, 461, 465, 469, 473, 477,
                          481, 485, 489, 493, 497, 501, 505, 509]

        # get result calculated by set_pos_indices
        res = self.gather_obj.set_pos_indices(run_idx)

        # compare result to reference
        self.assertTrue(np.all(res[0][0] == ref_row_top))
        self.assertTrue(np.all(res[0][1] == ref_col_top))
        self.assertTrue(np.all(res[1][0] == ref_row_bottom))
        self.assertTrue(np.all(res[1][1] == ref_col_bottom))

    # asic location on module
    #  ____ ____ ____ ____ ____ ____ ____ ____
    # |    |    |    |    |    |    |    |    |
    # | 16 | 15 | 14 | 13 | 12 | 11 | 10 |  9 |
    # |____|____|____|____|____|____|____|____|
    # |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |
    # |____|____|____|____|____|____|____|____|
    #
    def test_pos_indices_asic1_0(self):
        self.gather_obj = GatherDrscsTest(asic=1)

        run_idx = 0
        # reference values
        ref_row = slice(None)
        ref_col = [0, 4, 8, 12, 16, 20, 24, 28,
                   32, 36, 40, 44, 48, 52, 56, 60]

        # get result calculated by set_pos_indices
        res = self.gather_obj.set_pos_indices(run_idx)[0]

        # compare result to reference
        self.assertTrue(np.all(res[0] == ref_row))
        self.assertTrue(np.all(res[1] == ref_col))

    def test_pos_indices_asic1_1(self):
        self.gather_obj = GatherDrscsTest(asic=1)

        run_idx = 1
        # reference values
        ref_row = slice(None)
        ref_col = [1, 5, 9, 13, 17, 21, 25, 29,
                   33, 37, 41, 45, 49, 53, 57, 61]

        # get result calculated by set_pos_indices
        res = self.gather_obj.set_pos_indices(run_idx)[0]

        # compare result to reference
        self.assertTrue(np.all(res[0] == ref_row))
        self.assertTrue(np.all(res[1] == ref_col))

    def test_pos_indices_asic10_0(self):
        self.gather_obj = GatherDrscsTest(asic=10)

        run_idx = 0
        # reference values
        ref_row = slice(None)
        ref_col = [3, 7, 11, 15, 19, 23, 27, 31,
                   35, 39, 43, 47, 51, 55, 59, 63]

        # get result calculated by set_pos_indices
        res = self.gather_obj.set_pos_indices(run_idx)[0]

        # compare result to reference
        self.assertTrue(np.all(res[0] == ref_row))
        self.assertTrue(np.all(res[1] == ref_col))

    def test_pos_indices_asic10_1(self):
        self.gather_obj = GatherDrscsTest(asic=10)

        run_idx = 1
        # reference values
        ref_row = slice(None)
        ref_col = [2, 6, 10, 14, 18, 22, 26, 30,
                   34, 38, 42, 46, 50, 54, 58, 62]

        # get result calculated by set_pos_indices
        res = self.gather_obj.set_pos_indices(run_idx)[0]

        # compare result to reference
        self.assertTrue(np.all(res[0] == ref_row))
        self.assertTrue(np.all(res[1] == ref_col))


if __name__ == "__main__":
    # Run only the tests in the specified classes

    test_classes_to_run = [
        GatherPcdrsTests,
        GatherDrscsTests
    ]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)

    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
