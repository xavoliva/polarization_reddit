import unittest

import scipy.sparse as sp
import numpy as np

from preprocessing.constants import (
    get_event_regex,
)
from preprocessing.utils import (
    tokenize_comment,
)

from polarization.utils import (
    calculate_leaveout_polarization,
)


class Preprocessing(unittest.TestCase):
    def test_get_event_regex_and(self):
        expected = "(?:vot|elect).*(?:presid elect|ballot)"
        actual = get_event_regex(
            ["voting", "elections"], ["president election", "ballots"], "and"
        )

        self.assertEqual(expected, actual)

    def test_get_event_regex_or(self):
        expected = "vot|elect|presid"
        actual = get_event_regex(["vote", "election"], ["president"], "or")
        self.assertEqual(expected, actual)

    def test_tokenize_comment(self):
        expected = "thi is a test com"
        actual = tokenize_comment("This is a test comment.")

        self.assertEqual(expected, actual)

    def test_calculate_leaveout_polarization(self):
        c_0_D = [0, 1, 0, 3]
        c_1_D = [1, 1, 0, 3]
        c_2_D = [1, 2, 0, 2]
        dem_user_term_matrix = sp.csr_matrix(
            np.array(
                [
                    c_0_D,
                    c_1_D,
                    c_2_D,
                ]
            )
        )

        q_0_D = np.array(c_0_D) / sum(c_0_D)
        q_D_0 = (np.array(c_1_D) + np.array(c_2_D)) / 10

        q_1_D = np.array(c_1_D) / sum(c_1_D)
        q_D_1 = (np.array(c_0_D) + np.array(c_2_D)) / 9

        q_2_D = np.array(c_2_D) / sum(c_2_D)
        q_D_2 = (np.array(c_0_D) + np.array(c_1_D)) / 9

        q_D = (np.array(c_0_D) + np.array(c_1_D) + np.array(c_2_D)) / 14

        c_0_R = [1, 0, 1, 2]
        c_1_R = [1, 0, 0, 3]

        rep_user_term_matrix = sp.csr_matrix(
            [
                c_0_R,
                c_1_R,
            ]
        )

        q_0_R = np.array(c_0_R) / sum(c_0_R)
        q_R_0 = np.array(c_1_R) / 4

        q_1_R = np.array(c_1_R) / sum(c_1_R)
        q_R_1 = np.array(c_0_R) / 4

        q_R = (np.array(c_0_R) + np.array(c_1_R)) / 8

        pol_0_D = np.dot(q_0_D, np.nan_to_num(q_D_0 / (q_D_0 + q_R)))
        pol_1_D = np.dot(q_1_D, np.nan_to_num(q_D_1 / (q_D_1 + q_R)))
        pol_2_D = np.dot(q_2_D, np.nan_to_num(q_D_2 / (q_D_2 + q_R)))

        pol_0_R = np.dot(q_0_R, np.nan_to_num(q_R_0 / (q_R_0 + q_D)))
        pol_1_R = np.dot(q_1_R, np.nan_to_num(q_R_1 / (q_R_1 + q_D)))

        expected_total_polarization = 0.5 * (
            1 / 3 * (pol_0_D + pol_1_D + pol_2_D) + 1 / 2 * (pol_0_R + pol_1_R)
        )

        (
            total_polarization,
            dem_polarization,
            rep_polarization,
        ) = calculate_leaveout_polarization(
            dem_user_term_matrix,
            rep_user_term_matrix,
        )

        self.assertEqual([pol_0_D, pol_1_D, pol_2_D], dem_polarization)
        self.assertEqual([pol_0_R, pol_1_R], rep_polarization)
        self.assertAlmostEqual(expected_total_polarization, total_polarization)


if __name__ == "__main__":
    unittest.main()
