import unittest

import scipy.sparse as sp
import numpy as np
import pandas as pd

from preprocessing.constants import (
    get_event_regex,
)
from preprocessing.utils import (
    tokenize_comment,
)

from polarization.utils import (
    calculate_leaveout_polarization,
    calculate_polarization,
)


class Preprocessing(unittest.TestCase):
    def test_get_event_regex_and(self):
        expected = r"\b(?:vot|elect)\b.*\b(?:presid hous|ballot)\b|\b(?:presid hous|ballot)\b.*\b(?:vot|elect)\b"
        actual = get_event_regex(
            ["voting", "elections"], ["president houses", "ballots"], "and"
        )

        self.assertEqual(expected, actual)

    def test_get_event_regex_or(self):
        expected = r"\b(?:vot|elect|presid)\b"
        actual = get_event_regex(["vote", "election"], ["president"], "or")
        self.assertEqual(expected, actual)

    def test_contains_regex(self):
        reg = r"\b(?:vot|elect)\b.*\b(?:presid|ballot)\b"
        df = pd.DataFrame(
            {"A": ["elect house ballot", "elect house", "vot presid", "vot elect"]}
        )

        actual = df["A"].str.contains(reg, regex=True)

        self.assertEqual(True, actual[0])
        self.assertEqual(False, actual[1])
        self.assertEqual(True, actual[2])
        self.assertEqual(False, actual[3])

    def test_tokenize_comment(self):
        expected = "thi test com"
        actual = tokenize_comment(comment="This is a test comment.")

        self.assertEqual(expected, actual)


class Polarization(unittest.TestCase):
    def test_calculate_polarization(self):
        comments = pd.DataFrame(
            {
                "tokens": [
                    "ball day",
                    "day day",
                    "apple ball car day day day",
                    "apple ball ball day day",
                    "apple car day day",
                    "apple day day day",
                ],
                "id": [0, 1, 2, 3, 4, 5],
                "author": ["A", "A", "B", "C", "D", "E"],
                "party": ["dem", "dem", "dem", "dem", "rep", "rep"],
            }
        )

        (
            total_polarization,
            _,
            _,
        ), (
            _,
            _,
        ) = calculate_polarization(
            comments,
            ngram_range=(1, 1),
            event_vocab={"ball": 0, "day": 1, "apple": 2, "car": 3},
            equalize_users=False,
        )

        self.assertAlmostEqual(0.5239817492622929, total_polarization)

    def test_calculate_leaveout_polarization(self):
        c_0_D = [0, 1, 0, 3]
        c_1_D = [1, 1, 1, 3]
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

        nr_dem_users = dem_user_term_matrix.shape[0]

        q_0_D = np.array(c_0_D) / sum(c_0_D)
        q_D_0 = (np.array(c_1_D) + np.array(c_2_D)) / (sum(c_1_D) + sum(c_2_D))

        q_1_D = np.array(c_1_D) / sum(c_1_D)
        q_D_1 = (np.array(c_0_D) + np.array(c_2_D)) / (sum(c_0_D) + sum(c_2_D))

        q_2_D = np.array(c_2_D) / sum(c_2_D)
        q_D_2 = (np.array(c_0_D) + np.array(c_1_D)) / (sum(c_0_D) + sum(c_1_D))

        q_D = (np.array(c_0_D) + np.array(c_1_D) + np.array(c_2_D)) / (
            sum(c_0_D) + sum(c_1_D) + sum(c_2_D)
        )

        c_0_R = [1, 0, 1, 2]
        c_1_R = [1, 0, 0, 3]

        rep_user_term_matrix = sp.csr_matrix(
            [
                c_0_R,
                c_1_R,
            ]
        )

        nr_rep_users = rep_user_term_matrix.shape[0]

        q_0_R = np.array(c_0_R) / sum(c_0_R)
        q_R_0 = np.array(c_1_R) / sum(c_1_R)

        q_1_R = np.array(c_1_R) / sum(c_1_R)
        q_R_1 = np.array(c_0_R) / sum(c_0_R)

        q_R = (np.array(c_0_R) + np.array(c_1_R)) / (sum(c_0_R) + sum(c_1_R))

        pol_0_D = np.dot(q_0_D, q_D_0 / (q_D_0 + q_R))
        pol_1_D = np.dot(q_1_D, q_D_1 / (q_D_1 + q_R))
        pol_2_D = np.dot(q_2_D, q_D_2 / (q_D_2 + q_R))

        pol_0_R = np.dot(q_0_R, q_R_0 / (q_R_0 + q_D))
        pol_1_R = np.dot(q_1_R, q_R_1 / (q_R_1 + q_D))

        expected_total_polarization = 0.5 * (
            1 / nr_dem_users * (pol_0_D + pol_1_D + pol_2_D)
            + 1 / nr_rep_users * (pol_0_R + pol_1_R)
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
