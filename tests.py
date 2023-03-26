import unittest

from preprocessing.constants import (
    get_event_regex,
)
from preprocessing.utils import (
    tokenize_comment,
)


class Preprocessing(unittest.TestCase):
    def test_get_event_regex_and(self):
        expected = "(vot|elect)&(presid)"
        actual = get_event_regex(["voting", "elections"], ["president"], "and")

        self.assertEqual(expected, actual)

    def test_get_event_regex_or(self):
        expected = "vot|elect|presid"
        actual = get_event_regex(["vote", "election"], ["president"], "or")
        self.assertEqual(expected, actual)

    def test_tokenize_comment(self):
        expected = "thi is a test com"
        actual = tokenize_comment("This is a test comment.")

        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
