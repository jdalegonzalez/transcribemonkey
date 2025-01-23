import unittest

from utils import separate_english

class TestUtilityMethods(unittest.TestCase):
    def test_separate(self):

        # Test to make sure the separation works as expected
        str = "  Okay,好。 反正,anyways, 你去了。"
        expected_str = "Okay, 好。 反正, anyways, 你去了。"
        result_str, _ = separate_english(str)
        self.assertEqual(expected_str, result_str)

        # Test to make sure non-chinese strings are largely
        # left alone
        str = "Hey Dad, great to see you."
        self.assertEqual(("  " + str, ""), separate_english("  " + str, False))
        self.assertEqual((str, ""), separate_english("   " + str))

if __name__ == "__main__": unittest.main()