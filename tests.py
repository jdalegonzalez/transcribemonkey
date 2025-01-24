import unittest

from utils import separate_english, WhisperSegment, PropertyDict
from transcribe import pyannote_collapse_convert_segments

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

class TestWhisperSegment(unittest.TestCase):
    def test_simple_assignment(self):
        seg = WhisperSegment()
        id = 23
        seg.id = id
        self.assertEqual(seg.id, id)
        seg.text = "Charlie"
        self.assertEqual(seg.text, "")

        seg.words = ["Hi", " ", "Mom"]
        self.assertEqual(seg.text, "Hi Mom")
        seg = WhisperSegment({'start': 0, 'end': 10, 'speaker': 'Tom', 'words': []})
        self.assertEqual(seg.words, [])
        seg = WhisperSegment({'start': 0, 'end': 10, 'speaker': 'Tom', 'words': ["hi"]})
        self.assertEqual(seg.words, [{'word': "hi"}])
        self.assertEqual(seg.text, "hi")
        seg.words = ['Me', 'You']
        self.assertEqual(seg.words, [{'word': "Me"},{'word': "You"}])
        self.assertEqual(seg.text, "MeYou")
        seg.words = []
        self.assertEqual(seg.words, [])
        self.assertFalse(seg.words)
        word = PropertyDict({'start': 0, 'end': 10, 'word': "Me"})
        seg.words = [word]
        self.assertTrue(seg.text, 'Me')
        seg.words = [{'start': 0, 'end': 10, 'word': "Us"}]
        self.assertTrue(seg.text, 'Us')
        seg.words = ["Them"]
        self.assertTrue(seg.text, 'Them')

class TestCollapsePyannote(unittest.TestCase):
    def test_collapse_segments(self):
        segments = [
            WhisperSegment({'start': 0, 'end': 10, 'speaker': 'Dale', 'speaker_confidence': 1}),
            WhisperSegment({'start': 10, 'end': 20, 'speaker': 'Dale', 'speaker_confidence': 1, 'words': ['Hi', ' Dad']}),
            WhisperSegment({'start': 20, 'end': 30, 'speaker': 'Dale', 'speaker_confidence': 1, 'words': []}),
            WhisperSegment({'start': 30, 'end': 40, 'speaker': 'Dale', 'speaker_confidence': 1, 'words': ['Hi', ' Mom']}),
            WhisperSegment({'start': 40, 'end': 50, 'speaker': 'Dale', 'speaker_confidence': 1, 'words': []}),
        ]
        results = pyannote_collapse_convert_segments(None, segments)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1].id, str(3))
        self.assertEqual(results[1].text, "Hi Mom")

if __name__ == "__main__": unittest.main()