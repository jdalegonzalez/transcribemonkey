import unittest

from utils import is_a_number, separate_english, combine_words, WhisperSegment, PropertyDict
from subsegment import SubSegment
from transcribe import pyannote_collapse_convert_segments

class TestUtilityMethods(unittest.TestCase):
    def test_separate(self):

        # Test to make sure the separation works as expected
        str = "  Okay,好。 反正,anyways, 你去了。"
        expected_str = "Okay, 好。 反正, anyways, 你去了。"
        result_str, _ = separate_english(str)
        self.assertEqual(expected_str, result_str.lstrip())

        # Test to make sure non-chinese strings are largely
        # left alone
        str = "Hey Dad, great to see you."
        self.assertEqual(("  "+str,""), separate_english("  "+str))
        str, _ = combine_words( ['I', "'", 'm', ' ', '很', '开心', '啊', '。'])
        self.assertEqual(str, "I'm 很开心啊。")

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
        seg = WhisperSegment({'start': 0, 'end': 10, 'probability': 1, 'speaker': 'Tom', 'words': []})
        self.assertEqual(seg.words, [])
        seg = WhisperSegment({'start': 0, 'end': 10, 'probability': 1, 'speaker': 'Tom', 'words': ["hi"]})
        self.assertEqual(seg.words, [{'word': "hi"}])
        self.assertEqual(seg.text, "hi")
        seg.words = ['Me', 'You']
        self.assertEqual(seg.words, [{'word': "Me"},{'word': "You"}])
        self.assertEqual(seg.text, "MeYou")
        seg.words = []
        self.assertEqual(seg.words, [])
        self.assertFalse(seg.words)
        word = PropertyDict({'start': 0, 'end': 10, 'probability': 1, 'word': "Me"})
        seg.words = [word]
        self.assertTrue(seg.text, 'Me')
        seg.words = [{'start': 0, 'end': 10, 'word': "Us", 'probability': 1, }]
        self.assertTrue(seg.text, 'Us')
        seg.words = ["Them"]
        self.assertTrue(seg.text, 'Them')

class TestCollapsePyannote(unittest.TestCase):
    def test_collapse_segments(self):
        segments = [
            WhisperSegment({'start': 0, 'end': 10, 'speaker': 'Debbie', 'speaker_confidence': 1}),
            WhisperSegment({'start': 10, 'end': 20, 'speaker': 'Dale', 'speaker_confidence': 1, 
                'words': [
                    {'start':10, 'end': 15, 'word': ' Hi', 'probability': 1},
                    {'start': 15, 'end': 20, 'word': ' Dad', 'probability': 1}
                ]}),
            WhisperSegment({'start': 20, 'end': 30, 'speaker': 'Jeffry', 'speaker_confidence': 1, 
                'words': []}),
            WhisperSegment({'start': 30, 'end': 40, 'speaker': 'Dale', 'speaker_confidence': 1, 
                'words': [
                    {'start': 30, 'end': 35, 'word': ' Hi', 'probability': 1}, 
                    {'start': 35, 'end': 40, 'word':' Mom', 'probability': 1}
                ]}),
            WhisperSegment({'start': 40, 'end': 50, 'speaker': 'Sarah', 'speaker_confidence': 1, 
                'words': []}),
        ]
        results = pyannote_collapse_convert_segments(None, segments)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, str(0))
        self.assertEqual(results[0].text, " Hi Dad Hi Mom")
        left = WhisperSegment({
            'id': 0,
            'start': 30, 'end': 40, 'speaker': 'Dale', 'speaker_confidence': 1, 
            'words': [{'probability': 1, 'start': 30, 'end': 35, 'word': 'Hi'},{'probability': 1, 'start': 35, 'end': 40, 'word': ' Mom'}]
        })
        right = WhisperSegment({
            'id': 1,
            'start': 40, 'end': 50, 'speaker': 'Dale', 'speaker_confidence': 1, 
            'words': [{'probability': 1, 'start': 40, 'end': 45, 'word':' and'},{'probability': 1, 'start': 45, 'end': 50, 'word': ' Dad'}]
        })
        merge = WhisperSegment({
            'id': 0,
            'start': 30, 'end': 50, 'speaker': 'Dale', 'speaker_confidence': 1, 
            'words': [
                {'probability': 1, 'start': 30, 'end': 35, 'word':'Hi'},
                {'probability': 1, 'start': 35, 'end': 40, 'word':' Mom'},
                {'probability': 1, 'start': 40, 'end': 45, 'word':' and'},
                {'probability': 1, 'start': 45, 'end': 50, 'word': ' Dad'}
        ]})
        left.merge(right)
        self.assertEqual(left.text,merge.text)
        left = WhisperSegment({
            'id': 0,
            'start': 30, 'end': 40, 'speaker': 'Dale', 'speaker_confidence': 1, 'words': [' Hi', ' Mom']
        })
        right = WhisperSegment({
            'id': 1,
            'start': 40, 'end': 50, 'speaker': 'Dale', 'speaker_confidence': 1, 'words': [' and', ' Dad']
        })
        merge = WhisperSegment({
            'id': 0,
            'start': 30, 'end': 50, 'speaker': 'Dale', 'speaker_confidence': 1, 
            'words': [
                {'probability': 1, 'start': 30, 'end': 35, 'word':' Hi'},
                {'probability': 1, 'start': 35, 'end': 40, 'word': ' Mom'},
                {'probability': 1, 'start': 40, 'end': 45, 'word':' and'},
                {'probability': 1, 'start': 45, 'end': 50, 'word':' Dad'},
            ]})
        right.merge(left)
        self.assertEqual(right.text,merge.text)

class TestVariousUtilFunctions(unittest.TestCase):
    def test_is_a_number(self):
        vals = [
            ("1234.0",True),("95",True),("", False),(None, False),("-0.0",False),("192.0.0.1",False),("192.0 - 10",False),
            ("Thirty-One",True),("One Hundred",True),("Twelve",True),("Oranges",False)
        ]
        for val,expected in vals:
            self.assertEqual(is_a_number(val), expected, msg=f'Is "{val}" a number?')

if __name__ == "__main__": unittest.main()