import unittest
from tools.tests.dependencies import require
require('torch','tokenizers','regex','numpy')
from tools.correction.byte_candidates import package,summarize
from tools.correction.byte_data import encode

class ByteCandidatesTests(unittest.TestCase):
    def test_deduplicates_and_always_keeps_copy_candidate(self):
        rows=package('আমি যাব।',[encode('আমি যাই।'),encode('আমি যাই।')],[-.1,-.2])
        self.assertEqual([r['answer'] for r in rows],['আমি যাই।','আমি যাব।'])
        self.assertIsNone(rows[-1]['beam_score'])
        with self.assertRaises(ValueError):package('ক',[encode('খ')],[float('nan')])

    def test_protected_text_invalid_utf8_and_oracle_are_separate(self):
        rows=package('abc 123',[encode('abc 124'),[0,200,1],encode('abc 123')],[-.1,-.2,-.3])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['answer'],'abc 123')
        report=summarize([dict(source='ক',target='খ',kind='error',candidates=[dict(answer='গ'),dict(answer='খ'),dict(answer='ক')])])
        self.assertEqual(report['all']['top1_repairs'],0)
        self.assertEqual(report['all']['oracle_repairs'],1)
