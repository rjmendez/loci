"""findings.jsonl is a mixed append log.

Alongside real findings it carries access-tracking rows, written once per read.
They hold no text, and being the newest rows they crowd out any findings[-N:]
slice: measured across the corpus, 3,681 of 6,610 records (55.7%) are access
rows, all text-less. For one investigation all 20 of the last 20 records were
access rows, so the summary ladder handed the model twenty empty bullets and got
back a summary invented from the title.

The same slice feeds recent_findings on every full-fidelity investigation_load,
and total_findings counted them too.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from investigation_tools import _only_findings


class OnlyFindingsTest(unittest.TestCase):
    def test_drops_access_rows(self):
        recs = [
            {"record_type": "observed", "text": "a real finding"},
            {"record_type": "access", "last_accessed": 1787766238, "query": "q"},
            {"record_type": "inferred", "text": "another"},
        ]
        self.assertEqual([r["record_type"] for r in _only_findings(recs)],
                         ["observed", "inferred"])

    def test_keeps_every_real_finding_type(self):
        # The corpus census: observed, inferred, gap, procedure, assumed.
        kinds = ["observed", "inferred", "gap", "procedure", "assumed"]
        recs = [{"record_type": k, "text": "t"} for k in kinds]
        self.assertEqual(len(_only_findings(recs)), len(kinds))

    def test_falls_back_to_the_type_key(self):
        # Older records carry "type" rather than "record_type".
        self.assertEqual(_only_findings([{"type": "access"}]), [])
        self.assertEqual(len(_only_findings([{"type": "observed", "text": "t"}])), 1)

    def test_keeps_records_with_no_type_at_all(self):
        # Unknown is not the same as non-finding; dropping these would lose data.
        self.assertEqual(len(_only_findings([{"text": "no type given"}])), 1)

    def test_ignores_non_dict_rows(self):
        self.assertEqual(_only_findings(["junk", None, 3]), [])

    def test_an_all_access_tail_leaves_nothing(self):
        # The exact shape that produced the invented summary.
        recs = [{"record_type": "access"} for _ in range(20)]
        self.assertEqual(_only_findings(recs), [])


if __name__ == "__main__":
    unittest.main()
