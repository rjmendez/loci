"""drop_self_echo: a memory item that is the same question returning, or a bare-JSON answer, is dropped.

Mutation runs (each reverted after):
  - filter returns context unchanged -> test_same_question_echo_is_dropped FAILED
  - JSON-answer rule removed          -> test_bare_json_answer_is_dropped FAILED
"""
import importlib.util, os, unittest

spec = importlib.util.spec_from_file_location("loci_rag", os.path.join(os.path.dirname(__file__), "__init__.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
drop = mod.drop_self_echo

QUESTION = "What did we decide about the relay earlier? Check Loci memory before answering."
ECHO = ('[1 id=16671ef6 score=1.000 src=litellm origin=hermes_memory confidence=low] Q: What did we decide '
        'about the relay earlier? Check Loci memory before answering. A: {"title": "Check Loci Memory"}\n')
REAL = ('[2 id=44c2e899 score=0.812 src=dama_gotchi_code] Q: How is the relay firmware flashed? '
        'A: Use the 921600 baud loader on ttyUSB0, then verify the checksum.\n')
JSON_ANSWER = ('[3 id=aaaa score=0.9 src=litellm] Q: which port serves the relay admin page '
               'A: {"title": "Relay port"}\n')


class SelfEchoTest(unittest.TestCase):
    def test_same_question_echo_is_dropped(self):
        out = drop(ECHO + REAL, QUESTION)
        self.assertNotIn("Check Loci Memory", out)
        self.assertIn("921600 baud loader", out)            # positive twin: the real answer stays

    def test_bare_json_answer_is_dropped(self):
        out = drop(JSON_ANSWER + REAL, "Which port serves the relay admin page?")
        self.assertNotIn('"title"', out)
        self.assertIn("921600 baud loader", out)

    def test_different_question_with_real_answer_is_kept(self):
        self.assertEqual(drop(REAL, QUESTION).strip(), REAL.strip())

    def test_empty_context_is_unchanged(self):
        self.assertEqual(drop("", QUESTION), "")


if __name__ == "__main__":
    unittest.main()
