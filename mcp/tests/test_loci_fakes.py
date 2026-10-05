"""The shared fakes must not hide misuse: an unscripted model call is recorded, not raised."""
from loci_fakes import fake_lazy_generate


def test_fake_lazy_generate_answers_in_order_then_records_overflow():
    gen = fake_lazy_generate([{"text": "a", "ok": True}, {"text": "b", "ok": True}])
    assert [gen("p1"), gen("p2")] == [{"text": "a", "ok": True}, {"text": "b", "ok": True}]
    assert gen.overflow == []
    # A third call is answered like an unavailable model and recorded, so code that
    # wraps the model call in `except Exception` cannot swallow it into a pass.
    assert gen("p3") == {"text": "", "ok": False, "why": "fake_lazy_generate: no scripted response left"}
    assert gen.prompts == ["p1", "p2", "p3"]
    assert gen.overflow == ["p3"]


def test_fake_lazy_generate_callable_answers_every_call():
    gen = fake_lazy_generate(lambda prompt: {"text": prompt.upper(), "ok": True})
    assert gen("x") == {"text": "X", "ok": True}
    assert (gen.prompts, gen.overflow) == (["x"], [])
