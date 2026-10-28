"""Preregistration harness for learned routing policies (Tripwire-Governor S15). Offline, stdlib only.

* ``policy_prereg.stats``    -- the pass-rule arithmetic: required n per arm, Wilson bound, Holm, guards.
* ``policy_prereg.registry`` -- registration records committed by hash (write once) and the
  held-out-scored-once ledger.

Import as ``policy_prereg`` with ``scripts/`` on ``sys.path`` (``python scripts/...`` already has it).
"""
