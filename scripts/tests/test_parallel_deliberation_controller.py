from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts'))

from parallel_deliberation_controller import ParallelDeliberationController


def test_parallel_deliberation_prefers_high_confidence_conflict_resolution():
    controller = ParallelDeliberationController()
    result = controller.deliberate(
        'auth and cache policy',
        [
            {
                'agent_id': 'agent-a',
                'subtask': 'Check auth MFA',
                'claim': 'Auth does not require MFA.',
                'confidence': 'high',
                'evidence': 'Config says MFA optional',
                'provenance': {'source': 'cheap-tier', 'model': 'qwen2.5:3b'},
            },
            {
                'agent_id': 'agent-b',
                'subtask': 'Check auth MFA',
                'claim': 'Auth requires MFA for privileged access.',
                'confidence': 'medium',
                'evidence': 'Policy check',
                'provenance': {'source': 'escalated-tier', 'model': 'strong-model'},
            },
        ],
    )

    assert result['degraded'] is False
    assert result['decision'] == 'arbitrated'
    assert result['subtasks'][0]['winner']['agent_id'] == 'agent-a'
    assert result['subtasks'][0]['winner']['claim'] == 'Auth does not require MFA.'
    assert len(result['conflicts']) == 1
    assert result['provenance'][0]['agent_id'] == 'agent-a'
    assert result['provenance'][1]['agent_id'] == 'agent-b'


def test_confidence_outranks_input_order_and_agent_name():
    # In the test above the high-confidence agent is also first in the input and
    # first alphabetically, so a ranking that ignored confidence still picked it.
    # Here the high-confidence opinion comes from agent-z, listed last.
    controller = ParallelDeliberationController()
    result = controller.deliberate(
        'auth and cache policy',
        [
            {'agent_id': 'agent-a', 'subtask': 'Check auth MFA', 'claim': 'Auth requires MFA for privileged access.',
             'confidence': 'low', 'evidence': 'hunch', 'provenance': {'source': 'cheap-tier'}},
            {'agent_id': 'agent-m', 'subtask': 'Check auth MFA', 'claim': 'Auth requires MFA for privileged access.',
             'confidence': 'medium', 'evidence': 'Policy check', 'provenance': {'source': 'cheap-tier'}},
            {'agent_id': 'agent-z', 'subtask': 'Check auth MFA', 'claim': 'Auth does not require MFA.',
             'confidence': 'high', 'evidence': 'Config says MFA optional', 'provenance': {'source': 'escalated-tier'}},
        ],
    )

    assert result['subtasks'][0]['winner']['agent_id'] == 'agent-z'
    assert result['subtasks'][0]['winner']['claim'] == 'Auth does not require MFA.'


def test_parallel_deliberation_fails_open_on_empty_input():
    controller = ParallelDeliberationController()
    result = controller.deliberate('cache policy', [])

    assert result['degraded'] is True
    assert result['decision'] == 'no_reliable_opinion'
    assert result['participants'] == 0
    assert result['subtasks'] == []
    assert result['conflicts'] == []
