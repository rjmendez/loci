#!/usr/bin/env python3
"""
Weak supervision label generation from gate outcomes.

Implements deterministic, rule-based weak label generation from normalized
trainlog records. Labels are generated from gate outcomes:
- abstain_required: model should abstain
- escalate_required: escalation to larger model needed
- provenance_risk: provenance/traceability issues
- contradiction_risk: contradictory inputs or outputs
- promotion_risk: model should not be promoted

Privacy contract: input records MUST be scrubbed and normalized per
bc-trainlog-privacy-scrub spec. Fail-closed if contract not met.
"""

import json
import sys
import hashlib
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime


class WeakLabel(Enum):
    """Weak supervision label types."""
    ABSTAIN_REQUIRED = "abstain_required"
    ESCALATE_REQUIRED = "escalate_required"
    PROVENANCE_RISK = "provenance_risk"
    CONTRADICTION_RISK = "contradiction_risk"
    PROMOTION_RISK = "promotion_risk"


@dataclass
class WeakLabelRule:
    """Rule for generating a weak label."""
    label: WeakLabel
    criteria: str  # human-readable description
    trigger_fields: List[str]  # trainlog fields that trigger this rule
    conditions: Dict[str, Any]  # field-level conditions


@dataclass
class WeakLabelResult:
    """Result of weak label generation for a trainlog record."""
    trainlog_id: str  # deterministic ID of source trainlog record
    weak_labels: List[WeakLabel]
    rule_triggers: Dict[str, List[str]]  # label -> rule names that triggered
    confidence: float  # 0.0-1.0, based on number of triggers
    timestamp: str
    schema_version: str = "1.0"


class PrivacyScrubContract:
    """Enforces privacy scrub contract on trainlog records."""

    REQUIRED_SCRUB_MARKERS = {
        "_scrubbed": True,
        "_scrubbed_timestamp": None,  # will be checked for presence
        "_scrubbed_version": None,  # will be checked for presence
    }

    REQUIRED_NORMAL_MARKERS = {
        "_normalized": True,
        "_normalized_timestamp": None,
        "_normalized_version": None,
    }

    @staticmethod
    def validate(record: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that record meets privacy scrub contract.
        
        Returns: (is_valid, reason)
        """
        # Check scrub markers
        for marker, expected_val in PrivacyScrubContract.REQUIRED_SCRUB_MARKERS.items():
            if marker not in record:
                return False, f"Missing scrub marker: {marker}"
            if expected_val is not None and record[marker] != expected_val:
                return False, f"Scrub marker {marker} value mismatch"
        
        # Check normalization markers
        for marker, expected_val in PrivacyScrubContract.REQUIRED_NORMAL_MARKERS.items():
            if marker not in record:
                return False, f"Missing normalization marker: {marker}"
            if expected_val is not None and record[marker] != expected_val:
                return False, f"Normalization marker {marker} value mismatch"
        
        # Check for raw payload leakage (fail-closed)
        sensitive_patterns = ["password", "token", "secret", "api_key", "credential"]
        for key in record.keys():
            key_lower = key.lower()
            for pattern in sensitive_patterns:
                if pattern in key_lower and not key.startswith("_"):
                    # Check value is not a sensitive-looking string
                    val = record[key]
                    if isinstance(val, str) and len(val) > 20:
                        return False, f"Potential raw sensitive data in field: {key}"
        
        return True, "OK"


class WeakLabelRules:
    """Rule definitions for weak label generation."""

    @staticmethod
    def get_all_rules() -> List[WeakLabelRule]:
        """Return all weak label rules."""
        return [
            # ABSTAIN_REQUIRED rules
            WeakLabelRule(
                label=WeakLabel.ABSTAIN_REQUIRED,
                criteria="Low model confidence score",
                trigger_fields=["confidence_score"],
                conditions={"confidence_score": ("lt", 0.5)}
            ),
            WeakLabelRule(
                label=WeakLabel.ABSTAIN_REQUIRED,
                criteria="Input entropy above threshold",
                trigger_fields=["input_entropy"],
                conditions={"input_entropy": ("gt", 0.8)}
            ),
            WeakLabelRule(
                label=WeakLabel.ABSTAIN_REQUIRED,
                criteria="Gate output indicates uncertainty",
                trigger_fields=["gate_outcomes"],
                conditions={"gate_outcomes": ("contains", "uncertainty")}
            ),
            
            # ESCALATE_REQUIRED rules
            WeakLabelRule(
                label=WeakLabel.ESCALATE_REQUIRED,
                criteria="Gate explicitly marks for escalation",
                trigger_fields=["gate_outcomes"],
                conditions={"gate_outcomes": ("contains", "escalate_required")}
            ),
            WeakLabelRule(
                label=WeakLabel.ESCALATE_REQUIRED,
                criteria="Specialist model confidence too low",
                trigger_fields=["model_type", "confidence_score"],
                conditions={
                    "model_type": ("eq", "specialist"),
                    "confidence_score": ("lt", 0.3),
                }
            ),
            
            # PROVENANCE_RISK rules
            WeakLabelRule(
                label=WeakLabel.PROVENANCE_RISK,
                criteria="Gate indicates provenance issues",
                trigger_fields=["gate_outcomes"],
                conditions={"gate_outcomes": ("contains", "provenance_risk")}
            ),
            WeakLabelRule(
                label=WeakLabel.PROVENANCE_RISK,
                criteria="Missing provenance chain link",
                trigger_fields=["provenance_chain_length", "required_chain_length"],
                conditions={
                    "provenance_chain_length": ("lt_field", "required_chain_length"),
                }
            ),
            WeakLabelRule(
                label=WeakLabel.PROVENANCE_RISK,
                criteria="Provenance timestamp gap too large",
                trigger_fields=["provenance_max_gap_ms"],
                conditions={"provenance_max_gap_ms": ("gt", 60000)}  # 60 seconds
            ),
            
            # CONTRADICTION_RISK rules
            WeakLabelRule(
                label=WeakLabel.CONTRADICTION_RISK,
                criteria="Gate indicates contradictions",
                trigger_fields=["gate_outcomes"],
                conditions={"gate_outcomes": ("contains", "contradiction_risk")}
            ),
            WeakLabelRule(
                label=WeakLabel.CONTRADICTION_RISK,
                criteria="Input and output agreement below threshold",
                trigger_fields=["input_output_agreement"],
                conditions={"input_output_agreement": ("lt", 0.6)}
            ),
            WeakLabelRule(
                label=WeakLabel.CONTRADICTION_RISK,
                criteria="Contradictory specialist opinions",
                trigger_fields=["specialist_vote_entropy"],
                conditions={"specialist_vote_entropy": ("gt", 0.9)}
            ),
            
            # PROMOTION_RISK rules
            WeakLabelRule(
                label=WeakLabel.PROMOTION_RISK,
                criteria="Gate indicates promotion risk",
                trigger_fields=["gate_outcomes"],
                conditions={"gate_outcomes": ("contains", "promotion_risk")}
            ),
            WeakLabelRule(
                label=WeakLabel.PROMOTION_RISK,
                criteria="Holdout performance below threshold",
                trigger_fields=["holdout_accuracy"],
                conditions={"holdout_accuracy": ("lt", 0.70)}
            ),
            WeakLabelRule(
                label=WeakLabel.PROMOTION_RISK,
                criteria="High error rate in recent examples",
                trigger_fields=["recent_error_rate"],
                conditions={"recent_error_rate": ("gt", 0.1)}
            ),
            WeakLabelRule(
                label=WeakLabel.PROMOTION_RISK,
                criteria="Policy violation recorded",
                trigger_fields=["policy_violations"],
                conditions={"policy_violations": ("gt", 0)}
            ),
        ]


class WeakLabelGenerator:
    """Generate weak supervision labels from normalized trainlog records."""

    def __init__(self):
        self.rules = WeakLabelRules.get_all_rules()
        self.contract = PrivacyScrubContract()

    def generate_labels(self, record: Dict[str, Any]) -> WeakLabelResult:
        """
        Generate weak labels for a trainlog record.
        
        Args:
            record: Normalized and scrubbed trainlog record
            
        Returns:
            WeakLabelResult with labels, triggers, and confidence
            
        Raises:
            ValueError: if privacy scrub contract violated
        """
        # Enforce privacy contract (fail-closed)
        is_valid, reason = self.contract.validate(record)
        if not is_valid:
            raise ValueError(f"Privacy scrub contract violated: {reason}")
        
        # Generate deterministic ID for this record
        trainlog_id = self._generate_deterministic_id(record)
        
        # Evaluate all rules
        triggered_labels: Dict[WeakLabel, List[str]] = {}
        for rule in self.rules:
            if self._evaluate_rule(rule, record):
                if rule.label not in triggered_labels:
                    triggered_labels[rule.label] = []
                triggered_labels[rule.label].append(rule.criteria)
        
        # Convert to result format
        weak_labels = list(triggered_labels.keys())
        confidence = self._compute_confidence(triggered_labels, record)
        
        rule_triggers_dict = {label.value: triggers for label, triggers in triggered_labels.items()}
        
        result = WeakLabelResult(
            trainlog_id=trainlog_id,
            weak_labels=weak_labels,
            rule_triggers=rule_triggers_dict,
            confidence=confidence,
            timestamp=datetime.now(datetime.UTC).isoformat() if hasattr(datetime, 'UTC') else datetime.utcnow().isoformat() + "Z",
        )
        
        return result

    def _evaluate_rule(self, rule: WeakLabelRule, record: Dict[str, Any]) -> bool:
        """Evaluate if a rule's conditions are met in the record."""
        for field_name, condition_tuple in rule.conditions.items():
            op, value = condition_tuple
            field_value = self._get_nested_value(record, field_name)
            
            if field_value is None:
                return False
            
            if op == "contains":
                if isinstance(field_value, list) and value in field_value:
                    continue
                elif isinstance(field_value, str) and value in field_value:
                    continue
                else:
                    return False
            elif op == "lt":
                if isinstance(field_value, (int, float)) and isinstance(value, (int, float)):
                    if field_value < value:
                        continue
                    else:
                        return False
                else:
                    return False
            elif op == "lt_field":
                # Compare two fields
                other_value = self._get_nested_value(record, value)
                if other_value is not None and isinstance(field_value, (int, float)) and isinstance(other_value, (int, float)):
                    if field_value < other_value:
                        continue
                    else:
                        return False
                else:
                    return False
            elif op == "gt":
                if isinstance(field_value, (int, float)) and isinstance(value, (int, float)):
                    if field_value > value:
                        continue
                    else:
                        return False
                else:
                    return False
            elif op == "eq":
                if field_value == value:
                    continue
                else:
                    return False
            else:
                return False
        
        return True

    def _get_nested_value(self, record: Dict[str, Any], field: str) -> Any:
        """Get nested field value from record using dot notation."""
        parts = field.split(".")
        value = record
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value

    def _generate_deterministic_id(self, record: Dict[str, Any]) -> str:
        """Generate deterministic ID for trainlog record."""
        # Use _normalized_id if available, otherwise generate from content hash
        if "_normalized_id" in record:
            return record["_normalized_id"]
        
        # Hash relevant fields to create deterministic ID
        hash_fields = ["_normalized_timestamp", "_source_log_id", "_event_sequence"]
        hash_input = json.dumps(
            {k: record.get(k, "") for k in hash_fields},
            sort_keys=True,
            default=str
        )
        content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return f"trainlog-{content_hash}"

    def _compute_confidence(self, triggered_labels: Dict[WeakLabel, List[str]], record: Dict[str, Any]) -> float:
        """Compute confidence score for the weak labels."""
        if not triggered_labels:
            return 1.0  # High confidence in no-labels case
        
        # Average trigger count across all labels
        total_triggers = sum(len(triggers) for triggers in triggered_labels.values())
        num_labels = len(triggered_labels)
        avg_triggers = total_triggers / num_labels if num_labels > 0 else 0
        
        # Confidence based on redundancy (more triggers = higher confidence)
        max_possible_triggers = len([r for r in self.rules if r.label in triggered_labels])
        if max_possible_triggers > 0:
            confidence = min(0.99, total_triggers / max_possible_triggers)
        else:
            confidence = 0.5
        
        return confidence

    def process_batch(self, records: List[Dict[str, Any]]) -> Tuple[List[WeakLabelResult], List[Dict]]:
        """Process batch of trainlog records."""
        results = []
        errors = []
        
        for i, record in enumerate(records):
            try:
                result = self.generate_labels(record)
                results.append(result)
            except ValueError as e:
                errors.append({
                    "record_index": i,
                    "error": str(e),
                    "record_id": record.get("_normalized_id", "unknown")
                })
        
        return results, errors

    def output_jsonl(self, results: List[WeakLabelResult], errors: List[Dict]) -> str:
        """Output results in JSONL format (machine-readable)."""
        lines = []
        
        # Output results
        for result in results:
            obj = asdict(result)
            obj["weak_labels"] = [label.value for label in result.weak_labels]
            lines.append(json.dumps(obj, default=str))
        
        # Output errors with schema
        for error in errors:
            error_obj = {
                "type": "weak_label_error",
                "record_index": error["record_index"],
                "record_id": error["record_id"],
                "error": error["error"],
                "schema_version": "1.0",
            }
            lines.append(json.dumps(error_obj))
        
        return "\n".join(lines)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate weak supervision labels from normalized trainlog records"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with scrubbed/normalized trainlog records",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file with weak label results",
    )
    parser.add_argument(
        "--rules",
        action="store_true",
        help="Print all weak label rules and exit",
    )
    
    args = parser.parse_args()
    
    generator = WeakLabelGenerator()
    
    if args.rules:
        print("Weak Label Rules:")
        print("=" * 80)
        for rule in generator.rules:
            print(f"\n{rule.label.value}:")
            print(f"  Criteria: {rule.criteria}")
            print(f"  Fields: {', '.join(rule.trigger_fields)}")
            print(f"  Conditions: {rule.conditions}")
        return 0
    
    # Process input file
    records = []
    with open(args.input, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    results, errors = generator.process_batch(records)
    
    # Write output
    output_lines = generator.output_jsonl(results, errors)
    with open(args.output, "w") as f:
        f.write(output_lines)
    
    # Print summary
    print(f"Processed {len(records)} records")
    print(f"Generated {len(results)} weak label outputs")
    print(f"Errors: {len(errors)}")
    print(f"Output written to: {args.output}")
    
    return 0 if len(errors) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
