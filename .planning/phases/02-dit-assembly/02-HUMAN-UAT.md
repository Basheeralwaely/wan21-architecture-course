---
status: complete
phase: 02-dit-assembly
source: [02-VERIFICATION.md]
started: 2026-04-24T15:45:00Z
updated: 2026-04-24T16:10:00Z
---

## Current Test

[testing complete]

## Tests

### 1. WR-01: Wrong LoRA formula in NB-06 Exercise 3
expected: Exercise 3 should use `r * (in_features + out_features)` formula for LoRA parameter count calculation, not `2 * max(in, out) * r`. For ffn.0 (1536->8960) at rank 16, correct answer is 167,936 not 286,720.
result: pass

### 2. IN-01: GateModule.forward line citation in NB-06 summary
expected: Summary table should cite GateModule.forward at line 194 (the forward method), not line 190 (the class definition).
result: issue
reported: "GateModule.forward still cites line 190 instead of line 194"
severity: minor

### 3. IN-02: WanModel.__init__ citation range in NB-08
expected: WanModel.__init__ citation should be lines 274-338 (full body), not 274-328 (truncated before has_image_input/has_ref_conv/control_adapter setup).
result: pass

## Summary

total: 3
passed: 2
issues: 1
pending: 0
skipped: 0
blocked: 0

## Gaps

- truth: "Summary table should cite GateModule.forward at line 194 (the forward method), not line 190 (the class definition)"
  status: failed
  reason: "User reported: GateModule.forward still cites line 190 instead of line 194"
  severity: minor
  test: 2
  root_cause: ""
  artifacts: []
  missing: []
  debug_session: ""
