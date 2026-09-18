## What changed

<!-- One paragraph. What does the system do differently now? -->

## Why

<!-- The evidence, tagged as in docs/architecture_J.md:
     [P] paper  [B] benchmark/measured report  [E] we can demonstrate it  [D] team decision
     A [D] is fine — record it so it can be challenged later. -->

## Checklist

- [ ] `pytest -q` passes
- [ ] `ruff check src eval tests scripts` passes
- [ ] Rationale lives in the code or the docs, not only in this PR description

### If this touches `contracts/*_J.py`

- [ ] The other owner has reviewed it — contracts are **joint**, changed only by agreement
- [ ] The change is additive with a backward-compatible default, or the break is called out explicitly
- [ ] `docs/contracts_J.md` updated to match

### If this could move a metric

- [ ] Ran the gate: `python -m eval.run_gate_S --split dev --baseline <run>`
- [ ] Verdict pasted below, **including the sensitivity line** — "the gate went green" is not a
      finding; "the gate can resolve changes of at least X" is
- [ ] If the gate **refused** rather than rejected, say why. A refusal means the question could
      not be asked (degraded backend, stamp mismatch, too few queries) and is not a quality
      signal in either direction

```
<!-- paste the gate report -->
```

### If this changes agent behaviour

- [ ] Budgets still terminate — see `tests/unit/agents/test_routing_S.py`
- [ ] Any new violation kind is classified in `DEEP_VIOLATION_KINDS` or
      `SHALLOW_VIOLATION_KINDS` (the module refuses to import otherwise, on purpose)
- [ ] Any new escalation trigger has guidance in `NEXT_STEPS` — an escalation that cannot say
      what a human should do next discards the diagnosis the system already made

### Held-out discipline

- [ ] The `test` split was not used to choose a prompt, a threshold, or a model
