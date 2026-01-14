# Troubleshooting Guide

This file documents error patterns encountered and their solutions.

## Format

| Error Pattern | Symptom | Cause | Solution |
|---------------|---------|-------|----------|
| Pattern name | What you see | Why it happens | How to fix |

---

## Common Issues

<!-- Add troubleshooting entries below -->

| Error Pattern | Symptom | Cause | Solution |
|---------------|---------|-------|----------|
| (Template) | Describe the error message or behavior | Root cause analysis | Step-by-step fix |
| MMStar first-letter scoring | Gibberish outputs still score non-trivially (e.g., step 0 shows ~0.23-0.24). | `exact_match()` uses only the first character of the output, so any text starting with A-D can be counted correct. | No mitigation implemented yet; current eval pipeline does not enforce constrained decoding. |
