# AGENTS.md

## Repository expectations

- Keep edits minimal and local.
- Do not rename public functions unless required.
- Preserve existing scientific behavior unless fixing a clear bug.
- Prefer small interface fixes over large refactors.
- After editing Python files, run:
  - python -m compileall .
  - pytest -q if tests exist
- If no tests exist, run lightweight import/syntax smoke checks and report exactly what was validated.
- When reporting issues, pay special attention to:
  - array shape mismatches
  - inconsistent tensor dimension ordering
  - broken imports
  - circular imports
  - function signature mismatches across files
