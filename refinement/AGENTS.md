# Refinement code conventions

The `refinement/` package contains probabilistic EOF-coefficient correction, observation handling, normalized losses, and pressure-dependent regularization.

- Preserve function interfaces and tensor dimension conventions across modules.
- Maintain consistent pressure levels, valid-data masks, and EOF mode ordering.
- Keep changes to numerical algorithms explicit and verify scientific units.
- Check syntax with `python -m compileall refinement` and run available tests with `pytest -q refinement/tests`.
- Validate array shapes, imports, and function signatures when modifying the training or inference interfaces.
