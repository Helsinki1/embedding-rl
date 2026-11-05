# Local verification — 2026-11-04

- 14 automated tests passed, including a numeric check of the REINFORCE gradient,
  fixed-reference nDCG, KL, InfoNCE, split isolation, interrupted judge caching,
  all three training methods, explicit final testing, and real offline W&B log creation.
- Ruff passed; local CLI and all three Modal entrypoints import and show CLI help.
- Downloaded and read the real SciFact dataset: 5,183 documents; the default selection
  yields 200 training, 50 validation, and 300 official test queries.
- Loaded the real BGE-small checkpoint on CPU. A forward/backward check produced a
  384-dimensional vector with unit norm and nonzero encoder gradients.
- Installed this project into the task-local environment at
  `../../work/embedding-rl-venv`. The environment reuses existing system packages;
  use the clean-venv instructions in README for a fresh installation.

Tests used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` because this machine has unrelated pytest
plugins with missing dependencies. The project tests require no extra pytest plugins.

No Modal container build, paid GPU job, paid OpenAI judgment, online W&B upload, or real
benchmark training run has been executed. The tests establish pipeline correctness on
small fixtures; they do not establish retrieval improvement or cloud-runtime success.

Start with `config.pilot.json` after adding the account settings in README.
