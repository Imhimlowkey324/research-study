# Research Study

A small, reproducible Python project for running a research study. The work is
split into three independent pieces so each can be developed and tested on its
own:

- **`data_generation/`** — builds the dataset the study runs on.
- **`graders/`** — scores or evaluates results derived from that data.
- **`tests/`** — automated checks that the two folders above behave correctly.

## Project layout

```
research-study/
├── data_generation/    # Code that creates / collects the study data
│   ├── __init__.py
│   └── generate.py
├── graders/            # Code that scores or evaluates results
│   ├── __init__.py
│   └── grader.py
├── tests/              # Automated tests for the code above
│   ├── test_data_generation.py
│   └── test_graders.py
├── data/               # Generated data lands here (contents are git-ignored)
├── .venv/              # Local virtual environment (git-ignored)
├── pyproject.toml      # Project metadata + test configuration
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

## Setup

```powershell
# 1. Create the virtual environment
python -m venv .venv

# 2. Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

On macOS / Linux, activate the environment with `source .venv/bin/activate`.

## Usage

```powershell
# Generate a sample dataset into data/
python -m data_generation.generate

# Run the test suite
pytest
```
