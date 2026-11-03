# Contributing to LineCount AI

Thank you for your interest in contributing to LineCount AI — a real-time industrial object detection, tracking, and counting system.

## Table of Contents
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Standards](#code-standards)
- [Test Requirements](#test-requirements)
- [Submitting Changes](#submitting-changes)

---

## Development Setup

### Prerequisites
- Python 3.9+
- Git

### Steps

1. **Fork the repository** on GitHub.

2. **Clone your fork:**
   ```bash
   git clone https://github.com/<your-username>/linecount-ai.git
   cd linecount-ai
   ```

3. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   # or
   make install-dev
   ```

5. **Verify the test suite passes before making changes:**
   ```bash
   make test
   ```

---

## Project Structure

```
linecount-ai/
├── src/linecount/
│   ├── detection/      # YOLO-based object detector
│   ├── tracking/       # ByteTrack multi-object tracker
│   ├── counting/       # Line-crossing counter, anti-double-count
│   └── evaluation/     # Counting accuracy metrics
├── app/
│   ├── streamlit_app.py  # Real-time counting dashboard
│   └── api.py            # FastAPI REST endpoints
├── scripts/            # Detection, tracking, counting scripts
├── tests/              # Unit tests (pytest)
├── notebooks/          # Conveyor EDA and benchmark notebooks
└── docs/               # Architecture and API documentation
```

---

## Code Standards

- **Style:** Follow [PEP 8](https://peps.python.org/pep-0008/). Maximum line length: **100 characters**.
- **Formatting:** Use `black` (`make format`) before committing.
- **Imports:** Sort with `isort` (`make format`).
- **Docstrings:** All public classes and functions **must** have Google-style docstrings.
- **Type hints:** All new functions should include type annotations.
- **No fabricated results:** Do not hardcode FPS numbers or counting accuracy figures.

---

## Test Requirements

- All new features **must** include corresponding unit tests.
- All tests must pass: `make test`
- FastAPI tests use `pytest.importorskip("fastapi")` — install fastapi to run them.
- Tracking tests may use synthetic frame data to avoid requiring video files.

---

## Submitting Changes

1. Ensure all tests pass: `make test`
2. Run lint checks: `make lint`
3. Update `docs/` if your change affects architecture or API behaviour.
4. Push your branch and open a **Pull Request** with a clear description.

---

## Reporting Issues

Open a GitHub Issue with:
- A minimal reproducible example (or sample video frames if relevant).
- Your Python version and OS.
- The full error traceback.
