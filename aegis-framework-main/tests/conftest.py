# tests/conftest.py

import pytest
import torch
import os
import yaml
import shutil
from pathlib import Path

@pytest.fixture(scope="session")
def device():
    """Provides a torch device for tests, preferring CUDA if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def tmp_output_dir(tmp_path):
    """Provides a temporary directory for test outputs and cleans it up afterward."""
    # tmp_path is a built-in pytest fixture that provides a temporary directory
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    yield output_dir
    # Cleanup is handled automatically by pytest's tmp_path fixture