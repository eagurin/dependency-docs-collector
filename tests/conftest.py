"""
Конфигурация для тестов pytest.
"""

import os
import sys
import pytest
from pathlib import Path

# Добавляем корневую директорию в sys.path для доступа к main.py
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def test_data_dir():
    """Путь к директории с тестовыми данными."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def temp_output_dir(tmp_path):
    """Временная директория для выходных данных."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def temp_project_dir(tmp_path):
    """Временная директория для тестового проекта."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    return project_dir 