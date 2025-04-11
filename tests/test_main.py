"""
Тесты для основного функционала main.py.
"""

import pytest
import sys
import os
from pathlib import Path

# Импортируем класс DependencyAnalyzer напрямую из main.py
from main import DependencyAnalyzer, DocumentationCollector, DocumentationCombiner


def test_dependency_analyzer_init(temp_project_dir):
    """Тест инициализации анализатора зависимостей."""
    analyzer = DependencyAnalyzer(str(temp_project_dir))
    assert analyzer.project_path == temp_project_dir
    assert isinstance(analyzer.dependencies, set)


def test_clean_package_name():
    """Тест очистки имени пакета."""
    analyzer = DependencyAnalyzer(".")
    assert analyzer.clean_package_name("package==1.0.0") == "package"
    assert analyzer.clean_package_name("package>=1.0.0,<2.0.0") == "package"
    assert analyzer.clean_package_name("package;python_version>='3.6'") == "package"
    assert analyzer.clean_package_name("git+https://github.com/user/repo.git") == "repo"


def test_documentation_collector_init(temp_output_dir):
    """Тест инициализации сборщика документации."""
    collector = DocumentationCollector(str(temp_output_dir))
    assert collector.output_dir == temp_output_dir


def test_documentation_combiner_init(temp_output_dir):
    """Тест инициализации комбинатора документации."""
    combiner = DocumentationCombiner(str(temp_output_dir))
    assert combiner.output_dir == temp_output_dir
    assert hasattr(combiner, 'MAX_FILE_SIZE') 