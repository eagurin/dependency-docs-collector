"""
Тесты для основного функционала main.py.
"""

import os
import sys
from collections.abc import AsyncGenerator
from pathlib import Path

import aiohttp
import pytest

# Импортируем классы напрямую из main.py
from main import (
    DependencyAnalyzer,
    DocumentationCollector,
    DocumentationCombiner,
    LectureGenerator,
)


@pytest.fixture
async def aiohttp_session() -> AsyncGenerator[aiohttp.ClientSession, None]:
    """Фикстура для создания aiohttp сессии."""
    async with aiohttp.ClientSession() as session:
        yield session


@pytest.mark.asyncio
async def test_dependency_analyzer_init(temp_project_dir):
    """Тест инициализации анализатора зависимостей."""
    analyzer = DependencyAnalyzer(str(temp_project_dir))
    assert analyzer.project_path == temp_project_dir
    assert isinstance(analyzer.dependencies, set)


@pytest.mark.asyncio
async def test_clean_package_name():
    """Тест очистки имени пакета."""
    analyzer = DependencyAnalyzer(".")
    assert analyzer.clean_package_name("package==1.0.0") == "package"
    assert analyzer.clean_package_name("package>=1.0.0,<2.0.0") == "package"
    assert analyzer.clean_package_name("package;python_version>='3.6'") == "package"
    assert analyzer.clean_package_name("git+https://github.com/user/repo.git") == "repo"


@pytest.mark.asyncio
async def test_analyze_dependencies(temp_project_dir):
    """Тест анализа зависимостей."""
    # Создаем тестовый файл requirements.txt
    req_file = temp_project_dir / "requirements.txt"
    req_file.write_text("requests==2.31.0\nflask>=2.0.0\npandas")
    
    analyzer = DependencyAnalyzer(str(temp_project_dir))
    dependencies = await analyzer.analyze_dependencies()
    
    assert dependencies == {"requests", "flask", "pandas"}


@pytest.mark.asyncio
async def test_documentation_collector_init(temp_output_dir, aiohttp_session):
    """Тест инициализации сборщика документации."""
    collector = DocumentationCollector(str(temp_output_dir))
    collector.session = aiohttp_session
    assert collector.output_dir == temp_output_dir
    assert collector.concurrency == 3


@pytest.mark.asyncio
async def test_documentation_collector_get_package_info(temp_output_dir, aiohttp_session):
    """Тест получения информации о пакете."""
    collector = DocumentationCollector(str(temp_output_dir))
    collector.session = aiohttp_session
    
    info = await collector.get_package_info("requests")
    assert info is not None
    assert "info" in info
    assert info["info"]["name"] == "requests"


@pytest.mark.asyncio
async def test_documentation_combiner_init(temp_output_dir):
    """Тест инициализации комбинатора документации."""
    combiner = DocumentationCombiner(str(temp_output_dir))
    assert combiner.output_dir == temp_output_dir
    assert hasattr(combiner, 'MAX_FILE_SIZE')


@pytest.mark.asyncio
async def test_documentation_combiner_split_content(temp_output_dir):
    """Тест разделения большого контента."""
    combiner = DocumentationCombiner(str(temp_output_dir))
    content = ["# Test" * 1000]  # Создаем большой контент
    
    files = await combiner._split_content("test_package", temp_output_dir, content)
    assert len(files) > 0
    assert all(Path(f).exists() for f in files)


@pytest.mark.asyncio
async def test_lecture_generator_init(temp_output_dir):
    """Тест инициализации генератора лекций."""
    generator = LectureGenerator(str(temp_output_dir))
    assert generator.output_dir == temp_output_dir


@pytest.mark.asyncio
async def test_lecture_generator_generate(temp_output_dir):
    """Тест генерации лекции."""
    generator = LectureGenerator(str(temp_output_dir))
    
    # Создаем тестовые данные
    combined_docs = {
        "requests": [str(temp_output_dir / "requests_combined.txt")]
    }
    docs_info = {
        "requests": {
            "name": "requests",
            "version": "2.31.0",
            "description": "Test description",
            "github_url": "https://github.com/psf/requests",
            "pypi_url": "https://pypi.org/project/requests/",
        }
    }
    
    # Создаем тестовый файл документации
    (temp_output_dir / "requests_combined.txt").write_text("# Test Documentation")
    
    await generator.generate_lecture(combined_docs, docs_info)
    
    # Проверяем созданные файлы
    assert (temp_output_dir / "00_содержание.md").exists()
    assert (temp_output_dir / "01_requests.md").exists()
    assert (temp_output_dir / "index.html").exists() 