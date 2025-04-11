.PHONY: help clean install dev test lint format docs build all

# Переменные
PYTHON = python3
VENV = .venv
PACKAGE = main.py
TESTS = tests

help:
	@echo "Доступные команды:"
	@echo "  make help     - Показать эту справку"
	@echo "  make setup    - Создать виртуальную среду и установить зависимости"
	@echo "  make clean    - Удалить кеш Python и временные файлы"
	@echo "  make install  - Установить пакет"
	@echo "  make dev      - Установить пакет в режиме разработки"
	@echo "  make test     - Запустить тесты"
	@echo "  make lint     - Проверить код с помощью линтеров"
	@echo "  make format   - Отформатировать код"
	@echo "  make docs     - Сгенерировать документацию"
	@echo "  make build    - Собрать пакет"
	@echo "  make all      - Выполнить все команды (lint, test, build)"

setup:
	@echo "🔧 Создание виртуальной среды..."
	uv venv $(VENV)
	@echo "📦 Установка зависимостей..."
	uv pip install -e .[dev,test,docs]
	@echo "✅ Среда настроена!"

clean:
	@echo "🧹 Удаление временных файлов..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	@echo "✅ Очистка завершена!"

install:
	@echo "📦 Установка пакета..."
	uv pip install .
	@echo "✅ Установка завершена!"

dev:
	@echo "🔧 Установка пакета в режиме разработки..."
	uv pip install -e .[dev]
	@echo "✅ Установка в режиме разработки завершена!"

test:
	@echo "🧪 Запуск тестов..."
	pytest $(TESTS) -v
	@echo "✅ Тесты выполнены!"

lint:
	@echo "🔍 Проверка кода линтерами..."
	ruff check $(PACKAGE) $(TESTS)
	mypy $(PACKAGE) $(TESTS)
	@echo "✅ Проверка завершена!"

format:
	@echo "✨ Форматирование кода..."
	ruff format $(PACKAGE) $(TESTS)
	@echo "✅ Форматирование завершено!"

docs:
	@echo "📚 Генерация документации..."
	mkdocs build
	@echo "✅ Документация сгенерирована в директории site/"

build:
	@echo "🏗️ Сборка пакета..."
	uv pip install build
	$(PYTHON) -m build
	@echo "✅ Сборка завершена!"

all: lint test build
	@echo "🎉 Все команды выполнены!" 