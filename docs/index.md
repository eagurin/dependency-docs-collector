# 📚 Dependency Documentation Collector

> 🔍 **Автоматизированный сбор и упорядочивание документации для зависимостей Python-проектов**

[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./about/license.md)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/eagurin/dependency-docs-collector)

## 🌟 О проекте

**Dependency Documentation Collector** — это инструмент, который автоматизирует процесс анализа зависимостей Python-проектов и сбора документации к ним из GitHub-репозиториев. Он помогает разработчикам лучше понимать используемые библиотеки и создавать образовательные материалы на их основе.

## ✨ Возможности

- 📋 **Анализ зависимостей** из различных конфигурационных файлов
  - requirements.txt
  - setup.py
  - pyproject.toml
  - Pipfile
  - Poetry files
  - и другие

- 🧠 **Умное определение** используемых и неиспользуемых зависимостей
  - Анализ импортов в коде
  - Поиск неиспользуемых библиотек
  - Выявление скрытых зависимостей

- 📥 **Сбор документации** из GitHub-репозиториев
  - Автоматический поиск репозиториев
  - Клонирование и анализ документации
  - Умный отбор полезных файлов

- 📝 **Объединение документации** в удобочитаемые файлы
  - Форматирование содержимого
  - Создание структурированных документов
  - Разделение больших файлов

- 🎓 **Генерация учебных материалов**
  - Создание лекций на основе документации
  - Формирование оглавления
  - Структурирование примеров кода

## 🚀 Быстрый старт

```bash
# Установка
uv install dependency-docs-collector

# Анализ зависимостей текущего проекта
dependency-docs-collector --project-path . --output-dir ./docs_output

# Сбор документации для указанных библиотек
dependency-docs-collector --library-list "requests,flask,pandas" --create-lecture
```

## 📚 Структура документации

- **[Руководство](./guide/installation.md)** — подробные инструкции по установке и использованию
- **[Справочник API](./api/)** — документация по программному интерфейсу
- **[Разработка](./development/contributing.md)** — руководство для контрибьюторов
- **[О проекте](./about/authors.md)** — информация о авторах и лицензии

## 💡 Примеры использования

### Анализ зависимостей проекта

```python
# Импорт напрямую из main.py
from main import DependencyAnalyzer

# Создание анализатора
analyzer = DependencyAnalyzer("/path/to/project")

# Получение списка зависимостей
dependencies = await analyzer.analyze_dependencies()
print(f"Found {len(dependencies)} dependencies")

# Анализ импортов
if scan_imports:
    imports = await analyzer.find_imports_in_source()
    unused = analyzer.find_unused_dependencies(dependencies, imports)
    print(f"Found {len(unused)} unused dependencies")
```

### Сбор документации

```python
# Импорт напрямую из main.py
from main import DocumentationCollector

# Создание коллектора
collector = DocumentationCollector("./output")

# Сбор документации для зависимостей
docs_info = await collector.collect_documentation(dependencies)
print(f"Collected documentation for {len(docs_info)} packages")
```

## 📑 Лицензия

Этот проект распространяется под лицензией MIT. Подробнее смотрите в [файле лицензии](./about/license.md).
