# 🚀 Быстрый старт

Это руководство поможет вам быстро начать работу с **Dependency Documentation Collector** и познакомит вас с основными функциями.

## 📋 Основные сценарии использования

Инструмент поддерживает два основных сценария:

1. **Анализ зависимостей проекта** — сканирование проекта для выявления зависимостей с возможностью сбора их документации
2. **Работа с указанным списком библиотек** — сбор документации для конкретных библиотек без анализа проекта

## 🔎 Анализ зависимостей проекта

### Базовый анализ

```bash
# Анализ текущего проекта, вывод результатов в консоль
dependency-docs-collector --project-path .

# Анализ проекта в указанном каталоге с сохранением результатов
dependency-docs-collector --project-path /path/to/project --output-dir ./docs_output
```

### Анализ с проверкой используемых импортов

```bash
# Анализ зависимостей и поиск неиспользуемых библиотек
dependency-docs-collector --project-path . --scan-imports
```

### Только анализ зависимостей без сбора документации

```bash
# Только анализ без сбора документации
dependency-docs-collector --project-path . --analyze-only
```

## 📚 Работа с указанным списком библиотек

```bash
# Сбор документации для указанных библиотек
dependency-docs-collector --library-list "requests,flask,pandas" --output-dir ./docs_output

# С созданием учебных материалов (лекций)
dependency-docs-collector --library-list "requests,fastapi,sqlalchemy" --create-lecture
```

## 🖥️ Интерактивный пример

Посмотрим, как выглядит типичный сеанс работы с инструментом:

```bash
# Создаем каталог для выходных данных
$ mkdir docs_output

# Анализируем проект и собираем документацию
$ dependency-docs-collector --project-path . --output-dir ./docs_output
[INFO] Scanning project for dependency files...
[INFO] Found 3 dependency files
[INFO] Analyzing dependencies...
[INFO] Found 15 dependencies
[INFO] Collecting documentation for dependencies...
[INFO] Processing package: requests
[INFO] Processing package: flask
[INFO] Processing package: sqlalchemy
...
[INFO] Documentation collected for 12 packages
[INFO] Combining documentation...
[INFO] All done! Documentation saved to ./docs_output
```

## 📊 Анализ результатов

После выполнения команды в указанном выходном каталоге будут созданы:

1. Папки с документацией для каждой библиотеки
2. Объединенные файлы документации для каждой библиотеки
3. Если был указан флаг `--create-lecture`, будут созданы учебные материалы

```bash
docs_output/
├── requests/
│   ├── docs/
│   ├── examples/
│   └── ...
├── flask/
│   └── ...
├── requests_combined.txt
├── flask_combined.txt
├── 00_содержание.md  # если указан --create-lecture
├── 01_requests.md    # если указан --create-lecture
└── ...
```

## 🧩 Интеграция в скрипты

Вы можете использовать инструмент в своих скриптах:

```python
import asyncio
from dependency_docs_collector.analyzer import DependencyAnalyzer
from dependency_docs_collector.collector import DocumentationCollector
from dependency_docs_collector.combiner import DocumentationCombiner

async def main():
    # Анализ зависимостей
    analyzer = DependencyAnalyzer("/path/to/project")
    result = await analyzer.analyze_project(scan_imports=True)
    
    dependencies = result["dependencies"]
    print(f"Found {len(dependencies)} dependencies")
    
    # Сбор документации
    collector = DocumentationCollector("./output")
    docs_info = await collector.collect_documentation(dependencies)
    
    # Объединение документации
    combiner = DocumentationCombiner("./output")
    combined_docs = await combiner.combine_documentation(docs_info)
    
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🎯 Следующие шаги

Теперь, когда вы познакомились с основами, вы можете:

- Изучить [полное руководство по использованию](usage.md) для более глубокого понимания всех возможностей
- Ознакомиться со [справочником API](../api/) для программного использования
- Узнать о [поддерживаемых форматах](supported-formats.md) файлов зависимостей
