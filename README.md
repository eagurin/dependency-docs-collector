# 📚 Dependency Documentation Collector

![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-alpha-orange)

> 🔍 **Автоматизированный сбор и упорядочивание документации для зависимостей Python-проектов**

## ✨ Возможности

- 📋 **Анализ зависимостей** из различных конфигурационных файлов (requirements.txt, setup.py, pyproject.toml и др.)
- 🧠 **Умное определение** используемых и неиспользуемых зависимостей на основе анализа импортов
- 📥 **Сбор документации** из GitHub-репозиториев найденных зависимостей
- 📝 **Объединение документации** в удобочитаемые файлы
- 🎓 **Генерация учебных материалов** на основе собранной документации

## 🚀 Установка

### Используя uv (рекомендуется)

```bash
uv install dependency-docs-collector
```

### Используя pip

```bash
pip install dependency-docs-collector
```

### Из исходного кода

```bash
git clone https://github.com/eagurin/dependency-docs-collector.git
cd dependency-docs-collector
uv venv  # Создание виртуальной среды
# Активируем виртуальную среду
# На Linux/macOS: source .venv/bin/activate
# На Windows: .venv\Scripts\activate
uv pip install -e .
```

## 🔧 Использование

### Анализ зависимостей проекта

```bash
# Базовое использование - анализ зависимостей в текущем каталоге
dependency-docs-collector --project-path . --output-dir ./docs_output

# Анализ зависимостей с идентификацией используемых/неиспользуемых импортов
dependency-docs-collector --project-path /path/to/project --scan-imports

# Только анализ зависимостей без сбора документации
dependency-docs-collector --project-path /path/to/project --analyze-only
```

### Сбор документации для конкретных библиотек

```bash
# Сбор документации для указанных библиотек
dependency-docs-collector --library-list "requests,flask,pandas" --output-dir ./docs_output

# Сбор документации и создание учебных материалов
dependency-docs-collector --library-list "requests,flask,pandas" --create-lecture
```

### Дополнительные опции

```bash
# Задать максимальное количество пакетов для обработки
dependency-docs-collector --max-packages 10

# Задать уровень параллелизма при обработке пакетов
dependency-docs-collector --concurrency 5

# Пропустить пакеты, для которых уже есть документация
dependency-docs-collector --skip-existing

# Настроить уровень логирования
dependency-docs-collector --log-level DEBUG
```

## 📂 Результаты работы

Инструмент создает следующую структуру каталогов:

```
output_dir/
├── package1/
│   ├── docs/
│   ├── examples/
│   └── ...
├── package2/
│   ├── docs/
│   ├── examples/
│   └── ...
├── package1_combined.txt  # Объединенная документация
├── package2_combined.txt
├── 00_содержание.md       # Если указан флаг --create-lecture
├── 01_package1.md         # Если указан флаг --create-lecture
├── 02_package2.md         # Если указан флаг --create-lecture
└── index.html             # Если указан флаг --create-lecture
```

## 📋 Требования

- Python 3.8+
- aiohttp
- aiofiles
- tomli (для Python < 3.11)

## 🧩 Интеграция с uv

Этот проект полностью совместим с [uv](https://github.com/astral-sh/uv) - быстрым Python-пакетным менеджером, написанным на Rust. Рекомендуем использовать uv для установки и управления зависимостями.

## 🤝 Вклад в проект

Мы приветствуем любой вклад! Если вы хотите помочь улучшить этот проект:

1. Форкните репозиторий
2. Создайте ветку для вашей функции (`git checkout -b feature/amazing-feature`)
3. Зафиксируйте ваши изменения (`git commit -m 'Add some amazing feature'`)
4. Отправьте изменения (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📄 Лицензия

Распространяется под лицензией MIT. Смотрите файл `LICENSE` для дополнительной информации.

## 💖 Благодарности

- Всем разработчикам open-source библиотек, которые делают Python-экосистему такой замечательной
- Команде uv за создание отличного инструмента для управления пакетами
- Сообществу GitHub за поддержку
