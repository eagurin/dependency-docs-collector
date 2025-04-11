# 📥 Установка

Существует несколько способов установки **Dependency Documentation Collector**, в зависимости от ваших предпочтений и рабочего окружения.

## 🛠️ Требования

- Python 3.8 или выше
- Доступ в интернет для скачивания зависимостей и документации
- Установленный Git (для клонирования репозиториев)

## ⚡ Быстрая установка через uv (рекомендуется)

[uv](https://github.com/astral-sh/uv) - это быстрый и надежный Python-пакетный менеджер, написанный на Rust.

```bash
# Установка uv, если он еще не установлен
curl -LsSf https://astral.sh/uv/install.sh | sh

# Установка dependency-docs-collector
uv install dependency-docs-collector
```

## 🐍 Установка через pip

```bash
pip install dependency-docs-collector
```

## 🧪 Установка из исходного кода (для разработчиков)

```bash
# Клонирование репозитория
git clone https://github.com/eagurin/dependency-docs-collector.git
cd dependency-docs-collector

# Создание виртуальной среды с помощью uv
uv venv

# Активация виртуальной среды
# На Linux/macOS:
source .venv/bin/activate
# На Windows:
.venv\Scripts\activate

# Установка в режиме разработки
uv pip install -e ".[dev]"
```

## 🧩 Опциональные зависимости

Пакет поддерживает несколько групп дополнительных зависимостей:

- `dev`: зависимости для разработки (линтеры, форматтеры)
- `test`: зависимости для запуска тестов
- `docs`: зависимости для работы с документацией

```bash
# Установка с опциональными зависимостями
uv install "dependency-docs-collector[dev,test,docs]"
```

## 🔄 Обновление

```bash
# Обновление с помощью uv
uv pip install --upgrade dependency-docs-collector

# Обновление с помощью pip
pip install --upgrade dependency-docs-collector
```

## ✅ Проверка установки

После установки вы можете проверить, что пакет работает корректно:

```bash
# Проверка версии
dependency-docs-collector --version

# Запуск справки
dependency-docs-collector --help
```

## 🔍 Устранение проблем

Если у вас возникли проблемы с установкой:

1. Убедитесь, что используете правильную версию Python (3.8+)
2. Проверьте доступ в интернет и наличие прав на запись в каталог установки
3. Если вы используете виртуальную среду, убедитесь, что она активирована
4. Попробуйте очистить кэш pip перед установкой: `pip cache purge`

Если проблема не устранена, пожалуйста, [создайте issue](https://github.com/eagurin/dependency-docs-collector/issues) в репозитории проекта.

## 🚀 Далее

После установки перейдите к разделу [Быстрый старт](quickstart.md) для знакомства с основными функциями инструмента.
