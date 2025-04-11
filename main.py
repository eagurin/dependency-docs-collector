#!/usr/bin/env python3
"""
Комплексный инструмент для анализа зависимостей Python-проекта и сбора документации.

Этот скрипт выполняет следующие функции:
1. Анализирует зависимости Python-проекта из различных файлов конфигурации
2. Определяет используемые и неиспользуемые зависимости на основе анализа импортов
3. Собирает документацию для найденных зависимостей из их GitHub-репозиториев
4. Объединяет документацию в удобные для чтения файлы
5. Опционально создает учебные материалы (лекции) на основе собранной документации

Использование:
    python main.py --project-path /path/to/project --output-dir ./docs_output
    python main.py --library-list "requests,flask,pandas" --create-lecture
    python main.py --project-path /path/to/project --scan-imports --analyze-only
"""

import os
import sys
import json
import logging
import argparse
import asyncio
import re
import aiohttp
import aiofiles
import shutil
import random
import time
import importlib
import subprocess
import pkgutil
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("dependency_docs.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

# Создаем глобальный пул потоков для CPU-bound операций
thread_pool = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# Настройки для aiohttp
aiohttp_timeout = aiohttp.ClientTimeout(
    total=60, connect=30, sock_connect=30, sock_read=30
)

# Кеш для результатов запросов к PyPI
PYPI_CACHE_FILE = ".pypi_cache.json"
pypi_cache = {}

# Типы файлов, где обычно хранятся зависимости Python-проектов
DEPENDENCY_FILES = [
    "requirements.txt",  # стандартный файл зависимостей pip
    "setup.py",  # setup.py с install_requires
    "setup.cfg",  # setup.cfg с install_requires
    "Pipfile",  # Pipenv
    "Pipfile.lock",  # Pipenv lock file
    "pyproject.toml",  # Poetry, Hatch, PDM
    "poetry.lock",  # Poetry lock file
    "environment.yml",  # Conda
    "tox.ini",  # tox
    "conda-environment.yml",  # Conda environment file
    "constraints.txt",  # Constraints file
]

# Базовый список известных стандартных библиотек Python
KNOWN_STDLIB_MODULES = {
    "abc",
    "aifc",
    "argparse",
    "array",
    "ast",
    "asyncio",
    "atexit",
    "audioop",
    "base64",
    "bdb",
    "binascii",
    "binhex",
    "bisect",
    "builtins",
    "bz2",
    "cProfile",
    "calendar",
    "cgi",
    "cgitb",
    "chunk",
    "cmath",
    "cmd",
    "code",
    "codecs",
    "codeop",
    "collections",
    "colorsys",
    "compileall",
    "concurrent",
    "configparser",
    "contextlib",
    "contextvars",
    "copy",
    "copyreg",
    "crypt",
    "csv",
    "ctypes",
    "curses",
    "dataclasses",
    "datetime",
    "dbm",
    "decimal",
    "difflib",
    "dis",
    "distutils",
    "doctest",
    "email",
    "encodings",
    "ensurepip",
    "enum",
    "errno",
    "faulthandler",
    "fcntl",
    "filecmp",
    "fileinput",
    "fnmatch",
    "formatter",
    "fractions",
    "ftplib",
    "functools",
    "gc",
    "getopt",
    "getpass",
    "gettext",
    "glob",
    "grp",
    "gzip",
    "hashlib",
    "heapq",
    "hmac",
    "html",
    "http",
    "idlelib",
    "imaplib",
    "imghdr",
    "imp",
    "importlib",
    "inspect",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "keyword",
    "lib2to3",
    "linecache",
    "locale",
    "logging",
    "lzma",
    "macpath",
    "mailbox",
    "mailcap",
    "marshal",
    "math",
    "mimetypes",
    "mmap",
    "modulefinder",
    "msilib",
    "msvcrt",
    "multiprocessing",
    "netrc",
    "nis",
    "nntplib",
    "numbers",
    "operator",
    "optparse",
    "os",
    "ossaudiodev",
    "parser",
    "pathlib",
    "pdb",
    "pickle",
    "pickletools",
    "pipes",
    "pkgutil",
    "platform",
    "plistlib",
    "poplib",
    "posix",
    "pprint",
    "profile",
    "pstats",
    "pty",
    "pwd",
    "py_compile",
    "pyclbr",
    "pydoc",
    "queue",
    "quopri",
    "random",
    "re",
    "readline",
    "reprlib",
    "resource",
    "rlcompleter",
    "runpy",
    "sched",
    "secrets",
    "select",
    "selectors",
    "shelve",
    "shlex",
    "shutil",
    "signal",
    "site",
    "smtpd",
    "smtplib",
    "sndhdr",
    "socket",
    "socketserver",
    "spwd",
    "sqlite3",
    "ssl",
    "stat",
    "statistics",
    "string",
    "stringprep",
    "struct",
    "subprocess",
    "sunau",
    "symbol",
    "symtable",
    "sys",
    "sysconfig",
    "syslog",
    "tabnanny",
    "tarfile",
    "telnetlib",
    "tempfile",
    "termios",
    "test",
    "textwrap",
    "threading",
    "time",
    "timeit",
    "tkinter",
    "token",
    "tokenize",
    "trace",
    "traceback",
    "tracemalloc",
    "tty",
    "turtle",
    "turtledemo",
    "types",
    "typing",
    "unicodedata",
    "unittest",
    "urllib",
    "uu",
    "uuid",
    "venv",
    "warnings",
    "wave",
    "weakref",
    "webbrowser",
    "winreg",
    "winsound",
    "wsgiref",
    "xdrlib",
    "xml",
    "xmlrpc",
    "zipapp",
    "zipfile",
    "zipimport",
    "zlib",
    # Тестирование
    "pytest",
    "unittest",
    "nose",
    "mock",
    "hypothesis",
    "tox",
    "coverage",
    "doctest",
}

# Базовый словарь известных соответствий между импортами и пакетами
KNOWN_PACKAGE_ALIASES = {
    # Scrapy экосистема
    "itemadapter": "scrapy",
    "parsel": "scrapy",
    "twisted": "scrapy",
    "scrapy_splash": "scrapy",
    # Langchain экосистема
    "langchain_community": "langchain-community",
    "langchain_core": "langchain",
    "langchain_text_splitters": "langchain",
    "langchain_experimental": "langchain",
    "langchain_openai": "langchain-openai",
    # Некоторые известные псевдонимы
    "tensorfow_probability": "tensorflow",
    "tensorflow_datasets": "tensorflow",
    "keras": "tensorflow",
    # Общие псевдонимы
    "bs4": "beautifulsoup4",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "mx": "mxnet",
    "torch": "pytorch",
    "dotenv": "python-dotenv",
}

# Специальные случаи для модулей, которые можно определить только тут
SPECIAL_PACKAGES = {
    "setuptools": {"setuptools", "pkg_resources", "easy_install"},
    "wheel": {"wheel"},
    "pip": {"pip"},
    "twine": {"twine"},
    "pre-commit": {"pre_commit"},
    "ipython": {"IPython", "ipython"},
}

# Инициализация глобальных переменных
STDLIB_MODULES = KNOWN_STDLIB_MODULES.copy()
PACKAGE_ALIASES = KNOWN_PACKAGE_ALIASES.copy()

# Регулярные выражения для извлечения зависимостей
REGEX_PATTERNS = {
    "requirements.txt": r"^([a-zA-Z0-9_\-\.]+)(?:[~=<>!]+.*)?$",  # например: package==1.0.0 или package>=1.0.0,<2.0.0
    "setup.py": r"setup\(\s*.*?install_requires=\[([^\]]+)\]",  # ищем секцию install_requires внутри setup()
    "pyproject.toml": r"(?:dependencies|requires) = \[([^\]]+)\]",  # для poetry и других
    "import": r"^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)",  # для поиска import-ов в .py файлах
}


def get_stdlib_modules() -> Set[str]:
    """
    Получает список стандартных библиотек Python, используя различные методы
    в зависимости от версии Python.

    Returns:
        Set[str]: Множество имен стандартных библиотек
    """
    stdlib_modules = set()

    # Метод 1: Используем sys.stdlib_module_names, если доступно (Python 3.10+)
    if hasattr(sys, "stdlib_module_names"):
        logger.debug(
            "Получение списка стандартных библиотек через sys.stdlib_module_names"
        )
        stdlib_modules.update(sys.stdlib_module_names)
    else:
        # Метод 2: Используем pkgutil для получения встроенных модулей
        logger.debug("Получение списка стандартных библиотек через pkgutil")
        for module_info in pkgutil.iter_modules():
            if not module_info.ispkg:  # Только модули, не пакеты
                if is_stdlib_module(module_info.name):
                    stdlib_modules.add(module_info.name)

        # Метод 3: Добавляем известные стандартные библиотеки вручную
        stdlib_modules.update(KNOWN_STDLIB_MODULES)

    # Добавляем служебные и приватные модули
    private_modules = {f"_{module}" for module in stdlib_modules}
    stdlib_modules.update(private_modules)

    # Добавляем другие известные системные модули
    stdlib_modules.update(
        [
            "__future__",
            "__main__",
            "builtins",
            "StringIO",
            "typing_extensions",
            "attr",
            "async_timeout",
        ]
    )

    return stdlib_modules


def is_stdlib_module(module_name: str) -> bool:
    """
    Проверяет, является ли модуль частью стандартной библиотеки Python.

    Args:
        module_name (str): Имя модуля для проверки

    Returns:
        bool: True, если модуль входит в стандартную библиотеку, иначе False
    """
    try:
        # Попытка импортировать модуль
        module = importlib.import_module(module_name)
        # Проверяем путь к модулю
        if module.__file__:
            return (
                "site-packages" not in module.__file__
                and "dist-packages" not in module.__file__
            )
        return True
    except (ImportError, AttributeError):
        return False


def get_package_metadata(package_name: str) -> Optional[Dict]:
    """
    Получает метаданные пакета из PyPI.

    Args:
        package_name (str): Имя пакета

    Returns:
        Dict|None: Метаданные пакета или None, если пакет не найден
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        import urllib.request
        import urllib.error

        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as e:
        logger.debug(f"Не удалось получить метаданные для {package_name}: {e}")
        return None


def build_package_aliases_dict(package_names: List[str]) -> Dict[str, str]:
    """
    Создает словарь соответствия между именами импортов и пакетами.

    Args:
        package_names (List[str]): Список имен пакетов

    Returns:
        Dict[str, str]: Словарь {имя_импорта: имя_пакета}
    """
    aliases = {}

    # Добавляем известные псевдонимы
    aliases.update(KNOWN_PACKAGE_ALIASES)

    # Для каждого пакета пытаемся определить соответствующие модули импорта
    for package_name in package_names:
        # Если пакет уже в нашем словаре псевдонимов, пропускаем
        if package_name in aliases.values():
            continue

        # Нормализуем имя пакета для импорта (замена дефисов на подчеркивания)
        import_name = package_name.replace("-", "_")

        # Добавляем основное соответствие
        aliases[import_name] = package_name

        # Проверяем, есть ли дополнительная информация на PyPI
        metadata = get_package_metadata(package_name)
        if metadata:
            try:
                # Проверяем top_level.txt в метаданных, если доступно
                if "urls" in metadata:
                    for url_info in metadata["urls"]:
                        if url_info.get("packagetype") == "sdist":
                            # Можно было бы скачать sdist и извлечь top_level.txt,
                            # но это значительно сложнее и требует больше ресурсов
                            pass
            except Exception as e:
                logger.debug(
                    f"Ошибка при извлечении информации о модулях для {package_name}: {e}"
                )

    # Добавляем специальные случаи
    for pkg, modules in SPECIAL_PACKAGES.items():
        for module in modules:
            aliases[module] = pkg

    return aliases


def get_installed_packages() -> List[str]:
    """
    Получает список установленных пакетов Python.

    Returns:
        List[str]: Список имен установленных пакетов
    """
    try:
        output = subprocess.check_output(["pip", "list", "--format=json"], text=True)
        packages = json.loads(output)
        return [pkg["name"].lower() for pkg in packages]
    except (subprocess.SubprocessError, json.JSONDecodeError) as e:
        logger.warning(f"Не удалось получить список установленных пакетов: {e}")
        return []


def get_cached_or_build_stdlib_modules() -> Set[str]:
    """
    Возвращает список стандартных библиотек, используя кэширование
    для повышения производительности.

    Returns:
        Set[str]: Множество стандартных библиотек
    """
    cache_file = os.path.join(os.path.expanduser("~"), ".stdlib_modules_cache.json")

    # Если есть кэш и он актуален (не старше 30 дней), используем его
    if os.path.exists(cache_file):
        try:
            mtime = os.path.getmtime(cache_file)
            if (time.time() - mtime) < 30 * 24 * 60 * 60:  # 30 дней
                with open(cache_file, "r") as f:
                    return set(json.load(f))
        except Exception as e:
            logger.debug(f"Ошибка при чтении кэша стандартных библиотек: {e}")

    # Иначе собираем список и кэшируем
    stdlib_modules = get_stdlib_modules()

    # Сохраняем в кэш
    try:
        with open(cache_file, "w") as f:
            json.dump(list(stdlib_modules), f)
    except Exception as e:
        logger.debug(f"Ошибка при сохранении кэша стандартных библиотек: {e}")

    return stdlib_modules


class DependencyAnalyzer:
    """Анализ зависимостей проекта."""

    def __init__(self, project_path: str):
        """
        Инициализация анализатора зависимостей.

        Args:
            project_path: Путь к проекту для анализа
        """
        self.project_path = Path(project_path)
        self.dependencies: Set[str] = set()
        self.ignored_packages = {
            "pip",
            "setuptools",
            "wheel",
            "distribute",
            "pkg-resources",
            "python",
            "python-dateutil",
            "pytz",
            "six",
            "typing-extensions",
        }

    def clean_package_name(self, package: str) -> str:
        """
        Очищает имя пакета от версий и опций.

        Args:
            package: Исходное имя пакета с версией и опциями

        Returns:
            Очищенное имя пакета
        """
        # Удаляем все спецификаторы версий и опции
        package = re.sub(r"[<>=~!].+", "", package)
        # Удаляем все после точки с запятой (например, python_version<"3.9")
        package = package.split(";")[0]
        # Удаляем все после # (например, #egg=sphinx-paramlinks)
        package = package.split("#")[0]
        # Удаляем URL-схемы (например, git+https://)
        package = re.sub(r"^[a-z]+\+https?://.+?/", "", package)
        # Удаляем github.com/user/ из URL
        package = re.sub(r"github\.com/[^/]+/", "", package)
        # Удаляем .git в конце
        package = package.replace(".git", "")
        # Удаляем лишние пробелы
        package = package.strip()
        # Приводим к нижнему регистру для единообразия
        package = package.lower()
        return package

    async def analyze_requirements_txt(self, file_path: Path) -> Set[str]:
        """
        Анализ requirements.txt.

        Args:
            file_path: Путь к файлу requirements.txt

        Returns:
            Множество имен пакетов
        """
        deps = set()
        try:
            async with aiofiles.open(
                file_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                content = await f.read()
                for line in content.splitlines():
                    line = line.strip()
                    # Пропускаем комментарии и пустые строки
                    if not line or line.startswith("#"):
                        continue
                    # Пропускаем опции pip (например, --no-binary)
                    if line.startswith("-"):
                        continue
                    # Обрабатываем ссылки на другие файлы requirements
                    if line.startswith("-r"):
                        req_file = line.split(" ", 1)[1].strip()
                        req_path = file_path.parent / req_file
                        if req_path.exists():
                            nested_deps = await self.analyze_requirements_txt(req_path)
                            deps.update(nested_deps)
                        continue
                    # Убираем версии и опции
                    package = self.clean_package_name(line)
                    if package and package not in self.ignored_packages:
                        deps.add(package)
        except Exception as e:
            logger.error(f"Ошибка при чтении {file_path}: {e}")
        return deps

    async def analyze_setup_py(self, file_path: Path) -> Set[str]:
        """
        Анализ setup.py.

        Args:
            file_path: Путь к файлу setup.py

        Returns:
            Множество имен пакетов
        """
        deps = set()
        try:
            async with aiofiles.open(
                file_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                content = await f.read()

                # Ищем install_requires
                if "install_requires" in content:
                    # Используем регулярное выражение для более надежного парсинга
                    install_requires = re.search(
                        r"install_requires\s*=\s*\[(.*?)\]", content, re.DOTALL
                    )
                    if install_requires:
                        # Извлекаем список зависимостей
                        deps_list = install_requires.group(1)
                        # Ищем все строки в кавычках
                        for match in re.finditer(r'[\'"]([^\'"]+)[\'"]', deps_list):
                            package = self.clean_package_name(match.group(1))
                            if package and package not in self.ignored_packages:
                                deps.add(package)

                # Ищем extras_require
                extras_require = re.search(
                    r"extras_require\s*=\s*{(.*?)}", content, re.DOTALL
                )
                if extras_require:
                    # Извлекаем словарь дополнительных зависимостей
                    extras_dict = extras_require.group(1)
                    # Ищем все строки в кавычках
                    for match in re.finditer(r'[\'"]([^\'"]+)[\'"]', extras_dict):
                        package = self.clean_package_name(match.group(1))
                        if package and package not in self.ignored_packages:
                            deps.add(package)
        except Exception as e:
            logger.error(f"Ошибка при чтении {file_path}: {e}")
        return deps

    async def analyze_pyproject_toml(self, file_path: Path) -> Set[str]:
        """
        Анализ pyproject.toml.

        Args:
            file_path: Путь к файлу pyproject.toml

        Returns:
            Множество имен пакетов
        """
        deps = set()
        try:
            # Используем aiofiles для асинхронного чтения файла
            async with aiofiles.open(file_path, "rb") as f:
                content = await f.read()
                try:
                    import tomli

                    data = tomli.loads(content.decode("utf-8"))
                except ImportError:
                    logger.warning(
                        "Модуль tomli не установлен. Используем регулярные выражения для парсинга pyproject.toml"
                    )
                    # Используем регулярные выражения для парсинга
                    content_str = content.decode("utf-8")
                    # Ищем секции с зависимостями
                    for section in ["dependencies", "requires", "dev-dependencies"]:
                        pattern = rf"{section}\s*=\s*\[(.*?)\]"
                        for match in re.finditer(pattern, content_str, re.DOTALL):
                            deps_section = match.group(1)
                            # Извлекаем имена пакетов
                            for pkg_match in re.finditer(
                                r'[\'"]([^\'"]+)[\'"]', deps_section
                            ):
                                package = self.clean_package_name(pkg_match.group(1))
                                if package and package not in self.ignored_packages:
                                    deps.add(package)
                    return deps

                # Проверяем разные форматы pyproject.toml

                # PEP 621 формат
                if "project" in data:
                    # Основные зависимости
                    if "dependencies" in data["project"]:
                        for dep in data["project"]["dependencies"]:
                            package = self.clean_package_name(dep)
                            if package and package not in self.ignored_packages:
                                deps.add(package)

                    # Опциональные зависимости
                    if "optional-dependencies" in data["project"]:
                        for group, group_deps in data["project"][
                            "optional-dependencies"
                        ].items():
                            for dep in group_deps:
                                package = self.clean_package_name(dep)
                                if package and package not in self.ignored_packages:
                                    deps.add(package)

                # Poetry формат
                if "tool" in data and "poetry" in data["tool"]:
                    poetry_data = data["tool"]["poetry"]

                    # Основные зависимости
                    if "dependencies" in poetry_data:
                        for dep_name, dep_info in poetry_data["dependencies"].items():
                            if dep_name != "python":  # Пропускаем python
                                package = self.clean_package_name(dep_name)
                                if package and package not in self.ignored_packages:
                                    deps.add(package)

                    # Зависимости для разработки
                    if "dev-dependencies" in poetry_data:
                        for dep_name, dep_info in poetry_data[
                            "dev-dependencies"
                        ].items():
                            package = self.clean_package_name(dep_name)
                            if package and package not in self.ignored_packages:
                                deps.add(package)

                    # Группы зависимостей (Poetry >= 1.2.0)
                    if "group" in poetry_data:
                        for group_name, group_info in poetry_data["group"].items():
                            if "dependencies" in group_info:
                                for dep_name, dep_info in group_info[
                                    "dependencies"
                                ].items():
                                    package = self.clean_package_name(dep_name)
                                    if package and package not in self.ignored_packages:
                                        deps.add(package)

                # Flit формат
                if "tool" in data and "flit" in data["tool"]:
                    flit_data = data["tool"]["flit"]

                    # Основные зависимости
                    if "metadata" in flit_data and "requires" in flit_data["metadata"]:
                        for dep in flit_data["metadata"]["requires"]:
                            package = self.clean_package_name(dep)
                            if package and package not in self.ignored_packages:
                                deps.add(package)

                    # Опциональные зависимости
                    if (
                        "metadata" in flit_data
                        and "requires-extra" in flit_data["metadata"]
                    ):
                        for group, group_deps in flit_data["metadata"][
                            "requires-extra"
                        ].items():
                            for dep in group_deps:
                                package = self.clean_package_name(dep)
                                if package and package not in self.ignored_packages:
                                    deps.add(package)

                # PDM формат
                if "tool" in data and "pdm" in data["tool"]:
                    pdm_data = data["tool"]["pdm"]

                    # Зависимости
                    if "dependencies" in pdm_data:
                        for dep in pdm_data["dependencies"]:
                            package = self.clean_package_name(dep)
                            if package and package not in self.ignored_packages:
                                deps.add(package)

                    # Опциональные зависимости
                    if "optional-dependencies" in pdm_data:
                        for group, group_deps in pdm_data[
                            "optional-dependencies"
                        ].items():
                            for dep in group_deps:
                                package = self.clean_package_name(dep)
                                if package and package not in self.ignored_packages:
                                    deps.add(package)

                # Hatch формат
                if "tool" in data and "hatch" in data["tool"]:
                    hatch_data = data["tool"]["hatch"]

                    # Зависимости
                    if "dependencies" in hatch_data:
                        for dep in hatch_data["dependencies"]:
                            package = self.clean_package_name(dep)
                            if package and package not in self.ignored_packages:
                                deps.add(package)

                    # Опциональные зависимости
                    if "optional-dependencies" in hatch_data:
                        for group, group_deps in hatch_data[
                            "optional-dependencies"
                        ].items():
                            for dep in group_deps:
                                package = self.clean_package_name(dep)
                                if package and package not in self.ignored_packages:
                                    deps.add(package)
        except Exception as e:
            logger.error(f"Ошибка при чтении {file_path}: {e}")
        return deps

    async def analyze_pipfile(self, file_path: Path) -> Set[str]:
        """
        Анализ Pipfile.

        Args:
            file_path: Путь к файлу Pipfile

        Returns:
            Множество имен пакетов
        """
        deps = set()
        try:
            async with aiofiles.open(
                file_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                content = await f.read()

                # Ищем секцию [packages]
                if "[packages]" in content:
                    packages_section = content.split("[packages]")[1].split("[")[0]
                    for line in packages_section.split("\n"):
                        if "=" in line:
                            package = line.split("=")[0].strip()
                            if package and package not in self.ignored_packages:
                                package = self.clean_package_name(package)
                                if package:
                                    deps.add(package)

                # Ищем секцию [dev-packages]
                if "[dev-packages]" in content:
                    dev_section = content.split("[dev-packages]")[1].split("[")[0]
                    for line in dev_section.split("\n"):
                        if "=" in line:
                            package = line.split("=")[0].strip()
                            if package and package not in self.ignored_packages:
                                package = self.clean_package_name(package)
                                if package:
                                    deps.add(package)
        except Exception as e:
            logger.error(f"Ошибка при чтении {file_path}: {e}")
        return deps

    async def analyze_pipfile_lock(self, file_path: Path) -> Set[str]:
        """
        Анализ Pipfile.lock.

        Args:
            file_path: Путь к файлу Pipfile.lock

        Returns:
            Множество имен пакетов
        """
        deps = set()
        try:
            async with aiofiles.open(
                file_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                content = await f.read()
                data = json.loads(content)

                # Обрабатываем основные пакеты
                if "default" in data:
                    for package_name in data["default"].keys():
                        if package_name not in self.ignored_packages:
                            package = self.clean_package_name(package_name)
                            if package:
                                deps.add(package)

                # Обрабатываем пакеты для разработки
                if "develop" in data:
                    for package_name in data["develop"].keys():
                        if package_name not in self.ignored_packages:
                            package = self.clean_package_name(package_name)
                            if package:
                                deps.add(package)
        except Exception as e:
            logger.error(f"Ошибка при чтении {file_path}: {e}")
        return deps

    async def analyze_poetry_lock(self, file_path: Path) -> Set[str]:
        """
        Анализ poetry.lock.

        Args:
            file_path: Путь к файлу poetry.lock

        Returns:
            Множество имен пакетов
        """
        deps = set()
        try:
            async with aiofiles.open(
                file_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                content = await f.read()

                # Ищем все пакеты в формате [[package]]
                package_sections = re.findall(
                    r"\[\[package\]\](.*?)(?=\[\[package\]\]|\Z)", content, re.DOTALL
                )

                for section in package_sections:
                    # Извлекаем имя пакета
                    name_match = re.search(r'name\s*=\s*"([^"]+)"', section)
                    if name_match:
                        package_name = name_match.group(1)
                        if package_name not in self.ignored_packages:
                            package = self.clean_package_name(package_name)
                            if package:
                                deps.add(package)
        except Exception as e:
            logger.error(f"Ошибка при чтении {file_path}: {e}")
        return deps

    def find_dependency_files(self) -> List[Tuple[Path, str]]:
        """
        Поиск файлов с зависимостями в проекте.

        Returns:
            Список кортежей (путь к файлу, тип файла)
        """
        dependency_files = []

        # Шаблоны файлов для поиска
        file_patterns = [
            ("requirements.txt", "requirements"),
            ("requirements-*.txt", "requirements"),
            ("requirements/*.txt", "requirements"),
            ("setup.py", "setup"),
            ("pyproject.toml", "pyproject"),
            ("Pipfile", "pipfile"),
            ("Pipfile.lock", "pipfile_lock"),
            ("poetry.lock", "poetry_lock"),
        ]

        # Ищем файлы по шаблонам
        for pattern, file_type in file_patterns:
            if "*" in pattern:
                # Для шаблонов с подстановочными знаками
                for file_path in self.project_path.glob("**/" + pattern):
                    if not any(part.startswith(".") for part in file_path.parts):
                        dependency_files.append((file_path, file_type))
            else:
                # Для точных имен файлов
                for file_path in self.project_path.rglob(pattern):
                    if not any(part.startswith(".") for part in file_path.parts):
                        dependency_files.append((file_path, file_type))

        return dependency_files

    async def analyze_dependencies(self) -> Set[str]:
        """
        Анализ всех зависимостей проекта.

        Returns:
            Множество имен пакетов
        """
        dependency_files = self.find_dependency_files()
        if not dependency_files:
            logger.warning(f"Не найдены файлы с зависимостями в {self.project_path}")
            return set()

        tasks = []

        for file_path, file_type in dependency_files:
            logger.info(f"Анализ {file_path}")
            if file_type == "requirements":
                tasks.append(self.analyze_requirements_txt(file_path))
            elif file_type == "setup":
                tasks.append(self.analyze_setup_py(file_path))
            elif file_type == "pyproject":
                tasks.append(self.analyze_pyproject_toml(file_path))
            elif file_type == "pipfile":
                tasks.append(self.analyze_pipfile(file_path))
            elif file_type == "pipfile_lock":
                tasks.append(self.analyze_pipfile_lock(file_path))
            elif file_type == "poetry_lock":
                tasks.append(self.analyze_poetry_lock(file_path))

        # Выполняем все задачи параллельно
        results = await asyncio.gather(*tasks)

        # Объединяем результаты
        for deps in results:
            self.dependencies.update(deps)

        logger.info(f"Найдено {len(self.dependencies)} уникальных зависимостей")
        return self.dependencies

    async def find_imports_in_source(self, max_files=1000) -> Set[str]:
        """
        Ищет импорты в исходных файлах Python.

        Args:
            max_files (int): Максимальное количество файлов для анализа

        Returns:
            Set[str]: Множество найденных импортов
        """
        imports = set()

        # Собираем все .py файлы, кроме файлов в директориях, которые нужно исключить
        exclude_dirs = {
            ".git",
            ".github",
            ".venv",
            "venv",
            "env",
            "__pycache__",
            "node_modules",
            "build",
            "dist",
            "site-packages",
        }

        all_py_files = []
        for root, dirs, files in os.walk(self.project_path):
            # Исключаем нежелательные директории
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith(".py") and file != os.path.basename(__file__):
                    all_py_files.append(os.path.join(root, file))

        logger.info(f"Найдено {len(all_py_files)} Python-файлов для анализа импортов")

        # Ограничиваем количество файлов
        if len(all_py_files) > max_files:
            logger.info(f"Ограничиваем анализ до {max_files} файлов")
            all_py_files = all_py_files[:max_files]

        # Анализируем импорты асинхронно
        async def analyze_file(file_path):
            try:
                async with aiofiles.open(
                    file_path, "r", encoding="utf-8", errors="ignore"
                ) as f:
                    content = await f.read()

                file_imports = set()
                # Ищем строки импорта
                for line in content.split("\n"):
                    match = re.match(REGEX_PATTERNS["import"], line)
                    if match:
                        # Извлекаем только базовое имя модуля
                        module_name = match.group(1).split(".")[0]
                        if module_name not in STDLIB_MODULES and module_name != "self":
                            file_imports.add(module_name)

                return file_imports

            except Exception as e:
                logger.debug(f"Не удалось проанализировать файл {file_path}: {str(e)}")
                return set()

        # Создаем задачи для анализа файлов
        # Выполняем задачи с ограничением параллельности
        semaphore = asyncio.Semaphore(
            20
        )  # Ограничиваем количество одновременно открытых файлов

        async def analyze_with_semaphore(file_path):
            async with semaphore:
                return await analyze_file(file_path)

        # Запускаем задачи и собираем результаты
        results = await asyncio.gather(
            *[analyze_with_semaphore(file_path) for file_path in all_py_files]
        )

        # Объединяем результаты
        for file_imports in results:
            imports.update(file_imports)

        return imports

    def normalize_dependencies(self, dependencies: Set[str]) -> Set[str]:
        """
        Нормализует список зависимостей, преобразуя псевдонимы и подмодули в основные пакеты.

        Args:
            dependencies (set): Множество зависимостей

        Returns:
            set: Нормализованное множество зависимостей
        """
        normalized = set()

        for dep in dependencies:
            # Проверяем, есть ли пакет в словаре псевдонимов
            if dep in PACKAGE_ALIASES:
                normalized.add(PACKAGE_ALIASES[dep])
            else:
                normalized.add(dep)

        return normalized

    def find_unused_dependencies(
        self, dependencies: Set[str], used_imports: Set[str]
    ) -> Set[str]:
        """
        Определяет неиспользуемые зависимости на основе анализа импортов.

        Args:
            dependencies (set): Множество зависимостей из файлов requirements
            used_imports (set): Множество импортированных модулей из кода

        Returns:
            set: Множество неиспользуемых зависимостей
        """
        unused = set()

        # Создаем обратный словарь для пакетов и их импортов
        reverse_aliases = {}
        for module, package in PACKAGE_ALIASES.items():
            if package not in reverse_aliases:
                reverse_aliases[package] = set()
            reverse_aliases[package].add(module)

        # Добавляем соответствия "сам на себя" для каждого пакета
        for package in set(PACKAGE_ALIASES.values()):
            if package not in reverse_aliases:
                reverse_aliases[package] = set()
            reverse_aliases[package].add(package)
            # Добавляем вариант с заменой дефисов на подчеркивания
            if "-" in package:
                reverse_aliases[package].add(package.replace("-", "_"))

        for dep in dependencies:
            # Определяем все возможные имена импорта для данной зависимости
            possible_imports = set()

            # 1. Непосредственно имя пакета
            possible_imports.add(dep)

            # 2. Имя пакета с дефисом, замененным на подчеркивание
            if "-" in dep:
                possible_imports.add(dep.replace("-", "_"))

            # 3. Известные псевдонимы и подмодули для данного пакета
            if dep in reverse_aliases:
                possible_imports.update(reverse_aliases[dep])

            # Проверяем, используется ли хоть один из возможных импортов
            is_used = False
            for imp in possible_imports:
                if imp in used_imports:
                    is_used = True
                    break

            if not is_used:
                unused.add(dep)

        # Исключаем из списка неиспользуемых те зависимости,
        # которые могут использоваться непрямыми способами
        safe_list = {
            "setuptools",
            "wheel",
            "pip",
            "twine",
            "build",
            "pytest",
            "pytest-cov",
            "mypy",
            "pylint",
            "flake8",
            "black",
            "isort",
            "pre-commit",
            "coverage",
            "tox",
            "sphinx",
            "mkdocs",
            "jupyter",
            "ipykernel",
            "nbconvert",
            "python-dotenv",
            "gunicorn",
            "uvicorn",
        }

        return unused - safe_list

    async def analyze_project(
        self, scan_imports=False, max_files=1000
    ) -> Dict[str, Any]:
        """
        Полный анализ проекта: зависимости и импорты.

        Args:
            scan_imports (bool): Анализировать ли импорты в коде
            max_files (int): Максимальное количество файлов для анализа импортов

        Returns:
            Dict[str, Any]: Результаты анализа
        """
        result = {
            "dependency_files": self.find_dependency_files(),
            "dependencies": set(),
            "probable_dependencies": None,
            "unused_dependencies": None,
            "imports": None,
        }

        # Анализируем зависимости
        dependencies = await self.analyze_dependencies()
        result["dependencies"] = dependencies

        # Если нужно, анализируем импорты
        if scan_imports:
            imports = await self.find_imports_in_source(max_files)
            result["imports"] = imports

            # Нормализуем импорты и зависимости
            normalized_imports = self.normalize_dependencies(imports)
            normalized_dependencies = self.normalize_dependencies(dependencies)

            # Находим вероятные дополнительные зависимости
            result["probable_dependencies"] = (
                normalized_imports - normalized_dependencies
            )

            # Находим неиспользуемые зависимости
            result["unused_dependencies"] = self.find_unused_dependencies(
                normalized_dependencies, imports
            )

        return result


class DocumentationCollector:
    """Сбор документации для зависимостей."""

    # Директории, где обычно находится полезная документация
    DOCS_DIRS = {
        "docs",
        "doc",
        "documentation",
        "cookbook",
        "examples",
        "tutorials",
        "guides",
        "notebooks",
        "recipes",
        "howto",
        "getting-started",
        "quickstart",
        "reference",
        "api",
        "best-practices",
        "samples",
        "demo",
        "walkthrough",
    }

    # Расширения файлов, которые мы ищем
    DOC_EXTENSIONS = {
        ".md",
        ".ipynb",
        ".rst",
        ".txt",
        ".py",  # Для примеров кода и docstrings
        ".json",  # Для конфигурационных примеров
        ".yaml",
        ".yml",
    }

    # Ключевые слова для определения полезной документации
    DOC_KEYWORDS = {
        "tutorial",
        "guide",
        "example",
        "howto",
        "cookbook",
        "documentation",
        "usage",
        "getting started",
        "installation",
        "api reference",
        "configuration",
        "quickstart",
        "features",
        "overview",
        "introduction",
        "basics",
        "concepts",
        "pattern",
        "best practice",
        "faq",
        "troubleshooting",
        "debugging",
        "solution",
        "walkthrough",
        "cheatsheet",
        "demo",
        "sample",
        "advanced",
        "optimization",
        "performance",
        "security",
        "testing",
        "deployment",
        "integration",
    }

    def __init__(self, output_dir: str, concurrency: int = 3):
        """
        Инициализация сборщика документации.

        Args:
            output_dir: Директория для сохранения документации
            concurrency: Количество одновременно обрабатываемых пакетов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = None  # Будет инициализирован в методе collect_documentation
        self.semaphore = asyncio.Semaphore(
            5
        )  # Ограничиваем количество одновременных запросов
        self.concurrency = concurrency

        # Загружаем кеш PyPI, если он существует
        global pypi_cache
        if os.path.exists(PYPI_CACHE_FILE):
            try:
                with open(PYPI_CACHE_FILE, "r") as f:
                    pypi_cache = json.load(f)
                logger.info(
                    f"Загружен кеш PyPI с информацией о {len(pypi_cache)} пакетах"
                )
            except Exception as e:
                logger.warning(f"Не удалось загрузить кеш PyPI: {e}")
                pypi_cache = {}

    async def get_package_info(self, package: str) -> Dict[str, Any]:
        """
        Получение информации о пакете из PyPI с использованием кеша.

        Args:
            package: Имя пакета

        Returns:
            Словарь с информацией о пакете
        """
        try:
            # Используем только имя пакета без версий и опций
            clean_package = (
                package.split("~=")[0]
                .split("==")[0]
                .split(">=")[0]
                .split("<=")[0]
                .split(";")[0]
                .strip()
            )

            # Проверяем кеш
            if clean_package in pypi_cache:
                logger.debug(f"Используем кешированную информацию для {clean_package}")
                return pypi_cache[clean_package]

            # Если нет в кеше, делаем запрос к PyPI
            async with self.semaphore:
                async with self.session.get(
                    f"https://pypi.org/pypi/{clean_package}/json",
                    timeout=aiohttp_timeout,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Сохраняем в кеш
                        pypi_cache[clean_package] = data
                        # Периодически сохраняем кеш на диск
                        if (
                            random.random() < 0.1
                        ):  # 10% шанс сохранения после каждого запроса
                            await self._save_pypi_cache()
                        return data
                    else:
                        logger.error(
                            f"Ошибка при получении информации о {package}: {response.status}"
                        )
                        return {}
        except asyncio.TimeoutError:
            logger.error(f"Таймаут при получении информации о {package}")
            return {}
        except Exception as e:
            logger.error(f"Ошибка при получении информации о {package}: {e}")
            return {}

    async def _save_pypi_cache(self):
        """Сохраняет кеш PyPI на диск."""
        try:
            async with aiofiles.open(PYPI_CACHE_FILE, "w") as f:
                await f.write(json.dumps(pypi_cache))
            logger.debug(f"Кеш PyPI сохранен ({len(pypi_cache)} записей)")
        except Exception as e:
            logger.warning(f"Не удалось сохранить кеш PyPI: {e}")

    def get_github_repo(
        self, package_info: Dict[str, Any], package_name: str
    ) -> Optional[str]:
        """
        Получение URL GitHub репозитория.

        Args:
            package_info: Информация о пакете из PyPI
            package_name: Имя пакета

        Returns:
            URL GitHub репозитория или None
        """
        if not package_info:
            return None

        # Словарь известных репозиториев для популярных пакетов
        known_repos = {
            "sqlalchemy": "https://github.com/sqlalchemy/sqlalchemy",
            "numpy": "https://github.com/numpy/numpy",
            "pandas": "https://github.com/pandas-dev/pandas",
            "requests": "https://github.com/psf/requests",
            "flask": "https://github.com/pallets/flask",
            "django": "https://github.com/django/django",
            "tensorflow": "https://github.com/tensorflow/tensorflow",
            "pytorch": "https://github.com/pytorch/pytorch",
            "scikit-learn": "https://github.com/scikit-learn/scikit-learn",
            "matplotlib": "https://github.com/matplotlib/matplotlib",
            "beautifulsoup4": "https://github.com/wention/BeautifulSoup4",
            "aiofiles": "https://github.com/Tinche/aiofiles",
            "tqdm": "https://github.com/tqdm/tqdm",
            "markdownify": "https://github.com/matthewwithanm/python-markdownify",
            "isort": "https://github.com/PyCQA/isort",
            "asyncio": "https://github.com/python/asyncio",
            "tomli": "https://github.com/hukkin/tomli",
            "mako": "https://github.com/sqlalchemy/mako",
            "fastapi": "https://github.com/tiangolo/fastapi",
            "pydantic": "https://github.com/pydantic/pydantic",
            "uvicorn": "https://github.com/encode/uvicorn",
            "starlette": "https://github.com/encode/starlette",
            "httpx": "https://github.com/encode/httpx",
            "pytest": "https://github.com/pytest-dev/pytest",
            "sphinx": "https://github.com/sphinx-doc/sphinx",
            "celery": "https://github.com/celery/celery",
            "redis": "https://github.com/redis/redis-py",
            "aiohttp": "https://github.com/aio-libs/aiohttp",
            "jinja2": "https://github.com/pallets/jinja",
            "werkzeug": "https://github.com/pallets/werkzeug",
            "click": "https://github.com/pallets/click",
            "pillow": "https://github.com/python-pillow/Pillow",
            "sqlmodel": "https://github.com/tiangolo/sqlmodel",
            "typer": "https://github.com/tiangolo/typer",
            "rich": "https://github.com/Textualize/rich",
            "black": "https://github.com/psf/black",
            "flake8": "https://github.com/PyCQA/flake8",
            "mypy": "https://github.com/python/mypy",
            "poetry": "https://github.com/python-poetry/poetry",
            "pipenv": "https://github.com/pypa/pipenv",
            "alembic": "https://github.com/sqlalchemy/alembic",
            "pymongo": "https://github.com/mongodb/mongo-python-driver",
            "motor": "https://github.com/mongodb/motor",
            "psycopg2": "https://github.com/psycopg/psycopg2",
            "asyncpg": "https://github.com/MagicStack/asyncpg",
            "tortoise-orm": "https://github.com/tortoise/tortoise-orm",
            "sqlalchemy-utils": "https://github.com/kvesteri/sqlalchemy-utils",
            "pyjwt": "https://github.com/jpadilla/pyjwt",
            "passlib": "https://github.com/glic3rinu/passlib",
            "python-jose": "https://github.com/mpdavis/python-jose",
            "python-multipart": "https://github.com/andrew-d/python-multipart",
            "email-validator": "https://github.com/JoshData/python-email-validator",
            "orjson": "https://github.com/ijl/orjson",
            "ujson": "https://github.com/ultrajson/ultrajson",
            "python-dotenv": "https://github.com/theskumar/python-dotenv",
            "pyyaml": "https://github.com/yaml/pyyaml",
            "toml": "https://github.com/uiri/toml",
            "ruff": "https://github.com/charliermarsh/ruff",
        }

        # Проверяем, есть ли пакет в списке известных репозиториев
        package_lower = package_name.lower()
        if package_lower in known_repos:
            return known_repos[package_lower]

        info = package_info.get("info", {})
        project_urls = info.get("project_urls") or {}

        # Поиск в разных полях
        for key in [
            "GitHub",
            "Source",
            "Source Code",
            "Repository",
            "Code",
            "Homepage",
        ]:
            if url := project_urls.get(key):
                if "github.com" in url:
                    # Очистка URL от лишних частей пути
                    parts = url.split("github.com/")
                    if len(parts) > 1:
                        repo_path = parts[1].split("/tree/")[0]
                        return f"https://github.com/{repo_path}"

        # Поиск в homepage и других URL
        for url in [
            info.get("home_page"),
            info.get("package_url"),
            info.get("project_url"),
        ]:
            if url and "github.com" in url:
                parts = url.split("github.com/")
                if len(parts) > 1:
                    repo_path = parts[1].split("/tree/")[0]
                    return f"https://github.com/{repo_path}"

        # Поиск в описании и других текстовых полях
        for text_field in [info.get("description", ""), info.get("summary", "")]:
            if text_field and "github.com" in text_field:
                # Ищем URL GitHub в тексте
                github_urls = re.findall(
                    r"https?://github\.com/[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+", text_field
                )
                if github_urls:
                    return github_urls[0]

        # Проверяем альтернативные источники кода
        for key, url in project_urls.items():
            if url and ("gitlab.com" in url or "bitbucket.org" in url):
                logger.info(f"Найден не-GitHub репозиторий для {package_name}: {url}")
                return None  # Пока работаем только с GitHub

        return None

    async def is_useful_documentation(self, file_path: Path) -> bool:
        """
        Проверяет, является ли файл полезной документацией на английском или русском языке.

        Args:
            file_path: Путь к файлу

        Returns:
            True, если файл содержит полезную документацию
        """
        try:
            # Проверка размера файла
            if file_path.stat().st_size > 10_000_000:  # Пропускаем файлы больше 10MB
                return False

            # Проверка содержимого файла
            async with aiofiles.open(
                file_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                content = await f.read(8000)  # Читаем первые 8KB для анализа
                content_lower = content.lower()

                # Проверка языка - ищем признаки других языков (не английский и не русский)
                # Список распространенных слов и символов на других языках
                other_languages_markers = [
                    "über",
                    "für",
                    "schön",  # немецкий
                    "à",
                    "é",
                    "è",
                    "ê",
                    "ç",
                    "œ",  # французский
                    "你好",
                    "中文",
                    "汉语",  # китайский
                    "안녕하세요",
                    "한국어",  # корейский
                    "こんにちは",
                    "おはよう",  # японский
                    "hola",
                    "cómo",
                    "está",  # испанский
                    "olá",
                    "obrigado",  # португальский
                    "ciao",
                    "grazie",  # итальянский
                ]

                # Если найдены маркеры других языков и нет достаточного количества английских/русских слов
                english_russian_markers = [
                    "the",
                    "and",
                    "is",
                    "in",
                    "to",
                    "of",
                    "for",
                    "with",
                    "это",
                    "для",
                    "как",
                    "что",
                    "или",
                    "при",
                    "если",
                ]

                # Подсчитываем количество маркеров других языков и английских/русских маркеров
                other_lang_count = sum(
                    1 for marker in other_languages_markers if marker in content_lower
                )
                eng_rus_count = sum(
                    1 for marker in english_russian_markers if marker in content_lower
                )

                # Если маркеров других языков больше, чем английских/русских, пропускаем файл
                if other_lang_count > eng_rus_count and other_lang_count > 3:
                    return False

                # Проверка на наличие ключевых слов
                if any(keyword in content_lower for keyword in self.DOC_KEYWORDS):
                    return True

                # Специальная обработка для разных форматов
                if file_path.suffix == ".rst":
                    # Проверка на RST заголовки и директивы
                    if any(
                        pattern in content
                        for pattern in ["===", "---", ".. code-block::", ".. note::"]
                    ):
                        return True
                elif file_path.suffix == ".md":
                    # Проверка на Markdown заголовки и блоки кода
                    if any(pattern in content for pattern in ["##", "```", "---"]):
                        return True
                elif file_path.suffix == ".ipynb":
                    # Проверка на наличие markdown ячеек и кода
                    if (
                        '"cell_type": "markdown"' in content
                        or '"cell_type": "code"' in content
                    ):
                        return True
                elif file_path.suffix == ".py":
                    # Проверка на наличие docstrings и примеров
                    if '"""' in content or "'''" in content or "Example:" in content:
                        return True

            return False
        except Exception as e:
            logger.warning(f"Ошибка при анализе {file_path}: {e}")
            return False

    async def clone_repo(self, github_url: str, temp_dir: Path) -> bool:
        """
        Клонирование репозитория с использованием asyncio.

        Args:
            github_url: URL GitHub репозитория
            temp_dir: Временная директория для клонирования

        Returns:
            True, если клонирование успешно
        """
        try:
            # Создаем процесс для клонирования репозитория
            process = await asyncio.create_subprocess_exec(
                "git",
                "clone",
                "--depth",
                "1",
                github_url,
                str(temp_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Ждем завершения процесса с таймаутом
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=540
                )
                if process.returncode != 0:
                    logger.error(
                        f"Ошибка при клонировании {github_url}: {stderr.decode()}"
                    )
                    return False
                return True
            except asyncio.TimeoutError:
                # Убиваем процесс, если он не завершился за 540 секунд
                process.kill()
                logger.error(f"Таймаут при клонировании {github_url}")
                return False

        except Exception as e:
            logger.error(f"Ошибка при клонировании {github_url}: {e}")
            return False

    async def process_package(self, package: str) -> Optional[Dict[str, Any]]:
        """
        Обработка одного пакета.

        Args:
            package: Имя пакета

        Returns:
            Словарь с информацией о собранной документации или None
        """
        try:
            logger.info(f"Сбор документации для {package}")

            # Используем только имя пакета без версий и опций для поиска в PyPI
            clean_package = (
                package.split("~=")[0]
                .split("==")[0]
                .split(">=")[0]
                .split("<=")[0]
                .split(";")[0]
                .strip()
            )

            # Пропускаем URL-зависимости
            if clean_package.startswith(("http://", "https://", "git+")):
                logger.warning(f"Пропускаем URL-зависимость: {package}")
                return None

            # Получаем информацию о пакете из PyPI
            package_info = await self.get_package_info(clean_package)
            if not package_info:
                logger.warning(f"Нет информации о пакете {clean_package}")
                return None

            # Получаем URL GitHub репозитория
            github_url = self.get_github_repo(package_info, clean_package)
            if not github_url:
                logger.warning(f"Не найден GitHub репозиторий для {clean_package}")
                return None

            # Создание временной директории для клонирования
            temp_dir = self.output_dir / f"temp_{clean_package}"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(exist_ok=True)

            try:
                # Клонирование репозитория
                clone_success = await self.clone_repo(github_url, temp_dir)
                if not clone_success:
                    return None

                # Поиск нужных директорий и файлов
                useful_files = []
                total_files = 0

                # Собираем потенциальные файлы документации
                potential_doc_files = []
                for path in temp_dir.rglob("*"):
                    if path.is_file() and path.suffix.lower() in self.DOC_EXTENSIONS:
                        total_files += 1
                        # Проверяем, находится ли файл в нужной директории
                        if any(
                            parent.name.lower() in self.DOCS_DIRS
                            for parent in path.parents
                        ):
                            potential_doc_files.append(path)

                # Если не нашли файлы в документационных директориях, ищем в корне проекта
                if not potential_doc_files:
                    for path in temp_dir.glob("*"):
                        if (
                            path.is_file()
                            and path.suffix.lower() in self.DOC_EXTENSIONS
                        ):
                            potential_doc_files.append(path)

                # Асинхронно проверяем каждый потенциальный файл документации
                # Ограничиваем количество одновременных проверок
                file_semaphore = asyncio.Semaphore(
                    20
                )  # Проверяем до 20 файлов одновременно

                async def check_file(file_path):
                    async with file_semaphore:
                        if await self.is_useful_documentation(file_path):
                            return file_path
                        return None

                # Запускаем все проверки параллельно
                file_check_tasks = [
                    check_file(file_path) for file_path in potential_doc_files
                ]
                file_check_results = await asyncio.gather(*file_check_tasks)

                # Фильтруем None результаты
                useful_files = [
                    file_path
                    for file_path in file_check_results
                    if file_path is not None
                ]
                useful_files_found = len(useful_files)

                logger.info(
                    f"Найдено {useful_files_found} полезных файлов из {total_files} для {clean_package}"
                )

                # Фильтруем файлы по языку
                language_filtered_files = []
                for file_path in useful_files:
                    try:
                        # Проверяем первые 10KB файла на наличие признаков других языков
                        async with aiofiles.open(
                            file_path, "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            content = await f.read(10000)

                            # Проверка на наличие большого количества символов других языков
                            non_eng_rus_chars = sum(
                                1
                                for char in content
                                if ord(char) > 127 and not (1040 <= ord(char) <= 1103)
                            )
                            total_chars = len(content)

                            # Если более 30% символов не английские и не русские, пропускаем файл
                            if (
                                non_eng_rus_chars > 0
                                and non_eng_rus_chars / total_chars > 0.3
                            ):
                                logger.info(
                                    f"Пропускаем файл на другом языке: {file_path}"
                                )
                                continue

                            language_filtered_files.append(file_path)
                    except Exception as e:
                        logger.warning(
                            f"Ошибка при проверке языка файла {file_path}: {e}"
                        )

                # Обновляем список полезных файлов
                useful_files = language_filtered_files
                useful_files_found = len(useful_files)

                logger.info(
                    f"После фильтрации по языку осталось {useful_files_found} файлов для {clean_package}"
                )

                if useful_files:
                    # Создание директории для документации пакета
                    package_dir = self.output_dir / clean_package
                    if package_dir.exists():
                        shutil.rmtree(package_dir)
                    package_dir.mkdir(exist_ok=True)

                    # Асинхронное копирование всех полезных файлов
                    async def copy_file(file_path):
                        relative_path = file_path.relative_to(temp_dir)
                        target_path = package_dir / relative_path
                        target_path.parent.mkdir(parents=True, exist_ok=True)

                        async with aiofiles.open(file_path, "rb") as src:
                            content = await src.read()
                            async with aiofiles.open(target_path, "wb") as dst:
                                await dst.write(content)
                        return str(relative_path)

                    # Копируем файлы параллельно с ограничением
                    copy_semaphore = asyncio.Semaphore(
                        10
                    )  # Ограничиваем количество одновременных копирований

                    async def copy_with_semaphore(file_path):
                        async with copy_semaphore:
                            return await copy_file(file_path)

                    copied_files = await asyncio.gather(
                        *[copy_with_semaphore(file_path) for file_path in useful_files]
                    )

                    # Сохранение информации о пакете
                    return {
                        "name": clean_package,
                        "version": package_info.get("info", {}).get("version", ""),
                        "description": package_info.get("info", {}).get("summary", ""),
                        "long_description": package_info.get("info", {}).get(
                            "description", ""
                        )[:1000]
                        + "..."
                        if package_info.get("info", {}).get("description", "")
                        else "",
                        "github_url": github_url,
                        "pypi_url": f"https://pypi.org/project/{clean_package}/",
                        "docs_path": str(package_dir),
                        "files": copied_files,
                        "stats": {
                            "total_files": total_files,
                            "useful_files": useful_files_found,
                        },
                    }

                else:
                    logger.warning(
                        f"Не найдено полезной документации для {clean_package}"
                    )
                    return None

            finally:
                # Очистка временной директории
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

        except Exception as e:
            logger.error(f"Ошибка при обработке пакета {package}: {e}")
            return None

    async def collect_documentation(
        self, dependencies: Set[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Сбор документации для всех зависимостей.

        Args:
            dependencies: Множество имен пакетов

        Returns:
            Словарь с информацией о собранной документации
        """
        docs_info = {}

        # Создаем HTTP-сессию
        async with aiohttp.ClientSession(timeout=aiohttp_timeout) as session:
            self.session = session

            # Создаем задачи для обработки пакетов
            tasks = []
            for package in dependencies:
                tasks.append(self.process_package(package))

            # Выполняем задачи с ограничением параллельности
            semaphore = asyncio.Semaphore(
                self.concurrency
            )  # Ограничиваем количество одновременных клонирований

            async def process_with_semaphore(package):
                async with semaphore:
                    return await self.process_package(package)

            # Запускаем задачи и собираем результаты
            results = await asyncio.gather(
                *[process_with_semaphore(package) for package in dependencies]
            )

            # Обрабатываем результаты
            for package, result in zip(dependencies, results):
                if result:
                    docs_info[result["name"]] = result

        # Сохраняем кеш PyPI перед завершением
        await self._save_pypi_cache()

        return docs_info


class DocumentationCombiner:
    """Combines documentation files for each library into unified text files."""

    # Maximum size for combined documentation files (1.8MB in bytes)
    MAX_FILE_SIZE = 1.8 * 1024 * 1024

    def __init__(self, output_dir: str):
        """
        Инициализация комбинатора документации.

        Args:
            output_dir: Директория для сохранения объединенных файлов
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def combine_documentation(
        self, docs_info: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """
        Combines documentation for each library into unified text files.
        If documentation exceeds MAX_FILE_SIZE, it splits into multiple parts.

        Args:
            docs_info: Dictionary with information about collected documentation

        Returns:
            Dictionary mapping package names to lists of combined documentation file paths
        """
        result = {}

        # Создаем задачи для обработки пакетов
        tasks = []
        for package, info in docs_info.items():
            tasks.append(self.process_package(package, info))

        # Выполняем задачи параллельно
        results = await asyncio.gather(*tasks)

        # Обрабатываем результаты
        for package, combined_files in results:
            if combined_files:
                result[package] = combined_files

        return result

    async def _read_file_content(
        self, file_path: Path, package_dir: Path
    ) -> Optional[str]:
        """
        Асинхронно читает содержимое файла и проверяет язык.

        Args:
            file_path: Путь к файлу
            package_dir: Корневая директория пакета

        Returns:
            Содержимое файла с заголовком или None
        """
        try:
            async with aiofiles.open(
                file_path, "r", encoding="utf-8", errors="ignore"
            ) as f:
                content = await f.read()

                # Проверка языка - пропускаем файлы не на английском/русском
                non_eng_rus_chars = sum(
                    1
                    for char in content
                    if ord(char) > 127 and not (1040 <= ord(char) <= 1103)
                )
                total_chars = len(content)

                # Если более 30% символов не английские и не русские, пропускаем файл
                if non_eng_rus_chars > 0 and non_eng_rus_chars / total_chars > 0.3:
                    logger.info(
                        f"Пропускаем файл на другом языке при объединении: {file_path}"
                    )
                    return None

                # Add file header with path information
                header = (
                    f"\n\n{'='*80}\n{file_path.relative_to(package_dir)}\n{'='*80}\n\n"
                )
                return header + content
        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")
            return None

    async def process_package(
        self, package: str, info: Dict[str, Any]
    ) -> Tuple[str, Optional[List[str]]]:
        """
        Обработка одного пакета.

        Args:
            package: Имя пакета
            info: Информация о пакете

        Returns:
            Кортеж (имя пакета, список путей к объединенным файлам)
        """
        logger.info(f"Combining documentation for {package}")

        # Используем корневую директорию для объединенных файлов
        combined_dir = self.output_dir

        # Get all documentation files for this package
        package_dir = Path(info.get("docs_path", ""))
        if not package_dir.exists():
            logger.warning(f"No documentation directory found for {package}")
            return package, None

        # Находим все файлы документации для обработки
        doc_file_paths = []
        for file_path in sorted(package_dir.glob("**/*")):
            if file_path.is_file() and file_path.suffix.lower() in [
                ".md",
                ".rst",
                ".txt",
                ".py",
                ".ipynb",
            ]:
                doc_file_paths.append(file_path)

        if not doc_file_paths:
            logger.warning(f"No documentation files found for {package}")
            return package, None

        # Асинхронно читаем содержимое каждого файла
        file_semaphore = asyncio.Semaphore(
            20
        )  # Ограничиваем количество одновременно открытых файлов

        async def read_file_with_semaphore(file_path):
            async with file_semaphore:
                return await self._read_file_content(file_path, package_dir)

        # Запускаем асинхронное чтение всех файлов
        content_tasks = [
            read_file_with_semaphore(file_path) for file_path in doc_file_paths
        ]
        content_results = await asyncio.gather(*content_tasks)

        # Фильтруем None результаты
        all_content = [content for content in content_results if content is not None]

        if not all_content:
            logger.warning(f"No content found for {package}")
            return package, None

        # Combine and split content if needed
        combined_files = await self._split_content(package, combined_dir, all_content)

        logger.info(
            f"Created {len(combined_files)} combined documentation file(s) for {package}"
        )

        return package, combined_files

    async def _split_content(
        self, package: str, output_dir: Path, content_blocks: List[str]
    ) -> List[str]:
        """
        Splits content into multiple files if total size exceeds MAX_FILE_SIZE.

        Args:
            package: Package name
            output_dir: Directory to save combined files
            content_blocks: List of content blocks to combine

        Returns:
            List of paths to created files
        """
        created_files = []
        current_content = []
        current_size = 0
        part_num = 1

        # Добавляем таймаут для обработки каждого блока
        for block in content_blocks:
            try:
                block_size = len(block.encode("utf-8"))

                # If adding this block would exceed the limit, save current content and start a new file
                if current_size + block_size > self.MAX_FILE_SIZE and current_content:
                    file_path = await self._save_combined_file(
                        package, output_dir, current_content, part_num
                    )
                    created_files.append(file_path)
                    current_content = []
                    current_size = 0
                    part_num += 1

                # If a single block is larger than the limit, we need to split it
                if block_size > self.MAX_FILE_SIZE:
                    logger.info(
                        f"Large documentation block found for {package}, splitting into multiple parts"
                    )

                    # Добавляем таймаут для разделения больших блоков
                    try:
                        # Выполняем разделение блока с таймаутом в отдельном потоке
                        loop = asyncio.get_event_loop()
                        split_task = loop.run_in_executor(
                            thread_pool, self._split_large_block, block
                        )
                        split_blocks = await asyncio.wait_for(split_task, timeout=60)

                        # Сохраняем разделенные блоки
                        for split_block in split_blocks:
                            file_path = await self._save_combined_file(
                                package, output_dir, [split_block], part_num
                            )
                            created_files.append(file_path)
                            part_num += 1
                    except asyncio.TimeoutError:
                        logger.error(
                            f"Timeout while splitting large block for {package}. Skipping this block."
                        )
                        # Сохраняем сообщение об ошибке вместо блока
                        error_message = "Error: This documentation block was too large to process within the time limit."
                        file_path = await self._save_combined_file(
                            package, output_dir, [error_message], part_num
                        )
                        created_files.append(file_path)
                        part_num += 1
                else:
                    current_content.append(block)
                    current_size += block_size
            except Exception as e:
                logger.error(f"Error processing content block for {package}: {e}")
                # Продолжаем с следующим блоком
                continue

        # Save any remaining content
        if current_content:
            try:
                file_path = await self._save_combined_file(
                    package, output_dir, current_content, part_num
                )
                created_files.append(file_path)
            except Exception as e:
                logger.error(f"Error saving final content for {package}: {e}")

        return created_files

    def _split_large_block(self, block: str) -> List[str]:
        """
        Splits a large content block into smaller chunks that fit within MAX_FILE_SIZE.
        Оптимизированная версия для больших блоков.

        Args:
            block: Large content block to split

        Returns:
            List of smaller content blocks
        """
        result = []

        # Если блок не очень большой, используем стандартный алгоритм
        if len(block.encode("utf-8")) < self.MAX_FILE_SIZE * 5:
            remaining = block

            while remaining:
                # Calculate how many bytes we can safely take
                safe_size = int(self.MAX_FILE_SIZE * 0.95)

                # Find a good splitting point (end of paragraph or line)
                if len(remaining.encode("utf-8")) > safe_size:
                    # Try to find paragraph break
                    split_pos = remaining[:safe_size].rfind("\n\n")
                    if split_pos == -1:
                        # If no paragraph break, try line break
                        split_pos = remaining[:safe_size].rfind("\n")
                    if split_pos == -1:
                        # If no line break, just split at character boundary
                        split_pos = safe_size

                    chunk = remaining[:split_pos]
                    remaining = remaining[split_pos:]
                else:
                    chunk = remaining
                    remaining = ""

                result.append(chunk)
        else:
            # Для очень больших блоков используем более эффективный алгоритм
            # Разбиваем текст на строки
            lines = block.split("\n")
            current_chunk = []
            current_size = 0

            for line in lines:
                line_size = len((line + "\n").encode("utf-8"))

                # Если добавление этой строки превысит лимит, сохраняем текущий чанк
                if current_size + line_size > self.MAX_FILE_SIZE and current_chunk:
                    result.append("\n".join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Если одна строка больше лимита, разбиваем ее на части
                if line_size > self.MAX_FILE_SIZE:
                    # Разбиваем длинную строку на части
                    remaining_line = line
                    while remaining_line:
                        safe_size = int(self.MAX_FILE_SIZE * 0.95)
                        if len(remaining_line.encode("utf-8")) > safe_size:
                            # Находим безопасную точку разделения
                            # Для простоты разделяем по символам
                            split_pos = (
                                safe_size // 4
                            )  # Примерно учитываем UTF-8 кодирование
                            chunk = remaining_line[:split_pos]
                            remaining_line = remaining_line[split_pos:]
                        else:
                            chunk = remaining_line
                            remaining_line = ""

                        # Добавляем часть строки как отдельный чанк
                        result.append(chunk)
                else:
                    # Добавляем строку к текущему чанку
                    current_chunk.append(line)
                    current_size += line_size

            # Добавляем последний чанк, если он не пустой
            if current_chunk:
                result.append("\n".join(current_chunk))

        # Проверяем, что все чанки не превышают максимальный размер
        for i, chunk in enumerate(result):
            chunk_size = len(chunk.encode("utf-8"))
            if chunk_size > self.MAX_FILE_SIZE:
                logger.warning(
                    f"Chunk {i} is still too large: {chunk_size} bytes. Truncating."
                )
                # Обрезаем чанк до безопасного размера
                safe_chars = (
                    int(self.MAX_FILE_SIZE * 0.9) // 4
                )  # Примерная оценка для UTF-8
                result[i] = (
                    chunk[:safe_chars]
                    + "\n\n... (content truncated due to size limits)"
                )

        return result

    async def _save_combined_file(
        self, package: str, output_dir: Path, content_blocks: List[str], part_num: int
    ) -> str:
        """
        Saves combined content to a file.

        Args:
            package: Package name
            output_dir: Directory to save file
            content_blocks: Content blocks to combine
            part_num: Part number for filename

        Returns:
            Path to created file
        """
        if part_num == 1:
            filename = f"{package}_combined.txt"
        else:
            filename = f"{package}_combined_part{part_num}.txt"

        file_path = output_dir / filename

        # Add header with package information
        header = f"Combined documentation for {package}"
        if part_num > 1:
            header += f" (Part {part_num})"
        header = f"{header}\n{'='*len(header)}\n\n"

        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
            await f.write(header)
            await f.write("\n".join(content_blocks))

        return str(file_path)


class LectureGenerator:
    """Генератор учебных материалов на основе собранной документации."""

    def __init__(self, output_dir: str):
        """
        Инициализация генератора лекций.

        Args:
            output_dir: Директория для сохранения лекций
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def generate_lecture(
        self, combined_docs: Dict[str, List[str]], docs_info: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Создает лекцию по библиотекам Python на основе собранной документации.

        Args:
            combined_docs: Словарь с комбинированными файлами документации
            docs_info: Словарь с информацией о собранной документации
        """
        logger.info("Создание лекции на основе собранной документации...")

        # Создаем файл с содержанием лекции
        toc_path = self.output_dir / "00_содержание.md"

        # Формируем содержание
        toc_content = """# Библиотеки Python: Обзор, примеры использования и лучшие практики

## Содержание

"""

        # Добавляем разделы для каждой библиотеки
        chapter_num = 1
        for package_name in sorted(docs_info.keys()):
            package_info = docs_info[package_name]
            description = package_info.get("description", "")
            short_desc = (
                description[:100] + "..." if len(description) > 100 else description
            )
            toc_content += f"{chapter_num}. [{package_name.capitalize()}]({chapter_num:02d}_{package_name}.md) - {short_desc}\n"
            chapter_num += 1

        # Сохраняем содержание
        toc_task = aiofiles.open(toc_path, "w", encoding="utf-8")

        # Асинхронная функция для создания лекции по одной библиотеке
        async def create_package_lecture(
            chapter_num: int, package_name: str, package_info: Dict[str, Any]
        ) -> None:
            # Путь к файлу лекции для текущей библиотеки
            lecture_file = self.output_dir / f"{chapter_num:02d}_{package_name}.md"

            # Формируем содержимое лекции
            lecture_content = f"""# {package_name.capitalize()}: Обзор и примеры использования

## Описание

{package_info.get('description', 'Нет описания')}

## Ссылки

- [GitHub репозиторий]({package_info.get('github_url', '')})
- [PyPI страница]({package_info.get('pypi_url', '')})

## Версия

{package_info.get('version', 'Неизвестна')}

## Основные возможности и примеры использования

"""

            # Если есть комбинированные документы, используем их для создания лекции
            if package_name in combined_docs:
                # Берем первый файл (основной или единственный)
                doc_file = combined_docs[package_name][0]

                try:
                    # Читаем файл документации
                    async with aiofiles.open(
                        doc_file, "r", encoding="utf-8", errors="ignore"
                    ) as f:
                        doc_content = await f.read()

                    # Ищем примеры кода и основные возможности
                    import re

                    # Ищем примеры кода (блоки кода в Markdown и RST)
                    code_examples = re.findall(
                        r"```(?:python)?\s*(.*?)\s*```", doc_content, re.DOTALL
                    )
                    if not code_examples:
                        # Ищем примеры кода в RST формате
                        code_examples = re.findall(
                            r".. code-block:: python\s*\n\n(.*?)(?=\n\n\S|\Z)",
                            doc_content,
                            re.DOTALL,
                        )

                    # Ищем разделы с заголовками
                    sections = re.findall(
                        r"#+\s+(.*?)\n+(.*?)(?=\n#+\s+|\Z)", doc_content, re.DOTALL
                    )
                    if not sections:
                        # Ищем заголовки в RST формате
                        sections = re.findall(
                            r"(.*?)\n([=\-~]+)\n+(.*?)(?=\n\S+\n[=\-~]+|\Z)",
                            doc_content,
                            re.DOTALL,
                        )
                        sections = [(title, content) for title, _, content in sections]

                    # Добавляем информацию о основных возможностях
                    features_added = False
                    for section_title, section_content in sections:
                        # Пропускаем слишком длинные разделы
                        if len(section_content) > 1000:
                            section_content = section_content[:1000] + "...\n"

                        # Ищем разделы, связанные с возможностями, примерами, руководствами
                        if any(
                            keyword in section_title.lower()
                            for keyword in [
                                "features",
                                "usage",
                                "guide",
                                "tutorial",
                                "example",
                                "installation",
                                "getting started",
                                "quick start",
                                "overview",
                                "introduction",
                                "basic",
                                "how to",
                            ]
                        ):
                            lecture_content += (
                                f"### {section_title}\n\n{section_content}\n\n"
                            )
                            features_added = True

                    # Если не нашли подходящих разделов, добавляем несколько примеров кода
                    if not features_added and code_examples:
                        lecture_content += "### Примеры кода\n\n"
                        # Добавляем до 3 примеров кода
                        for i, example in enumerate(code_examples[:3]):
                            if len(example) > 500:  # Ограничиваем длину примера
                                example = example[:500] + "...\n"
                            lecture_content += (
                                f"Пример {i+1}:\n\n```python\n{example}\n```\n\n"
                            )

                except Exception as e:
                    logger.error(f"Ошибка при создании лекции для {package_name}: {e}")
                    lecture_content += (
                        "Не удалось обработать документацию для этой библиотеки.\n"
                    )

            # Если нет документации, добавляем заглушку
            else:
                lecture_content += "Для этой библиотеки не удалось собрать документацию. Пожалуйста, обратитесь к официальной документации.\n"

            # Добавляем рекомендации по использованию
            lecture_content += """
## Рекомендации по использованию

- Изучите официальную документацию библиотеки для получения полной информации
- Проверьте наличие актуальных версий библиотеки перед использованием
- Обратитесь к GitHub репозиторию для просмотра исходного кода и примеров
- Присоединяйтесь к сообществу пользователей библиотеки для получения поддержки
"""

            # Сохраняем лекцию
            async with aiofiles.open(lecture_file, "w", encoding="utf-8") as f:
                await f.write(lecture_content)

            logger.info(f"Создана лекция для библиотеки {package_name}")

        # Создаем задачи для обработки каждой библиотеки
        package_tasks = []
        for chapter_num, package_name in enumerate(sorted(docs_info.keys()), 1):
            package_info = docs_info[package_name]
            package_tasks.append(
                create_package_lecture(chapter_num, package_name, package_info)
            )

        # Запускаем сохранение содержания и создание лекций для всех библиотек параллельно
        async with toc_task as f:
            await f.write(toc_content)

        await asyncio.gather(*package_tasks)

        # Создаем индексный файл HTML для удобного просмотра
        await self._create_html_index(docs_info)

        logger.info(f"Лекция по библиотекам Python создана в {self.output_dir}")

    async def _create_html_index(self, docs_info: Dict[str, Dict[str, Any]]) -> None:
        """
        Создает HTML-индекс для удобного просмотра лекций.

        Args:
            docs_info: Словарь с информацией о собранной документации
        """
        index_path = self.output_dir / "index.html"

        index_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Лекция по библиотекам Python</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            max-width: 1200px;
            margin: 0 auto;
            color: #333;
            background-color: #f8f9fa;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .toc {
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .toc ol {
            padding-left: 20px;
        }
        .toc a {
            text-decoration: none;
            color: #3498db;
        }
        .toc a:hover {
            text-decoration: underline;
            color: #2980b9;
        }
        .footer {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 0.9em;
            color: #666;
        }
        .package-description {
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>Лекция по библиотекам Python</h1>
    
    <div class="toc">
        <h2>Содержание</h2>
        <ol>
"""

        # Добавляем ссылки на лекции для каждой библиотеки
        for chapter_num, package_name in enumerate(sorted(docs_info.keys()), 1):
            package_info = docs_info[package_name]
            description = package_info.get("description", "")
            short_desc = (
                description[:100] + "..." if len(description) > 100 else description
            )
            index_content += f'            <li><a href="{chapter_num:02d}_{package_name}.md">{package_name.capitalize()}</a> <span class="package-description">- {short_desc}</span></li>\n'

        index_content += (
            """
        </ol>
    </div>
    
    <p>Эта лекция сгенерирована автоматически на основе документации библиотек Python.</p>
    <p>Для просмотра Markdown файлов используйте любой Markdown просмотрщик или конвертер.</p>
    
    <div class="footer">
        <p>Сгенерировано с помощью Python Documentation Collector</p>
        <p>Дата создания: """
            + time.strftime("%Y-%m-%d %H:%M:%S")
            + """</p>
    </div>
</body>
</html>
"""
        )

        # Сохраняем индексный файл
        async with aiofiles.open(index_path, "w", encoding="utf-8") as f:
            await f.write(index_content)

        logger.info(f"Индексный файл HTML создан: {index_path}")


async def main_async():
    """Основная асинхронная функция."""
    parser = argparse.ArgumentParser(
        description="Анализ зависимостей и сбор документации"
    )
    parser.add_argument("--project-path", default=".", help="Путь к проекту")
    parser.add_argument(
        "--output-dir",
        default="./dependency_docs",
        help="Директория для сохранения результатов",
    )
    parser.add_argument(
        "--max-packages",
        type=int,
        default=0,
        help="Максимальное количество пакетов для обработки (0 = все)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Количество одновременно обрабатываемых пакетов",
    )
    parser.add_argument(
        "--library-list",
        type=str,
        default="",
        help="Список библиотек через запятую для сбора документации (если не указаны, используются зависимости проекта)",
    )
    parser.add_argument(
        "--create-lecture",
        action="store_true",
        help="Создать лекцию на основе собранной документации",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Уровень логирования",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Пропускать пакеты, для которых уже собрана документация",
    )
    parser.add_argument(
        "--scan-imports",
        action="store_true",
        help="Анализировать импорты в коде для определения используемых/неиспользуемых зависимостей",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Только анализировать зависимости, без сбора документации",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1000,
        help="Максимальное количество файлов для анализа импортов",
    )
    parser.add_argument(
        "--dynamic-scan",
        action="store_true",
        help="Динамическое определение стандартных библиотек и зависимостей",
    )
    parser.add_argument(
        "--markdown", action="store_true", help="Форматировать вывод анализа в Markdown"
    )
    args = parser.parse_args()

    # Устанавливаем уровень логирования
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Если запрошено динамическое сканирование, получаем списки библиотек
    global STDLIB_MODULES, PACKAGE_ALIASES

    if args.dynamic_scan:
        logger.info("Выполняется динамическое определение библиотек...")

        # Получаем список стандартных библиотек
        STDLIB_MODULES = get_cached_or_build_stdlib_modules()
        logger.info(f"Определено {len(STDLIB_MODULES)} стандартных библиотек")

        # Получаем список установленных пакетов
        installed_packages = get_installed_packages()
        logger.info(f"Найдено {len(installed_packages)} установленных пакетов")

        # Строим словарь псевдонимов
        PACKAGE_ALIASES = build_package_aliases_dict(installed_packages)

    # Определяем список библиотек
    dependencies = set()

    if args.library_list:
        # Используем предоставленный список библиотек
        dependencies = {lib.strip() for lib in args.library_list.split(",")}
        logger.info(f"Используется указанный список из {len(dependencies)} библиотек")
    else:
        # Анализ зависимостей проекта
        analyzer = DependencyAnalyzer(args.project_path)
        analysis_result = await analyzer.analyze_project(
            scan_imports=args.scan_imports, max_files=args.max_files
        )
        dependencies = analysis_result["dependencies"]
        logger.info(f"Найдено {len(dependencies)} зависимостей в проекте")

        # Выводим результаты анализа
        if args.scan_imports:
            if args.markdown:
                print_markdown_report(analysis_result)
            else:
                print_text_report(analysis_result)

        # Если указан флаг --analyze-only, завершаем работу
        if args.analyze_only:
            # Сохраняем результаты анализа в файл
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            analysis_path = output_dir / "dependencies_analysis.json"

            # Преобразуем множества в списки для сериализации JSON
            serializable_result = {
                "dependencies": list(analysis_result["dependencies"]),
                "dependency_files": {
                    k: [str(p) for p in v]
                    for k, v in analysis_result["dependency_files"]
                },
            }

            if args.scan_imports:
                serializable_result.update(
                    {
                        "imports": list(analysis_result["imports"]),
                        "probable_dependencies": list(
                            analysis_result["probable_dependencies"]
                        ),
                        "unused_dependencies": list(
                            analysis_result["unused_dependencies"]
                        ),
                    }
                )

            async with aiofiles.open(analysis_path, "w") as f:
                await f.write(json.dumps(serializable_result, indent=2))

            logger.info(f"Результаты анализа сохранены в {analysis_path}")
            return

    # Ограничение количества пакетов для обработки
    if args.max_packages > 0 and len(dependencies) > args.max_packages:
        dependencies_list = list(dependencies)
        # Сортируем для стабильности результатов
        dependencies_list.sort()
        dependencies = set(dependencies_list[: args.max_packages])
        logger.info(f"Ограничено до {len(dependencies)} пакетов")

    # Создаем директории для результатов
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_docs_dir = output_dir / "raw_docs"
    raw_docs_dir.mkdir(exist_ok=True)

    combined_docs_dir = output_dir / "combined_docs"
    combined_docs_dir.mkdir(exist_ok=True)

    # Проверяем наличие существующей документации, если указан флаг --skip-existing
    if args.skip_existing:
        existing_packages = set()
        report_path = output_dir / "report.json"
        if report_path.exists():
            try:
                with open(report_path, "r") as f:
                    report_data = json.load(f)
                    existing_packages = set(report_data.get("packages_info", {}).keys())
                    logger.info(
                        f"Найдена существующая документация для {len(existing_packages)} пакетов"
                    )
            except Exception as e:
                logger.warning(f"Не удалось загрузить существующий отчет: {e}")

        # Фильтруем зависимости, оставляя только те, для которых нет документации
        dependencies = dependencies - existing_packages
        logger.info(
            f"После фильтрации осталось {len(dependencies)} пакетов для обработки"
        )

    # Сбор документации
    collector = DocumentationCollector(str(raw_docs_dir), args.concurrency)
    docs_info = await collector.collect_documentation(dependencies)
    logger.info(f"Собрана документация для {len(docs_info)} пакетов")

    # Создание комбинированных файлов документации
    combiner = DocumentationCombiner(str(combined_docs_dir))
    combined_docs = await combiner.combine_documentation(docs_info)
    logger.info(
        f"Созданы комбинированные файлы документации для {len(combined_docs)} пакетов"
    )

    # Объединяем с существующим отчетом, если он есть и указан флаг --skip-existing
    if args.skip_existing:
        report_path = output_dir / "report.json"
        if report_path.exists():
            try:
                with open(report_path, "r") as f:
                    existing_report = json.load(f)
                    # Объединяем информацию о пакетах
                    existing_packages_info = existing_report.get("packages_info", {})
                    docs_info.update(existing_packages_info)
                    # Объединяем информацию о комбинированных файлах
                    existing_combined_docs = existing_report.get("combined_docs", {})
                    combined_docs.update(existing_combined_docs)
                    logger.info("Объединены данные с существующим отчетом")
            except Exception as e:
                logger.warning(f"Не удалось объединить с существующим отчетом: {e}")

    # Сохранение итогового отчета
    report = {
        "total_dependencies": len(dependencies),
        "processed_packages": len(docs_info),
        "packages_info": docs_info,
        "combined_docs": combined_docs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {
            "project_path": args.project_path,
            "output_dir": args.output_dir,
            "max_packages": args.max_packages,
            "concurrency": args.concurrency,
            "library_list": args.library_list,
            "create_lecture": args.create_lecture,
            "skip_existing": args.skip_existing,
            "scan_imports": args.scan_imports,
        },
    }

    report_path = output_dir / "report.json"
    async with aiofiles.open(report_path, "w") as f:
        await f.write(json.dumps(report, indent=2))

    logger.info(f"Отчет сохранен в {report_path}")

    # Создание лекции на основе собранной документации
    if args.create_lecture:
        lecture_path = output_dir / "lecture"
        lecture_path.mkdir(exist_ok=True)
        lecture_generator = LectureGenerator(str(lecture_path))
        await lecture_generator.generate_lecture(combined_docs, docs_info)
        logger.info(f"Лекция по библиотекам Python создана в {lecture_path}")


def print_text_report(analysis_result):
    """
    Выводит результаты анализа зависимостей в текстовом формате.

    Args:
        analysis_result (dict): Результаты анализа зависимостей
    """
    print("\nОбнаруженные файлы с зависимостями:")
    print("-" * 60)
    for file_type, file_paths in analysis_result["dependency_files"]:
        print(f"{file_type}: {len(file_paths)} шт.")
        for path in file_paths:
            print(f"  - {path}")

    print("\nЗависимости:")
    print("-" * 60)
    for dep in sorted(analysis_result["dependencies"]):
        print(f"- {dep}")

    if analysis_result["probable_dependencies"]:
        print("\nВероятные дополнительные зависимости (из импортов):")
        print("-" * 60)
        for dep in sorted(analysis_result["probable_dependencies"]):
            print(f"- {dep}")
    else:
        print("\nВероятных дополнительных зависимостей не обнаружено")

    if analysis_result["unused_dependencies"]:
        print("\nПотенциально неиспользуемые зависимости:")
        print("-" * 60)
        for dep in sorted(analysis_result["unused_dependencies"]):
            print(f"- {dep}")
    else:
        print("\nНеиспользуемых зависимостей не обнаружено")


def print_markdown_report(analysis_result):
    """
    Выводит результаты анализа зависимостей в формате Markdown.

    Args:
        analysis_result (dict): Результаты анализа зависимостей
    """
    print("# Анализ зависимостей Python-проекта")
    print()

    print("## Обнаруженные файлы с зависимостями")
    for file_type, file_paths in analysis_result["dependency_files"]:
        print(f"\n### {file_type} ({len(file_paths)} шт.)")
        for path in file_paths:
            print(f"* `{path}`")

    print("\n## Зависимости")
    if analysis_result["dependencies"]:
        for dep in sorted(analysis_result["dependencies"]):
            print(f"* `{dep}`")
    else:
        print("*Зависимости не обнаружены*")

    print("\n## Дополнительный анализ")

    print("\n### Вероятные дополнительные зависимости")
    if analysis_result["probable_dependencies"]:
        for dep in sorted(analysis_result["probable_dependencies"]):
            print(f"* `{dep}`")
    else:
        print("*Вероятных дополнительных зависимостей не обнаружено*")

    print("\n### Потенциально неиспользуемые зависимости")
    if analysis_result["unused_dependencies"]:
        for dep in sorted(analysis_result["unused_dependencies"]):
            print(
                f"* `{dep}` - указана в файлах зависимостей, но не используется в коде"
            )
    else:
        print("*Неиспользуемых зависимостей не обнаружено*")

    print("\n## Сводная информация")
    print(
        f"* Всего файлов с зависимостями: **{sum(len(files) for _, files in analysis_result['dependency_files'])}**"
    )
    print(f"* Всего зависимостей: **{len(analysis_result['dependencies'])}**")
    print(
        f"* Вероятных дополнительных зависимостей: **{len(analysis_result['probable_dependencies'])}**"
    )
    print(
        f"* Потенциально неиспользуемых зависимостей: **{len(analysis_result['unused_dependencies'])}**"
    )


def main():
    """Точка входа для синхронного запуска."""
    # Устанавливаем политику для Windows, если нужно
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Запускаем асинхронный код с настройками для обработки больших файлов
    try:
        start_time = time.time()
        asyncio.run(main_async(), debug=False)
        elapsed_time = time.time() - start_time
        logger.info(f"Выполнение завершено за {elapsed_time:.2f} секунд")
    except KeyboardInterrupt:
        logger.info("Выполнение прервано пользователем")
    except Exception as e:
        logger.error(f"Ошибка при выполнении: {e}", exc_info=True)
    finally:
        # Закрываем пул потоков
        thread_pool.shutdown(wait=False)


if __name__ == "__main__":
    main()
