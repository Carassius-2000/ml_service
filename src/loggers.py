"""
Модуль предоставляет два логгера для обработки информационных и ошибочных сообщений.

Логгеры предварительно настроены с индивидуальным форматированием и потоками вывода:
- `info_logger` записывает сообщения уровня INFO в stdout
- `error_logger` записывает сообщения уровня ERROR в stderr

Attributes
----------
info_logger : logging.Logger
    Экземпляр логгера для информационных сообщений.
    - Уровень: INFO
    - Вывод: stdout
error_logger : logging.Logger
    Экземпляр логгера для сообщений об ошибках.
    - Уровень: ERROR
    - Вывод: stderr
"""

import logging
import sys

info_logger = logging.getLogger("info")
info_logger.setLevel(logging.INFO)
info_handler = logging.StreamHandler(sys.stdout)
info_handler.setFormatter(logging.Formatter("%(asctime)s - INFO: %(message)s"))
info_logger.addHandler(info_handler)

error_logger = logging.getLogger("error")
error_logger.setLevel(logging.ERROR)
error_handler = logging.StreamHandler(sys.stderr)
error_handler.setFormatter(logging.Formatter("%(asctime)s - ERROR: %(message)s"))
error_logger.addHandler(error_handler)
