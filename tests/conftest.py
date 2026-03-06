import shutil
import uuid
from pathlib import Path

import pytest


@pytest.fixture
def local_tmp_path():
    base = Path(".pytest_temp")
    base.mkdir(exist_ok=True)
    path = base / f"tmp_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
