import os
from pathlib import Path

list_of_files = [
    f"STT/__init__.py",
    f"STT/cloud_storage/__init__.py",
    f"STT/components/__init__.py",
    f"STT/configuration/__init__.py",
    f"STT/constants/__init__.py",
    f"STT/entity/__init__.py",
    f"STT/exceptions/__init__.py",
    f"STT/logger/__init__.py",
    f"STT/models/__init__.py",
    f"STT/pipeline/__init__.py",
    f"STT/utils/__init__.py",
    f'setup.py',
    f'requirements.txt'
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")
