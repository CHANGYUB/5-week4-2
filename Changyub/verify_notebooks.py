"""현재 Python을 커널로 사용해 두 과제 노트북을 전체 실행한다."""
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent


def verify():
    with tempfile.TemporaryDirectory(prefix="week4-notebooks-") as folder:
        for name, sub in [("JUPYTER_RUNTIME_DIR", "jupyter"), ("IPYTHONDIR", "ipython"), ("MPLCONFIGDIR", "mpl")]:
            directory = Path(folder) / sub
            directory.mkdir()
            os.environ[name] = str(directory)
        for filename in ["prompt_assignment.ipynb", "rag_assignment.ipynb"]:
            path = ROOT / filename
            notebook = nbformat.read(path, as_version=4)
            manager = KernelManager(kernel_name="python3", transport="ipc" if os.name == "posix" else "tcp")
            if os.name == "posix":
                manager.ip = str(Path(folder) / "kernel")
            manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
            NotebookClient(notebook, km=manager, timeout=180,
                           resources={"metadata": {"path": str(ROOT)}}).execute()
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            print(f"실행 성공: {filename}", flush=True)


if __name__ == "__main__":
    verify()
