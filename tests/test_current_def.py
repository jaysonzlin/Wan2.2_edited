from pathlib import Path


def test_current_def_declares_a_matching_utonia_runtime():
    definition = Path("current.def").read_text()

    assert "../Utonia /opt/Utonia" in definition
    assert "https://download.pytorch.org/whl/cu126" in definition
    assert "torch-2.5.1+cu126.html" in definition
    assert "spconv-cu126==2.3.8" in definition
    assert "pip install --no-deps -e /opt/Utonia" in definition
    assert "flash-attn==2.6.3\" --no-build-isolation" in definition
    assert "import utonia" in definition
    assert "import torch_scatter" in definition
    assert "torch2.4-cp310" not in definition
