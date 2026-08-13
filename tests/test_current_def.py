from pathlib import Path


def test_current_def_declares_a_matching_utonia_runtime():
    definition = Path("current.def").read_text()

    assert "../utonia /opt/Utonia" not in definition
    assert "/opt/Utonia" not in definition
    assert '"torch==2.4.1" "torchvision==0.19.1"' in definition
    assert "https://download.pytorch.org/whl/cu124" in definition
    assert "torch-2.4.0+cu124.html" in definition
    assert "spconv-cu124" in definition
    assert "spconv-cu126" not in definition
    assert "torch-2.5.1+cu126.html" not in definition
    assert "pip install --no-deps --no-build-isolation -e" not in definition
    assert "pip install --upgrade pip setuptools wheel" in definition
    assert "flash_attn-2.6.3+cu126torch2.4-cp310-cp310-linux_x86_64.whl" in definition
    assert "$ENV_BIN/python - <<'PY'" not in definition
    assert "wget -qO /tmp/miniforge.sh" in definition
    assert "bash /tmp/miniforge.sh -b -p /opt/conda" in definition
    assert "rm /tmp/miniforge.sh" in definition
    assert "./Miniforge3-Linux-x86_64.sh /opt/miniforge.sh" not in definition
