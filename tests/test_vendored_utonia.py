from pathlib import Path


def test_vendored_utonia_contains_only_the_inference_runtime():
    root = Path("wan/utonia")
    required = {
        "__init__.py",
        "data.py",
        "model.py",
        "module.py",
        "registry.py",
        "structure.py",
        "transform.py",
        "utils.py",
        "serialization/__init__.py",
        "serialization/default.py",
        "serialization/hilbert.py",
        "serialization/z_order.py",
        "LICENSE",
    }

    assert all((root / path).is_file() for path in required)
    assert "Apache License" in (root / "LICENSE").read_text()
    assert not (root / "pca_trajectory.py").exists()
    assert not (root / "joint_trajectory_pca.py").exists()
    assert not (root / "rgb_trajectory.py").exists()


def test_packaging_discovers_wan_subpackages():
    definition = Path("pyproject.toml").read_text()

    assert "[tool.setuptools.packages.find]" in definition
    assert 'include = ["wan", "wan.*"]' in definition
    assert '"wan" = ["**/*.py", "utonia/LICENSE"]' in definition
