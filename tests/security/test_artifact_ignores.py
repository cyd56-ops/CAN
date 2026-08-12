"""本地敏感或大型研究产物的配置防护测试。"""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_sensitive_artifact_patterns_are_ignored() -> None:
    """秘密载体、模型产物和本地论文应默认排除在版本控制外。"""
    configured_patterns = {
        line.strip()
        for line in (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    required_patterns = {
        ".venv/",
        "artifacts/",
        "checkpoints/",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "paper/*.pdf",
    }

    assert required_patterns <= configured_patterns
