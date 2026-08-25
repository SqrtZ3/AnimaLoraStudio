"""tools/check_missing_deps.py —— --no-venv 模式下"缺什么"的判据。

按发行版名（importlib.metadata）判，不按模块名 import：pyyaml 的模块是 yaml、
Pillow 是 PIL，按模块名猜会误报一堆。
"""
import importlib.metadata as md

import pytest

from tools import check_missing_deps as cmd


REQ = """\
# Core
torch>=2.0.0
requests[socks]>=2.31.0
pyyaml>=6.0.1   # inline comment
-r other.txt
--index-url https://example.invalid/simple

Pillow>=10.0.0
prodigy-plus-schedule-free>=2.0.0
torch>=2.0.0
"""


def test_parses_names_stripping_extras_versions_and_options():
    assert cmd.parse_requirement_names(REQ) == [
        "torch", "requests", "pyyaml", "Pillow", "prodigy-plus-schedule-free",
    ]


def test_dedupes_case_and_separator_insensitively():
    names = cmd.parse_requirement_names("Foo_Bar>=1\nfoo-bar>=2\nFOO-BAR\n")
    assert names == ["Foo_Bar"]


def test_reports_only_absent_dists(monkeypatch):
    present = {"torch", "pyyaml"}

    def _version(dist):
        if dist in present:
            return "1.0"
        raise md.PackageNotFoundError(dist)

    monkeypatch.setattr(cmd.md, "version", _version)
    assert cmd.missing_from(REQ) == ["requests", "Pillow", "prodigy-plus-schedule-free"]


def test_broken_metadata_counts_as_installed(monkeypatch):
    """元数据损坏当"装了"——宁可漏报也别制造假警报把人赶去 pip。"""

    def _boom(dist):
        raise RuntimeError("corrupt dist-info")

    monkeypatch.setattr(cmd.md, "version", _boom)
    assert cmd.missing_from("torch>=2.0.0\n") == []


def test_all_present_prints_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cmd.md, "version", lambda d: "1.0")
    req = tmp_path / "r.txt"
    req.write_text(REQ, encoding="utf-8")
    assert cmd.main([str(req)]) == 0
    assert capsys.readouterr().out == ""


def test_missing_file_is_not_fatal(capsys):
    assert cmd.main(["/no/such/requirements.txt"]) == 0
    assert capsys.readouterr().out == ""
