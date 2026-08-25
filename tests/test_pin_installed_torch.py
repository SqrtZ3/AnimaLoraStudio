"""tools/pin_installed_torch.py —— 别让 pip 把厂商定制 torch 换成 PyPI 同名包。

场景是真实的：海光 DTK 的 `torch==2.9.0+das.opt1.dtk2604`、ROCm 的 `+rocm`、
自编译的 `+local` 都不在 PyPI 上。pip 一旦认为需要重装 torch，会拉一个同名但
完全不同的 wheel 盖上去，**并且成功退出** —— 之后 torch.cuda.is_available()
变 False，报错全指向别处。
"""
import importlib.metadata as md

import pytest

from tools import pin_installed_torch as pin


@pytest.fixture
def fake_installed(monkeypatch):
    """让 importlib.metadata.version 只认我们给的这张表。"""

    def _apply(table: dict[str, str]):
        def _version(dist: str) -> str:
            if dist in table:
                return table[dist]
            raise md.PackageNotFoundError(dist)

        monkeypatch.setattr(pin.md, "version", _version)

    return _apply


def test_pins_installed_packages_only(fake_installed):
    """没装的包不写进去 —— 否则会造出永远无法满足的约束。"""
    fake_installed({"torch": "2.9.0+das.opt1.dtk2604"})
    assert pin.build_constraints() == ["torch==2.9.0+das.opt1.dtk2604"]


def test_pins_the_whole_family_when_present(fake_installed):
    fake_installed({
        "torch": "2.9.0+das.opt1.dtk2604",
        "torchvision": "0.24.0+das.dtk2604",
        "triton": "3.2.0",
    })
    assert pin.build_constraints() == [
        "torch==2.9.0+das.opt1.dtk2604",
        "torchvision==0.24.0+das.dtk2604",
        "triton==3.2.0",
    ]


def test_no_torch_means_no_constraints(fake_installed):
    """全新 venv 首装：没有东西要保护，让 pip 正常走（行为与本工具加入前一致）。"""
    fake_installed({})
    assert pin.build_constraints() == []


def test_broken_metadata_does_not_raise(monkeypatch):
    """元数据损坏不能让 bootstrap 挂掉 —— 这是保护措施，不是失败点。"""

    def _boom(dist: str):
        raise RuntimeError("corrupt dist-info")

    monkeypatch.setattr(pin.md, "version", _boom)
    assert pin.build_constraints() == []


@pytest.mark.parametrize("version,expected", [
    ("2.9.0+das.opt1.dtk2604", True),
    ("2.9.1+cu130", True),
    ("2.6.0+rocm6.2", True),
    ("2.5.1", False),
    (None, False),
])
def test_vendor_build_detection(version, expected):
    assert pin.is_vendor_build(version) is expected


def test_writes_file_and_prints_path(tmp_path, fake_installed, capsys):
    fake_installed({"torch": "2.9.0+das.opt1.dtk2604"})
    out = tmp_path / "tc.txt"
    assert pin.main(["--out", str(out), "--quiet"]) == 0
    body = out.read_text(encoding="utf-8")
    assert "torch==2.9.0+das.opt1.dtk2604" in body
    assert body.lstrip().startswith("#"), "要有注释头说明这文件是干什么的"
    assert capsys.readouterr().out.strip() == str(out)


def test_prints_nothing_when_no_torch(tmp_path, fake_installed, capsys):
    """无输出是调用方判断「不加 -c」的依据，不能退化成打印空路径。"""
    fake_installed({})
    out = tmp_path / "tc.txt"
    assert pin.main(["--out", str(out), "--quiet"]) == 0
    assert capsys.readouterr().out == ""
    assert not out.exists()


def test_unwritable_out_is_not_fatal(fake_installed, capsys, tmp_path):
    fake_installed({"torch": "2.9.0+das.opt1.dtk2604"})
    unwritable = tmp_path / "no-such-dir" / "tc.txt"
    assert pin.main(["--out", str(unwritable), "--quiet"]) == 0
    assert capsys.readouterr().out == "", "写失败时不能打印路径，否则调用方会 -c 一个不存在的文件"
