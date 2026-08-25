#!/usr/bin/env python
"""列出 requirements.txt 里**当前解释器上没装**的包，一行一个（空格分隔输出）。

给 ``studio.sh --no-venv`` 用：在厂商定制栈（海光 DTK / 昇腾 / 其它国产加速卡）上，
依赖是随镜像装在系统 site-packages 里的，而且不少包**没有对应架构的 wheel**。
那种环境下 ``pip install -r requirements.txt`` 不是"补几个包"，是"就地源码编译"，
慢且经常直接失败。所以先只**报告**缺什么，由人决定怎么补（通常是重建镜像）。

判据用 ``importlib.metadata``（按**发行版名**查），不是 ``import``：
- ``pyyaml`` 的模块名是 ``yaml``、``Pillow`` 是 ``PIL``、``opencv-python`` 是 ``cv2``…
  按模块名猜会误报一堆。
- extras（``requests[socks]``）与版本约束会被剥掉，只按发行版名判在不在。

**只判"装没装"，不判版本是否满足约束。** 版本冲突交给 pip 自己在真装的时候说，
这里的目的是让用户一眼看出"镜像里缺哪几个"，而不是复刻一个 resolver。

Stdlib only。永远 exit 0 —— 这是诊断输出，不该成为启动的失败点。

用法::

    python tools/check_missing_deps.py requirements.txt
    # stdout: "fastapi uvicorn spandrel"（全都装了则无输出）
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import re
import sys

# 行首是这些的都不是包（pip 选项 / 引用其它文件）
_NOT_A_REQUIREMENT = ("-", "--")

# 从 "pkg[extra1,extra2] >= 1.2, <2 ; python_version<'3.12'" 里抠出 "pkg"
_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def parse_requirement_names(text: str) -> list[str]:
    """从 requirements.txt 文本里取出发行版名，按出现顺序、去重。"""
    names: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith(_NOT_A_REQUIREMENT):
            continue
        m = _NAME_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        key = name.lower().replace("_", "-")
        if key not in seen:
            seen.add(key)
            names.append(name)
    return names


def is_installed(dist: str) -> bool:
    try:
        md.version(dist)
        return True
    except md.PackageNotFoundError:
        return False
    except Exception:  # noqa: BLE001 —— 元数据损坏当成"装了"，别制造假警报
        return True


def missing_from(text: str) -> list[str]:
    return [name for name in parse_requirement_names(text) if not is_installed(name)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", help="requirements.txt 路径")
    args = parser.parse_args(argv)

    try:
        with open(args.requirements, encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        print(f"[check-deps] 读不到 {args.requirements}: {exc}", file=sys.stderr)
        return 0

    missing = missing_from(text)
    if missing:
        print(" ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
