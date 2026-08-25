#!/usr/bin/env python
"""Bootstrap helper：把**已装好的** torch 钉进一个 pip constraints 文件。

解决的问题
----------
`pip install -r requirements.txt` 里有一条 ``torch>=2.0.0``。在绝大多数机器上这没事
（已装的 torch 满足约束，pip 不动它），但在**厂商定制构建**的环境里会出人命：

- 海光 DCU：DTK 镜像自带 ``torch==2.9.0+das.opt1.dtk2604``（HIP 后端）。
- ROCm / 国产加速卡 / 自编译 torch：同类情况。

这些构建**不在 PyPI 上**。只要有任何一步让 pip 认为需要重装 torch（换了 venv、
装了个依赖 torch 的包并触发解析、有人手滑加了 ``--upgrade``），pip 就会从 PyPI
拉一个**同名但完全不同**的 wheel 覆盖上去，**而且成功退出**。之后
``torch.cuda.is_available()`` 变 False，报错全指向别处 —— 排查成本极高。

做法
----
读当前解释器里已装的 torch/torchvision/torchaudio/triton 的**精确版本**（含
``+local`` 段），写成 ``torch==2.9.0+das.opt1.dtk2604`` 这样的 constraints。
pip 拿到 ``-c`` 之后，任何想改动这些包的解析都会**直接失败并说明原因**，
而不是静默替换。

只钉**已经装了的**包 —— 没装 torchvision 就不写它，不会平白制造无法满足的约束。

Stdlib only —— venv 刚建好（只有 pip + setuptools）时也要能跑。

用法::

    python tools/pin_installed_torch.py --out venv/.torch-constraints.txt
    # stdout：写出的文件路径；没有可钉的包则无输出
    pip install -r requirements.txt -c venv/.torch-constraints.txt

永远 exit 0：这是保护措施，不该成为 bootstrap 的失败点。
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import sys

# 顺序即写入顺序。torch 打头，其余按依赖紧密度。
_PINNED = ("torch", "torchvision", "torchaudio", "triton", "pytorch-triton-rocm")


def installed_version(dist: str) -> str | None:
    """返回已装 dist 的精确版本；没装返回 None。"""
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None
    except Exception:  # noqa: BLE001 —— 元数据损坏不该让 bootstrap 挂掉
        return None


def build_constraints() -> list[str]:
    """收集需要钉住的 ``name==version`` 行。"""
    lines: list[str] = []
    for dist in _PINNED:
        version = installed_version(dist)
        if version:
            lines.append(f"{dist}=={version}")
    return lines


def is_vendor_build(version: str | None) -> bool:
    """版本号带 local 段（``+das`` / ``+rocm`` / ``+cu128`` …）即视为定制构建。

    仅用于给用户一句更明确的提示，不影响是否钉住 —— 钉住是无条件的。
    """
    return bool(version) and "+" in str(version)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="constraints 文件写到哪")
    parser.add_argument("--quiet", action="store_true", help="不往 stderr 打提示")
    args = parser.parse_args(argv)

    lines = build_constraints()
    if not lines:
        # 还没装 torch（全新 venv 首装）——没有东西要保护，让 pip 正常走。
        return 0

    header = [
        "# 由 tools/pin_installed_torch.py 自动生成，勿手改。",
        "# 作用：不让 pip 把已装好的 torch 换成 PyPI 上的同名包。",
        "# 想换 torch 请删掉本文件或直接 pip install，不要绕过 requirements.txt。",
    ]
    try:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(header + lines) + "\n")
    except OSError as exc:
        if not args.quiet:
            print(f"[pin-torch] 写 {args.out} 失败：{exc}（跳过保护）", file=sys.stderr)
        return 0

    torch_version = installed_version("torch")
    if not args.quiet and is_vendor_build(torch_version):
        print(
            f"[pin-torch] 检测到定制构建 torch=={torch_version}，已钉住不让 pip 替换。",
            file=sys.stderr,
        )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
