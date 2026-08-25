"""device_backend=dcu 的 schema / 请求判定 / fail-fast 守卫。

不需要 DCU 硬件：``enable()`` 那条真机路径不在这里测（它要 HIP 构建的 torch），
这里锁的是**关掉时逐字节不变**与**打开时该拦的都拦得住**。
"""
import argparse

import pytest

from studio.domain import TrainingConfig
from utils import dcu_compat


# ── schema ────────────────────────────────────────────────────────────────────


def test_device_backend_defaults_to_auto():
    cfg = TrainingConfig(data_dir="d", output_dir="o")
    assert cfg.device_backend == "auto"


def test_device_backend_rejects_unknown_value():
    with pytest.raises(ValueError):
        TrainingConfig(data_dir="d", output_dir="o", device_backend="npu")


def test_device_backend_reaches_cli():
    from studio.infrastructure.argparse_bridge import build_parser

    parser = build_parser(TrainingConfig, prog="t")
    assert parser.parse_args(["--device-backend", "dcu"]).device_backend == "dcu"
    assert parser.parse_args([]).device_backend == "auto"


# ── 请求判定 ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("value", ["dcu", "DCU", "hygon", "rocm", " dcu "])
def test_dcu_requested_true(value):
    assert dcu_compat.dcu_requested(value) is True


@pytest.mark.parametrize("value", ["auto", "cuda", "", None])
def test_dcu_requested_false(value):
    assert dcu_compat.dcu_requested(value) is False


def test_dcu_requested_via_env(monkeypatch):
    monkeypatch.setenv("ANIMA_DCU", "1")
    assert dcu_compat.dcu_requested("auto") is True
    monkeypatch.setenv("ANIMA_DCU", "0")
    assert dcu_compat.dcu_requested("auto") is False


# ── 能力探测与守卫 ────────────────────────────────────────────────────────────
#
# 这一节锁的是一条政策：**不按平台名字预判，实际跑一次再下结论**，而且只拦
# 「缺了会训到一半才崩」的那一类。海光技术支持出过 xformers/flash-attn 的专用
# 编译版（并带 triton），任何「DCU 上就是没有」的硬编码都是错的。


@pytest.fixture
def dcu_on(monkeypatch):
    """假装已确认在 DCU 上（不碰真硬件）。"""
    monkeypatch.setattr(dcu_compat, "_DCU_ENABLED", True)
    yield


def _args(**kw):
    base = dict(attention_backend="none", navit_packing=False, optimizer_type="adamw")
    base.update(kw)
    return argparse.Namespace(**base)


def test_guard_is_noop_when_not_enabled():
    """未启用 DCU 时守卫必须完全不管事 —— 这是「关掉即等价」的锁。"""
    dcu_compat.guard_unsupported(_args(attention_backend="flash_attn", navit_packing=True))


def test_guard_does_not_block_attention_backends(dcu_on, monkeypatch):
    """attention_backend 缺加速库会被上游优雅回落到 SDPA，DCU 侧不再额外收紧。

    上游 enable_xformers() / set_flash_attn_enabled() 本来就是 warn 后走 SDPA；
    在 DCU 上单独抛错等于比 CUDA 更严，没有依据。
    """
    monkeypatch.setattr(dcu_compat, "_probe_xformers_varlen", lambda: "ImportError: no xformers")
    dcu_compat.guard_unsupported(_args(attention_backend="xformers"))
    dcu_compat.guard_unsupported(_args(attention_backend="flash_attn"))


def test_guard_does_not_block_8bit_optimizer(dcu_on, monkeypatch):
    """8-bit 优化器缺 bnb 由优化器自己的加载路径报错，不在这里预判。"""
    monkeypatch.setattr(dcu_compat, "_probe_xformers_varlen", lambda: "ImportError: no xformers")
    dcu_compat.guard_unsupported(_args(optimizer_type="adamw8bit"))


def test_guard_blocks_navit_packing_when_varlen_probe_fails(dcu_on, monkeypatch):
    """navit 的 varlen 内核是 lazy import，缺了会在第一个训练步才崩 —— 值得前置拦。"""
    monkeypatch.setattr(dcu_compat, "_probe_xformers_varlen",
                        lambda: "ImportError: No module named 'xformers'")
    with pytest.raises(ValueError) as exc:
        dcu_compat.guard_unsupported(_args(navit_packing=True))
    msg = exc.value.args[0]
    assert "navit_packing" in msg
    assert "ImportError" in msg, "错误里要带上实测到的真实异常，而不是一句泛泛的不支持"
    assert "不是" in msg, "要说明这不是「DCU 不支持」的一般结论"


def test_guard_allows_navit_packing_when_varlen_probe_passes(dcu_on, monkeypatch):
    """装了可用的 xformers 构建（如海光专用编译版）就该放行 —— 判据是实测不是平台名。"""
    monkeypatch.setattr(dcu_compat, "_probe_xformers_varlen", lambda: "")
    dcu_compat.guard_unsupported(_args(navit_packing=True))


def test_capability_report_observes_without_raising(dcu_on, monkeypatch):
    """能力报告只观测、不改配置、不抛错，哪怕什么都不可用。"""
    monkeypatch.setattr(dcu_compat, "_probe_xformers_varlen", lambda: "RuntimeError: nope")
    report = dcu_compat.capability_report()
    assert set(report) == {"xformers_varlen", "flash_attn", "triton", "bitsandbytes"}
    assert report["xformers_varlen"] == "RuntimeError: nope"


def test_varlen_probe_swallows_everything():
    """探测本身不能把训练带崩 —— 没有 CUDA 的机器上也只该返回错误摘要。"""
    err = dcu_compat._probe_xformers_varlen()
    assert isinstance(err, str)


# ── allocator env ─────────────────────────────────────────────────────────────


def test_set_allocator_env_does_not_override_user_value(monkeypatch):
    monkeypatch.setenv("PYTORCH_HIP_ALLOC_CONF", "garbage_collection_threshold:0.8")
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    dcu_compat.set_allocator_env()
    import os

    assert os.environ["PYTORCH_HIP_ALLOC_CONF"] == "garbage_collection_threshold:0.8"
    assert os.environ["PYTORCH_ALLOC_CONF"] == "expandable_segments:True"
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
