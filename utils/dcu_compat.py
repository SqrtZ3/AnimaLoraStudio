"""海光 DCU（K100-AI / 深算系列）兼容层 —— opt-in，默认关，CUDA 路径逐字节不变。

启用方式（二选一，等价）：
    * 环境变量 ``ANIMA_DCU=1``
    * YAML / CLI ``device_backend: dcu``

为什么这个文件这么短
--------------------
DCU 的 DTK 软件栈是 ROCm/HIP 的衍生版，海光适配版 PyTorch
（``torch==2.x+das.optN.dtkXXXX``）**本身就是 HIP 后端的 torch**：
``torch.cuda.is_available()`` / ``device_count()`` / ``get_device_name()`` /
``torch.device("cuda")`` 直接就是 DCU，``torch.version.hip`` 非空、
``torch.version.cuda`` 为空。

所以本文件**不做设备重映射** —— runtime 里既有的 ``torch.cuda.*`` 调用在 DCU 上
本来就是对的。只做三件事：

1. **确认真的在 DCU 上**（``enable()``）—— 防止"以为在国产卡上跑，其实 torch 被
   pip 覆盖成了 CUDA 构建"这种静默失败。真机踩过。
2. **可用性修复**（``configure_sdpa_backends()``）—— DTK 的 SDPA 有一条会直接抛错
   而不回落的路径，实测一次再决定关谁。
3. **能力实测与记录**（``capability_report()`` / ``guard_unsupported()``）——
   **实际跑一次再下结论，不按平台名字预判**，且只拦「缺了会训到一半才崩」的那一类。
4. **环境提示**（allocator / MIOpen 缓存）。

平台参考：DTK ≥ 24.04 才支持 K100-AI（gfx928）；镜像来自光源 sourcefind；
``rocm-smi``（DTK 侧）/ ``hy-smi``（驱动侧）看卡；``/opt/dtk`` 是 DTK 根目录。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DCU_ENABLED = False
_DCU_INFO: dict = {}

# K100-AI 的 GPU arch。DTK 的 torch 若不是为 gfx928 编的，需要 HSA_OVERRIDE_GFX_VERSION=9.2.8。
_K100AI_ARCH = "gfx928"


def dcu_requested(device_backend: str | None = None) -> bool:
    """是否请求了 DCU 后端（env 或配置字段任一命中）。"""
    if os.environ.get("ANIMA_DCU", "").strip().lower() in ("1", "true", "yes", "dcu"):
        return True
    return (device_backend or "").strip().lower() in ("dcu", "hygon", "rocm")


def is_dcu() -> bool:
    """当前进程是否已确认跑在 DCU 上。"""
    return _DCU_ENABLED


def device_str() -> str:
    """训练主设备字符串。

    DCU 上**就是** ``"cuda"`` —— HIP 后端复用了 CUDA 的设备命名空间，写 ``"hip"`` 反而报错。
    这不是"没适配"，是 DTK 的既定约定。
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _torch_flavor(torch) -> str:
    """判定当前 torch 是 HIP 构建、CUDA 构建还是 CPU-only。"""
    if getattr(torch.version, "hip", None):
        return "hip"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return "cpu"


def _arch_name(torch) -> str:
    """取设备 0 的 gcnArchName（如 ``gfx928:sramecc+:xnack-``），取不到返回空串。"""
    try:
        return str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or "")
    except Exception:
        return ""


def _safe_device_name() -> str:
    import torch

    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "unknown"


def enable() -> dict:
    """确认运行环境确实是 DCU。失败即抛错，不静默回退。

    必须在**第一次真正使用设备之前**调用。返回一份环境信息 dict 供日志/telemetry 用。
    """
    global _DCU_ENABLED, _DCU_INFO
    if _DCU_ENABLED:
        return _DCU_INFO

    # ★ 顺序要紧：allocator 配置必须在**本进程第一次碰设备之前**设好。
    # PyTorch 的 caching allocator 只在首次分配时解析一次 PYTORCH_*_ALLOC_CONF，
    # 之后再改环境变量是静默无效的（不报错，只是没生效）。下面的 _arch_name() /
    # configure_sdpa_backends() 都会初始化 CUDA/HIP 上下文并分配张量，所以放在它们前面。
    set_allocator_env()

    import torch

    flavor = _torch_flavor(torch)
    if flavor != "hip":
        raise RuntimeError(
            f"device_backend=dcu（或 ANIMA_DCU=1）要求海光适配版 PyTorch（HIP 后端），"
            f"但当前 torch {torch.__version__} 是 {flavor} 构建"
            f"（torch.version.hip={getattr(torch.version, 'hip', None)!r}）。\n"
            "最常见原因：在 DTK 镜像里执行过 `pip install torch`，把适配版覆盖成了 PyPI 的\n"
            "CUDA/CPU 构建。修复：重建容器，或从光源（sourcefind）重装 torch-*+das*.dtk* 轮子；\n"
            "**不要**用 pip 从 PyPI 装 torch/torchvision。"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch 是 HIP 构建，但 torch.cuda.is_available() 为 False —— 容器里没有可见的 DCU。\n"
            "检查：(1) `rocm-smi` / `hy-smi` 能否看到卡；(2) 环境变量 HIP_VISIBLE_DEVICES /\n"
            "ROCR_VISIBLE_DEVICES 是否把卡屏蔽了；(3) /opt/dtk 的环境变量是否 source 过\n"
            "（LD_LIBRARY_PATH 缺 /opt/dtk/lib 会表现为找不到 libhip*.so）。"
        )

    arch = _arch_name(torch)
    _DCU_ENABLED = True
    _DCU_INFO = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device_count": torch.cuda.device_count(),
        "device_name": _safe_device_name(),
        "arch": arch,
        "dtk_root": os.environ.get("DTK_HOME") or ("/opt/dtk" if os.path.isdir("/opt/dtk") else ""),
    }
    logger.info(
        "[dcu] 已确认海光 DCU：torch=%s hip=%s devices=%s (%s, arch=%s)",
        _DCU_INFO["torch"], _DCU_INFO["hip"], _DCU_INFO["device_count"],
        _DCU_INFO["device_name"], arch or "未知",
    )
    if arch and _K100AI_ARCH not in arch:
        # 不是错误 —— DCU 有 Z100/K100/K100-AI 多个型号，只是提醒别把别的型号的实测结论套过来。
        logger.info(
            "[dcu] 注意：设备 arch=%s，不是 K100-AI 的 %s。换型号需重新实测。",
            arch, _K100AI_ARCH,
        )
    _DCU_INFO["sdpa"] = configure_sdpa_backends()
    return _DCU_INFO


def configure_sdpa_backends() -> dict:
    """把**实际不可用**的 SDPA 后端关掉，让 dispatcher 落到能跑的那个。

    ★ 这不是性能调优，是可用性修复。【实测 · scnet BW(gfx936) / torch 2.9.0+das.dtk2604】::

        F.scaled_dot_product_attention(q, k, v)        # 不给 mask
        RuntimeError: No matching libraries found for flash_attn_2_cuda*.so

    DTK 的 torch 把**无 mask 的 SDPA** 派发给一个外部的 flash-attn 动态库，而基础镜像里
    根本没有那个 .so（``find / -name '*flash_attn*'`` 为空）。它不会优雅回退到 math，
    而是直接抛 RuntimeError —— 于是「给了 mask 就能跑、不给 mask 就崩」。

    ``torch.backends.cuda.enable_flash_sdp(False)`` 一行即可绕开：实测关掉之后
    同一个调用立刻通过（fwd+bwd 都对）。

    做法上**不写死"DCU 没有 flash"** —— 光源在持续补 flash-attn 的 das 版轮子，装上以后
    这条路是通的。所以这里实测一次：能跑就什么都不动，跑不通才逐个关。

    返回一份 dict 记录最终各后端的开关状态，供日志/telemetry 用。
    """
    import torch
    import torch.nn.functional as F

    backends = getattr(torch.backends, "cuda", None)
    if backends is None or not hasattr(backends, "enable_flash_sdp"):
        return {}

    def _probe() -> str:
        """跑一次无 mask 的小 SDPA，返回 "" 表示通过，否则返回错误摘要。"""
        try:
            q = torch.randn(1, 2, 64, 32, device="cuda", dtype=torch.bfloat16)
            F.scaled_dot_product_attention(q, q, q)
            return ""
        except Exception as exc:  # noqa: BLE001 —— 任何异常都算"这条路不通"
            return f"{type(exc).__name__}: {exc}"

    disabled = []
    err = _probe()
    if err and backends.flash_sdp_enabled():
        backends.enable_flash_sdp(False)
        disabled.append("flash")
        logger.warning("[dcu] 无 mask SDPA 失败（%s），已关闭 flash 后端后重试", err[:120])
        err = _probe()
    if err and backends.mem_efficient_sdp_enabled():
        backends.enable_mem_efficient_sdp(False)
        disabled.append("mem_efficient")
        logger.warning("[dcu] 仍失败（%s），已关闭 mem_efficient 后端后重试", err[:120])
        err = _probe()

    state = {
        "flash": bool(backends.flash_sdp_enabled()),
        "mem_efficient": bool(backends.mem_efficient_sdp_enabled()),
        "math": bool(backends.math_sdp_enabled()),
        "disabled_by_studio": disabled,
        "still_failing": err,
    }
    if err:
        # 到这一步还不通就别静默继续 —— 注意力是每一层都要走的路。
        raise RuntimeError(
            "DCU 上所有 SDPA 后端都跑不通，最后一次错误：" + err + "\n"
            "这台机器的注意力路径不可用，先确认 DTK 环境（source /opt/dtk/env.sh）与 torch 构建。"
        )
    if disabled:
        logger.info(
            "[dcu] SDPA 后端最终状态：flash=%s mem_efficient=%s math=%s（本次关掉了 %s）"
            " → 只剩 math 时注意力会物化 O(S^2) 分数矩阵，长序列的显存要按实测压。",
            state["flash"], state["mem_efficient"], state["math"], "/".join(disabled),
        )
    return state


def set_allocator_env() -> None:
    """DCU 侧 allocator 提示。

    变量名随 torch 版本变过三轮：新版统一成 ``PYTORCH_ALLOC_CONF``（旧的两个会打
    deprecation 警告），ROCm 后端历史上读 ``PYTORCH_HIP_ALLOC_CONF``，更早是
    ``PYTORCH_CUDA_ALLOC_CONF``。三个都设（只在用户没显式设过时），谁被认就是谁生效
    —— DTK 的 torch 认哪个没有文档，实测才知道。
    ``expandable_segments`` 在 DTK 上是否受支持随版本变化，设置失败不影响训练，
    所以这里不做断言。

    **不加 ``_DCU_ENABLED`` 守卫**：本函数由 ``enable()`` 在确认设备之前调用
    （见那里的顺序说明），此时标志位还没置上。唯一调用点就是 ``enable()``，
    而 ``enable()`` 只在 ``device_backend=dcu`` 时被调用，所以 CUDA 路径不受影响。
    只覆盖用户没有显式设过的变量。
    """
    for var in ("PYTORCH_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        if var not in os.environ:
            os.environ[var] = "expandable_segments:True"
            logger.info("[dcu] %s=expandable_segments:True", var)


def miopen_cache_hint() -> str:
    """MIOpen 内核缓存位置提示。

    DTK 迁移的已知坑：``~/.cache/miopen`` 里的旧编译缓存会让**算子结果出错**（不是变慢），
    换 DTK 版本 / 换卡型号后尤其容易撞上。这里只返回提示串，不擅自删用户的缓存
    （删除属于难以撤销的操作）。
    """
    path = os.path.expanduser(os.environ.get("MIOPEN_USER_DB_PATH", "~/.cache/miopen"))
    return (f"MIOpen 缓存目录: {path}"
            f"（换 DTK 版本/卡型号后若出现数值异常，先 rm -rf 它再复现一次）")


# ── 能力探测 ──────────────────────────────────────────────────────────────────
#
# 原则（本节存在的全部理由）：**实际跑一次再下结论，不按平台名字预判。**
#
# 反面教材是本文件的上一版：它按「xformers/flash-attn 没有 HIP 轮子」这个一般性认知
# 在构造期硬拦。这个前提是错的 —— 海光技术支持出过专用编译版（并带 triton），
# 用它打的镜像跑完过完整训练。按平台名字写死会把本来能用的东西永久拦住。
#
# 反过来「import 成功」也不等于「能跑」：DTK 上无 mask 的 SDPA 就是 import 一切正常、
# 调用时才抛缺 .so（见 configure_sdpa_backends）。所以探测一律是**功能探测**：
# 真的构造张量、真的调一次。
#
# 拦不拦的分界线：
#   * 缺了会**优雅回落**的 —— 不拦。上游 enable_xformers() / set_flash_attn_enabled()
#     本来就是 warn 后走 SDPA，DCU 上没理由比 CUDA 更严。
#   * 缺了会**训到一半才崩**的 —— 构造期拦（navit_packing 的 varlen 内核就是这类：
#     xformers 是在第一个训练步才 lazy import 的）。


def _probe_xformers_varlen() -> str:
    """真的跑一次 BlockDiagonalMask + memory_efficient_attention。

    Returns:
        "" 表示这条路通；否则返回错误摘要。
    """
    try:
        import torch
        from xformers.ops import memory_efficient_attention
        from xformers.ops.fmha import BlockDiagonalMask

        q = torch.randn(1, 8, 2, 16, device="cuda", dtype=torch.bfloat16)
        bias = BlockDiagonalMask.from_seqlens([4, 4])
        memory_efficient_attention(q, q, q, attn_bias=bias)
        return ""
    except Exception as exc:  # noqa: BLE001 —— 任何异常都算"这条路不通"
        return f"{type(exc).__name__}: {exc}"


def capability_report() -> dict:
    """把本机**实测**的可选加速路径记一份，供日志/telemetry。

    只观测、不改配置、不抛错。写进日志是为了让"为什么这台机器慢"这种问题
    有据可查，而不是靠猜平台。
    """
    report = {"xformers_varlen": _probe_xformers_varlen()}
    for name, mod in (("flash_attn", "flash_attn"), ("triton", "triton"),
                      ("bitsandbytes", "bitsandbytes")):
        try:
            __import__(mod)
            report[name] = ""
        except Exception as exc:  # noqa: BLE001
            report[name] = f"{type(exc).__name__}: {exc}"
    ok = [k for k, v in report.items() if not v]
    bad = [k for k, v in report.items() if v]
    logger.info("[dcu] 可选加速路径实测：可用=%s 不可用=%s",
                ", ".join(ok) or "无", ", ".join(bad) or "无")
    return report


def guard_unsupported(args) -> None:
    """只拦「缺了会训到一半才崩」的那一类，且判据是功能探测。

    目前只有一条：``navit_packing`` 的块对角前向走 xformers 的 varlen 内核
    （``modeling/anima/cosmos_predict2_modeling.py`` 里 lazy import），
    缺它不会回落、会在第一个训练步 ImportError。
    """
    if not _DCU_ENABLED:
        return

    if bool(getattr(args, "navit_packing", False)):
        err = _probe_xformers_varlen()
        if err:
            raise ValueError(
                "navit_packing=true 需要 xformers 的 BlockDiagonalMask varlen 内核，"
                f"但本机实测跑不通：{err}\n"
                "  这不是「DCU 不支持」的一般结论 —— 装上可用的 xformers 构建即可；"
                "海光技术支持有专用编译版。\n"
                "  暂时的替代路径：navit_packing: false，走原有 ARB 分桶。"
            )
