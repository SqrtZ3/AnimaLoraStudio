#!/usr/bin/env bash
# ==============================================================================
#   AnimaLoraStudio · 国家超算互联网（scnet）海光 DCU 启动脚本
#
# 只做 scnet 环境特有的三件事，然后把活交给 studio.sh：
#
#   1. source DTK 环境。scnet 的 /etc/profile.d/ 里**没有**它，交互 shell 与
#      非登录 ssh 的 LD_LIBRARY_PATH 都是空的 → `import torch` 直接报
#      `ImportError: libgalaxyhip.so.5`，看起来像"torch 装坏了"。
#   2. 捞出内网 HTTP 代理。容器没有直连出网，代理变量**只存在于 jupyter-lab
#      进程的环境里**，sshd 不继承 → SSH 会话里 pip/git 全超时，Jupyter 终端
#      里却正常。这里从 jupyter 进程的 /proc/<pid>/environ 里读回来。
#   3. 监听 0.0.0.0 与控制台登记的内网端口。scnet 用平台域名路由到内网端口
#      （不是路径前缀代理），所以前端不需要 base/root_path 改造，改端口即可。
#
# 用法:
#   bash studio_scnet.sh                  用默认端口 $STUDIO_PORT 启动
#   STUDIO_PORT=6006 bash studio_scnet.sh 指定端口（要与控制台登记的一致）
#   bash studio_scnet.sh --check          只做环境体检，不启动
#
# 其余参数原样透传给 studio.sh。
# ==============================================================================

set -u

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DTK_ENV="${DTK_ENV:-/opt/dtk/env.sh}"
STUDIO_HOST="${STUDIO_HOST:-0.0.0.0}"
STUDIO_PORT="${STUDIO_PORT:-6006}"

CHECK_ONLY=false
PASSTHROUGH=()
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=true ;;
        *)       PASSTHROUGH+=("$arg") ;;
    esac
done

# 强制 UTF-8，避免非 UTF-8 locale 下体检输出变乱码
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8

say() { echo "[scnet] $*"; }
die() { echo "[scnet] ERROR: $*" >&2; exit 1; }

# ── 1. DTK 环境 ───────────────────────────────────────────────────────────────
if [ -f "$DTK_ENV" ]; then
    # shellcheck disable=SC1090
    . "$DTK_ENV"
    say "已 source $DTK_ENV"
else
    say "WARNING: 找不到 $DTK_ENV —— 若 import torch 报 libgalaxyhip.so 缺失，就是这里"
fi

# ── 2. 内网代理（只在用户没设过时才捞） ───────────────────────────────────────
if [ -z "${http_proxy:-}${HTTP_PROXY:-}" ]; then
    _jpid="$(pgrep -f jupyter-lab 2>/dev/null | head -1 || true)"
    if [ -n "$_jpid" ] && [ -r "/proc/$_jpid/environ" ]; then
        while IFS= read -r _line; do
            case "$_line" in
                http_proxy=*|https_proxy=*|HTTP_PROXY=*|HTTPS_PROXY=*|no_proxy=*|NO_PROXY=*)
                    export "${_line?}" ;;
            esac
        done < <(tr '\0' '\n' < "/proc/$_jpid/environ")
        say "已从 jupyter-lab(pid=$_jpid) 继承代理：${http_proxy:-${HTTP_PROXY:-未取到}}"
    else
        say "WARNING: 没找到 jupyter-lab 进程，代理未继承 —— pip/下载可能全超时"
    fi
else
    say "代理已由环境提供：${http_proxy:-$HTTP_PROXY}"
fi

# ── 3. 体检 ───────────────────────────────────────────────────────────────────
_py="${PYTHON_BIN:-python}"
command -v "$_py" >/dev/null 2>&1 || die "找不到解释器 $_py"

"$_py" - <<'PYEOF' || die "torch 体检未通过（详见上面的输出）"
import sys
try:
    import torch
except Exception as exc:
    print(f"[scnet] import torch 失败: {type(exc).__name__}: {exc}")
    print("[scnet]   多半是没 source /opt/dtk/env.sh（LD_LIBRARY_PATH 缺 /opt/dtk/lib）")
    sys.exit(1)

hip = getattr(torch.version, "hip", None)
cuda = getattr(torch.version, "cuda", None)
print(f"[scnet] torch={torch.__version__} hip={hip} cuda={cuda}")
if not hip:
    print("[scnet] ★ 这不是 HIP 构建的 torch —— 极可能是被 `pip install torch` 覆盖过。")
    print("[scnet]   修复：重建容器，或从光源重装 torch-*+das*.dtk* 轮子。")
    print("[scnet]   之后用 ./studio.sh --system-site-packages 让 venv 看得见它。")
    sys.exit(1)
if not torch.cuda.is_available():
    print("[scnet] ★ HIP 构建，但看不到卡。检查 rocm-smi / hy-smi 与 HIP_VISIBLE_DEVICES。")
    sys.exit(1)
print(f"[scnet] 设备 x{torch.cuda.device_count()}: {torch.cuda.get_device_name(0)}")
PYEOF

if [ "$CHECK_ONLY" = "true" ]; then
    say "体检通过（--check，不启动）"
    exit 0
fi

# ── 4. 起服务 ─────────────────────────────────────────────────────────────────
say "启动 Studio：http://$STUDIO_HOST:$STUDIO_PORT"
say "  端口必须与 scnet 控制台登记的内网端口一致，否则平台域名路由不过来。"
cd "$REPO_DIR" || die "cd $REPO_DIR 失败"
exec bash studio.sh --system-site-packages \
    --host "$STUDIO_HOST" --port "$STUDIO_PORT" "${PASSTHROUGH[@]:-}"
