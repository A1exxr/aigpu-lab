#!/usr/bin/env bash
# =============================================================================
#  aigpu-lab 环境检查
#
#  用法（在 WSL 里）：
#      bash bootstrap.sh                    # 用默认环境 triton
#      AIGPU_ENV=myenv bash bootstrap.sh    # 指定别的环境
#      bash bootstrap.sh --install          # 把缺失的包装上（默认不装）
#
#  ⚠️ 设计原则（改过一次，这里说明原因）：
#      这个脚本**不会自动创建 conda 环境**，也**不会自动安装任何包**。
#      早先的版本会 `conda create` 一个新环境，结果在一台已经有完整环境的机器上
#      白建了 7GB。所以现在的规则是：
#          「先报告缺什么，装不装由你决定。」
#      只有加 --install 才会动手，而且只装缺的那几个。
# =============================================================================
set -uo pipefail

ENV_NAME="${AIGPU_ENV:-triton}"
DO_INSTALL=0
[ "${1:-}" = "--install" ] && DO_INSTALL=1

if [ -t 1 ]; then
  B=$'\033[1;34m'; G=$'\033[1;32m'; Y=$'\033[1;33m'; R=$'\033[1;31m'; N=$'\033[0m'
else
  B=""; G=""; Y=""; R=""; N=""
fi
step() { printf '\n%s==>%s %s\n' "$B" "$N" "$*"; }
ok()   { printf '%s  [+]%s %s\n' "$G" "$N" "$*"; }
warn() { printf '%s  [!]%s %s\n' "$Y" "$N" "$*"; }
err()  { printf '%s  [x]%s %s\n' "$R" "$N" "$*"; }

printf '%s\n' "=============================================================="
printf '%s\n' " aigpu-lab 环境检查（不会自动安装任何东西）"
printf '%s\n' "=============================================================="

# ---------------------------------------------------------------- 1. GPU
step "1/4  检查 GPU"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  err "nvidia-smi 不可用。确认在 WSL2 里、Windows 侧驱动正常。"
  exit 1
fi
ok "$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ---------------------------------------------------------------- 2. 项目位置
step "2/4  项目位置"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$HERE" in
  /mnt/*)
    warn "项目在 Windows 盘上（$HERE）—— 编译 / IO 会慢几倍。"
    echo "     同步到 WSL：bash $HERE/sync.sh"
    ;;
  *)
    ok "在 WSL 原生文件系统上：$HERE"
    ;;
esac

# ---------------------------------------------------------------- 3. 找环境
step "3/4  定位 conda 环境 '$ENV_NAME'"

# 非交互 shell 下 conda 常不在 PATH（初始化写在 ~/.bashrc 里），主动找出来
if ! command -v conda >/dev/null 2>&1; then
  for _base in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
               "$HOME/mambaforge" /opt/conda /opt/miniconda3; do
    if [ -x "$_base/bin/conda" ]; then
      # shellcheck disable=SC1091
      . "$_base/etc/profile.d/conda.sh" 2>/dev/null || export PATH="$_base/bin:$PATH"
      break
    fi
  done
fi

CONDA_BASE=""
command -v conda >/dev/null 2>&1 && CONDA_BASE="$(conda info --base 2>/dev/null)"

if [ -n "$CONDA_BASE" ] && [ -x "$CONDA_BASE/envs/$ENV_NAME/bin/python" ]; then
  PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
  ok "找到：$PY"
elif [ "$ENV_NAME" = "base" ] && [ -x "$HOME/miniconda3/bin/python" ]; then
  PY="$HOME/miniconda3/bin/python"
  ok "找到 base：$PY"
else
  err "找不到环境 '$ENV_NAME'。"
  echo "     可用的环境："
  [ -n "$CONDA_BASE" ] && ls -d "$CONDA_BASE"/envs/*/ 2>/dev/null | sed 's|.*/envs/||; s|/$||; s|^|       |'
  echo "     换一个：AIGPU_ENV=<名字> bash bootstrap.sh"
  echo "     ⚠️ 这个脚本**不会**替你新建环境。"
  exit 1
fi

# ---------------------------------------------------------------- 4. 差什么包
step "4/4  检查依赖（只报告，不擅自安装）"

RESULT=$("$PY" - <<'PY'
import importlib.util as u
req = [("torch", "整个项目的基础"),
       ("triton", "W5 写 FlashAttention"),
       ("numpy", "workloads.py 用"),
       ("ninja", "load_inline 的构建后端（编译 CUDA kernel 必需）")]
miss = []
for mod, why in req:
    if u.find_spec(mod) is None:
        miss.append(mod)
        print(f"  [缺] {mod:<10} {why}")
    else:
        try:
            v = getattr(__import__(mod), "__version__", "?")
        except Exception:
            v = "?"
        print(f"  [有] {mod:<10} {v}")
has_mpl = u.find_spec("matplotlib") is not None
print(f"  [{'有' if has_mpl else '缺'}] matplotlib  画 Roofline 图 / 曲线（可选）")
print("MISSING=" + ",".join(miss))
PY
)
echo "$RESULT" | grep -v '^MISSING='
MISS_LIST=$(echo "$RESULT" | grep '^MISSING=' | cut -d= -f2)

echo
if [ -z "$MISS_LIST" ]; then
  ok "依赖齐全。"
else
  warn "缺：$MISS_LIST"
  if [ "$DO_INSTALL" = "1" ]; then
    echo "     正在安装（因为你传了 --install）..."
    # shellcheck disable=SC2086
    "$PY" -m pip install $MISS_LIST
    ok "装完了。重跑一次确认：bash bootstrap.sh"
  else
    echo "     要装的话，二选一："
    echo "       $PY -m pip install $MISS_LIST"
    echo "       bash bootstrap.sh --install"
    echo "     ⚠️ 这会改动你的 conda 环境 '$ENV_NAME'，所以默认不自动执行。"
  fi
fi

# ---------------------------------------------------------------- 5. 自检
step "额外：完整环境自检"
if [ -f check_env.py ]; then
  "$PY" check_env.py
else
  echo "     （当前目录没有 check_env.py，跳过）"
fi

cat <<EOF

==============================================================
 以后每次开始工作：

     conda activate $ENV_NAME
     cd ~/projects/aigpu-lab

 跑第一批数字：

     python gpu_watch.py -- python run_baseline.py

 在 Windows 侧改完代码要同步过来：

     bash /mnt/e/DSH_project/aigpu-lab/sync.sh
==============================================================
EOF
