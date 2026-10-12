#!/usr/bin/env bash
# =============================================================================
#  把 Windows 侧的 aigpu-lab 同步到 WSL 的工作目录
#
#  用法（在 WSL 里）：
#      bash /mnt/e/DSH_project/aigpu-lab/sync.sh
#
#  为什么需要它：
#      Windows 侧（/mnt/e/DSH_project/aigpu-lab）是**源**，
#      WSL 里（~/projects/aigpu-lab）是**工作副本**。
#      在 Windows 改完代码不跑这个，WSL 里跑的还是旧代码。
#      —— 这个坑已经踩过一次：改了 specs.py 的峰值算力，
#         但 WSL 里跑出来的报告还在用旧值。
#
#  幂等，可重复运行。不会删除你在 WSL 里新增的文件（比如你写的 kernel）。
# =============================================================================
set -uo pipefail

SRC="${1:-/mnt/e/DSH_project/aigpu-lab}"
DEST="$HOME/projects/aigpu-lab"

if [ -t 1 ]; then G=$'\033[1;32m'; Y=$'\033[1;33m'; R=$'\033[1;31m'; N=$'\033[0m'
else G=""; Y=""; R=""; N=""; fi
ok()   { printf '%s[+]%s %s\n' "$G" "$N" "$*"; }
warn() { printf '%s[!]%s %s\n' "$Y" "$N" "$*"; }
err()  { printf '%s[x]%s %s\n' "$R" "$N" "$*"; }

if [ ! -d "$SRC" ]; then
  err "源目录不存在：$SRC"
  echo "    如果你已经把项目只放在 WSL 里了，就不用跑这个脚本。"
  exit 1
fi

# 安全检查：DEST 必须是正常的项目目录
if [ -z "$DEST" ] || [ "$DEST" = "$HOME" ] || [ "$DEST" = "/" ] \
   || [ "$(basename "$DEST")" != "aigpu-lab" ]; then
  err "目标路径异常（DEST=$DEST），中止。"
  exit 1
fi

if [ "$SRC" = "$DEST" ]; then
  warn "源和目标是同一个目录，无需同步。"
  exit 0
fi

mkdir -p "$DEST"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude 'results/' \
        "$SRC"/ "$DEST"/ || { err "rsync 失败"; exit 1; }
else
  ( cd "$SRC" && tar cf - --exclude='__pycache__' --exclude='*.pyc' \
      --exclude='./results' . ) | ( cd "$DEST" && tar xf - ) \
    || { err "复制失败"; exit 1; }
fi

ok "已同步 $SRC → $DEST"
echo
echo "已同步的关键文件时间戳："
for f in gpulab/specs.py kernels/bench_kernels.py run_attention_scaling.py; do
  if [ -f "$DEST/$f" ]; then
    printf '    %-32s %s\n' "$f" "$(date -r "$DEST/$f" '+%Y-%m-%d %H:%M:%S')"
  fi
done
echo
echo "（results/ 目录被排除，不会覆盖你跑出来的结果）"
