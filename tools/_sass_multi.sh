#!/usr/bin/env bash
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:$PATH
PY=$HOME/miniconda3/envs/triton/bin/python
B=$HOME/.cache/torch_extensions/py311_cu128
S=/mnt/e/DSH_project/_tools/_sass.py
for d in k_gelu k_gemm_tc k_gemm_reg k_softmax k_reduce wmma_demo; do
  echo "================ $d ================"
  "$PY" "$S" "$B/$d/$d.so"
  echo
done
