#!/bin/bash
# Khung chạy chung cho TA_GP / MA_GP / PI_GP.
# File gọi nó phải set trước: MODULE, ATTACK, CKPT_DIR, DATASETS.
#
# Thiết kế (chốt trong plans/260721-1730-closed-source-multiarch-pipeline):
#   - 1 topology = star. Lý do: star là topology DUY NHẤT có defense gain dương
#     cho PI (random/chain/tree đều âm), và thắng cả 4 tiêu chí ở MA.
#   - Không gen data / không train: tái dùng test dataset + checkpoint Llama.
#   - Mỗi (backbone, dataset) chạy trong subshell riêng -> hỏng 1 cái không kéo
#     sập phần còn lại; cuối cùng in danh sách FAILED.

# ── env ───────────────────────────────────────────────────────────────────
[ -f ../.env ] && source ../.env
PY="../.venv/bin/python"
[ -x "$PY" ] || PY="python3"

TOPO="${TOPO:-star}"
SAMPLES="${SAMPLES:-60}"
BACKBONES=(
  # Bỏ comment dòng dưới nếu muốn chạy lại baseline Llama (cần vLLM server ở $BASE_URL).
  # Để nguyên comment thì chỉ chạy 2 backbone closed-source.
  # "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
  "deepseek.v3-v1:0"
  "anthropic.claude-haiku-4-5"
)

while [ $# -gt 0 ]; do
  case "$1" in
    --smoke)      SAMPLES=2; SMOKE=1; shift ;;
    --samples)    SAMPLES="$2"; shift 2 ;;
    --topo)       TOPO="$2"; shift 2 ;;
    --backbones)  read -r -a BACKBONES <<< "$2"; shift 2 ;;
    *) echo "tham số lạ: $1"; exit 64 ;;
  esac
done

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  CSV="$(cd .. && pwd)/evaluation_results_SMOKE.csv"
else
  CSV="$(cd .. && pwd)/evaluation_results.csv"
fi

LOG_FILE="run_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================================="
echo " $MODULE | topology=$TOPO | samples=$SAMPLES | smoke=$SMOKE"
echo " CSV -> $CSV"
echo " log -> $LOG_FILE"
echo "=========================================================="

# ── preflight: mọi backbone phải gọi được TRƯỚC khi tốn tiền ──────────────
echo "=== PREFLIGHT ==="
PRE_OK=()
for BK in "${BACKBONES[@]}"; do
  if $PY llm_provider.py "$BK"; then
    PRE_OK+=("$BK")
  else
    echo "   ↑ bỏ backbone này khỏi lượt chạy"
  fi
done
if [ ${#PRE_OK[@]} -eq 0 ]; then
  echo "❌ Không backbone nào gọi được — dừng. Kiểm tra .env (AWS_BEARER_TOKEN_BEDROCK / BASE_URL)."
  exit 1
fi
BACKBONES=("${PRE_OK[@]}")
echo

# ── tìm checkpoint tự động (v1 = MyGAT, v2 = *tgat*) ──────────────────────
find_ckpt () {  # $1 = thư mục checkpoint, $2 = v1|v2
  local dir="$1" kind="$2"
  if [ "$kind" = "v2" ]; then
    ls -1t "$dir"/*tgat*.pth 2>/dev/null | head -1
  else
    ls -1t "$dir"/*.pth 2>/dev/null | grep -v tgat | head -1
  fi
}

FAILED=()

run_one () {   # $1 = backbone, $2 = dataset ("" nếu module không có nhiều dataset)
  local BK="$1" DS="$2"
  local tag="${BK##*/}${DS:+/$DS}"
  local ds_arg="" ck_dir="$CKPT_DIR" atk="$ATTACK"
  if [ -n "$DS" ]; then
    ds_arg="--dataset $DS"
    ck_dir="checkpoint/$DS"
    atk="pi_${DS}_escalation"
  fi

  local CK1 CK2
  CK1=$(find_ckpt "$ck_dir" v1)
  CK2=$(find_ckpt "$ck_dir" v2)
  [ -n "$CK1" ] && [ -n "$CK2" ] || { echo "❌ [$tag] thiếu checkpoint trong $ck_dir"; return 1; }

  (
    set -e
    echo "--- [$tag] DEFENSE v1 (MyGAT) ---"
    OUT1=$($PY main_defense_for_different_topology.py $ds_arg \
             --graph_type "$TOPO" --model_type "$BK" \
             --gnn_checkpoint_path "$CK1" --samples "$SAMPLES" | tee /dev/stderr)
    NODF1=$(echo "$OUT1" | grep '^NO_DEFENSE_FILE:' | cut -d':' -f2- | xargs)
    DF1=$(echo "$OUT1"   | grep '^DEFENSE_FILE:'    | cut -d':' -f2- | xargs)
    [ -n "$NODF1" ] && [ -n "$DF1" ] || { echo "❌ [$tag] v1 không ra file"; exit 1; }

    echo "--- [$tag] DEFENSE v2 (TemporalGAT) ---"
    OUT2=$($PY main_defense_for_different_topology_v2.py $ds_arg \
             --graph_type "$TOPO" --model_type "$BK" \
             --gnn_checkpoint_path "$CK2" --samples "$SAMPLES" | tee /dev/stderr)
    NODF2=$(echo "$OUT2" | grep '^NO_DEFENSE_FILE:' | cut -d':' -f2- | xargs)
    DF2=$(echo "$OUT2"   | grep '^DEFENSE_FILE:'    | cut -d':' -f2- | xargs)
    [ -n "$NODF2" ] && [ -n "$DF2" ] || { echo "❌ [$tag] v2 không ra file"; exit 1; }

    echo "--- [$tag] EVALUATE -> CSV ---"
    $PY evaluate_output.py --no_defense_file "$NODF1" --defense_file "$DF1" \
        --attack "$atk" --defense_model "MyGAT_v1"       --output_csv "$CSV"
    $PY evaluate_output.py --no_defense_file "$NODF2" --defense_file "$DF2" \
        --attack "$atk" --defense_model "TemporalGAT_v2" --output_csv "$CSV"
  )
}

for BK in "${BACKBONES[@]}"; do
  for DS in "${DATASETS[@]}"; do
    tag="${BK##*/}${DS:+/$DS}"
    echo "############################################################"
    echo "###  BACKBONE: $BK   ${DS:+DATASET: $DS}"
    echo "############################################################"
    if run_one "$BK" "$DS"; then
      echo "✅ [$tag] DONE"
    else
      echo "⚠️  [$tag] FAILED — bỏ qua, chạy tiếp."
      FAILED+=("$tag")
    fi
  done
done

echo "=========================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "=== $MODULE DONE — tất cả thành công ==="
else
  echo "=== $MODULE DONE — bị skip: ${FAILED[*]} ==="
fi
echo "CSV: $CSV"
echo "=========================================================="
[ ${#FAILED[@]} -eq 0 ]
