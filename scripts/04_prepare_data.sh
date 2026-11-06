#!/bin/bash
# Download the official MLPerf Llama 3.1 8B data (preprocessed C4 subset, ~84 GB) and the
# Llama 3.1 tokenizer with MLCommons' R2 downloader, then make a tokenizer-only copy.
# Login node (Internet). Downloading 84 GB took several hours on Hangang (0.7-8 MB/s):
#   setsid nohup ./scripts/04_prepare_data.sh > results/data-download.log 2>&1 < /dev/null &
# If the data already exists on /scratch (copied from another user), only check sizes:
#   ./scripts/04_prepare_data.sh --verify
set -euo pipefail
source "$(dirname "$0")/../env.sh"
VERIFY_ONLY=0; [ "${1:-}" = "--verify" ] && VERIFY_ONLY=1
DL=https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh
if [ $VERIFY_ONLY = 0 ]; then
mkdir -p "$DATA_ROOT/data" "$DATA_ROOT/tokenizer"
curl -fsSL "$DL" -o "$DATA_ROOT/mlc-r2-downloader.sh"

if [ ! -f "$PREPROC_DATA/c4-train.en_6_text_document.bin" ]; then
  ( cd "$DATA_ROOT/data" && bash "$DATA_ROOT/mlc-r2-downloader.sh" -d llama3_1_8b_preprocessed_c4_dataset \
      https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri )
fi
TOK_FULL=$DATA_ROOT/tokenizer/llama3_1_8b_tokenizer
if [ ! -f "$TOK_FULL/tokenizer.json" ]; then
  ( cd "$DATA_ROOT/tokenizer" && bash "$DATA_ROOT/mlc-r2-downloader.sh" -d llama3_1_8b_tokenizer \
      https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri )
fi
# Pretraining starts from random weights: mount only the 6 tokenizer files NVIDIA expects
mkdir -p "$TOKENIZER"
cp -p "$TOK_FULL"/{tokenizer.json,tokenizer_config.json,special_tokens_map.json,LICENSE,README.md,USE_POLICY.md} "$TOKENIZER/"

fi
echo "== size check against expected-mounts-8b.csv"
declare -A EXP=(
 ["$PREPROC_DATA/c4-train.en_6_text_document.bin"]=83976774280
 ["$PREPROC_DATA/c4-train.en_6_text_document.idx"]=912172262
 ["$PREPROC_DATA/c4-validation-91205-samples.en_text_document.bin"]=166568364
 ["$PREPROC_DATA/c4-validation-91205-samples.en_text_document.idx"]=1824142
 ["$TOKENIZER/tokenizer.json"]=9085658
 ["$TOKENIZER/tokenizer_config.json"]=50500 )
ok=1
for p in "${!EXP[@]}"; do
  s=$(stat -L -c %s "$p" 2>/dev/null || echo missing)
  if [ "$s" = "${EXP[$p]}" ]; then echo "OK       $p"; else echo "MISMATCH $p ($s != ${EXP[$p]})"; ok=0; fi
done
[ $ok = 1 ] && echo "DATA_READY" || { echo "DATA_NOT_READY"; exit 1; }
[ $VERIFY_ONLY = 1 ] && exit 0
echo "Optional: the tokenizer download also holds ~30 GB of model weights that are never used:"
echo "  rm -rf $TOK_FULL/model-0000*-of-00004.safetensors $TOK_FULL/original"
