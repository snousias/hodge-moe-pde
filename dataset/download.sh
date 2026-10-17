#!/usr/bin/env bash
# Download the preprocessed AirFRANS mesh pool from Hugging Face into dataset/processed_augmented/.
#
#   bash dataset/download.sh            # from any directory
#
# Dataset page: https://huggingface.co/datasets/snousias/hodge-moe-pde-airfrans (ODbL)
set -euo pipefail

BASE="https://huggingface.co/datasets/snousias/hodge-moe-pde-airfrans/resolve/main"
SHARDS=(train_0 train_1 train_2 train_3 test)
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

fetch() {
    if command -v wget >/dev/null 2>&1; then wget -q --show-progress -O "$2" "$1"
    else curl -fL --progress-bar -o "$2" "$1"; fi
}

fetch "$BASE/SHA256SUMS" SHA256SUMS
for f in "${SHARDS[@]}"; do
    echo "downloading $f.tar"
    fetch "$BASE/$f.tar" "$f.tar"
done

sha256sum -c SHA256SUMS
for f in "${SHARDS[@]}"; do
    tar -xf "$f.tar" && rm "$f.tar"
done
rm SHA256SUMS

n=$(find processed_augmented -maxdepth 1 -type f | wc -l)
echo "processed_augmented/: $n files (expected 6601)"
[ "$n" -eq 6601 ]
