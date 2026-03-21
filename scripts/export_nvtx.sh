#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
RESULT_DIR="$PROJECT_ROOT/result"

SQLITE_DIR="$RESULT_DIR/sqlite"
CSV_DIR="$RESULT_DIR/csv"

mkdir -p "$SQLITE_DIR"
mkdir -p "$CSV_DIR"

shopt -s nullglob
rep_files=("$RESULT_DIR"/*.nsys-rep)

if [ ${#rep_files[@]} -eq 0 ]; then
    echo "未发现 .nsys-rep 文件于: $RESULT_DIR"
    exit 0
fi

for rep_file in "${rep_files[@]}"; do
    base_name=$(basename "$rep_file" .nsys-rep)
    
    csv_output_prefix="$CSV_DIR/$base_name"
    
    echo "------------------------------------------------"
    echo "正在处理: $base_name"

    nsys stats --report nvtx_sum \
               --format csv \
               --force-overwrite true \
               --output "$csv_output_prefix" \
               "$rep_file"
    if [ -f "$RESULT_DIR/$base_name.sqlite" ]; then
        mv "$RESULT_DIR/$base_name.sqlite" "$SQLITE_DIR/"
    fi
done

echo "------------------------------------------------"
echo "导出完成！"
echo "SQLite 文件已存入: $SQLITE_DIR"
echo "CSV 文件已存入: $CSV_DIR"