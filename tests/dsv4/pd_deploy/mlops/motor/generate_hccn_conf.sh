#!/bin/bash
# Query this node's NPU NICs before starting Prefill or Decode.
set -eo pipefail

output=${1:-/etc/hccn.conf}
tool=${HCCN_TOOL:-/usr/local/Ascend/driver/tools/hccn_tool}
count=${HCCN_DEVICE_COUNT:-16}
temporary=$(mktemp "${output}.XXXXXX")
trap 'rm -f "$temporary"' EXIT

for ((i=0; i<count; i++)); do
  "$tool" -i "$i" -ip -g |
    sed -n -e "s/^ipaddr:/address_${i}=/p" -e "s/^netmask:/netmask_${i}=/p"
done > "$temporary"
test "$(wc -l < "$temporary")" -eq "$((count * 2))"
chmod 644 "$temporary"
mv "$temporary" "$output"
