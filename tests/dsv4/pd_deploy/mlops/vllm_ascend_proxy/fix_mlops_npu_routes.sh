#!/usr/bin/env bash

set -euo pipefail

target=${1:?Usage: bash fix_mlops_npu_routes.sh SERVICE_OR_POD [fix|rollback|verify]}
if [[ "$target" =~ ^(msrv-[0-9]{14}-[a-z0-9]+)(-|$) ]]; then
  service=${BASH_REMATCH[1]}
else
  echo "无法从参数提取服务 ID：$target" >&2
  exit 2
fi
mode=${2:-fix}
case "$mode" in fix|rollback|verify) ;; *) exit 2 ;; esac
state="./.${service}.npu-route-nodes"

discover_nodes() {
  kubectl get pods -n ark-service --no-headers \
    -o custom-columns=NAME:.metadata.name,NODE:.spec.nodeName |
    awk -v prefix="$service-" '
      index($1, prefix) == 1 {
        role = substr($1, length(prefix) + 1)
        sub(/-.*/, "", role)
        if (role == "decode" || role == "prefill") print role, $2
      }' | sort -u
}

find_plugin() {
  kubectl get pods -n kube-system --no-headers \
    --field-selector "spec.nodeName=$1" \
    -o custom-columns=NAME:.metadata.name |
    awk '/^npu-a-device-plugin-/ {print; exit}'
}

if [[ "$mode" == verify ]]; then
  verify_tmp=$(mktemp -d)
  trap 'rm -rf "$verify_tmp"' EXIT
  discover_nodes > "$verify_tmp/nodes"
  for role in decode prefill; do
    if ! grep -q "^$role " "$verify_tmp/nodes"; then
      echo "FAIL: 服务中未找到 $role 节点" >&2
      exit 1
    fi
  done

  # 同一个只读程序分两步运行：先收集设备 IP，再验证对端路由和连通性。
  cat > "$verify_tmp/check.sh" <<'VERIFY'
set -eu
action=$1
role=$2
node=$3
tool=/usr/local/Ascend/driver/tools/hccn_tool
hccn() {
  for attempt in 1 2 3; do
    if output=$("$tool" "$@" 2>&1); then
      printf '%s\n' "$output"
      return 0
    else rc=$?; fi
    printf '%s\n' "$output" >&2
    [ "$rc" = 11 ] || return "$rc"
    sleep 1
  done
  return "$rc"
}

if [ "$action" = inventory ]; then
  for device in $(seq 0 15); do
    info=$(hccn -i "$device" -ip -g)
    ip=$(printf '%s\n' "$info" | awk -F: '/^ipaddr:/ {print $2}')
    test -n "$ip"
    printf '%s %s %s\n' "$node" "$device" "$ip"
  done
  exit 0
fi

peers=$4
own=$5
total=0
route_ok=0
ping_ok=0
failed=0
while read -r own_node device ip; do
  table=$((100 + device))
  if ! rules=$(hccn -i "$device" -ip_rule -g) ||
     ! routes=$(hccn -i "$device" -ip_route -g table "$table") ||
     ! gateway_info=$(hccn -i "$device" -gateway -g); then
    printf 'FAIL READ %s %s NPU%s: 无法读取设备路由信息\n' "$role" "$node" "$device"
    failed=$((failed + 1))
    continue
  fi
  gateway=$(printf '%s\n' "$gateway_info" | awk -F'[:, ]+' '/^default gateway:/ {print $3}')
  nic=$(printf '%s\n' "$gateway_info" | awk -F'[:, ]+' '/^default gateway:/ {print $5}')
  rule_ok=0
  if printf '%s\n' "$rules" | awk -v ip="$ip" -v t="$table" \
    '$2=="from" && $3==ip && $4=="lookup" && $5==t {ok=1} END {exit !ok}'; then
    rule_ok=1
  fi

  # 每个对端节点验证同卡号路径；两个方向分别由各节点执行。
  while read -r peer_node peer_device peer_ip; do
    total=$((total + 1))
    prefix=$(printf '%s\n' "$peer_ip" | awk -F. '{print $1 "." $2 ".0.0/16"}')
    if [ "$rule_ok" != 1 ] || ! printf '%s\n' "$routes" |
      awk -v p="$prefix" -v g="$gateway" -v n="$nic" \
        '$1==p && $2=="via" && $3==g && $4=="dev" && $5==n {ok=1} END {exit !ok}'; then
      printf 'FAIL ROUTE %s NPU%s (%s) -> %s NPU%s (%s): table=%s 期望 %s via %s dev %s，且源规则匹配\n' \
        "$node" "$device" "$ip" "$peer_node" "$peer_device" "$peer_ip" "$table" "$prefix" "$gateway" "$nic"
      failed=$((failed + 1))
      continue
    fi
    route_ok=$((route_ok + 1))
    if ping=$(hccn -i "$device" -ping -g address "$peer_ip") &&
      printf '%s\n' "$ping" | grep -Fq ', 0.00% packet loss'; then
      ping_ok=$((ping_ok + 1))
    else
      printf 'FAIL PING %s NPU%s (%s) -> %s NPU%s (%s)\n%s\n' \
        "$node" "$device" "$ip" "$peer_node" "$peer_device" "$peer_ip" "$ping"
      failed=$((failed + 1))
    fi
  done <<EOF
$(printf '%s\n' "$peers" | awk -v d="$device" '$2==d')
EOF
done <<EOF
$own
EOF
printf 'SUMMARY %s %s: routes=%s/%s ping=%s/%s failures=%s\n' \
  "$role" "$node" "$route_ok" "$total" "$ping_ok" "$total" "$failed"
test "$failed" -eq 0
VERIFY

  pids=()
  while read -r role node; do
    plugin=$(find_plugin "$node")
    printf '%s %s %s\n' "$role" "$node" "$plugin" >> "$verify_tmp/plugins"
    kubectl exec -i -n kube-system "$plugin" -- sh -s -- inventory "$role" "$node" \
      < "$verify_tmp/check.sh" > "$verify_tmp/$role.$node" &
    pids+=("$!")
  done < "$verify_tmp/nodes"
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  if [[ "$status" != 0 ]]; then
    echo 'FAIL: NPU 地址收集失败，未执行连通性检查' >&2
    exit 1
  fi

  echo '检查全部源卡的 /16 策略路由，以及各 P/D 节点对同卡号的双向 ping。'
  pids=()
  while read -r role node plugin; do
    if [[ "$role" == decode ]]; then peer_role=prefill; else peer_role=decode; fi
    peers=$(cat "$verify_tmp/$peer_role."*)
    own=$(cat "$verify_tmp/$role.$node")
    kubectl exec -i -n kube-system "$plugin" -- sh -s -- check "$role" "$node" "$peers" "$own" \
      < "$verify_tmp/check.sh" &
    pids+=("$!")
  done < "$verify_tmp/plugins"
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  if [[ "$status" == 0 ]]; then
    echo 'PASS: 全部策略路由及所测双向设备路径通过（不代表 Mooncake KV/推理验收）。'
  else
    echo 'FAIL: 存在路由或设备连通性问题，见上述节点和卡号。'
  fi
  exit "$status"
fi

# 保存原节点，回滚不依赖业务 Pod 继续存在。
if [[ "$mode" == fix && ! -f "$state" ]]; then
  discover_nodes > "$state"
fi
test -s "$state"

while read -r role node; do
  plugin=$(find_plugin "$node")
  printf '%s %s %s\n' "$mode" "$role" "$node"

  kubectl exec -i -n kube-system "$plugin" -- sh -s -- "$mode" "$role" "$service" <<'REMOTE'
set -eu
mode=$1
role=$2
work="/tmp/mlops-routes-$3"
original=/usr/local/Ascend/driver/tools/hccn_tool
tool="$work/hccn_tool"

# 只读管理容器里使用临时工具副本和工作配置，原工具保持不变。
if [ "$mode" = fix ] && [ ! -x "$tool" ]; then
  mkdir -p "$work"
  conf=$(mktemp /tmp/mcXXXXXXX) # 与 /etc/hccn.conf 同为 14 字节
  printf '%s\n' "$conf" > "$work/config-path"
  HCCN_ROUTE_CONF="$conf" perl -0777 -pe \
    's@/etc/hccn[.]conf@$ENV{HCCN_ROUTE_CONF}@g' "$original" > "$tool"
  chmod 700 "$tool"
  touch "$work/added"
fi

# hccn_tool 偶发 busy；仅重试这一种错误，避免污染路由读取结果。
hccn() {
  for attempt in 1 2 3; do
    if output=$("$tool" "$@" 2>&1); then
      printf '%s\n' "$output"
      return 0
    else rc=$?; fi
    printf '%s\n' "$output" >&2
    [ "$rc" = 11 ] || return "$rc"
    sleep 1
  done
  return "$rc"
}

if [ "$mode" = rollback ]; then
  [ -f "$work/added" ] || exit 0
  while read -r device table target; do
    routes=$(hccn -i "$device" -ip_route -g table "$table")
    if printf '%s\n' "$routes" | awk -v p="$target/16" '$1==p {found=1} END {exit !found}'; then
      hccn -i "$device" -ip_route -d ip "$target" ip_mask 16 table "$table"
    fi
  done < "$work/added"
  rm -f "$(cat "$work/config-path")" "$tool" "$work/added" "$work/config-path"
  exit 0
fi

case "$role" in
  decode) targets='28.19.0.0 28.102.0.0' ;;
  prefill) targets='28.103.0.0' ;;
esac

for device in $(seq 0 15); do
  table=$((100 + device))
  routes=$(hccn -i "$device" -ip_route -g table "$table")
  read -r gateway nic <<EOF
$(printf '%s\n' "$routes" | awk '$2=="via" {print $3, $5; exit}')
EOF
  for target in $targets; do
    if printf '%s\n' "$routes" | awk -v p="$target/16" '$1==p {found=1} END {exit !found}'; then
      continue
    fi
    printf '%s %s %s\n' "$device" "$table" "$target" >> "$work/added"
    hccn -i "$device" -ip_route -a ip "$target" ip_mask 16 \
      via "$gateway" dev "$nic" table "$table"
    hccn -i "$device" -ip_route -g table "$table" |
      grep -F "$target/16 via $gateway dev $nic"
  done
done
REMOTE
done < "$state"

if [[ "$mode" == rollback ]]; then rm -f "$state"; fi
