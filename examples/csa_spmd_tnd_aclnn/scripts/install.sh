#!/usr/bin/env bash
set -euo pipefail
package=${1:?Built operator .run package required}
destination=$(realpath -m -- "${2:?External installation directory required}")
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
case "$destination" in
  "$root"|"$root"/*) echo 'Install outside the source repository' >&2; exit 1 ;;
esac
mkdir -p -- "$destination"
bash "$package" --quiet --install-path="$destination"
printf 'Source %s/vendors/csa_spmd_tnd_pto/bin/set_env.bash before importing torch_npu.\n' "$destination"
