#!/usr/bin/env bash
# Fetch only the public simulator dependency, into an explicitly chosen location.
set -euo pipefail
target="${1:?Usage: bash scripts/bootstrap_rpent.sh /absolute/path/to/RPent}"
repository="https://github.com/RLinf/RPent.git"
commit="3c2516e350d85dc3e2ca5a3c19b21faffaf28e3c"
[[ "$target" = /* && "$target" != / ]] || { echo "Use an absolute dependency directory" >&2; exit 2; }
if [[ -e "$target" ]]; then
  [[ -d "$target/.git" ]] || { echo "Existing target is not a checkout" >&2; exit 2; }
  [[ "$(git -C "$target" remote get-url origin)" = "$repository" ]] || { echo "Origin mismatch" >&2; exit 2; }
  [[ "$(git -C "$target" rev-parse HEAD)" = "$commit" ]] || { echo "Commit mismatch; existing checkout was not modified" >&2; exit 2; }
else
  mkdir -p "$target"
  git -C "$target" init -q
  git -C "$target" remote add origin "$repository"
  git -C "$target" fetch --depth 1 origin "$commit"
  git -C "$target" checkout --detach FETCH_HEAD
fi
echo "RPent ready: $target ($commit)"
