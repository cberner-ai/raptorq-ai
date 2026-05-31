#!/usr/bin/env bash
set -euo pipefail

# Snapshot of the Rust files that existed directly under src/, plus the fuzzer
# target, when this check was introduced. The just lint stage fails if any of
# these file contents change.
protected_files=(
  "9039f92017cbed696ad5308d58e4c8704d771ca9db21f68bf83eefd2afcc35c5  src/base.rs"
  "8feafd3a9f4f221261933f626b8f97565f33e25a077b09f72171a8aec2cc25c1  src/decoder.rs"
  "c44a4a5aceaf512ae269bea69f6237137809a06866e1b7637493351c6a90cf1d  src/encoder.rs"
  "c355f6daa05650a03ffbfc5b3cca1be0753d86436aae87b9bc4670f9da36f082  src/lib.rs"
  "e8e7ac5d4505a10b42b610be8f0050978101ada81f4b0134f19ff36b31f13b1b  src/python.rs"
  "0909789afe942fe2e7fea2c6caa11560f8b67ca25cce675efb79663a6b272b00  src/symbol.rs"
  "cfbb29904b5f7e4c3c38783147c445a8db09e1ae558a8e4c1ff48b48dd4cf223  fuzz/fuzz_targets/fuzz_raptorq.rs"
)

if ! command -v shasum >/dev/null; then
  echo "Unable to find shasum, which is required for the protected Rust file check." >&2
  exit 2
fi

failures=()

for protected_file in "${protected_files[@]}"; do
  expected_hash="${protected_file%%  *}"
  path="${protected_file#*  }"

  if [[ ! -f "$path" ]]; then
    failures+=("${path} (missing)")
    continue
  fi

  actual_hash="$(shasum -a 256 "$path")"
  actual_hash="${actual_hash%% *}"

  if [[ "$actual_hash" != "$expected_hash" ]]; then
    failures+=("${path} (modified)")
  fi
done

if [[ "${#failures[@]}" -gt 0 ]]; then
  echo "Protected Rust files changed:"
  printf '  %s\n' "${failures[@]}"
  echo
  echo "These files are locked by the just lint stage and must match the protected snapshot."
  exit 1
fi

echo "Protected Rust files were not changed."
