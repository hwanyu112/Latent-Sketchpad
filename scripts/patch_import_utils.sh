#!/usr/bin/env bash
set -euo pipefail

# 1) Locate the transformers Trainer's source file (trainer.py) that Python is using
import_utils_py=$(
python - <<'PY' 2>/dev/null
import os, transformers
print(os.path.join(os.path.dirname(os.path.abspath(transformers.__file__)), "utils/import_utils.py"), end="")
PY
)

echo "→ import_utils.py: $import_utils_py"

# 2) Verify we found a trainer.py
if [[ ! -f "$import_utils_py" ]]; then
  echo "✖ Could not locate transformers' trainer.py" >&2
  exit 1
fi

# 3) Create a backup of the original trainer.py
cp "$import_utils_py" "${import_utils_py}.bak"
echo "✔ Created backup of configuration_utils.py at ${import_utils_py}.bak"
sed -i 's/if not is_torch_greater_or_equal("2.6")/if not is_torch_greater_or_equal("2.6", accept_dev=True)/' "$import_utils_py"
echo "✔ Replaced 'accept_dev=True' in import_utils.py"
echo "✔ Patched $import_utils_py"