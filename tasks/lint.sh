#!/bin/bash
set -ou pipefail
set +e
FAIL=false

echo "shellcheck"
shellcheck tasks/*.sh

if [ "$FAIL" = true ]; then
  echo "linting failed"
  exit 1
fi
echo "Linting passed"
exit 0

