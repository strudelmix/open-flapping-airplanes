#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m compileall -q experiments
python3 -m unittest discover -s experiments/tests -p 'test_*.py' -v
c++ -std=c++17 -O2 -Wall -Wextra -Werror -I. experiments/tests/test_limits.cpp -o /tmp/ofa_limit_tests_$$
/tmp/ofa_limit_tests_$$
rm -f /tmp/ofa_limit_tests_$$
echo "CPU tests passed; CUDA build and B200 run are NOT tested by this script."
