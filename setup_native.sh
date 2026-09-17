#!/bin/bash
# Build the native C++ core. No sudo, no OpenCV C++ needed (dependency-free C++17).
# Output: sih_dr/netradr_core*.so  (gitignored; rebuilt per machine)
set -e
cd "$(dirname "$0")"
command -v cmake >/dev/null || { echo "need cmake"; exit 1; }
command -v g++ >/dev/null || { echo "need g++"; exit 1; }
python3 -c "import pybind11" 2>/dev/null || pip install --break-system-packages pybind11
cmake -B native/build -S native -DCMAKE_BUILD_TYPE=Release
cmake --build native/build -j"$(nproc)"
cp native/build/netradr_core*.so sih_dr/
echo "built: $(ls sih_dr/netradr_core*.so)"
NETRADR_IMPL=cxx python3 -c "
from sih_dr import _native
print('native available:', _native.available())"
