#pragma once
// Minimal contiguous image views. No OpenCV dependency.
#include <cmath>
#include <cstdint>
#include <vector>

namespace netradr {

struct ImgU8 {  // HxW(xC) contiguous
  std::vector<uint8_t> v;
  int h = 0, w = 0, c = 1;
  ImgU8() = default;
  ImgU8(int h_, int w_, int c_ = 1) : v((size_t)h_ * w_ * c_, 0), h(h_), w(w_), c(c_) {}
  uint8_t* row(int y) { return v.data() + (size_t)y * w * c; }
  const uint8_t* row(int y) const { return v.data() + (size_t)y * w * c; }
};

struct ImgF32 {
  std::vector<float> v;
  int h = 0, w = 0;
  ImgF32() = default;
  ImgF32(int h_, int w_) : v((size_t)h_ * w_, 0.0f), h(h_), w(w_) {}
  float* row(int y) { return v.data() + (size_t)y * w; }
  const float* row(int y) const { return v.data() + (size_t)y * w; }
};

inline int clampi(int x, int a, int b) { return x < a ? a : (x > b ? b : x); }

// OpenCV BORDER_DEFAULT (reflect101) index mapping
inline int ref101(int x, int n) {
  if (n <= 1) return 0;
  while (x < 0 || x >= n) x = (x < 0) ? -x : 2 * n - 2 - x;
  return x;
}

}  // namespace netradr
