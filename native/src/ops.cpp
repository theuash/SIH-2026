#include "netradr/ops.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <vector>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace netradr {

// ---------------- color ----------------
void split_green(const ImgU8& rgb, ImgU8& g) {
  g = ImgU8(rgb.h, rgb.w);
  for (int y = 0; y < rgb.h; y++) {
    const uint8_t* s = rgb.row(y);
    uint8_t* d = g.row(y);
    for (int x = 0; x < rgb.w; x++) d[x] = s[3 * x + 1];
  }
}

void max_channel(const ImgU8& rgb, ImgU8& v) {
  v = ImgU8(rgb.h, rgb.w);
  for (int y = 0; y < rgb.h; y++) {
    const uint8_t* s = rgb.row(y);
    uint8_t* d = v.row(y);
    for (int x = 0; x < rgb.w; x++)
      d[x] = std::max(s[3 * x], std::max(s[3 * x + 1], s[3 * x + 2]));
  }
}

static inline double srgb_lin(double c) {
  c /= 255.0;
  return c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
}
static inline double lab_f(double t) {
  const double e = 216.0 / 24389.0, k = 24389.0 / 27.0;
  return t > e ? std::cbrt(t) : (k * t + 16.0) / 116.0;
}

static double lin_lut[256];
static double labf_lut[4097];
static double finv_lut[4097];
static uint8_t enc_lut[4097];
static bool lab_lut_init = false;
static void lab_lut_ensure() {
  if (lab_lut_init) return;
  for (int i = 0; i < 256; i++) {
    double c = i / 255.0;
    lin_lut[i] = c <= 0.04045 ? c / 12.92 : std::pow((c + 0.055) / 1.055, 2.4);
  }
  const double e = 216.0 / 24389.0, k = 24389.0 / 27.0;
  for (int i = 0; i <= 4096; i++) {
    double t = i / 4096.0 * 1.2;
    labf_lut[i] = t > e ? std::cbrt(t) : (k * t + 16.0) / 116.0;
    double t3 = t * t * t;
    finv_lut[i] = t3 > e ? t3 : (116 * t - 16) / k;
  }
  for (int i = 0; i <= 4096; i++) {  // linear-domain encode table
    double c = i / 4096.0 * 1.5;
    double s = c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1 / 2.4) - 0.055;
    enc_lut[i] = (uint8_t)clampi((int)std::lround((s < 0 ? 0 : (s > 1 ? 1 : s)) * 255), 0, 255);
  }
  lab_lut_init = true;
}
static inline double lab_f_fast(double t) {
  return labf_lut[clampi((int)(t / 1.2 * 4096 + 0.5), 0, 4096)];
}
static inline double lab_finv_fast(double t) {
  return finv_lut[clampi((int)(t / 1.2 * 4096 + 0.5), 0, 4096)];
}
static inline uint8_t srgb_enc_fast(double c) {
  return enc_lut[clampi((int)(c / 1.5 * 4096 + 0.5), 0, 4096)];
}

void rgb_to_lab8(const ImgU8& rgb, ImgU8& L, ImgU8& A, ImgU8& B) {
  lab_lut_ensure();
  L = ImgU8(rgb.h, rgb.w);
  A = ImgU8(rgb.h, rgb.w);
  B = ImgU8(rgb.h, rgb.w);
  for (int y = 0; y < rgb.h; y++) {
    const uint8_t* s = rgb.row(y);
    uint8_t *l = L.row(y), *a = A.row(y), *b = B.row(y);
    for (int x = 0; x < rgb.w; x++) {
      double r = lin_lut[s[3 * x]], g = lin_lut[s[3 * x + 1]], bl = lin_lut[s[3 * x + 2]];
      double X = (0.4124 * r + 0.3576 * g + 0.1805 * bl) / 0.95047;
      double Y = 0.2126 * r + 0.7152 * g + 0.0722 * bl;
      double Z = (0.0193 * r + 0.1192 * g + 0.9505 * bl) / 1.08883;
      double fx = lab_f_fast(X), fy = lab_f_fast(Y), fz = lab_f_fast(Z);
      l[x] = (uint8_t)clampi((int)std::lround((116 * fy - 16) * 255 / 100), 0, 255);
      a[x] = (uint8_t)clampi((int)std::lround(500 * (fx - fy) + 128), 0, 255);
      b[x] = (uint8_t)clampi((int)std::lround(200 * (fy - fz) + 128), 0, 255);
    }
  }
}

static inline double lab_finv(double t) {
  const double e = 216.0 / 24389.0, k = 24389.0 / 27.0;
  double t3 = t * t * t;
  return t3 > e ? t3 : (116 * t - 16) / k;
}
static inline uint8_t srgb_enc(double c) {
  c = c < 0 ? 0 : (c > 1 ? 1 : c);
  double s = c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1 / 2.4) - 0.055;
  return (uint8_t)clampi((int)std::lround(s * 255), 0, 255);
}

void lab_to_rgb8(const ImgU8& L, const ImgU8& A, const ImgU8& B, ImgU8& rgb) {
  lab_lut_ensure();
  rgb = ImgU8(L.h, L.w, 3);
  for (int y = 0; y < L.h; y++) {
    const uint8_t *l = L.row(y), *a = A.row(y), *b = B.row(y);
    uint8_t* d = rgb.row(y);
    for (int x = 0; x < L.w; x++) {
      double fy = (l[x] * 100.0 / 255.0 + 16.0) / 116.0;
      double fx = fy + (a[x] - 128) / 500.0;
      double fz = fy - (b[x] - 128) / 200.0;
      double X = 0.95047 * lab_finv_fast(fx), Y = lab_finv_fast(fy), Z = 1.08883 * lab_finv_fast(fz);
      d[3 * x] = srgb_enc_fast(3.2406 * X - 1.5372 * Y - 0.4986 * Z);
      d[3 * x + 1] = srgb_enc_fast(-0.9689 * X + 1.8758 * Y + 0.0415 * Z);
      d[3 * x + 2] = srgb_enc_fast(0.0557 * X - 0.2040 * Y + 1.0570 * Z);
    }
  }
}

// ---------------- stats ----------------
double laplacian_var(const ImgU8& g) {
  double sum = 0, sum2 = 0;
  long n = 0;
  for (int y = 1; y + 1 < g.h; y++) {
    const uint8_t* p = g.row(y - 1);
    const uint8_t* c = g.row(y);
    const uint8_t* nx = g.row(y + 1);
    for (int x = 1; x + 1 < g.w; x++) {
      double v = p[x] + c[x - 1] - 4.0 * c[x] + c[x + 1] + nx[x];
      sum += v;
      sum2 += v * v;
      n++;
    }
  }
  double m = sum / n;
  return sum2 / n - m * m;
}

void histogram(const ImgU8& g, uint64_t hist[256]) {
  for (int i = 0; i < 256; i++) hist[i] = 0;
  for (int y = 0; y < g.h; y++) {
    const uint8_t* r = g.row(y);
    for (int x = 0; x < g.w; x++) hist[r[x]]++;
  }
}

int otsu_of_hist(const uint64_t hist[256]) {
  double total = 0, sum = 0;
  for (int i = 0; i < 256; i++) {
    total += (double)hist[i];
    sum += (double)i * hist[i];
  }
  double sumB = 0, wB = 0, best = -1;
  int thr = 0;
  for (int i = 0; i < 256; i++) {
    wB += hist[i];
    if (!wB) continue;
    double wF = total - wB;
    if (!wF) break;
    sumB += (double)i * hist[i];
    double mB = sumB / wB, mF = (sum - sumB) / wF;
    double between = wB * wF * (mB - mF) * (mB - mF);
    if (between > best) {
      best = between;
      thr = i;
    }
  }
  return thr;
}

double percentile_u8(const ImgU8& g, double p) {
  std::vector<uint8_t> v;
  v.reserve((size_t)g.h * g.w);
  for (int y = 0; y < g.h; y++) {
    const uint8_t* r = g.row(y);
    v.insert(v.end(), r, r + g.w);
  }
  size_t k = (size_t)(p / 100.0 * (v.size() - 1));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

double percentile_masked(const ImgU8& g, const ImgU8& m, double p) {
  std::vector<uint8_t> v;
  v.reserve(1 << 16);
  for (int y = 0; y < g.h; y++) {
    const uint8_t* r = g.row(y);
    const uint8_t* mk = m.row(y);
    for (int x = 0; x < g.w; x++)
      if (mk[x]) v.push_back(r[x]);
  }
  if (v.empty()) return 0;
  size_t k = (size_t)(p / 100.0 * (v.size() - 1));
  std::nth_element(v.begin(), v.begin() + k, v.end());
  return v[k];
}

double median_masked(const ImgU8& g, const ImgU8& m) { return percentile_masked(g, m, 50.0); }

// ---------------- CLAHE ----------------
void clahe(ImgU8& l, double clip, int tiles) {
  int h = l.h, w = l.w;
  int tw = (w + tiles - 1) / tiles, th = (h + tiles - 1) / tiles;
  // LUT per tile
  std::vector<std::vector<uint8_t>> luts(tiles * tiles, std::vector<uint8_t>(256));
  for (int ty = 0; ty < tiles; ty++)
    for (int tx = 0; tx < tiles; tx++) {
      int x0 = tx * tw, y0 = ty * th;
      int x1 = std::min(x0 + tw, w), y1 = std::min(y0 + th, h);
      uint64_t hist[256] = {0};
      for (int y = y0; y < y1; y++) {
        const uint8_t* r = l.row(y);
        for (int x = x0; x < x1; x++) hist[r[x]]++;
      }
      int npix = (x1 - x0) * (y1 - y0);
      int limit = std::max(1, (int)(clip * npix / 256));
      int excess = 0;
      for (int i = 0; i < 256; i++)
        if (hist[i] > (uint64_t)limit) {
          excess += (int)(hist[i] - limit);
          hist[i] = limit;
        }
      int add = excess / 256, rem = excess % 256;
      for (int i = 0; i < 256; i++) {
        hist[i] += add + (i < rem ? 1 : 0);
      }
      uint64_t acc = 0;
      auto& lut = luts[ty * tiles + tx];
      for (int i = 0; i < 256; i++) {
        acc += hist[i];
        lut[i] = (uint8_t)clampi((int)((double)acc / npix * 255), 0, 255);
      }
    }
  ImgU8 out(h, w);
  for (int y = 0; y < h; y++) {
    const uint8_t* s = l.row(y);
    uint8_t* d = out.row(y);
    double fy = (double)y / th - 0.5;
    int ty0 = clampi((int)std::floor(fy), 0, tiles - 2);
    double dy = clampi(fy - ty0, 0.0, 1.0);
    for (int x = 0; x < w; x++) {
      double fx = (double)x / tw - 0.5;
      int tx0 = clampi((int)std::floor(fx), 0, tiles - 2);
      double dx = clampi(fx - tx0, 0.0, 1.0);
      int v = s[x];
      double top = luts[ty0 * tiles + tx0][v] * (1 - dx) + luts[ty0 * tiles + tx0 + 1][v] * dx;
      double bot = luts[(ty0 + 1) * tiles + tx0][v] * (1 - dx) + luts[(ty0 + 1) * tiles + tx0 + 1][v] * dx;
      d[x] = (uint8_t)(top * (1 - dy) + bot * dy + 0.5);
    }
  }
  l.v.swap(out.v);
}

// ---------------- gaussian / box / bilateral ----------------
static std::vector<double> gauss_kernel(double sigma, int& r) {
  r = std::max(1, (int)std::ceil(3 * sigma));
  std::vector<double> k(2 * r + 1);
  double s = 0;
  for (int i = -r; i <= r; i++) {
    k[i + r] = std::exp(-0.5 * i * i / (sigma * sigma));
    s += k[i + r];
  }
  for (auto& v : k) v /= s;
  return k;
}

void gauss_blur(const ImgU8& src, ImgU8& dst, double sigma) {
  int r;
  auto k = gauss_kernel(sigma, r);
  int h = src.h, w = src.w, c = src.c;
  std::vector<double> tmp((size_t)h * w * c);
  for (int y = 0; y < h; y++) {
    const uint8_t* s = src.row(y);
    for (int x = 0; x < w; x++)
      for (int ch = 0; ch < c; ch++) {
        double a = 0;
        for (int i = -r; i <= r; i++) a += k[i + r] * s[ref101(x + i, w) * c + ch];
        tmp[((size_t)y * w + x) * c + ch] = a;
      }
  }
  dst = ImgU8(h, w, c);
  for (int y = 0; y < h; y++) {
    uint8_t* d = dst.row(y);
    for (int x = 0; x < w; x++)
      for (int ch = 0; ch < c; ch++) {
        double a = 0;
        for (int i = -r; i <= r; i++) a += k[i + r] * tmp[((size_t)ref101(y + i, h) * w + x) * c + ch];
        d[x * c + ch] = (uint8_t)clampi((int)(a + 0.5), 0, 255);
      }
  }
}

void gauss_blur_f32(const ImgF32& src, ImgF32& dst, double sigma) {
  int r;
  auto k = gauss_kernel(sigma, r);
  int h = src.h, w = src.w;
  std::vector<double> tmp((size_t)h * w);
  for (int y = 0; y < h; y++) {
    const float* s = src.row(y);
    for (int x = 0; x < w; x++) {
      double a = 0;
      for (int i = -r; i <= r; i++) a += k[i + r] * s[ref101(x + i, w)];
      tmp[(size_t)y * w + x] = a;
    }
  }
  dst = ImgF32(h, w);
  for (int y = 0; y < h; y++) {
    float* d = dst.row(y);
    for (int x = 0; x < w; x++) {
      double a = 0;
      for (int i = -r; i <= r; i++) a += k[i + r] * tmp[(size_t)ref101(y + i, h) * w + x];
      d[x] = (float)a;
    }
  }
}

// one box pass, reflect101 borders, radius r
static void box_pass_u8(const ImgU8& src, ImgU8& dst, int r, bool horiz) {
  int h = src.h, w = src.w, c = src.c;
  dst = ImgU8(h, w, c);
  int win = 2 * r + 1;
  if (horiz) {
    for (int y = 0; y < h; y++) {
      const uint8_t* s = src.row(y);
      uint8_t* d = dst.row(y);
      for (int ch = 0; ch < c; ch++) {
        long acc = 0;
        for (int i = -r; i <= r; i++) acc += s[ref101(i, w) * c + ch];
        for (int x = 0; x < w; x++) {
          d[x * c + ch] = (uint8_t)(acc / win);
          acc += s[ref101(x + r + 1, w) * c + ch] - s[ref101(x - r, w) * c + ch];
        }
      }
    }
  } else {
    for (int x = 0; x < w; x++)
      for (int ch = 0; ch < c; ch++) {
        long acc = 0;
        for (int i = -r; i <= r; i++) acc += src.row(ref101(i, h))[x * c + ch];
        for (int y = 0; y < h; y++) {
          dst.row(y)[x * c + ch] = (uint8_t)(acc / win);
          acc += src.row(ref101(y + r + 1, h))[x * c + ch] - src.row(ref101(y - r, h))[x * c + ch];
        }
      }
  }
}

void box_background(const ImgU8& src, ImgU8& dst, int radius) {
  ImgU8 a, b;
  box_pass_u8(src, a, radius, true);
  box_pass_u8(a, b, radius, false);
  box_pass_u8(b, a, radius, true);
  box_pass_u8(a, b, radius, false);
  box_pass_u8(b, a, radius, true);
  box_pass_u8(a, dst, radius, false);
}

void bilateral3_rgb(const ImgU8& src, ImgU8& dst, double sC, double sS) {
  static double lut[512];
  static bool init = false;
  if (!init) {
    for (int i = 0; i < 512; i++) lut[i] = std::exp(-0.5 * i * i / (sC * sC));
    init = true;
  }
  double sp[3][3];
  for (int j = -1; j <= 1; j++)
    for (int i = -1; i <= 1; i++) sp[j + 1][i + 1] = std::exp(-0.5 * (i * i + j * j) / (sS * sS));
  int h = src.h, w = src.w;
  dst = ImgU8(h, w, 3);
  for (int y = 0; y < h; y++) {
    uint8_t* d = dst.row(y);
    for (int x = 0; x < w; x++) {
      const uint8_t* c0 = src.row(y) + 3 * x;
      double ws = 0, acc[3] = {0, 0, 0};
      for (int j = -1; j <= 1; j++) {
        const uint8_t* r = src.row(ref101(y + j, h));
        for (int i = -1; i <= 1; i++) {
          const uint8_t* p = r + 3 * ref101(x + i, w);
          double wc = lut[std::abs(p[0] - c0[0])] * lut[std::abs(p[1] - c0[1])] * lut[std::abs(p[2] - c0[2])];
          double wgt = wc * sp[j + 1][i + 1];
          ws += wgt;
          acc[0] += wgt * p[0];
          acc[1] += wgt * p[1];
          acc[2] += wgt * p[2];
        }
      }
      d[3 * x] = (uint8_t)(acc[0] / ws + 0.5);
      d[3 * x + 1] = (uint8_t)(acc[1] / ws + 0.5);
      d[3 * x + 2] = (uint8_t)(acc[2] / ws + 0.5);
    }
  }
}

// ---------------- morphology: deque sliding min/max, rect kernels ----------------
static void slide_1d_scalar(const uint8_t* s, uint8_t* d, int n, int k, bool is_min) {
  // Block Van Herk: O(N) rect min/max, replicate borders via padded copy.
  // Scratch reused across calls (thread-local) — no per-row allocation.
  int r = k / 2;
  int m = n + 2 * r, wsz = 2 * r + 1;
  thread_local std::vector<uint8_t> buf;
  if ((int)buf.size() < 3 * m) buf.resize(3 * m);
  uint8_t *p = buf.data(), *fwd = p + m, *bwd = fwd + m;
  for (int i = 0; i < m; i++) {
    int x = i - r;
    p[i] = s[x < 0 ? 0 : (x >= n ? n - 1 : x)];
  }
  for (int i = 0; i < m; i++) {
    if (i % wsz == 0)
      fwd[i] = p[i];
    else
      fwd[i] = is_min ? (p[i] < fwd[i - 1] ? p[i] : fwd[i - 1])
                      : (p[i] > fwd[i - 1] ? p[i] : fwd[i - 1]);
  }
  for (int i = m - 1; i >= 0; i--) {
    if ((i + 1) % wsz == 0)
      bwd[i] = p[i];
    else
      bwd[i] = is_min ? (p[i] < bwd[i + 1] ? p[i] : bwd[i + 1])
                      : (p[i] > bwd[i + 1] ? p[i] : bwd[i + 1]);
  }
  for (int x = 0; x < n; x++) {
    uint8_t a = bwd[x], b = fwd[x + 2 * r];
    d[x] = is_min ? (a < b ? a : b) : (a > b ? a : b);
  }
}
#ifdef __AVX2__
// Brute-force windowed min/max, 32 px per iteration. k vector loads per
// block — beats scans for k<=41 and is branch-free.
static void slide_1d_avx2(const uint8_t* s, uint8_t* d, int n, int k, bool is_min) {
  int r = k / 2;
  thread_local std::vector<uint8_t> pad;
  if ((int)pad.size() < n + 2 * r) pad.resize(n + 2 * r);
  uint8_t* p = pad.data();
  for (int i = 0; i < n + 2 * r; i++) {
    int x = i - r;
    p[i] = s[x < 0 ? 0 : (x >= n ? n - 1 : x)];
  }
  int x = 0;
  for (; x + 32 <= n; x += 32) {
    __m256i acc = _mm256_loadu_si256((const __m256i*)(p + x));
    for (int j = 1; j < k; j++) {
      __m256i v = _mm256_loadu_si256((const __m256i*)(p + x + j));
      acc = is_min ? _mm256_min_epu8(acc, v) : _mm256_max_epu8(acc, v);
    }
    _mm256_storeu_si256((__m256i*)(d + x), acc);
  }
  // scalar tail (also covers n < 32)
  for (; x < n; x++) {
    uint8_t best = p[x];
    for (int j = 1; j < k; j++)
      best = is_min ? (p[x + j] < best ? p[x + j] : best)
                    : (p[x + j] > best ? p[x + j] : best);
    d[x] = best;
  }
}
#endif

static void slide_1d(const uint8_t* s, uint8_t* d, int n, int k, bool is_min) {
#ifdef __AVX2__
  if (n >= 64 && k <= 64) {
    slide_1d_avx2(s, d, n, k, is_min);
    return;
  }
#endif
  slide_1d_scalar(s, d, n, k, is_min);
}

static void morph_rect(const ImgU8& src, ImgU8& dst, int k, bool is_min) {
  int h = src.h, w = src.w;
  ImgU8 tmp(h, w);
  std::vector<uint8_t> line(w), oline(w);
  for (int y = 0; y < h; y++) {
    slide_1d(src.row(y), tmp.row(y), w, k, is_min);
  }
  std::vector<uint8_t> col(h), ocol(h);
  dst = ImgU8(h, w);
  for (int x = 0; x < w; x++) {
    for (int y = 0; y < h; y++) col[y] = tmp.row(y)[x];
    slide_1d(col.data(), ocol.data(), h, k, is_min);
    for (int y = 0; y < h; y++) dst.row(y)[x] = ocol[y];
  }
}

void erode_rect(const ImgU8& src, ImgU8& dst, int k) { morph_rect(src, dst, k, true); }
void dilate_rect(const ImgU8& src, ImgU8& dst, int k) { morph_rect(src, dst, k, false); }

void morph_close(const ImgU8& src, ImgU8& dst, int k) {
  ImgU8 t;
  dilate_rect(src, t, k);
  erode_rect(t, dst, k);
}

void tophat_gray(const ImgU8& src, ImgU8& dst, int k) {
  ImgU8 opened, tmp;
  erode_rect(src, tmp, k);
  dilate_rect(tmp, opened, k);
  dst = ImgU8(src.h, src.w);
  for (int y = 0; y < src.h; y++) {
    const uint8_t* s = src.row(y);
    const uint8_t* o = opened.row(y);
    uint8_t* d = dst.row(y);
    for (int x = 0; x < src.w; x++) d[x] = s[x] > o[x] ? (uint8_t)(s[x] - o[x]) : 0;
  }
}

void thresh_otsu(const ImgU8& src, ImgU8& dst) {
  uint64_t hist[256];
  histogram(src, hist);
  int t = otsu_of_hist(hist);
  dst = ImgU8(src.h, src.w);
  for (int y = 0; y < src.h; y++) {
    const uint8_t* s = src.row(y);
    uint8_t* d = dst.row(y);
    for (int x = 0; x < src.w; x++) d[x] = s[x] > t ? 1 : 0;
  }
}

// ---------------- connected components (two-pass + union-find) ----------------
static int uf_find(std::vector<int>& p, int x) {
  while (p[x] != x) {
    p[x] = p[p[x]];
    x = p[x];
  }
  return x;
}

std::vector<Comp> cc_stats(const ImgU8& mask, std::vector<int>& labels) {
  int h = mask.h, w = mask.w;
  labels.assign((size_t)h * w, 0);
  std::vector<int> parent = {0};
  int next = 1;
  for (int y = 0; y < h; y++) {
    const uint8_t* r = mask.row(y);
    for (int x = 0; x < w; x++) {
      if (!r[x]) continue;
      int a = (x > 0) ? labels[(size_t)y * w + x - 1] : 0;
      int b = (y > 0) ? labels[(size_t)(y - 1) * w + x] : 0;
      if (!a && !b) {
        parent.push_back(next);
        labels[(size_t)y * w + x] = next++;
      } else if (a && !b) {
        labels[(size_t)y * w + x] = a;
      } else if (!a && b) {
        labels[(size_t)y * w + x] = b;
      } else {
        labels[(size_t)y * w + x] = a;
        int ra = uf_find(parent, a), rb = uf_find(parent, b);
        if (ra != rb) parent[rb] = ra;
      }
    }
  }
  std::vector<int> remap(parent.size(), 0);
  std::vector<Comp> comps(1);  // index 0 = background
  for (size_t i = 0; i < (size_t)h * w; i++) {
    int l = labels[i];
    if (!l) continue;
    int root = uf_find(parent, l);
    if (!remap[root]) {
      remap[root] = (int)comps.size();
      comps.push_back(Comp());
      comps.back().x0 = w;
      comps.back().y0 = h;
    }
    int id = remap[root];
    labels[i] = id;
    Comp& cp = comps[id];
    int x = (int)(i % w), y = (int)(i / w);
    cp.area++;
    cp.cx += x;
    cp.cy += y;
    if (x < cp.x0) cp.x0 = x;
    if (x > cp.x1) cp.x1 = x;
    if (y < cp.y0) cp.y0 = y;
    if (y > cp.y1) cp.y1 = y;
  }
  for (size_t i = 1; i < comps.size(); i++) {
    comps[i].cx /= comps[i].area;
    comps[i].cy /= comps[i].area;
  }
  return comps;
}

void keep_area_range(ImgU8& mask, int min_area, int max_area) {
  std::vector<int> labels;
  auto comps = cc_stats(mask, labels);
  std::vector<char> keep(comps.size(), 0);
  for (size_t i = 1; i < comps.size(); i++)
    keep[i] = comps[i].area >= min_area && (max_area < 0 || comps[i].area < max_area);
  for (int y = 0; y < mask.h; y++) {
    uint8_t* r = mask.row(y);
    for (int x = 0; x < mask.w; x++)
      if (r[x]) {
        int id = labels[(size_t)y * mask.w + x];
        r[x] = (id > 0 && keep[id]) ? 1 : 0;
      }
  }
}

void draw_filled_circle(ImgU8& mask, int cx, int cy, int r, uint8_t v) {
  for (int y = std::max(0, cy - r); y <= std::min(mask.h - 1, cy + r); y++) {
    uint8_t* row = mask.row(y);
    int dx = (int)std::sqrt((double)(r * r - (y - cy) * (y - cy)));
    for (int x = std::max(0, cx - dx); x <= std::min(mask.w - 1, cx + dx); x++) row[x] = v;
  }
}

// ---------------- geometry / candidates ----------------
Circle ritter_circle(const std::vector<int>& labels, int w, int h, int id) {
  // bbox extremes for seed pair
  int x0 = w, y0 = h, x1 = 0, y1 = 0;
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      if (labels[(size_t)y * w + x] == id) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
  auto dist2 = [](double ax, double ay, double bx, double by) {
    return (ax - bx) * (ax - bx) + (ay - by) * (ay - by);
  };
  // farthest from (x0,y0) then farthest from that
  double ax = x0, ay = y0, bx = x0, by = y0, best = -1;
  for (int y = y0; y <= y1; y++)
    for (int x = x0; x <= x1; x++)
      if (labels[(size_t)y * w + x] == id) {
        double d = dist2(x, y, ax, ay);
        if (d > best) {
          best = d;
          bx = x;
          by = y;
        }
      }
  ax = bx;
  ay = by;
  best = -1;
  double cx = ax, cy = ay;
  for (int y = y0; y <= y1; y++)
    for (int x = x0; x <= x1; x++)
      if (labels[(size_t)y * w + x] == id) {
        double d = dist2(x, y, ax, ay);
        if (d > best) {
          best = d;
          bx = x;
          by = y;
        }
      }
  cx = (ax + bx) / 2;
  cy = (ay + by) / 2;
  double r = std::sqrt(dist2(ax, ay, bx, by)) / 2;
  for (int y = y0; y <= y1; y++)
    for (int x = x0; x <= x1; x++)
      if (labels[(size_t)y * w + x] == id) {
        double d = std::sqrt(dist2(x, y, cx, cy));
        if (d > r) {
          double nr = (r + d) / 2, k = (d - r) / (2 * d + 1e-12);
          cx += (x - cx) * k;
          cy += (y - cy) * k;
          r = nr;
        }
      }
  return {cx, cy, r};
}

std::vector<Cand> weighted_centroids(const ImgU8& mask, const ImgU8& gray, double bg) {
  std::vector<int> labels;
  auto comps = cc_stats(mask, labels);
  std::vector<Cand> out;
  for (size_t id = 1; id < comps.size(); id++) {
    const Comp& cp = comps[id];
    double sw = 0, sx = 0, sy = 0, sg = 0;
    for (int y = cp.y0; y <= cp.y1; y++) {
      const uint8_t* gr = gray.row(y);
      for (int x = cp.x0; x <= cp.x1; x++)
        if (labels[(size_t)y * mask.w + x] == (int)id) {
          double wgt = 255.0 - gr[x] + 1.0;
          sw += wgt;
          sx += x * wgt;
          sy += y * wgt;
          sg += gr[x];
        }
    }
    double mean = sg / cp.area;
    double conf = (bg - mean) / 60.0;
    conf = conf < 0 ? 0 : (conf > 1 ? 1 : conf);
    out.push_back({cp.cx, cp.cy, sx / sw, sy / sw, (double)cp.area, conf});
  }
  return out;
}

// ---------------- explain ----------------
void add_gauss_splat(const ImgU8& mask, ImgF32& acc, double sigma) {
  ImgF32 tmp(mask.h, mask.w);
  for (int y = 0; y < mask.h; y++) {
    const uint8_t* r = mask.row(y);
    float* d = tmp.row(y);
    for (int x = 0; x < mask.w; x++) d[x] = r[x] ? 1.0f : 0.0f;
  }
  ImgF32 bl;
  gauss_blur_f32(tmp, bl, sigma);
  float mx = 1e-9f;
  for (auto v : bl.v)
    if (v > mx) mx = v;
  for (int y = 0; y < acc.h; y++) {
    float* d = acc.row(y);
    const float* s = bl.row(y);
    for (int x = 0; x < acc.w; x++) {
      float v = s[x] / mx;
      if (v > d[x]) d[x] = v;
    }
  }
}

void upsample_bilinear_f32(const ImgF32& src, ImgF32& dst /*pre-sized*/) {
  int sh = src.h, sw = src.w, dh = dst.h, dw = dst.w;
  if (sh == dh && sw == dw) {
    dst.v = src.v;
    return;
  }
  double sy = (double)sh / dh, sx = (double)sw / dw;
  for (int y = 0; y < dh; y++) {
    float* d = dst.row(y);
    double fyy = (y + 0.5) * sy - 0.5;
    int y0 = clampi((int)fyy, 0, sh - 2);
    double dy = clampi(fyy - y0, 0.0, 1.0);
    if (sh == 1) {
      y0 = 0;
      dy = 0;
    }
    const float* r0 = src.row(y0);
    const float* r1 = src.row(std::min(y0 + 1, sh - 1));
    for (int x = 0; x < dw; x++) {
      double fxx = (x + 0.5) * sx - 0.5;
      int x0 = clampi((int)fxx, 0, sw - 2);
      double dx = clampi(fxx - x0, 0.0, 1.0);
      if (sw == 1) {
        x0 = 0;
        dx = 0;
      }
      double top = r0[x0] * (1 - dx) + r0[x0 + 1] * dx;
      double bot = r1[x0] * (1 - dx) + r1[x0 + 1] * dx;
      d[x] = (float)(top * (1 - dy) + bot * dy);
    }
  }
}

static void jet(double t, uint8_t& r, uint8_t& g, uint8_t& b) {
  t = t < 0 ? 0 : (t > 1 ? 1 : t);
  r = (uint8_t)(255 * std::min(1.0, std::max(0.0, 1.5 - std::fabs(4 * t - 3))));
  g = (uint8_t)(255 * std::min(1.0, std::max(0.0, 1.5 - std::fabs(4 * t - 2))));
  b = (uint8_t)(255 * std::min(1.0, std::max(0.0, 1.5 - std::fabs(4 * t - 1))));
}

void jet_blend(const ImgU8& enh, const ImgF32& cam01, ImgU8& out) {
  out = ImgU8(enh.h, enh.w, 3);
  for (int y = 0; y < enh.h; y++) {
    const uint8_t* s = enh.row(y);
    const float* c = cam01.row(y);
    uint8_t* d = out.row(y);
    for (int x = 0; x < enh.w; x++) {
      uint8_t jr, jg, jb;
      jet(c[x], jr, jg, jb);
      d[3 * x] = (uint8_t)(s[3 * x] * 0.55 + jr * 0.45 + 0.5);
      d[3 * x + 1] = (uint8_t)(s[3 * x + 1] * 0.55 + jg * 0.45 + 0.5);
      d[3 * x + 2] = (uint8_t)(s[3 * x + 2] * 0.55 + jb * 0.45 + 0.5);
    }
  }
}

}  // namespace netradr
