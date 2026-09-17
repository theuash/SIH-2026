#include "netradr/modules.h"
#include <algorithm>
#include <vector>

namespace netradr {

static double overlap_frac(const ImgU8& m, const ImgU8& hot) {
  long inter = 0, tot = 0;
  for (int y = 0; y < m.h; y++) {
    const uint8_t* a = m.row(y);
    const uint8_t* b = hot.row(y);
    for (int x = 0; x < m.w; x++)
      if (a[x]) {
        tot++;
        inter += b[x];
      }
  }
  return tot ? (double)inter / tot : 0.0;
}

M4Out m4_pixels(const ImgU8& enh, const ImgU8& ma, const ImgU8& ex, const ImgU8& he,
                const ImgU8& vessel) {
  M4Out o;
  int h = enh.h, w = enh.w;
  // Heatmaps are smooth: accumulate at 1/4 res (16x less work), upsample once.
  const int S = 4;
  int cw = (w + S - 1) / S, ch = (h + S - 1) / S;
  ImgF32 cam_c(ch, cw);
  auto splat_coarse = [&](const ImgU8& mask, double sigma) {
    bool any = false;
    for (auto v : mask.v)
      if (v) {
        any = true;
        break;
      }
    if (!any) return;
    ImgF32 coarse(ch, cw);
    for (int cy = 0; cy < ch; cy++) {
      float* d = coarse.row(cy);
      for (int cx = 0; cx < cw; cx++) {
        int s = 0, n = 0;
        for (int y = cy * S; y < std::min(cy * S + S, h); y++) {
          const uint8_t* r = mask.row(y);
          for (int x = cx * S; x < std::min(cx * S + S, w); x++) {
            s += r[x];
            n++;
          }
        }
        d[cx] = (float)s / n;
      }
    }
    ImgF32 bl;
    gauss_blur_f32(coarse, bl, sigma / S);
    float mx = 1e-9f;
    for (auto v : bl.v)
      if (v > mx) mx = v;
    for (int y = 0; y < ch; y++) {
      float* d = cam_c.row(y);
      const float* s = bl.row(y);
      for (int x = 0; x < cw; x++) {
        float v = s[x] / mx;
        if (v > d[x]) d[x] = v;
      }
    }
  };
  splat_coarse(ma, 18);
  splat_coarse(ex, 26);
  splat_coarse(he, 30);
  {  // vessel bed at coarse res
    ImgF32 coarse(ch, cw);
    bool any = false;
    for (int cy = 0; cy < ch; cy++) {
      float* d = coarse.row(cy);
      for (int cx = 0; cx < cw; cx++) {
        int s = 0, n = 0;
        for (int y = cy * S; y < std::min(cy * S + S, h); y++) {
          const uint8_t* r = vessel.row(y);
          for (int x = cx * S; x < std::min(cx * S + S, w); x++) {
            s += r[x];
            n++;
          }
        }
        d[cx] = (float)s / n;
        any = any || s > 0;
      }
    }
    if (any) {
      ImgF32 bl;
      gauss_blur_f32(coarse, bl, 12.0 / S);
      float mx = 1e-9f;
      for (auto v : bl.v)
        if (v > mx) mx = v;
      for (int y = 0; y < ch; y++) {
        float* d = cam_c.row(y);
        const float* s = bl.row(y);
        for (int x = 0; x < cw; x++) {
          float v = 0.35f * s[x] / mx;
          if (v > d[x]) d[x] = v;
        }
      }
    }
  }
  ImgF32 cam(h, w);
  upsample_bilinear_f32(cam_c, cam);
  o.cam = cam;
  // hot = top 30%
  std::vector<float> vals = cam.v;
  size_t k = (size_t)(0.70 * (vals.size() - 1));
  std::nth_element(vals.begin(), vals.begin() + k, vals.end());
  float t = vals[k];
  ImgU8 hot(h, w);
  for (int y = 0; y < h; y++) {
    const float* c = cam.row(y);
    uint8_t* d = hot.row(y);
    for (int x = 0; x < w; x++) d[x] = c[x] >= t ? 1 : 0;
  }
  o.ov_ma = overlap_frac(ma, hot);
  o.ov_ex = overlap_frac(ex, hot);
  o.ov_he = overlap_frac(he, hot);
  jet_blend(enh, cam, o.overlay);
  return o;
}

}  // namespace netradr
