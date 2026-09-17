#include "netradr/modules.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace netradr {

M2Out m2_segment(const ImgU8& enh) {
  M2Out o;
  int h = enh.h, w = enh.w;
  double scale = (double)h * w / (512.0 * 512.0);
  ImgU8 green, L, A, B;
  split_green(enh, green);
  rgb_to_lab8(enh, L, A, B);

  // interior guard (eroded retinal mask)
  ImgU8 base(h, w), interior(h, w);
  for (int y = 0; y < h; y++) {
    const uint8_t* l = L.row(y);
    uint8_t* d = base.row(y);
    for (int x = 0; x < w; x++) d[x] = l[x] > 12 ? 1 : 0;
  }
  erode_rect(base, interior, 31);

  // vessels
  ImgU8 cg = green;
  clahe(cg, 2.0, 8);
  ImgU8 th, vraw;
  tophat_gray(cg, th, 9);
  thresh_otsu(th, vraw);
  o.vessel = vraw;
  keep_area_range(o.vessel, (int)(60 * scale), -1);
  long vn = 0;
  for (auto v : o.vessel.v) vn += v;
  o.vessel_density = (double)vn / (h * w);

  // optic disc: top-percentile red, largest comp, Ritter circle
  ImgU8 red(h, w);
  for (int y = 0; y < h; y++) {
    const uint8_t* s = enh.row(y);
    uint8_t* d = red.row(y);
    for (int x = 0; x < w; x++) d[x] = s[3 * x];
  }
  double thr = percentile_u8(red, 99.2);
  ImgU8 bright(h, w);
  for (int y = 0; y < h; y++) {
    const uint8_t* s = red.row(y);
    uint8_t* d = bright.row(y);
    for (int x = 0; x < w; x++) d[x] = s[x] > thr ? 1 : 0;
  }
  keep_area_range(bright, (int)(800 * scale), -1);
  o.od_cx = w * 0.35;
  o.od_cy = h * 0.5;
  o.od_r = 0;
  o.od_mask = ImgU8(h, w);
  {
    std::vector<int> lab;
    auto comps = cc_stats(bright, lab);
    if (comps.size() > 1) {
      size_t bi = 1;
      for (size_t i = 2; i < comps.size(); i++)
        if (comps[i].area > comps[bi].area) bi = i;
      Circle c = ritter_circle(lab, w, h, (int)bi);
      o.od_cx = c.x;
      o.od_cy = c.y;
      o.od_r = c.r;
      draw_filled_circle(o.od_mask, (int)c.x, (int)c.y, (int)c.r, 1);
    }
  }

  // fovea: temporal side, darkest blurred patch in ROI
  double dd = std::max(o.od_r * 2, 20.0);
  double sl = 0, sr = 0;
  for (int y = 0; y < h; y++) {
    const uint8_t* g = green.row(y);
    for (int x = 0; x < w / 2; x++) sl += g[x];
    for (int x = w / 2; x < w; x++) sr += g[x];
  }
  double dir = (sl < sr) ? -1 : 1;
  double fx = std::min(std::max(o.od_cx + dir * 2.5 * dd, dd), w - dd);
  double fy = std::min(std::max(o.od_cy, dd), h - dd);
  {
    int x0 = (int)std::max(fx - dd, 0.0), x1 = (int)std::min(fx + dd, (double)w);
    int y0 = (int)std::max(fy - dd, 0.0), y1 = (int)std::min(fy + dd, (double)h);
    int rw = std::max(x1 - x0, 1), rh = std::max(y1 - y0, 1);
    ImgU8 roi(rh, rw);
    for (int y = 0; y < rh; y++) {
      const uint8_t* s = green.row(y0 + y);
      uint8_t* d = roi.row(y);
      for (int x = 0; x < rw; x++) d[x] = s[x0 + x];
    }
    ImgU8 bl;
    gauss_blur(roi, bl, 2.6);
    int bx = 0, by = 0, bv = 256;
    for (int y = 0; y < rh; y++) {
      const uint8_t* r = bl.row(y);
      for (int x = 0; x < rw; x++)
        if (r[x] < bv) {
          bv = r[x];
          bx = x;
          by = y;
        }
    }
    fx = x0 + bx;
    fy = y0 + by;
  }
  o.fx = fx;
  o.fy = fy;

  // OD dilation guard
  ImgU8 od_dil;
  dilate_rect(o.od_mask, od_dil, 21);

  // exudates
  double lt = percentile_u8(L, 97.5), bt = percentile_u8(B, 75.0);
  o.ex = ImgU8(h, w);
  for (int y = 0; y < h; y++) {
    const uint8_t* l = L.row(y);
    const uint8_t* b = B.row(y);
    const uint8_t* od = od_dil.row(y);
    const uint8_t* inr = interior.row(y);
    uint8_t* d = o.ex.row(y);
    for (int x = 0; x < w; x++) d[x] = (l[x] > lt && b[x] > bt && !od[x] && inr[x]) ? 1 : 0;
  }
  keep_area_range(o.ex, (int)(25 * scale), -1);
  {
    std::vector<int> lab;
    auto comps = cc_stats(o.ex, lab);
    o.ex_n = (long)comps.size() - 1;
    long a = 0;
    for (auto v : o.ex.v) a += v;
    o.ex_a = a;
  }

  // dark lesions
  ImgU8 inv(h, w);
  for (int y = 0; y < h; y++) {
    const uint8_t* g = green.row(y);
    uint8_t* d = inv.row(y);
    for (int x = 0; x < w; x++) d[x] = 255 - g[x];
  }
  double bg = median_masked(green, interior);
  {
    ImgU8 ts, raw;
    tophat_gray(inv, ts, 7);
    thresh_otsu(ts, raw);
    o.ma = ImgU8(h, w);
    for (int y = 0; y < h; y++) {
      const uint8_t* r = raw.row(y);
      const uint8_t* od = od_dil.row(y);
      const uint8_t* vs = o.vessel.row(y);
      const uint8_t* inr = interior.row(y);
      uint8_t* d = o.ma.row(y);
      for (int x = 0; x < w; x++) d[x] = (r[x] && !od[x] && !vs[x] && inr[x]) ? 1 : 0;
    }
    keep_area_range(o.ma, (int)(3 * scale), (int)(220 * scale));
    auto all = weighted_centroids(o.ma, green, bg);
    o.ma_a = 0;
    for (auto& cd : all)
      if (cd.conf >= 0.2) {
        o.ma_c.push_back(cd);
        o.ma_a += (long)cd.area;
      }
  }
  {
    ImgU8 ts, raw;
    tophat_gray(inv, ts, 17);
    thresh_otsu(ts, raw);
    ImgU8 hem(h, w);
    for (int y = 0; y < h; y++) {
      const uint8_t* r = raw.row(y);
      const uint8_t* od = od_dil.row(y);
      const uint8_t* inr = interior.row(y);
      uint8_t* d = hem.row(y);
      for (int x = 0; x < w; x++) d[x] = (r[x] && !od[x] && inr[x]) ? 1 : 0;
    }
    keep_area_range(hem, (int)(220 * scale), -1);
    o.he = ImgU8(h, w);  // minus vessels
    for (int y = 0; y < h; y++) {
      const uint8_t* s = hem.row(y);
      const uint8_t* vs = o.vessel.row(y);
      uint8_t* d = o.he.row(y);
      for (int x = 0; x < w; x++) d[x] = (s[x] && !vs[x]) ? 1 : 0;
    }
    auto all = weighted_centroids(o.he, green, bg);
    o.he_a = 0;
    for (auto& cd : all)
      if (cd.conf >= 0.2) {
        o.he_c.push_back(cd);
        o.he_a += (long)cd.area;
      }
  }

  // neovascularization: dense fine vessels in peripapillary ring
  ImgU8 dil41, ring(h, w);
  dilate_rect(o.od_mask, dil41, 41);
  long rn = 0, rv = 0;
  for (int y = 0; y < h; y++) {
    const uint8_t* d41 = dil41.row(y);
    const uint8_t* od = o.od_mask.row(y);
    const uint8_t* vs = o.vessel.row(y);
    uint8_t* rg = ring.row(y);
    for (int x = 0; x < w; x++) {
      rg[x] = (d41[x] && !od[x]) ? 1 : 0;
      if (rg[x]) {
        rn++;
        rv += vs[x];
      }
    }
  }
  o.ring_density = rn ? (double)rv / rn : 0;
  o.nv_flag = o.ring_density > 0.18 && ((long)o.ma_c.size() + (long)o.he_c.size() > 4);
  o.nv = ImgU8(h, w);
  if (o.nv_flag)
    for (int y = 0; y < h; y++) {
      const uint8_t* rg = ring.row(y);
      const uint8_t* vs = o.vessel.row(y);
      uint8_t* d = o.nv.row(y);
      for (int x = 0; x < w; x++) d[x] = (rg[x] && vs[x]) ? 1 : 0;
    }
  return o;
}

}  // namespace netradr
