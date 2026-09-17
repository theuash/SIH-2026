#include "netradr/modules.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace netradr {

static const double FOCUS_THR = 28.0, DARK_FRAC_THR = 0.25, BRIGHT_FRAC_THR = 0.12,
                    FOV_COVER_THR = 0.55, FOV_OFF_THR = 0.28;

M1Out m1_assess_enhance(const ImgU8& rgb) {
  M1Out o;
  int h = rgb.h, w = rgb.w;
  ImgU8 green, val;
  split_green(rgb, green);
  max_channel(rgb, val);

  ImgU8 mraw(h, w), mask(h, w);
  for (int y = 0; y < h; y++) {
    const uint8_t* v = val.row(y);
    uint8_t* d = mraw.row(y);
    for (int x = 0; x < w; x++) d[x] = v[x] > 12 ? 1 : 0;
  }
  morph_close(mraw, mask, 15);

  double lap = laplacian_var(green);
  o.focus_ok = lap >= FOCUS_THR;

  long n = 0, ndark = 0, nbright = 0;
  double sum = 0;
  for (int y = 0; y < h; y++) {
    const uint8_t* g = green.row(y);
    const uint8_t* mk = mask.row(y);
    for (int x = 0; x < w; x++)
      if (mk[x]) {
        n++;
        sum += g[x];
        if (g[x] < 30) ndark++;
        if (g[x] > 235) nbright++;
      }
  }
  if (!n) {  // degenerate: whole-frame stats
    n = (long)h * w;
    for (int y = 0; y < h; y++) {
      const uint8_t* g = green.row(y);
      for (int x = 0; x < w; x++) {
        sum += g[x];
        if (g[x] < 30) ndark++;
        if (g[x] > 235) nbright++;
      }
    }
  }
  double dark = (double)ndark / n, bright = (double)nbright / n, mean = sum / n;
  o.illum_ok = dark <= DARK_FRAC_THR && bright <= BRIGHT_FRAC_THR && mean >= 35 && mean <= 225;

  std::vector<int> labels;
  auto comps = cc_stats(mask, labels);
  double area = 0, off = 1;
  if (comps.size() > 1) {
    size_t bi = 1;
    for (size_t i = 2; i < comps.size(); i++)
      if (comps[i].area > comps[bi].area) bi = i;
    area = (double)comps[bi].area / (h * w);
    off = std::fabs(comps[bi].cx - w / 2.0) / w;
  }
  o.fov_ok = area >= FOV_COVER_THR && off <= FOV_OFF_THR;

  double p0 = std::min(lap / (FOCUS_THR * 3), 1.0);
  double p1 = 1.0 - std::min((dark + bright) / 0.5, 1.0);
  double p2 = std::min(area / 0.85, 1.0);
  o.score = (p0 + p1 + p2) / 3.0;
  o.accepted = o.focus_ok && o.illum_ok && o.fov_ok;

  // --- enhance: CLAHE-L + flatten + bilateral ---
  ImgU8 L, A, B;
  rgb_to_lab8(rgb, L, A, B);
  clahe(L, 2.5, 8);
  ImgU8 bg;
  box_background(L, bg, std::max(8, std::min(h, w) / 8));
  ImgU8 flat(h, w);
  for (int y = 0; y < h; y++) {
    const uint8_t* l = L.row(y);
    const uint8_t* bgr = bg.row(y);
    uint8_t* d = flat.row(y);
    for (int x = 0; x < w; x++)
      d[x] = (uint8_t)clampi((int)(1.15 * l[x] - 0.15 * bgr[x] + 8 + 0.5), 0, 255);
  }
  ImgU8 merged;
  lab_to_rgb8(flat, A, B, merged);
  bilateral3_rgb(merged, o.enh, 25.0, 25.0);
  return o;
}

}  // namespace netradr
