#pragma once
// M1/M2/M4 entry points. Same constants as the Python twins — keep in sync.
#include <vector>
#include "netradr/img.h"
#include "netradr/ops.h"

namespace netradr {

struct M1Out {
  bool accepted = false, focus_ok = false, illum_ok = false, fov_ok = false;
  double score = 0;
  ImgU8 enh;
};
M1Out m1_assess_enhance(const ImgU8& rgb);

struct M2Out {
  ImgU8 vessel, od_mask, ma, ex, he, nv;
  double od_cx = 0, od_cy = 0, od_r = 0, fx = 0, fy = 0, vessel_density = 0;
  bool nv_flag = false;
  std::vector<Cand> ma_c, he_c;
  long ma_a = 0, he_a = 0, ex_a = 0;
  long ex_n = 0;
  double ring_density = 0;
};
M2Out m2_segment(const ImgU8& enh);

struct M4Out {
  ImgF32 cam;
  ImgU8 overlay;
  double ov_ma = 0, ov_ex = 0, ov_he = 0;
};
M4Out m4_pixels(const ImgU8& enh, const ImgU8& ma, const ImgU8& ex, const ImgU8& he,
                const ImgU8& vessel);

}  // namespace netradr
