// Zero-copy numpy <-> Img bindings. GIL released inside compute.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "netradr/modules.h"

namespace py = pybind11;
using namespace netradr;

static ImgU8 from_rgb(py::array_t<uint8_t> a) {
  py::buffer_info b = a.request();
  if (b.ndim != 3 || b.shape[2] != 3) throw std::runtime_error("need RGB HxWx3 uint8");
  ImgU8 m((int)b.shape[0], (int)b.shape[1], 3);
  const uint8_t* s = (const uint8_t*)b.ptr;
  // force C-contiguous copy only if needed; request() with c_style already ensures it
  size_t n = m.v.size();
  for (size_t i = 0; i < n; i++) m.v[i] = s[i];
  return m;
}

static ImgU8 from_mask(py::array m) {
  py::array_t<uint8_t, py::array::c_style | py::array::forcecast> a(m);
  py::buffer_info b = a.request();
  if (b.ndim != 2) throw std::runtime_error("need HxW mask");
  ImgU8 out((int)b.shape[0], (int)b.shape[1]);
  const uint8_t* s = (const uint8_t*)b.ptr;
  size_t n = out.v.size();
  for (size_t i = 0; i < n; i++) out.v[i] = s[i] ? 1 : 0;
  return out;
}

template <typename T>
static py::array_t<T> to_np2d(const T* data, int h, int w) {
  py::array_t<T> a({h, w});
  T* p = (T*)a.mutable_data();
  for (long i = 0; i < (long)h * w; i++) p[i] = data[i];
  return a;
}

static py::array mask_np(const ImgU8& m) {
  py::array_t<bool> a({m.h, m.w});
  bool* p = (bool*)a.mutable_data();
  for (long i = 0; i < (long)m.h * m.w; i++) p[i] = m.v[i] != 0;
  return a;
}

static py::array cands_np(const std::vector<Cand>& cs) {
  py::array_t<double> a({(long)cs.size(), (long)6});
  double* p = (double*)a.mutable_data();
  for (size_t i = 0; i < cs.size(); i++) {
    p[6 * i] = cs[i].x;
    p[6 * i + 1] = cs[i].y;
    p[6 * i + 2] = cs[i].xs;
    p[6 * i + 3] = cs[i].ys;
    p[6 * i + 4] = cs[i].area;
    p[6 * i + 5] = cs[i].conf;
  }
  return a;
}

PYBIND11_MODULE(netradr_core, m) {
  m.doc() = "NetraDR native pixel core (M1/M2/M4). Same contracts as Python twins.";
  m.def("m1_assess", [](py::array_t<uint8_t> img) {
    ImgU8 rgb = from_rgb(img);
    M1Out o;
    {
      py::gil_scoped_release rel;
      o = m1_assess_enhance(rgb);
    }
    py::dict d;
    d["accepted"] = o.accepted;
    d["quality_score"] = o.score;
    d["focus_ok"] = o.focus_ok;
    d["illumination_ok"] = o.illum_ok;
    d["fov_ok"] = o.fov_ok;
    d["enhanced"] = to_np2d<uint8_t>(o.enh.v.data(), o.enh.h, o.enh.w * 3).attr("reshape")(o.enh.h, o.enh.w, 3);
    return d;
  });
  m.def("m2_segment", [](py::array_t<uint8_t> enh) {
    ImgU8 e = from_rgb(enh);
    M2Out o;
    {
      py::gil_scoped_release rel;
      o = m2_segment(e);
    }
    py::dict d;
    d["vessel_mask"] = mask_np(o.vessel);
    d["od_mask"] = mask_np(o.od_mask);
    d["od_centroid"] = py::make_tuple(o.od_cx, o.od_cy);
    d["od_radius"] = o.od_r;
    d["fovea"] = py::make_tuple(o.fx, o.fy);
    d["ma_mask"] = mask_np(o.ma);
    d["ma_cands"] = cands_np(o.ma_c);
    d["ma_area"] = o.ma_a;
    d["ex_mask"] = mask_np(o.ex);
    d["ex_count"] = o.ex_n;
    d["ex_area"] = o.ex_a;
    d["he_mask"] = mask_np(o.he);
    d["he_cands"] = cands_np(o.he_c);
    d["he_area"] = o.he_a;
    d["nv_flag"] = o.nv_flag;
    d["nv_regions"] = mask_np(o.nv);
    d["vessel_density"] = o.vessel_density;
    return d;
  });
  m.def("m4_pixels", [](py::array_t<uint8_t> enh, py::array ma, py::array ex, py::array he,
                        py::array vessel) {
    ImgU8 e = from_rgb(enh);
    ImgU8 m1 = from_mask(ma), m2 = from_mask(ex), m3 = from_mask(he), v = from_mask(vessel);
    M4Out o;
    {
      py::gil_scoped_release rel;
      o = m4_pixels(e, m1, m2, m3, v);
    }
    py::dict d;
    d["cam"] = to_np2d<float>(o.cam.v.data(), o.cam.h, o.cam.w);
    d["overlay"] =
        to_np2d<uint8_t>(o.overlay.v.data(), o.overlay.h, o.overlay.w * 3).attr("reshape")(o.overlay.h, o.overlay.w, 3);
    d["ov"] = py::make_tuple(o.ov_ma, o.ov_ex, o.ov_he);
    return d;
  });
}
