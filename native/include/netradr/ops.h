#pragma once
// Pixel ops used by M1/M2/M4. Rectangular kernels, Van Herk O(N) morphology.
#include <cstdint>
#include <vector>
#include "netradr/img.h"

namespace netradr {

struct Comp {
  int area = 0;
  double cx = 0, cy = 0;
  int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};
struct Circle {
  double x = 0, y = 0, r = 0;
};
struct Cand {
  double x, y, xs, ys, area, conf;
};

// --- color ---
void split_green(const ImgU8& rgb, ImgU8& g);
void max_channel(const ImgU8& rgb, ImgU8& v);
void rgb_to_lab8(const ImgU8& rgb, ImgU8& L, ImgU8& A, ImgU8& B);
void lab_to_rgb8(const ImgU8& L, const ImgU8& A, const ImgU8& B, ImgU8& rgb);

// --- stats ---
double laplacian_var(const ImgU8& g);
void histogram(const ImgU8& g, uint64_t hist[256]);
int otsu_of_hist(const uint64_t hist[256]);
double percentile_u8(const ImgU8& g, double p);                       // all pixels
double percentile_masked(const ImgU8& g, const ImgU8& m /*0/1*/, double p);
double median_masked(const ImgU8& g, const ImgU8& m /*0/1*/);

// --- filters ---
void clahe(ImgU8& l, double clip, int tiles);          // in-place, uint8
void gauss_blur(const ImgU8& src, ImgU8& dst, double sigma);
void gauss_blur_f32(const ImgF32& src, ImgF32& dst, double sigma);
void box_background(const ImgU8& src, ImgU8& dst, int radius);  // 3-pass box ~= big gauss
void bilateral3_rgb(const ImgU8& src, ImgU8& dst, double sC, double sS);

// --- binary morphology (rect kxk, 0/1) ---
void erode_rect(const ImgU8& src, ImgU8& dst, int k);
void dilate_rect(const ImgU8& src, ImgU8& dst, int k);
void morph_close(const ImgU8& src, ImgU8& dst, int k);
void tophat_gray(const ImgU8& src, ImgU8& dst, int k);  // src - open(src), grayscale
void thresh_otsu(const ImgU8& src, ImgU8& dst /*0/1*/);

// --- components ---
// labels: background 0, comps 1..n. Returns stats (index 0 = background, ignore).
std::vector<Comp> cc_stats(const ImgU8& mask /*0/1*/, std::vector<int>& labels);
void keep_area_range(ImgU8& mask, int min_area, int max_area);  // max_area<0: no cap
void draw_filled_circle(ImgU8& mask, int cx, int cy, int r, uint8_t v);

// --- geometry / candidates ---
Circle ritter_circle(const std::vector<int>& labels, int w, int h, int id);
std::vector<Cand> weighted_centroids(const ImgU8& mask, const ImgU8& gray, double bg);

// --- explain ---
void add_gauss_splat(const ImgU8& mask /*0/1*/, ImgF32& acc, double sigma);
void upsample_bilinear_f32(const ImgF32& src, ImgF32& dst /*pre-sized*/);
void jet_blend(const ImgU8& enh, const ImgF32& cam01, ImgU8& out);

}  // namespace netradr
