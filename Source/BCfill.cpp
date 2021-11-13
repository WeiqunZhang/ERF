#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

#include "ERF.H"
#include "prob.H"

struct ERFHypFillExtDir
{
  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& dest,
    const int dcomp,
    const int numcomp,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int bcomp,
    const int orig_comp) const
  {
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();
    const amrex::Real* prob_lo = geom.ProbLo();
//    const amrex::Real* prob_hi = geom.ProbHi();
    const amrex::Real* dx = geom.CellSize();
    const amrex::Real x[AMREX_SPACEDIM] = {AMREX_D_DECL(
      prob_lo[0] + (iv[0] + 0.5) * dx[0], prob_lo[1] + (iv[1] + 0.5) * dx[1],
      prob_lo[2] + (iv[2] + 0.5) * dx[2])};

    const int* bc = bcr->data();

    amrex::Real s_int[NVAR] = {0.0};
    amrex::Real s_ext[NVAR] = {0.0};

    // xlo and xhi
    int idir = 0;
    if ((bc[idir] == amrex::BCType::ext_dir) && (iv[idir] < domlo[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(domlo[idir], iv[1], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom);
      for (int n = 0; n < NVAR; n++) {
        dest(iv, n) = s_ext[n];
      }
    } else if (
      (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) &&
      (iv[idir] > domhi[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(domhi[idir], iv[1], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom);
      for (int n = 0; n < NVAR; n++) {
        dest(iv, n) = s_ext[n];
      }
    }
    // ylo and yhi
    idir = 1;
    if ((bc[idir] == amrex::BCType::ext_dir) && (iv[idir] < domlo[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(iv[0], domlo[idir], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom);
      for (int n = 0; n < NVAR; n++) {
        dest(iv, n) = s_ext[n];
      }
    } else if (
      (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) &&
      (iv[idir] > domhi[idir])) {
      amrex::IntVect loc(AMREX_D_DECL(iv[0], domhi[idir], iv[2]));
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(loc, n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom);
      for (int n = 0; n < NVAR; n++) {
        dest(iv, n) = s_ext[n];
      }
    }
    // zlo and zhi
    idir = 2;
    if ((bc[idir] == amrex::BCType::ext_dir) && (iv[idir] < domlo[idir])) {
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(iv[0], iv[1], domlo[idir], n);
      }
      bcnormal(x, s_int, s_ext, idir, +1, time, geom);
      for (int n = 0; n < NVAR; n++) {
        dest(iv, n) = s_ext[n];
      }
    } else if (
      (bc[idir + AMREX_SPACEDIM] == amrex::BCType::ext_dir) &&
      (iv[idir] > domhi[idir])) {
      for (int n = 0; n < NVAR; n++) {
        s_int[n] = dest(iv[0], iv[1], domhi[idir], n);
      }
      bcnormal(x, s_int, s_ext, idir, -1, time, geom);
      for (int n = 0; n < NVAR; n++) {
        dest(iv, n) = s_ext[n];
      }
    }
  }
};

/*
struct ERFReactFillExtDir
{
  AMREX_GPU_DEVICE
  void operator()(
    const amrex::IntVect& iv,
    amrex::Array4<amrex::Real> const& dest,
    const int dcomp,
    const int numcomp,
    amrex::GeometryData const& geom,
    const amrex::Real time,
    const amrex::BCRec* bcr,
    const int bcomp,
    const int orig_comp) const
  {
  }
};
*/

namespace {
static ERFHypFillExtDir erf_hyp_fill_ext_dir;
static amrex::GpuBndryFuncFab<ERFHypFillExtDir>
  hyp_bndry_func(erf_hyp_fill_ext_dir);
} // namespace

void
erf_bcfill_hyp(
  amrex::Box const& bx,
  amrex::FArrayBox& data,
  const int dcomp,
  const int numcomp,
  amrex::Geometry const& geom,
  const amrex::Real time,
  const amrex::Vector<amrex::BCRec>& bcr,
  const int bcomp,
  const int scomp)
{
  hyp_bndry_func(bx, data, dcomp, numcomp, geom, time, bcr, bcomp, scomp);
}

void
erf_nullfill(
  amrex::Box const& bx,
  amrex::FArrayBox& data,
  const int dcomp,
  const int numcomp,
  amrex::Geometry const& geom,
  const amrex::Real time,
  const amrex::Vector<amrex::BCRec>& bcr,
  const int bcomp,
  const int scomp)
{
}
