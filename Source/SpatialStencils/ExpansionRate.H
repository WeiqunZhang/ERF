#ifndef _COMPUTE_EXPANSIONRATE_H_
#define _COMPUTE_EXPANSIONRATE_H_

using namespace amrex;

AMREX_GPU_DEVICE
inline Real
ComputeExpansionRate(const int &i, const int &j, const int &k,
                     const Array4<const Real>& u, const Array4<const Real>& v, const Array4<const Real>& w,
                     const enum MomentumEqn &momentumEqn,
                     const enum DiffusionDir &diffDir,
                     const GpuArray<Real, AMREX_SPACEDIM>& cellSizeInv) {
    Real dx_inv = cellSizeInv[0];
    Real dy_inv = cellSizeInv[1];
    Real dz_inv = cellSizeInv[2];

    Real expansionRate = 0;

    //TODO: Check if it is more efficient to store computed divergence at cell-centers rather than
    // computing on the fly

    switch (momentumEqn) {
        case MomentumEqn::x:
            switch (diffDir) {
                case DiffusionDir::x: // D11
                    // D11 (i+1/2)
                    expansionRate = (u(i  , j  , k  ) - u(i-1, j, k))*dx_inv +
                                    (v(i-1, j+1, k  ) - v(i-1, j, k))*dy_inv +
                                    (w(i-1, j  , k+1) - w(i-1, j, k))*dz_inv;
                    break;
                case DiffusionDir::y: // D12
                    expansionRate = 0.0;
                    break;
                case DiffusionDir::z: // D13
                    expansionRate = 0.0;
                    break;
                default:
                    amrex::Abort("Error: Diffusion direction is unrecognized");
            }
            break;
        case MomentumEqn::y:
            switch (diffDir) {
                case DiffusionDir::x: // D21
                    expansionRate = 0.0;
                    break;
                case DiffusionDir::y: // D22
                    expansionRate = (u(i+1, j-1, k  ) - u(i, j-1, k))*dx_inv +
                                    (v(i  , j  , k  ) - v(i, j-1, k))*dy_inv +
                                    (w(i  , j-1, k+1) - w(i, j-1, k))*dz_inv;
                    break;
                case DiffusionDir::z: // D23
                    expansionRate = 0.0;
                    break;
                default:
                    amrex::Abort("Error: Diffusion direction is unrecognized");
            }
            break;
        case MomentumEqn::z:
            switch (diffDir) {
                case DiffusionDir::x: // D31
                    expansionRate = 0.0;
                    break;
                case DiffusionDir::y: // D32
                    expansionRate = 0.0;
                    break;
                case DiffusionDir::z: // D33
                    expansionRate = (u(i+1, j  , k-1) - u(i, j, k-1))*dx_inv +
                                    (v(i  , j+1, k-1) - v(i, j, k-1))*dy_inv +
                                    (w(i  , j  , k  ) - w(i, j, k-1))*dz_inv;
                    break;
                default:
                    amrex::Abort("Error: Diffusion direction is unrecognized");
            }
            break;
        default:
            amrex::Abort("Error: Momentum equation is unrecognized");
    }

    return (1.0/3.0) * expansionRate;
}
#endif
