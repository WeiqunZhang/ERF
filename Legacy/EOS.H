#ifndef _EOS_H_
#define _EOS_H_

#include <ERF_Constants.H>
#include <AMReX.H>

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real getTgivenRandRTh(const amrex::Real rho, const amrex::Real rhotheta)
{
    amrex::Real p_loc = p_0 * std::pow(R_d * rhotheta / p_0, Gamma);
    return p_loc / (R_d * rho);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real getPgivenRTh(const amrex::Real rhotheta)
{
    // diagnostic relation for the full pressure
    // see https://erf.readthedocs.io/en/latest/theory/NavierStokesEquations.html
    return p_0 * std::pow(R_d * rhotheta / p_0, Gamma);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real getdPdRgivenConstantTheta(const amrex::Real rho, const amrex::Real theta)
{
    return Gamma * p_0 * std::pow( (R_d * theta / p_0), Gamma) * std::pow(rho, Gamma-1.0) ;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real getPprimegivenRTh(const amrex::Real rhotheta, const amrex::Real pres_hse_at_k)
{
    // diagnostic relation for the full pressure
    // see https://erf.readthedocs.io/en/latest/theory/NavierStokesEquations.html
    return p_0 * std::pow(R_d * rhotheta / p_0, Gamma) - pres_hse_at_k;
}


AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real getRhoThetagivenP(const amrex::Real p)
{
    // diagnostic relation for the full pressure
    // see https://erf.readthedocs.io/en/latest/theory/NavierStokesEquations.html
    return std::pow(p*std::pow(p_0, Gamma-1), 1.0 / Gamma)/ R_d;
}
#endif
