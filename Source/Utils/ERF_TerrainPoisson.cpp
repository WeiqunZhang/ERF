#include "ERF_TerrainPoisson.H"

using namespace amrex;


TerrainPoisson::TerrainPoisson (amrex::Geometry const& geom, amrex::BoxArray const& ba,
                                amrex::DistributionMapping const& dm)
    : m_geom(geom),
      m_grids(ba),
      m_dmap(dm)
{
}

void TerrainPoisson::apply(amrex::MultiFab& lhs, amrex::MultiFab const& rhs)
{
    AMREX_ASSERT(rhs.nGrowVect().allGT(0));

    const_cast<MultiFab&>(rhs).FillBoundary(m_geom.periodicity());

    MultiFab zpa_mf;

    auto const& dx = m_geom.InvCellSizeArray();

    auto const& y = lhs.arrays();
    auto const& x = rhs.const_arrays();
    auto const& zpa = zpa_mf.const_arrays();

    amrex::ParallelFor(rhs, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
    {
        terrpoisson_adotx(i,j,k,y[b], x[b], zpa[b], dx[0], dx[1], dx[2]);
    });
}

void TerrainPoisson::assign(amrex::MultiFab& lhs, amrex::MultiFab const& rhs)
{
    MultiFab::Copy(lhs, rhs, 0, 0, 1, 0);
}

void TerrainPoisson::scale(amrex::MultiFab& lhs, amrex::Real fac)
{
    lhs.mult(fac);
}

Real TerrainPoisson::dotProduct(amrex::MultiFab const& v1, amrex::MultiFab const& v2)
{
    return MultiFab::Dot(v1, 0, v2, 0, 1, 0);
}

void TerrainPoisson::increment(amrex::MultiFab& lhs, amrex::MultiFab const& rhs, Real a)
{
    MultiFab::Saxpy(lhs, a, rhs, 0, 0, 1, 0);
}

void TerrainPoisson::linComb(amrex::MultiFab& lhs, Real a, amrex::MultiFab const& rhs_a,
                             Real b, amrex::MultiFab const& rhs_b)
{
    MultiFab::LinComb(lhs, a, rhs_a, 0, b, rhs_b, 0, 0, 1, 0);
}


MultiFab TerrainPoisson::makeVecRHS()
{
    return MultiFab(m_grids, m_dmap, 1, 0);
}

MultiFab TerrainPoisson::makeVecLHS()
{
    return MultiFab(m_grids, m_dmap, 1, 1);
}

Real TerrainPoisson::norm2(amrex::MultiFab const& v)
{
    return v.norm2();
}

void TerrainPoisson::precond(amrex::MultiFab& lhs, amrex::MultiFab const& rhs)
{
    MultiFab::Copy(lhs, rhs, 0, 0, 1, 0);
}

void TerrainPoisson::setToZero(amrex::MultiFab& v)
{
    v.setVal(0);
}
