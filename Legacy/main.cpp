#include <AMReX_CArena.H>
#include <AMReX_REAL.H>
#include <AMReX_Utility.H>
#include <AMReX_IntVect.H>
//#include <AMReX_Box.H>
#include <AMReX_Amr.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
//#include <AMReX_AmrLevel.H>

//#include "IO.H"
#include "ERF.H"

std::string inputs_name = "";

amrex::LevelBld* getLevelBld();

// Set the refine_grid_layout flags to (1,1,0) by default
// since the ERF default is different from the amrex default (1,1,1)
// Also set max_grid_size to very large since the only reason for
// chopping grids is if Nprocs > Ngrids
void add_par () {
   ParmParse pp("amr");
   pp.add("refine_grid_layout_x",1);
   pp.add("refine_grid_layout_y",1);
   pp.add("refine_grid_layout_z",0);

   // n_proper is the minimum number of coarse cells between coarse-fine boundaries
   // between levels (ell and ell+1) and levels (ell-1 and ell).   We want this to be
   // greater than or equal to the stencil width (a function of spatial order) divided by
   // ref_ratio (which can be 2,3 or 4).  This ensures that fillpatch at level (ell)
   // does not need to reach beyond level (ell-1). Here to be conservative we set this to 2
   // (rather than the amrex default of 1).
   pp.add("n_proper",2);

   pp.add("max_grid_size",2048);
   pp.add("blocking_factor",2);
}

int
main(int argc, char* argv[])
{
  // Use this to trap NaNs in C++
  // feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW);

  if (argc <= 1) {
    amrex::Abort("Error: no inputs file provided on command line.");
  }

  // check to see if it contains --describe
  if (argc >= 2) {
    for (auto i = 1; i < argc; i++) {
      if (std::string(argv[i]) == "--describe") {
        IOManager::writeBuildInfo(std::cout);
        return 0;
      }
    }
  }

  // Make sure to catch new failures.
  amrex::Initialize(argc, argv, true, MPI_COMM_WORLD, add_par);

  // Save the inputs file name for later.
  if (!strchr(argv[1], '=')) {
    inputs_name = argv[1];
  }

  BL_PROFILE_VAR("main()", pmain);

  amrex::Real dRunTime1 = amrex::ParallelDescriptor::second();

  amrex::Print() << std::setprecision(10);

  int max_step = -1;
  amrex::Real stop_time = -1.0;

  amrex::ParmParse pp;
  pp.query("max_step", max_step);
  pp.query("stop_time", stop_time);

  if (max_step < 0 && stop_time < 0.0) {
    amrex::Abort(
      "Exiting because neither max_step nor stop_time is non-negative.");
  }

  // Print the current date and time
  time_t time_type;
  struct tm* time_pointer;
  time(&time_type);
  time_pointer = gmtime(&time_type);

  if (amrex::ParallelDescriptor::IOProcessor())
    amrex::Print() << std::setfill('0') << "\nStarting run at " << std::setw(2)
                   << time_pointer->tm_hour << ":" << std::setw(2)
                   << time_pointer->tm_min << ":" << std::setw(2)
                   << time_pointer->tm_sec << " UTC on "
                   << time_pointer->tm_year + 1900 << "-" << std::setw(2)
                   << time_pointer->tm_mon + 1 << "-" << std::setw(2)
                   << time_pointer->tm_mday << "." << std::endl;

  // Initialize random seed after we're running in parallel.
  amrex::Amr* amrptr = new amrex::Amr(getLevelBld());

  amrex::Real strt_time = 0.0;
  amrptr->init(strt_time, stop_time);

  // If we set the regrid_on_restart flag and if we are *not* going to take
  // a time step then we want to go ahead and regrid here.
  if (
    amrptr->RegridOnRestart() &&
    ((amrptr->levelSteps(0) >= max_step) || (amrptr->cumTime() >= stop_time))) {
    // Regrid only!
    amrptr->RegridOnly(amrptr->cumTime());
  }

  amrex::Real dRunTime2 = amrex::ParallelDescriptor::second();

  while (amrptr->okToContinue() &&
         (amrptr->levelSteps(0) < max_step || max_step < 0) &&
         (amrptr->cumTime() < stop_time || stop_time < 0.0)) {
    // Do a timestep
    amrptr->coarseTimeStep(stop_time);
  }

  // Write final checkpoint
  if (amrptr->stepOfLastCheckPoint() < amrptr->levelSteps(0)) {
    amrptr->checkPoint();
  }

  // Write final plotfile
  if (amrptr->stepOfLastPlotFile() < amrptr->levelSteps(0)) {
    amrptr->writePlotFile();
  }

  time(&time_type);
  time_pointer = gmtime(&time_type);

  if (amrex::ParallelDescriptor::IOProcessor())
    amrex::Print() << std::setfill('0') << "\nEnding run at " << std::setw(2)
                   << time_pointer->tm_hour << ":" << std::setw(2)
                   << time_pointer->tm_min << ":" << std::setw(2)
                   << time_pointer->tm_sec << " UTC on "
                   << time_pointer->tm_year + 1900 << "-" << std::setw(2)
                   << time_pointer->tm_mon + 1 << "-" << std::setw(2)
                   << time_pointer->tm_mday << "." << std::endl;

  delete amrptr;

  // This MUST follow the above delete as ~Amr() may dump files to disk
  const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();

  amrex::Real dRunTime3 = amrex::ParallelDescriptor::second();

  amrex::Real runtime_total = dRunTime3 - dRunTime1;
  amrex::Real runtime_timestep = dRunTime3 - dRunTime2;

  amrex::ParallelDescriptor::ReduceRealMax(runtime_total, IOProc);
  amrex::ParallelDescriptor::ReduceRealMax(runtime_timestep, IOProc);

  if (amrex::ParallelDescriptor::IOProcessor()) {
    amrex::Print() << "Run time = " << runtime_total << std::endl;
    amrex::Print() << "Run time w/o init = " << runtime_timestep << std::endl;
  }

  if (amrex::CArena* arena = dynamic_cast<amrex::CArena*>(amrex::The_Arena())) {
    // A barrier to make sure our output follows that of RunStats.
    amrex::ParallelDescriptor::Barrier();
    // We're using a CArena -- output some FAB memory stats.
    // This'll output total # of bytes of heap space in the Arena.
    // It's actually the high water mark of heap space required by FABs.
    char buf[256];

    sprintf(
      buf, "CPU(%d): Heap Space (bytes) used by Coalescing FAB Arena: %ld",
      amrex::ParallelDescriptor::MyProc(), arena->heap_space_used());

    amrex::Print() << buf << std::endl;
  }

  BL_PROFILE_VAR_STOP(pmain);
  BL_PROFILE_SET_RUN_TIME(dRunTime2);

  amrex::Finalize();

  return 0;
}
