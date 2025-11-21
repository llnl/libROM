/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

//               libROM MFEM Example: differential evolution for Maxwell equation (adapted from ex3p.cpp)
//
// Description:  This example code uses MFEM and librom to define a projection-based
//               reduced-order model for a simple electromagnetic diffusion
//               problem corresponding to the second order definite Maxwell
//               equation: curl curl E + E = f. The boundary condition is given by
//               E x n = <given tangential field>. Here, we use a given exact
//               solution E and compute the corresponding r.h.s. f. Each component
//               of the vector field E is assumed to be a sin wave, with frequency
//               controlled by the parameter kappa.
//
//               The objective of this example is to first build a local
//               projection-based ROM using a greedy algorithm to sample the
//               (1D) parameter space for kappa. After building the database, we use
//               our ROM with the differential evolution (DE) algorithm. Given a
//               vector field E, applying the DE algorithm with our ROM allows us to
//               search the parameter space to find a value of kappa which closely
//               matches the value of kappa used to produce the given vector field E.
//
//               The build_database phase builds a ROM database using different
//               frequncies and a latin-hypercube sampling procedure. The fom phase
//               solves the full-order model for visualization and comparison.
//               The de phase applies the differential evolution algorithm.
//
// Example commands:
// Before running the simulations below, make sure to delete the parameters file with "rm -r parameters.txt"
//
// Small example:
// build_database phase: mpirun -np 8 de_parametric_maxwell_greedy -build_database -greedy-param-min 1.0 -greedy-param-max 1.2 -greedy-param-size 5 -greedysubsize 2 -greedyconvsize 3 -greedyrelerrortol 0.01
// FOM phase:            mpirun -np 8 de_parametric_maxwell_greedy -visit -fom -f 1.15 (create a new solution to compare with)
// DE phase:             mpirun -np 8 de_parametric_maxwell_greedy -f 1.15 -visit -de -de_f 0.9 -de_cr 0.9 -de_ps 50 -de_min_iter 10 -de_max_iter 100 -de_ct 0.001 -de_min_freq 1.0 -de_max_freq 1.2 (Run interpolative differential evolution to see if target FOM can be matched. -visit outputs visualization for the vector field E with lowest relative error)
//
// Results - Best minimal cost: 1.65478e-05        Best agent: 1.15
//
// Larger example:
// build_database phase: mpirun -np 8 de_parametric_maxwell_greedy -build_database -greedy-param-min 0.5 -greedy-param-max 1.5 -greedy-param-size 15 -greedysubsize 4 -greedyconvsize 6 -greedyrelerrortol 0.01
// FOM phase:            mpirun -np 8 de_parametric_maxwell_greedy -visit -fom -f X.XX (create a new solution to compare with, set X.XX to desired frequency)
// DE phase:             mpirun -np 8 de_parametric_maxwell_greedy  -f X.XX -visit -de -de_f 0.9 -de_cr 0.9 -de_ps 50 -de_min_iter 10 -de_max_iter 100 -de_ct 0.001 -de_min_freq 0.5 -de_max_freq 1.5 (Run interpolative differential evolution to see if target FOM can be matched)
//
// Results with frequency 0.8 - Best minimal cost: 0.000367194        Best agent: 0.800017

// This example runs in parallel with MPI, by using the same number of MPI ranks
// in all phases (build_database, fom, de).

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "linalg/Vector.h"
#include "linalg/BasisGenerator.h"
#include "linalg/BasisReader.h"
#include "mfem/Utilities.hpp"
#include <cmath>
#include <set>
#include <vector>
#include "algo/greedy/GreedyRandomSampler.h"
#include "algo/DifferentialEvolution.h"

#ifndef _WIN32
#include <sys/stat.h>  // mkdir
#else
#include <direct.h>    // _mkdir
#define mkdir(dir, mode) _mkdir(dir)
#endif

using std::vector;
using std::set;
using std::map;
using std::make_pair;
using std::string;
using std::cout;
using std::endl;
using std::flush;
using std::max;
using std::min;
using std::to_string;
using std::ifstream;
using std::ofstream;
using std::ostringstream;
using namespace mfem;

// Exact solution, E, and r.h.s., f. See below for implementation.
void E_exact(const Vector &, Vector &);
void f_exact(const Vector &, Vector &);
double freq = 1.0, kappa;
int dim;

int myid;
int num_procs;

// 2. Parse command-line options.
const char *mesh_file = "../data/star.mesh";
int order = 1;
bool static_cond = false;
bool pa = false;
const char *device_config = "cpu";
bool visualization = false;
bool visit = false;
bool fom = false;
bool offline = false;
bool online = false;
int precision = 8;
double coef = 1.0;

// Greedy algorithm variables
bool build_database = false;
bool use_database = false;
double greedy_param_space_min = 1.0;
double greedy_param_space_max = 1.2;
int greedy_param_space_size = 5;
double greedy_relative_error_tol = 0.01;
int greedy_subset_size = 2;
int greedy_convergence_subset_size = 3;
double local_rom_freq = 0.0;
double curr_freq = 0.0;
vector<string> basisIdentifiers;
CAROM::GreedySampler* greedy_sampler = NULL;
bool calc_rel_error;
std::string curr_basis_identifier;
bool calc_err_indicator = false;

// DE variables
bool de = false;
double de_min_freq= -DBL_MAX;
double de_max_freq = DBL_MAX;
double target_freq = freq;
double de_F = 0.8;
double de_CR = 0.9;
int de_PS = 50;
int de_min_iter = 10;
int de_max_iter = 100;
double de_ct = 0.001;
#ifdef MFEM_USE_AMGX
bool useAmgX = false;
#endif

// Define the process for solving Maxwell equation. We will repeatedly call this function later
double simulation()
{
    if (myid == 0) cout << "Running loop at frequency: " << freq << endl;

    kappa = freq * M_PI;

    //    Enable hardware devices such as GPUs, and programming models such as
    //    CUDA, OCCA, RAJA and OpenMP based on command line options.
    Device device(device_config);
    if (myid == 0 || !de) {
        device.Print();
    }

    // 6. Read the (serial) mesh from the given mesh file on all processors.  We
    //    can handle triangular, quadrilateral, tetrahedral, hexahedral, surface
    //    and volume meshes with the same code.
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    dim = mesh->Dimension();
    int sdim = mesh->SpaceDimension();

    // 7. Refine the serial mesh on all processors to increase the resolution. In
    //    this example we do 'ref_levels' of uniform refinement. We choose
    //    'ref_levels' to be the largest number that gives a final mesh with no
    //    more than 1,000 elements.
    {
        int ref_levels = (int)floor(log(1000./mesh->GetNE())/log(2.)/dim);
        for (int l = 0; l < ref_levels; l++)
        {
            mesh->UniformRefinement();
        }
    }

    // 8. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    {
        int par_ref_levels = 2;
        for (int l = 0; l < par_ref_levels; l++)
        {
            pmesh->UniformRefinement();
        }
    }

    // 9. Define a parallel finite element space on the parallel mesh. Here we
    //    use the Nedelec finite elements of the specified order.
    FiniteElementCollection *fec = new ND_FECollection(order, dim);
    ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
    HYPRE_BigInt size = fespace->GlobalTrueVSize();
    if (!de && myid == 0)
    {
        cout << "Number of finite element unknowns: " << size << endl;
    }

    // 10. Determine the list of true (i.e. parallel conforming) essential
    //    boundary dofs. In this example, the boundary conditions are defined
    //    by marking all the boundary attributes from the mesh as essential
    //    (Dirichlet) and converting them to a list of true dofs.
    Array<int> ess_tdof_list;
    Array<int> ess_bdr;
    if (pmesh->bdr_attributes.Size())
    {
        ess_bdr.SetSize(pmesh->bdr_attributes.Max());
        ess_bdr = 1;
        fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // 11. Initiate ROM related variables
    int max_num_snapshots = 100;
    bool update_right_SV = false;
    bool isIncremental = false;
    const std::string saveBasisName = "basis" + curr_basis_identifier;
    std::string loadBasisName = "basis";

    // 11a. If using the greedy algorithm or DE, load the global greedy basis.
    if (build_database || use_database || de)
    {
        loadBasisName += "_greedy";
    }

    std::unique_ptr<const CAROM::Matrix> spatialbasis;
    CAROM::Options* options;
    CAROM::BasisGenerator *generator;
    int numRowRB, numColumnRB;
    StopWatch solveTimer, assembleTimer, mergeTimer;

    // 12. Set BasisGenerator if offline
    if (offline)
    {
        options = new CAROM::Options(fespace->GetTrueVSize(), max_num_snapshots,
                                     update_right_SV);
        if (myid == 0) cout << "Saving basis to: " << saveBasisName << endl;
        generator = new CAROM::BasisGenerator(*options, isIncremental, saveBasisName);
    }

    // 13. Set up the parallel linear form b(.) which corresponds to the
    //    right-hand side of the FEM linear system, which in this case is
    //    (f,phi_i) where f is given by the function f_exact and phi_i are the
    //    basis functions in the finite element fespace.
    assembleTimer.Start();
    VectorFunctionCoefficient f(sdim, f_exact);
    ParLinearForm *b = new ParLinearForm(fespace);
    b->AddDomainIntegrator(new VectorFEDomainLFIntegrator(f));
    b->Assemble();

    // 14. Define the solution vector x as a parallel finite element grid function
    //     corresponding to fespace. Initialize x by projecting the exact
    //     solution. Note that only values from the boundary edges will be used
    //     when eliminating the non-homogeneous boundary condition to modify the
    //     r.h.s. vector b.
    ParGridFunction x(fespace);
    VectorFunctionCoefficient E(sdim, E_exact);
    x.ProjectCoefficient(E);

    // 15. Set up the parallel bilinear form corresponding to the EM diffusion
    //     operator curl muinv curl + sigma I, by adding the curl-curl and the
    //     mass domain integrators.
    Coefficient *muinv = new ConstantCoefficient(1.0);
    Coefficient *sigma = new ConstantCoefficient(1.0);
    ParBilinearForm *a = new ParBilinearForm(fespace);
    if (pa) {
        a->SetAssemblyLevel(AssemblyLevel::PARTIAL);
    }
    a->AddDomainIntegrator(new CurlCurlIntegrator(*muinv));
    a->AddDomainIntegrator(new VectorFEMassIntegrator(*sigma));

    // 16. Assemble the parallel bilinear form and the corresponding linear
    //     system, applying any necessary transformations such as: parallel
    //     assembly, eliminating boundary conditions, applying conforming
    //     constraints for non-conforming AMR, static condensation, etc.
    if (static_cond) {
        a->EnableStaticCondensation();
    }
    a->Assemble();

    OperatorPtr A;
    Vector B, X;
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
    assembleTimer.Stop();

    // 17. Offline Phase: Solve the system AX=B using PCG with an AMS preconditioner.
    if(fom || offline)
    {
        // 18. Solve the full order linear system A X = B
        if (pa)
        {
#ifdef MFEM_USE_AMGX
            MatrixFreeAMS ams(*a, *A, *fespace, muinv, sigma, NULL, ess_bdr, useAmgX);
#else
            MatrixFreeAMS ams(*a, *A, *fespace, muinv, sigma, NULL, ess_bdr);
#endif
            CGSolver cg(MPI_COMM_WORLD);
            cg.SetRelTol(1e-12);
            cg.SetMaxIter(1000);
            cg.SetPrintLevel(1);
            cg.SetOperator(*A);
            cg.SetPreconditioner(ams);
            solveTimer.Start();
            cg.Mult(B, X);
            solveTimer.Stop();
        }
        else
        {
            if (myid == 0)
            {
                cout << "Size of linear system: "
                     << A.As<HypreParMatrix>()->GetGlobalNumRows() << endl;
            }

            ParFiniteElementSpace *prec_fespace =
                (a->StaticCondensationIsEnabled() ? a->SCParFESpace() : fespace);
            HypreAMS ams(*A.As<HypreParMatrix>(), prec_fespace);
            HyprePCG pcg(*A.As<HypreParMatrix>());
            pcg.SetTol(1e-12);
            pcg.SetMaxIter(500);
            pcg.SetPrintLevel(2);
            pcg.SetPreconditioner(ams);
            solveTimer.Start();
            pcg.Mult(B, X);
            solveTimer.Stop();
        }
        // 19. take and write snapshot for ROM
        if (offline)
        {
            bool addSample = generator->takeSample(X.GetData());
            generator->writeSnapshot();
            basisIdentifiers.push_back(saveBasisName);
            delete generator;
            delete options;
        }
    }

    // 20. The online phase
    if (online || de) {
        // 21. read the reduced basis
        assembleTimer.Start();
        CAROM::BasisReader reader(loadBasisName);
        spatialbasis = reader.getSpatialBasis();
        numRowRB = spatialbasis->numRows();
        numColumnRB = spatialbasis->numColumns();

        if (!de && myid == 0)
        {
            printf("spatial basis dimension is %d x %d\n", numRowRB, numColumnRB);
        }

        // 22. form inverse ROM operator
        CAROM::Matrix invReducedA(numColumnRB, numColumnRB, false);
        ComputeCtAB(*A.As<HypreParMatrix>(), *spatialbasis, *spatialbasis,
                    invReducedA);
        invReducedA.inverse();

        CAROM::Vector B_carom(B.GetData(), B.Size(), true, false);
        CAROM::Vector X_carom(X.GetData(), X.Size(), true, false);
        std::unique_ptr<CAROM::Vector> reducedRHS = spatialbasis->transposeMult(
                    B_carom);
        CAROM::Vector reducedSol(numColumnRB, false);
        assembleTimer.Stop();

        // 23. solve ROM
        solveTimer.Start();
        invReducedA.mult(*reducedRHS, reducedSol);
        solveTimer.Stop();

        // 24. reconstruct FOM state
        spatialbasis->mult(reducedSol, X_carom);
    }

    // 25. Recover the parallel grid function corresponding to X. This is the
    //     local finite element solution on each processor.
    a->RecoverFEMSolution(X, *b, x);

    // 26. Save the refined mesh and the solution in parallel. This output can
    //     be viewed later using GLVis: "glvis -np <np> -m Mesh -g Sol".
    if (fom || offline)
    {
        ostringstream mesh_name, sol_name;
        mesh_name << "Mesh" << curr_basis_identifier << myid;
        sol_name << "Sol" << curr_basis_identifier << myid;

        ofstream mesh_ofs(mesh_name.str().c_str());
        mesh_ofs.precision(precision);
        pmesh->Print(mesh_ofs);

        ofstream sol_ofs(sol_name.str().c_str());
        if (myid == 0) cout << "Saving solution to: " << sol_name.str() << endl;

        sol_ofs.precision(precision);
        for (int i = 0; i < X.Size(); ++i)
        {
            sol_ofs << X[i] << endl;
        }
    }

    // 27. Save data in the VisIt format.
    DataCollection *dc = NULL;
    if (visit)
    {
        if (offline) dc = new VisItDataCollection("de_maxwell_local", pmesh);
        else if (fom) dc = new VisItDataCollection("de_maxwell_local_fom", pmesh);
        else if (online) dc = new VisItDataCollection("de_maxwell_local_rom", pmesh);
        dc->SetPrecision(precision);
        dc->RegisterField("solution", &x);
        dc->Save();
        delete dc;
    }

    // 28. Send the solution by socket to a GLVis server.
    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        socketstream sol_sock(vishost, visport);
        sol_sock << "parallel " << num_procs << " " << myid << "\n";
        sol_sock.precision(8);
        sol_sock << "solution\n" << *pmesh << x << flush;
    }

    double curr_error = 0;

    // 29a. Calculate the relative error as commanded by the greedy algorithm.
    if (calc_rel_error && !de)
    {
        Vector true_solution(X.Size());
        ifstream solution_file;
        std::string solution_filename = "Sol" + curr_basis_identifier + to_string(myid);
        if (myid == 0) cout << "Comparing current run to solution at: "
                                << solution_filename << endl;
        solution_file.open(solution_filename);
        true_solution.Load(solution_file, X.Size());
        solution_file.close();
        Vector residual(X.Size());
        subtract(X, true_solution, residual);

        const double abs_error = sqrt(InnerProduct(MPI_COMM_WORLD, residual, residual));
        const double sol_norm = sqrt(InnerProduct(MPI_COMM_WORLD, true_solution,
                                     true_solution));

        curr_error = abs_error / sol_norm;
        if (myid == 0) cout << "The relative error is: " << curr_error << endl;

        greedy_sampler->setPointRelativeError(curr_error);

        if (myid == 0)
        {
            std::ofstream fout;
            fout.open("parameters.txt", std::ios::app);
            fout << freq << std::endl;
            fout.close();
        }
    }
    // 29b. Or calculate the error indicator as commanded by the greedy algorithm.
    else if (calc_err_indicator && !de)
    {
        Vector AX(X.Size());
        A.As<HypreParMatrix>()->Mult(X, AX);
        Vector residual(X.Size());
        subtract(B, AX, residual);

        curr_error = sqrt(InnerProduct(MPI_COMM_WORLD, residual, residual));
        if (myid == 0) cout << "The error indicator is: "
                                << curr_error << endl;
        greedy_sampler->setPointErrorIndicator(curr_error, 1);
    }
    // 29c. Or if not using the greedy algorithm, output the relative error
    //     the error indicator of the point. (just for extra information)
    else if (!build_database && online && !de)
    {
        Vector true_solution(X.Size());
        ifstream solution_file;
        std::string solution_filename = "Sol" + curr_basis_identifier + to_string(myid);
        if (myid == 0) cout << "Comparing current run to solution at: "
                                << solution_filename << endl;
        solution_file.open(solution_filename);
        true_solution.Load(solution_file, X.Size());
        solution_file.close();
        Vector residual(X.Size());
        subtract(X, true_solution, residual);

        const double abs_error = sqrt(InnerProduct(MPI_COMM_WORLD, residual, residual));
        const double sol_norm = sqrt(InnerProduct(MPI_COMM_WORLD, true_solution,
                                     true_solution));

        curr_error = abs_error / sol_norm;
        if (myid == 0) cout << "The relative error is: " << curr_error << endl;

        Vector AX(X.Size());
        A.As<HypreParMatrix>()->Mult(X, AX);
        subtract(B, AX, residual);
        curr_error = sqrt(InnerProduct(MPI_COMM_WORLD, residual, residual));
        if (myid == 0) cout << "The error indicator is: "
                                << curr_error << endl;
    }
    // 29d. If using DE, just calculate error. Prints less info than other options
    else if (de)
    {
        Vector true_solution(X.Size());
        ifstream solution_file;
        std::string solution_filename = "Sol" + curr_basis_identifier + to_string(myid);
        solution_file.open(solution_filename);
        true_solution.Load(solution_file, X.Size());
        solution_file.close();
        Vector residual(X.Size());
        subtract(X, true_solution, residual);

        const double abs_error = sqrt(InnerProduct(MPI_COMM_WORLD, residual, residual));
        const double sol_norm = sqrt(InnerProduct(MPI_COMM_WORLD, true_solution,
                                     true_solution));

        curr_error = abs_error / sol_norm;
        if (myid == 0) cout << "The relative error is: " << curr_error << endl;

    }

    // 30. If calculating the relative error, or after we sampled our first point,
    //     create a global ROM basis.
    if (calc_rel_error || (offline && basisIdentifiers.size() == 1))
    {
        mergeTimer.Start();
        std::unique_ptr<CAROM::BasisGenerator> basis_generator;
        options = new CAROM::Options(fespace->GetTrueVSize(), max_num_snapshots,
                                     update_right_SV);
        generator = new CAROM::BasisGenerator(*options, isIncremental, loadBasisName);
        for (int i = 0; i < basisIdentifiers.size(); ++i)
        {
            std::string snapshot_filename = basisIdentifiers[i] + "_snapshot";
            generator->loadSamples(snapshot_filename,"snapshot");
        }
        generator->endSamples(); // save the merged basis file
        mergeTimer.Stop();
        if (myid == 0)
        {
            printf("Elapsed time for merging and building ROM basis: %e second\n",
                   mergeTimer.RealTime());
        }
        delete generator;
        delete options;
    }

    // 31. print timing info
    if (myid == 0)
    {
        if((fom || offline) && !de)
        {
            printf("Elapsed time for assembling FOM: %e second\n",
                   assembleTimer.RealTime());
            printf("Elapsed time for solving FOM: %e second\n", solveTimer.RealTime());
        }
        if(online && !de)
        {
            printf("Elapsed time for assembling ROM: %e second\n",
                   assembleTimer.RealTime());
            printf("Elapsed time for solving ROM: %e second\n", solveTimer.RealTime());
        }
    }

    // 32. Free the used memory.
    delete a;
    delete sigma;
    delete muinv;
    delete b;
    delete fespace;
    delete fec;
    delete pmesh;

    return curr_error;
}

// Define class for relative difference function, used for DE algorithm
class RelativeDifferenceCostFunction : public CAROM::IOptimizable
{
public:
    RelativeDifferenceCostFunction(unsigned int dims) :
        m_dim(dims)
    {

    }
    double EvaluateCost(std::vector<double> & inputs) const override
    {
        freq = inputs[0];

        return simulation();
    }

    unsigned int NumberOfParameters() const override
    {
        return m_dim;
    }

    std::vector<Constraints> GetConstraints() const override
    {
        std::fstream fin("parameters.txt", std::ios_base::in);
        double curr_param;
        bool first_line = true;
        double min_freq = 0.0;
        double max_freq = 0.0;

        while (fin >> curr_param)
        {
            double curr_freq = curr_param;

            if (first_line)
            {
                min_freq = curr_freq;
                max_freq = curr_freq;
                first_line = false;
            }
            else
            {
                min_freq = min(curr_freq, min_freq);
                max_freq = max(curr_freq, max_freq);
            }
        }
        fin.close();

        if (de_min_freq != -DBL_MAX) min_freq = de_min_freq;
        if (de_max_freq != DBL_MAX) max_freq = de_max_freq;

        MFEM_VERIFY(min_freq <= max_freq, "freq DE range is invalid.");

        if (myid == 0) cout << "DE freq range is: " << min_freq << " to " <<
                                max_freq << endl;

        std::vector<Constraints> constr(NumberOfParameters());
        constr[0] = Constraints(min_freq, max_freq, true);
        return constr;
    }

private:
    unsigned int m_dim;
};


int main(int argc, char *argv[])
{

    // 1. Initialize MPI and HYPRE
    Mpi::Init(argc, argv);
    num_procs = Mpi::WorldSize();
    myid = Mpi::WorldRank();
    Hypre::Init();

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                   "Finite element order (polynomial degree).");
    args.AddOption(&freq, "-f", "--frequency", "Set the frequency for the exact"
                   " solution.");
    args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                   "--no-static-condensation", "Enable static condensation.");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                   "--no-partial-assembly", "Enable Partial Assembly.");
    args.AddOption(&device_config, "-d", "--device",
                   "Device configuration string, see Device::Configure().");
    args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                   "--no-visit-datafiles",
                   "Save data files for VisIt (visit.llnl.gov) visualization.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
#ifdef MFEM_USE_AMGX
    args.AddOption(&useAmgX, "-amgx", "--useAmgX", "-no-amgx",
                   "--no-useAmgX",
                   "Enable or disable AmgX in MatrixFreeAMS.");
#endif
    args.AddOption(&fom, "-fom", "--fom", "-no-fom", "--no-fom",
                   "Enable or disable the fom phase.");
    args.AddOption(&offline, "-offline", "--offline", "-no-offline","--no-offline",
                   "Enable or disable the offline phase.");
    args.AddOption(&online, "-online", "--online", "-no-online", "--no-online",
                   "Enable or disable the online phase.");
    args.AddOption(&build_database, "-build_database", "--build_database",
                   "-no-build_database", "--no-build_database",
                   "Enable or disable the build_database phase of the greedy algorithm.");
    args.AddOption(&use_database, "-use_database", "--use_database",
                   "-no-use_database", "--no-use_database",
                   "Enable or disable the use_database phase of the greedy algorithm.");
    args.AddOption(&greedy_param_space_min, "-greedy-param-min",
                   "--greedy-param-min", "The minimum value of the parameter point space.");
    args.AddOption(&greedy_param_space_max, "-greedy-param-max",
                   "--greedy-param-max", "The maximum value of the parameter point space.");
    args.AddOption(&greedy_param_space_size, "-greedy-param-size",
                   "--greedy-param-size",
                   "The number of values to search in the parameter point space.");
    args.AddOption(&greedy_relative_error_tol, "-greedyrelerrortol",
                   "--greedyrelerrortol", "The greedy algorithm relative error tolerance.");
    args.AddOption(&greedy_subset_size, "-greedysubsize", "--greedysubsize",
                   "The greedy algorithm subset size.");
    args.AddOption(&greedy_convergence_subset_size, "-greedyconvsize",
                   "--greedyconvsize", "The greedy algorithm convergence subset size.");
    args.AddOption(&de_min_freq, "-de_min_freq", "--de_min_frequency",
                   "DE min freq.");
    args.AddOption(&de_max_freq, "-de_max_freq", "--de_max_frequency",
                   "DE max freq.");
    args.AddOption(&de_F, "-de_f", "--de_f",
                   "DE F.");
    args.AddOption(&de_CR, "-de_cr", "--de_cr",
                   "DE CR.");
    args.AddOption(&de_PS, "-de_ps", "--de_ps",
                   "DE Population size.");
    args.AddOption(&de_min_iter, "-de_min_iter", "--de_min_iter",
                   "DE Minimum number of iterations.");
    args.AddOption(&de_max_iter, "-de_max_iter", "--de_max_iter",
                   "DE Maximum number of iterations.");
    args.AddOption(&de_ct, "-de_ct", "--de_ct",
                   "DE Cost threshold.");
    args.AddOption(&de, "-de", "--de", "-no-de", "--no-de",
                   "Enable or disable the differential evolution phase.");

    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }

    MFEM_VERIFY(!(build_database
                  && de), "both build_database and de can not be true!");

    MFEM_VERIFY(!build_database
                || !use_database,
                "both build_database and use_database can not be used at the same time.");

    if (de)
    {
        calc_err_indicator = false;
        calc_rel_error = false;
        online = true;
        build_database = false;
        use_database = true;
    }

    // 3a. Set up the ROM database for the greedy algorithm to run.
    if (build_database)
    {
        MFEM_VERIFY(!offline
                    && !online,
                    "offline and online must be turned off during the build_database phase.");
        MFEM_VERIFY(!visit
                    && !visualization,
                    "visit and visualization must be turned off during the build_database phase.")
        std::ifstream infile("de_parametric_maxwell_greedy_algorithm_data");
        if (infile.good())
        {
            if (myid == 0) cout << "The database has already been built."
                                    << "Exiting." << endl;
            return 0;
        }
        infile.close();
        greedy_sampler = new CAROM::GreedyRandomSampler(greedy_param_space_min,
                greedy_param_space_max,
                greedy_param_space_size, false, greedy_relative_error_tol, 1.05,
                2.0, greedy_subset_size, greedy_convergence_subset_size,
                true, "de_parametric_maxwell_greedy_log.txt");
    }

    // 3b. Or use the database set up by the greedy algorithm.
    else if (use_database)
    {
        MFEM_VERIFY(!offline
                    && online,
                    "offline must be turned off and online must be turned on during the build_database phase.");
        std::ifstream infile("de_parametric_maxwell_greedy_algorithm_data");
        if (!infile.good())
        {
            if (myid == 0) cout << "The database has not been built. Exiting."
                                    << endl;
            return 0;
        }
        infile.close();
    }

    if (build_database)
    {
        // The simulation is wrapped in a do-while statement so that the greedy
        // algorithm (build_database) can run multiple simulations in succession.
        do
        {

            offline = false;
            online = false;

            calc_rel_error = false;
            calc_err_indicator = false;
            curr_basis_identifier = "";

            // 4a. Set the correct stage of the greedy algorithm (i.e. sampling new point,
            //    calculating relative error of the last sampled point, or calculating
            //    an error indicator at a new point).

            local_rom_freq = 0.0;
            curr_freq = 0.0;
            struct CAROM::GreedyErrorIndicatorPoint pointRequiringRelativeError =
                greedy_sampler->getNextPointRequiringRelativeError();
            CAROM::Vector* relativeErrorPointData = pointRequiringRelativeError.point.get();
            struct CAROM::GreedyErrorIndicatorPoint pointRequiringErrorIndicator =
                greedy_sampler->getNextPointRequiringErrorIndicator();
            CAROM::Vector* errorIndicatorPointData =
                pointRequiringErrorIndicator.point.get();

            if (relativeErrorPointData != NULL)
            {
                if (myid == 0) cout << "Calculating the relative error of "
                                        << "the last sampled point." << endl;

                local_rom_freq = pointRequiringRelativeError.localROM.get()->item(0);
                curr_freq = pointRequiringRelativeError.point.get()->item(0);
                if (myid == 0) cout << "Using the basis obtained at the frequency: " <<
                                        local_rom_freq << endl;
                online = true;
                calc_rel_error = true;
            }
            else if (errorIndicatorPointData != NULL)
            {
                if (myid == 0) cout << "Calculating an error indicator at "
                                        << "a new point." << endl;

                local_rom_freq = pointRequiringErrorIndicator.localROM.get()->item(0);
                curr_freq = pointRequiringErrorIndicator.point.get()->item(0);
                if (myid == 0) cout << "Using the basis obtained at the frequency: "
                                        << local_rom_freq << endl;
                online = true;
                calc_err_indicator = true;
            }
            else
            {
                std::shared_ptr<CAROM::Vector> nextSampleParameterPoint =
                    greedy_sampler->getNextParameterPoint();
                CAROM::Vector* samplePointData = nextSampleParameterPoint.get();
                if (samplePointData != NULL)
                {
                    if (myid == 0) cout << "Sampling a new point." << endl;
                    local_rom_freq = samplePointData->item(0);
                    curr_freq = samplePointData->item(0);
                    offline = true;
                }
                else
                {
                    if (myid == 0) cout << "The greedy algorithm has finished." << endl;
                    greedy_sampler->save("de_parametric_maxwell_greedy_algorithm_data");
                    build_database = false;
                    continue;
                }
            }

            // 4b. Set the correct frequency as commanded by the greedy algorithm.
            // Solve Maxwell equation using simulation() function
            curr_basis_identifier += "_" + to_string(curr_freq);
            freq = curr_freq;

            simulation();
        } while(build_database);

    }
    else if (fom)
    {
        simulation();
    }
    // 5.  If in differential evolution mode, run the DE algorithm using the
    //     greedy database.
    else if (de)
    {
        target_freq = freq;
        std::ifstream infile("de_parametric_maxwell_greedy_algorithm_data");
        if (!infile.good())
        {
            if (myid == 0) std::cout << "The database has not been built. Exiting." <<
                                         std::endl;
            return 0;
        }
        infile.close();

        //  Create 1D relative error cost function for kappa
        RelativeDifferenceCostFunction cost(1);

        // Create Differential Evolution optimizer with population size of de_ps
        // differential weight of de_F, and crossover probability of de_CR
        CAROM::DifferentialEvolution de_opt(cost, de_PS, de_F, de_CR);

        // Optimize for at least de_min_iter iterations, to a maximum of de_max_iter iterations with verbose output.
        // Stop early, after de_min_iter iterations, if the minimum cost did not improve by de_ct
        std::vector<double> optimal_parameters = de_opt.Optimize(de_min_iter,
                de_max_iter, de_ct, true);

        freq = optimal_parameters[0];

        online = true;
        de = false;
        simulation();
    }

    delete greedy_sampler;

    MPI_Finalize();

    return 0;
}

// 33. define Exact E
void E_exact(const Vector &x, Vector &E)
{
    if (dim == 3)
    {
        E(0) = sin(kappa * x(1));
        E(1) = sin(kappa * x(2));
        E(2) = sin(kappa * x(0));
    }
    else
    {
        E(0) = sin(kappa * x(1));
        E(1) = sin(kappa * x(0));
        if (x.Size() == 3) {
            E(2) = 0.0;
        }
    }
}

// 34. define spatially varying RHS function
void f_exact(const Vector &x, Vector &f)
{
    if (dim == 3)
    {
        f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
        f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
        f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
    }
    else
    {
        f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
        f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
        if (x.Size() == 3) {
            f(2) = 0.0;
        }
    }
}

