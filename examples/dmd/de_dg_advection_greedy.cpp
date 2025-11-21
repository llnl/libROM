/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

//                       libROM MFEM Example: DE DG Advection Greedy (adapted from ex9p.cpp)
//
// Compile with: make de_dg_advection_greedy
//
// =================================================================================
//
// Sample runs and results for DMD (It is possible to run these in parallel,
//   though you should use the same number of ranks for each subsequent run.):
//
// Command 1:  Build DMD database
//   de_dg_advection_greedy -p 3 -rp 1 -dt 0.005 -tf 1.0 -build_database -rdim 16
//   -greedyreldifftol 0.00000001 -greedy-param-f-factor-max 2. -greedy-param-f-factor-min 1.
//   -greedy-param-size 20 -greedysubsize 5 -greedyconvsize 8
//
// Output 1:
//   The greedy algorithm has finished.
//
// Command 2: Forward run at target f-factor
//   de_dg_advection_greedy -p 3 -rp 1 -dt 0.005 -tf 1.0 -run_dmd -ff 1.6
//
// Output 2:
//   Relative error of DMD temperature (u) at t_final: 1 is 0.0006565966583426298
//
// Command 3: Run differential evolution search for target f-factor
//   de_dg_advection_greedy -p 3 -rp 1 -dt 0.005 -tf 1.0 -de -ff 1.6 -de_min_ff 1.0
//   -de_max_ff 2.0 -de_f 0.9 -de_cr 0.9 -de_ps 50 -de_min_iter 1 -de_max_iter 100
//   -de_ct 0.001
//
// Output 3:
//  Iteration: 1            Current minimal cost: 0.004666763171916453              Best agent: 1.597618121565086
//  Terminated due to cost tolerance condition being met
// =================================================================================
//
// Description:  This example code solves the time-dependent advection equation
//               du/dt + v.grad(u) = 0, where v is a given fluid velocity, and
//               u0(x)=u(0,x) is a given initial condition.
//
//               This problem is parameterized by its initial condition.  Specifically,
//               The initial condition is given as:
//                      u0 = sin(pi*f_factor*X(0))*sin(pi*f_factor*X(1));
//
//               The example demonstrates the use of Discontinuous Galerkin (DG)
//               bilinear forms in MFEM (face integrators), the use of implicit
//               and explicit ODE time integrators, the definition of periodic
//               boundary conditions through periodic meshes, as well as the use
//               of GLVis for persistent visualization of a time-evolving
//               solution. Saving of time-dependent data files for visualization
//               with VisIt (visit.llnl.gov) and ParaView (paraview.org), as
//               well as the optional saving with ADIOS2 (adios2.readthedocs.io)
//               are also illustrated.

#include "mfem.hpp"
#include "algo/DMD.h"
#include "algo/greedy/GreedyRandomSampler.h"
#include "algo/DifferentialEvolution.h"
#include "linalg/Vector.h"
#include <cmath>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include "utils/CSVDatabase.h"


using std::vector;
using std::cout;
using std::endl;
using std::flush;
using std::max;
using std::min;
using std::to_string;
using std::ifstream;
using std::ofstream;
using std::ostringstream;
using std::setfill;
using std::setw;
using namespace mfem;

// Choice for the problem setup. The fluid velocity, initial condition and
// inflow boundary condition are chosen based on this parameter.
int problem = 3;

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v);

// Initial condition
double u0_function(const Vector &x);

// Inflow boundary condition
double inflow_function(const Vector &x);

// Mesh bounding box
Vector bb_min, bb_max;

// Type of preconditioner for implicit time integrator
enum class PrecType : int
{
    ILU = 0,
    AIR = 1
};

#if MFEM_HYPRE_VERSION >= 21800
// Algebraic multigrid preconditioner for advective problems based on
// approximate ideal restriction (AIR). Most effective when matrix is
// first scaled by DG block inverse, and AIR applied to scaled matrix.
// See https://doi.org/10.1137/17M1144350.
class AIR_prec : public Solver
{
private:
    const HypreParMatrix *A;
    // Copy of A scaled by block-diagonal inverse
    HypreParMatrix A_s;

    HypreBoomerAMG *AIR_solver;
    int blocksize;

public:
    AIR_prec(int blocksize_) : AIR_solver(NULL), blocksize(blocksize_) { }

    void SetOperator(const Operator &op)
    {
        width = op.Width();
        height = op.Height();

        A = dynamic_cast<const HypreParMatrix *>(&op);
        MFEM_VERIFY(A != NULL, "AIR_prec requires a HypreParMatrix.")

        // Scale A by block-diagonal inverse
        BlockInverseScale(A, &A_s, NULL, NULL, blocksize,
                          BlockInverseScaleJob::MATRIX_ONLY);
        delete AIR_solver;
        AIR_solver = new HypreBoomerAMG(A_s);
        AIR_solver->SetAdvectiveOptions(1, "", "FA");
        AIR_solver->SetPrintLevel(0);
        AIR_solver->SetMaxLevels(50);
    }

    virtual void Mult(const Vector &x, Vector &y) const
    {
        // Scale the rhs by block inverse and solve system
        HypreParVector z_s;
        BlockInverseScale(A, NULL, &x, &z_s, blocksize,
                          BlockInverseScaleJob::RHS_ONLY);
        AIR_solver->Mult(z_s, y);
    }

    ~AIR_prec()
    {
        delete AIR_solver;
    }
};
#endif

class DG_Solver : public Solver
{
private:
    HypreParMatrix &M, &K;
    SparseMatrix M_diag;
    HypreParMatrix *A;
    GMRESSolver linear_solver;
    Solver *prec;
    double dt;
public:
    DG_Solver(HypreParMatrix &M_, HypreParMatrix &K_, const FiniteElementSpace &fes,
              PrecType prec_type)
        : M(M_),
          K(K_),
          A(NULL),
          linear_solver(M.GetComm()),
          dt(-1.0)
    {
        int block_size = fes.GetFE(0)->GetDof();
        if (prec_type == PrecType::ILU)
        {
            prec = new BlockILU(block_size,
                                BlockILU::Reordering::MINIMUM_DISCARDED_FILL);
        }
        else if (prec_type == PrecType::AIR)
        {
#if MFEM_HYPRE_VERSION >= 21800
            prec = new AIR_prec(block_size);
#else
            MFEM_ABORT("Must have MFEM_HYPRE_VERSION >= 21800 to use AIR.\n");
#endif
        }
        linear_solver.iterative_mode = false;
        linear_solver.SetRelTol(1e-9);
        linear_solver.SetAbsTol(0.0);
        linear_solver.SetMaxIter(100);
        linear_solver.SetPrintLevel(0);
        linear_solver.SetPreconditioner(*prec);

        M.GetDiag(M_diag);
    }

    void SetTimeStep(double dt_)
    {
        if (dt_ != dt)
        {
            dt = dt_;
            // Form operator A = M - dt*K
            delete A;
            A = Add(-dt, K, 0.0, K);
            SparseMatrix A_diag;
            A->GetDiag(A_diag);
            A_diag.Add(1.0, M_diag);
            // this will also call SetOperator on the preconditioner
            linear_solver.SetOperator(*A);
        }
    }

    void SetOperator(const Operator &op)
    {
        linear_solver.SetOperator(op);
    }

    virtual void Mult(const Vector &x, Vector &y) const
    {
        linear_solver.Mult(x, y);
    }

    ~DG_Solver()
    {
        delete prec;
        delete A;
    }
};

/** A time-dependent operator for the right-hand side of the ODE. The DG weak
    form of du/dt = -v.grad(u) is M du/dt = K u + b, where M and K are the mass
    and advection matrices, and b describes the flow on the boundary. This can
    be written as a general ODE, du/dt = M^{-1} (K u + b), and this class is
    used to evaluate the right-hand side. */
class FE_Evolution : public TimeDependentOperator
{
private:
    OperatorHandle M, K;
    const Vector &b;
    Solver *M_prec;
    CGSolver M_solver;
    DG_Solver *dg_solver;

    mutable Vector z;

public:
    FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_, const Vector &b_,
                 PrecType prec_type);

    virtual void Mult(const Vector &x, Vector &y) const;
    virtual void ImplicitSolve(const double dt, const Vector &x, Vector &k);

    virtual ~FE_Evolution();
};

//MFEM variables
int num_procs, myid;
const char *mesh_file = "../data/periodic-hexagon.mesh";
int ser_ref_levels = 2;
int par_ref_levels = 0;
int order = 3;
bool pa = false;
bool ea = false;
bool fa = false;
const char *device_config = "cpu";
int ode_solver_type = 4;
double t_final = 10.0;
double dt = 0.01;
double ef = 0.9999;
bool visualization = false;
bool visit = false;
bool paraview = false;
bool adios2 = false;
bool binary = false;
int vis_steps = 5;
// libROM DMD variables
int rdim = -1;
bool run_dmd = false;
const char *temp_io_dir = "./de_advection_greedy";
std::string io_dir;
// libROM parameterization variables
double f_factor = 1.;
// libROM greedy variables
double closest_rbf_val = 0.9;
bool build_database = false;
bool calc_err_indicator = false;
double greedy_param_space_f_factor_min = 0.1;
double greedy_param_space_f_factor_max = 5.0;
int greedy_param_space_size = 8;
double greedy_relative_diff_tol = 0.01;
int greedy_subset_size = 2;
int greedy_convergence_subset_size = 3;
bool online = false;
bool offline = false;
double target_f_factor = f_factor; // May not need this yet.
int precision = 16;
// Differential evolution variables.
double de_min_f_factor = -DBL_MAX;
double de_max_f_factor = DBL_MAX;
double de_F = 0.8;
double de_CR = 0.9;
int de_PS = 50;
int de_min_iter = 10;
int de_max_iter = 100;
double de_ct = 0.001;
bool de = false;


CAROM::GreedySampler* greedy_sampler = NULL;
Vector* true_solution_u = NULL;
double tot_true_solution_u_norm = 0.0;

#if MFEM_HYPRE_VERSION >= 21800
PrecType prec_type = PrecType::AIR;
#else
PrecType prec_type = PrecType::ILU;
#endif


double simulation()
{
    // 5. Read the serial mesh from the given mesh file on all processors. We can
    //    handle geometrically periodic meshes in this code.
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // 6. Define the ODE solver used for time integration. Several explicit
    //    Runge-Kutta methods are available.
    std::unique_ptr<ODESolver> ode_solver;
    switch (ode_solver_type)
    {
    // Explicit methods
    case 1:
        ode_solver.reset(new ForwardEulerSolver);
        break;
    case 2:
        ode_solver.reset(new RK2Solver(1.0));
        break;
    case 3:
        ode_solver.reset(new RK3SSPSolver);
        break;
    case 4:
        ode_solver.reset(new RK4Solver);
        break;
    case 6:
        ode_solver.reset(new RK6Solver);
        break;
    // Implicit (L-stable) methods
    case 11:
        ode_solver.reset(new BackwardEulerSolver);
        break;
    case 12:
        ode_solver.reset(new SDIRK23Solver(2));
        break;
    case 13:
        ode_solver.reset(new SDIRK33Solver);
        break;
    // Implicit A-stable methods (not L-stable)
    case 22:
        ode_solver.reset(new ImplicitMidpointSolver);
        break;
    case 23:
        ode_solver.reset(new SDIRK23Solver);
        break;
    case 24:
        ode_solver.reset(new SDIRK34Solver);
        break;
    default:
        if (myid == 0)
        {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
        }
        delete mesh;
        return 3;
    }

    // 7. Refine the mesh in serial to increase the resolution. In this example
    //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
    //    a command-line parameter. If the mesh is of NURBS type, we convert it
    //    to a (piecewise-polynomial) high-order mesh.
    for (int lev = 0; lev < ser_ref_levels; lev++)
    {
        mesh->UniformRefinement();
    }
    if (mesh->NURBSext)
    {
        mesh->SetCurvature(max(order, 1));
    }
    mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));

    // 8. Define the parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh pmesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    for (int lev = 0; lev < par_ref_levels; lev++)
    {
        pmesh.UniformRefinement();
    }

    // 9. Define the parallel discontinuous DG finite element space on the
    //    parallel refined mesh of the given polynomial order.
    DG_FECollection fec(order, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace fes(&pmesh, &fec);

    HYPRE_BigInt global_vSize = fes.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of unknowns: " << global_vSize << endl;
    }

    // 10. Set up and assemble the parallel bilinear and linear forms (and the
    //    parallel hypre matrices) corresponding to the DG discretization. The
    //    DGTraceIntegrator involves integrals over mesh interior faces.
    VectorFunctionCoefficient velocity(dim, velocity_function);
    FunctionCoefficient inflow(inflow_function);
    FunctionCoefficient u0(u0_function);

    ParBilinearForm m(&fes);
    ParBilinearForm k(&fes);
    if (pa)
    {
        m.SetAssemblyLevel(AssemblyLevel::PARTIAL);
        k.SetAssemblyLevel(AssemblyLevel::PARTIAL);
    }
    else if (ea)
    {
        m.SetAssemblyLevel(AssemblyLevel::ELEMENT);
        k.SetAssemblyLevel(AssemblyLevel::ELEMENT);
    }
    else if (fa)
    {
        m.SetAssemblyLevel(AssemblyLevel::FULL);
        k.SetAssemblyLevel(AssemblyLevel::FULL);
    }

    m.AddDomainIntegrator(new MassIntegrator);
    constexpr double alpha = -1.0;
    k.AddDomainIntegrator(new ConvectionIntegrator(velocity, alpha));
    k.AddInteriorFaceIntegrator(
        new NonconservativeDGTraceIntegrator(velocity, alpha));
    k.AddBdrFaceIntegrator(
        new NonconservativeDGTraceIntegrator(velocity, alpha));

    ParLinearForm b(&fes);
    b.AddBdrFaceIntegrator(
        new BoundaryFlowIntegrator(inflow, velocity, alpha));

    int skip_zeros = 0;
    m.Assemble();
    k.Assemble(skip_zeros);
    b.Assemble();
    m.Finalize();
    k.Finalize(skip_zeros);

    std::unique_ptr<HypreParVector> B(b.ParallelAssemble());

    // 12. Define the initial conditions, save the corresponding grid function to
    //    a file and (optionally) save data in the VisIt format and initialize
    //    GLVis visualization.
    ParGridFunction u_gf(&fes);
    u_gf.ProjectCoefficient(u0);
    std::unique_ptr<HypreParVector> U(u_gf.GetTrueDofs());

    if (!online)
    {
        ostringstream mesh_name, sol_name;
        mesh_name << io_dir << "/dg_advection-mesh." << setfill('0') << setw(
                      6) << myid;
        sol_name << io_dir << "/dg_advection-init." << setfill('0') << setw(
                     6) << myid;
        ofstream omesh(mesh_name.str().c_str());
        omesh.precision(precision);
        pmesh.Print(omesh);
        ofstream osol(sol_name.str().c_str());
        osol.precision(precision);
        u_gf.Save(osol);
    }

    // Create data collection for solution output: either VisItDataCollection for
    // ascii data files, or SidreDataCollection for binary data files.
    DataCollection *dc = NULL;
    if (visit)
    {
        if (binary)
        {
#ifdef MFEM_USE_SIDRE
            dc = new SidreDataCollection("DG_Advection", &pmesh);
#else
            MFEM_ABORT("Must build with MFEM_USE_SIDRE=YES for binary output.");
#endif
        }
        else
        {
            dc = new VisItDataCollection("DG_Advection", &pmesh);
            dc->SetPrecision(precision);
            // To save the mesh using MFEM's parallel mesh format:
            // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
        }
        dc->RegisterField("solution", &u_gf);
        dc->SetCycle(0);
        dc->SetTime(0.0);
        dc->Save();
    }

    ParaViewDataCollection *pd = NULL;
    if (paraview)
    {
        pd = new ParaViewDataCollection("DG_Advection", &pmesh);
        pd->SetPrefixPath("ParaView");
        pd->RegisterField("solution", &u_gf);
        pd->SetLevelsOfDetail(order);
        pd->SetDataFormat(VTKFormat::BINARY);
        pd->SetHighOrderOutput(true);
        pd->SetCycle(0);
        pd->SetTime(0.0);
        pd->Save();
    }

    // Optionally output a BP (binary pack) file using ADIOS2. This can be
    // visualized with the ParaView VTX reader.
#ifdef MFEM_USE_ADIOS2
    ADIOS2DataCollection *adios2_dc = NULL;
    if (adios2)
    {
        std::string postfix(mesh_file);
        postfix.erase(0, std::string("../data/").size() );
        postfix += "_o" + std::to_string(order);
        const std::string collection_name = "dg_advection-p-" + postfix + ".bp";

        adios2_dc = new ADIOS2DataCollection(MPI_COMM_WORLD, collection_name, &pmesh);
        // output data substreams are half the number of mpi processes
        adios2_dc->SetParameter("SubStreams", std::to_string(num_procs/2) );
        // adios2_dc->SetLevelsOfDetail(2);
        adios2_dc->RegisterField("solution", &u_gf);
        adios2_dc->SetCycle(0);
        adios2_dc->SetTime(0.0);
        adios2_dc->Save();
    }
#endif

    socketstream sout;
    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        sout.open(vishost, visport);
        if (!sout)
        {
            if (myid == 0)
                cout << "Unable to connect to GLVis server at "
                     << vishost << ':' << visport << endl;
            visualization = false;
            if (myid == 0)
            {
                cout << "GLVis visualization disabled.\n";
            }
        }
        else
        {
            sout << "parallel " << num_procs << " " << myid << "\n";
            sout.precision(precision);
            sout << "solution\n" << pmesh << u_gf;
            sout << flush;
        }
    }

    // 13. Define the time-dependent evolution operator describing the ODE
    //     right-hand side, and perform time-integration (looping over the time
    //     iterations, ti, with a time-step dt).
    FE_Evolution adv(m, k, *B, prec_type);

    StopWatch fom_timer, dmd_training_timer, dmd_prediction_timer;

    fom_timer.Start();

    double t = 0.0;
    vector<double> ts;
    adv.SetTime(t);
    ode_solver->Init(adv);
    std::unique_ptr<CAROM::Vector> init;
    CAROM::CSVDatabase csv_db;

    fom_timer.Stop();

    std::unique_ptr<CAROM::DMD> dmd_U;

    if (de)
    {
        u_gf.SetFromTrueDofs(*U);
        init.reset(new CAROM::Vector(U->GetData(), U->Size(), true));

        ts.push_back(t);

        double rel_diff = 0.;

        //18. Create the parametric DMD.
        std::fstream fin(io_dir + "/parameters.txt", std::ios_base::in);
        double curr_param;
        std::vector<std::string> dmd_paths;
        std::vector<CAROM::Vector> param_vectors;

        while (fin >> curr_param)
        {
            double curr_f_factor = curr_param;

            // INPUT / OUTPUT potential issue
            dmd_paths.push_back(io_dir + "/" + to_string(curr_f_factor));
            CAROM::Vector param_vector(1, false);
            param_vector(0) = curr_f_factor;
            param_vectors.push_back(param_vector);
        }
        fin.close();

        CAROM::Vector desired_param(1, false);
        desired_param(0) = f_factor;

        dmd_training_timer.Start();

        CAROM::getParametricDMD(dmd_U, param_vectors, dmd_paths, desired_param,
                                "G", "LS", closest_rbf_val);

        dmd_U->projectInitialCondition(*init);

        dmd_training_timer.Stop();

        // Compare the DMD solution to the FOM Solution
        if (true_solution_u == NULL)
        {
            ifstream solution_file;
            ostringstream sol_name;
            ostringstream target_name;
            sol_name << io_dir << "/dg_advection_greedy" << to_string(
                         target_f_factor) << "-final." << setfill('0') << setw(6) << myid;

            solution_file.open(sol_name.str().c_str());

            true_solution_u = new Vector(U->Size());
            true_solution_u->Load(solution_file, U->Size());
            solution_file.close();
            tot_true_solution_u_norm = sqrt(InnerProduct(MPI_COMM_WORLD,
                                            *true_solution_u, *true_solution_u));
        }

        dmd_prediction_timer.Start();
        std::shared_ptr<CAROM::Vector> result_U = dmd_U->predict(t_final);
        dmd_prediction_timer.Stop();

        // 21. Calculate the relative error between the DMD final solution and the true solution.
        Vector dmd_solution_U(result_U->getData(),
                              result_U->dim());
        Vector diff_u(true_solution_u->Size());
        subtract(dmd_solution_U, *true_solution_u, diff_u);

        double tot_diff_norm_u = sqrt(InnerProduct(MPI_COMM_WORLD, diff_u, diff_u));
        double tot_true_solution_u_norm = sqrt(InnerProduct(MPI_COMM_WORLD,
                                               *true_solution_u, *true_solution_u));

        rel_diff = tot_diff_norm_u / tot_true_solution_u_norm;

        if (myid == 0)
        {
            std::cout << "Relative error of DMD temperature (u) at t_final: "
                      << t_final << " and f_factor: " << f_factor << " is " << rel_diff << std::endl;
            printf("Elapsed time for predicting DMD: %e second\n",
                   dmd_prediction_timer.RealTime());
        }

        return rel_diff;
    }

    // 14. If in offline mode, create DMD object and take initial sample.
    if(offline)
    {
        dmd_training_timer.Start();

        dmd_U.reset(new CAROM::DMD(U->Size(), dt));
        dmd_U->takeSample(U->GetData(), t);
        ts.push_back(t);
        if (myid == 0)
        {
            std::cout << "Taking snapshot at: " << t << std::endl;
        }

        dmd_training_timer.Stop();
    }

    // 15. If in online mode, save the initial vector.
    // 15. If in calc_err_indicator mode, load the current DMD database,
    //     and create a parametric DMD object at the current set of parameters.
    u_gf.SetFromTrueDofs(*U);
    init.reset(new CAROM::Vector(U->GetData(), U->Size(), true));

    if (calc_err_indicator)
    {
        std::fstream fin(io_dir+"/parameters.txt", std::ios_base::in);
        double curr_param;
        std::vector<std::string> dmd_paths;
        std::vector<CAROM::Vector> param_vectors;

        while (fin >> curr_param)
        {
            double curr_f_factor = curr_param;
            dmd_paths.push_back(io_dir + "/" + to_string(curr_f_factor));
            CAROM::Vector param_vector(1, false);
            param_vector(0) = curr_f_factor;
            param_vectors.push_back(param_vector);
        }
        fin.close();
        if (dmd_paths.size() > 1)
        {
            CAROM::Vector desired_param(1, false);
            desired_param(0) = f_factor;

            dmd_training_timer.Start();

            CAROM::getParametricDMD(dmd_U, param_vectors, dmd_paths, desired_param,
                                    "G", "LS", closest_rbf_val);
        }
        else
        {
            dmd_U.reset(new CAROM::DMD(dmd_paths[0]));
        }

        dmd_U->projectInitialCondition(*init);

        dmd_training_timer.Stop();

        // For the error indicator, load in the DMD predicted solution 10
        // steps before t_final, then run the FOM for the last 10 steps and
        // compare the final FOM solution to the DMD predicted solutioni

        // THIS IS LIKE COMPUTING A RESIDUAL.
        t = t_final - 10.*dt;

        std::shared_ptr<CAROM::Vector> carom_tf_u_minus_some = dmd_U->predict(t);

        for(int i = 0; i < carom_tf_u_minus_some->dim(); i++)
        {
            (*U)[i] = (*carom_tf_u_minus_some)(i);
        }

        u_gf.SetFromTrueDofs(*U);
    }

    ts.push_back(t);

    // 16. Iterate through the time loop.
    bool done = false;
    for (int ti = 0; !done; )
    {
        fom_timer.Start();

        double dt_real = min(dt, t_final - t);
        ode_solver->Step(*U, t, dt_real);
        ti++;
        done = (t >= t_final - 1e-8*dt);

        fom_timer.Stop();

        if (offline)
        {
            dmd_training_timer.Start();

            u_gf.SetFromTrueDofs(*U);
            dmd_U->takeSample(U->GetData(),t);

            if (myid == 0 && ti % vis_steps == 0)
            {
                std::cout << "Taking snapshot at: " << t << std::endl;
            }

            dmd_training_timer.Stop();
        }

        ts.push_back(t);

        if (done || ti % vis_steps == 0)
        {
            if (myid == 0)
            {
                cout << "time step: " << ti << ", time: " << t << endl;
            }

            // 11. Extract the parallel grid function corresponding to the finite
            //     element approximation U (the local solution on each processor).

            u_gf = *U;

            if (visualization)
            {
                sout << "parallel " << num_procs << " " << myid << "\n";
                sout << "solution\n" << pmesh << u_gf << flush;
            }

            if (visit)
            {
                dc->SetCycle(ti);
                dc->SetTime(t);
                dc->Save();
            }

            if (paraview)
            {
                pd->SetCycle(ti);
                pd->SetTime(t);
                pd->Save();
            }

#ifdef MFEM_USE_ADIOS2
            // transient solutions can be visualized with ParaView
            if (adios2)
            {
                adios2_dc->SetCycle(ti);
                adios2_dc->SetTime(t);
                adios2_dc->Save();
            }
#endif
        }
    }

    if (!build_database && myid == 0)
    {
        std::ofstream outFile(io_dir + "/ts.txt");
        for (int i = 0; i < ts.size(); i++)
        {
            outFile << ts[i] << "\n";
        }
    }


    // 17. Save the final solution in parallel. This output can be viewed later
    //     using GLVis: "glvis -np <np> -m dg_advection-mesh -g dg_advection-final".
    {
        Vector u_print(U->GetData(), U->Size());
        ostringstream sol_name;
        sol_name << io_dir << "/dg_advection_greedy" << to_string(
                     f_factor)  << "-final." <<
                 setfill('0') << setw(6) << myid;
        ofstream osol(sol_name.str().c_str());
        osol.precision(precision);
        u_print.Print(osol,1);
    }

    double rel_diff = 0.;

    // 18. If in online mode, create the parametric DMD.
    if (online)
    {
        std::fstream fin(io_dir+"/parameters.txt", std::ios_base::in);
        double curr_param;
        std::vector<std::string> dmd_paths;
        std::vector<CAROM::Vector> param_vectors;

        while (fin >> curr_param)
        {
            double curr_f_factor = curr_param;

            dmd_paths.push_back(io_dir + "/" + to_string(curr_f_factor));
            CAROM::Vector param_vector(1, false);
            param_vector(0) = curr_f_factor;
            param_vectors.push_back(param_vector);
        }
        fin.close();

        CAROM::Vector desired_param(1, false);
        desired_param(0) = f_factor;

        dmd_training_timer.Start();

        CAROM::getParametricDMD(dmd_U, param_vectors, dmd_paths, desired_param,
                                "G", "LS", closest_rbf_val);

        dmd_U->projectInitialCondition(*init);

        dmd_training_timer.Stop();
    }

    if (offline || calc_err_indicator)
    {
        // 19. If in offline mode, save the DMD object.
        if (offline)
        {
            if (myid == 0)
            {
                std::cout << "Creating DMD with rdim: " << rdim << std::endl;
            }

            dmd_training_timer.Start();

            dmd_U->train(rdim);

            dmd_training_timer.Stop();

            dmd_U->save(io_dir + "/" + to_string(f_factor));

            if (myid == 0)
            {
                std::ofstream fout;
                fout.open(io_dir+"/parameters.txt", std::ios::app);
                fout << to_string(f_factor) << std::endl;
                fout.close();
            }
        }

        if (true_solution_u == NULL)
        {

            true_solution_u = new Vector(U->Size());
            *true_solution_u = U->GetData();
            tot_true_solution_u_norm = sqrt(InnerProduct(MPI_COMM_WORLD,
                                            *true_solution_u, *true_solution_u));

        }

        std::shared_ptr<CAROM::Vector> result_U = dmd_U->predict(t_final);

        Vector dmd_solution_U(result_U->getData(),
                              result_U->dim());
        Vector diff_u(true_solution_u->Size());
        subtract(dmd_solution_U, *true_solution_u, diff_u);


        double tot_diff_norm_u = sqrt(InnerProduct(MPI_COMM_WORLD, diff_u, diff_u));
        rel_diff = tot_diff_norm_u / tot_true_solution_u_norm;


        if (myid == 0)
        {
            std::cout << "Rel. diff. of DMD temp. (u) at t_final at f_factor "
                      << f_factor << ": " << rel_diff << std::endl;
        }

        if (myid == 0)
        {
            printf("Elapsed time for training DMD: %e second\n",
                   dmd_training_timer.RealTime());
        }
    }
    else if (online)
    {
        dmd_prediction_timer.Start();
        std::shared_ptr<CAROM::Vector> result_U = dmd_U->predict(ts[0]);
        Vector initial_dmd_solution_U(result_U->getData(),
                                      result_U->dim());
        u_gf.SetFromTrueDofs(initial_dmd_solution_U);

        VisItDataCollection dmd_visit_dc("DMD_DG_Advection_Greedy_" +
                                         to_string(f_factor), &pmesh);
        dmd_visit_dc.RegisterField("temperature", &u_gf);
        if (visit)
        {
            dmd_visit_dc.SetCycle(0);
            dmd_visit_dc.SetTime(0.0);
            dmd_visit_dc.Save();
        }

        if (visit)
        {
            for (int i = 1; i < ts.size(); i++)
            {
                if (i == ts.size() - 1 || (i % vis_steps) == 0)
                {
                    result_U = dmd_U->predict(ts[i]);
                    if (myid == 0)
                    {
                        std::cout << "Predicting temperature using DMD at: " << ts[i] << std::endl;
                    }

                    Vector dmd_solution_U(result_U->getData(),
                                          result_U->dim());
                    u_gf.SetFromTrueDofs(dmd_solution_U);

                    dmd_visit_dc.SetCycle(i);
                    dmd_visit_dc.SetTime(ts[i]);
                    dmd_visit_dc.Save();
                }
            }
        }

        dmd_prediction_timer.Stop();

        if (true_solution_u == NULL)
        {

            true_solution_u = new Vector(U->Size());
            *true_solution_u = U->GetData();
            tot_true_solution_u_norm = sqrt(InnerProduct(MPI_COMM_WORLD,
                                            *true_solution_u, *true_solution_u));

        }

        result_U = dmd_U->predict(t_final);

        // 21. Calculate the relative error between the DMD final solution and the true solution.
        Vector dmd_solution_U(result_U->getData(),
                              result_U->dim());
        Vector diff_u(true_solution_u->Size());
        subtract(dmd_solution_U, *true_solution_u, diff_u);

        double tot_diff_norm_u = sqrt(InnerProduct(MPI_COMM_WORLD, diff_u, diff_u));
        double tot_true_solution_u_norm = sqrt(InnerProduct(MPI_COMM_WORLD,
                                               *true_solution_u, *true_solution_u));

        if (myid == 0)
        {
            std::cout << "Relative error of DMD temperature (u) at t_final: "
                      << t_final << " is " << tot_diff_norm_u / tot_true_solution_u_norm << std::endl;
            printf("Elapsed time for predicting DMD: %e second\n",
                   dmd_prediction_timer.RealTime());
        }

        if (myid == 0)
        {
            std::ofstream fout;
            fout.open(io_dir+"/error-output.txt", std::ios::app);
            fout << to_string(f_factor) << " : " << tot_diff_norm_u /
                 tot_true_solution_u_norm << std::endl;
            fout.close();
        }
    }
    // 22. Calculate the relative error as commanded by the greedy algorithm.
    if (offline)
    {
        if (myid == 0) std::cout << "The relative error is: " << rel_diff <<
                                     std::endl;
        greedy_sampler->setPointRelativeError(rel_diff);
    }
    // 23. Or calculate the error indicator as commanded by the greedy algorithm.
    else if (calc_err_indicator)
    {
        if (myid == 0) std::cout << "The error indicator is: " << rel_diff << std::endl;
        greedy_sampler->setPointErrorIndicator(rel_diff, 1);
    }

    if (!online && myid == 0)
    {
        printf("Elapsed time for solving FOM: %e second\n", fom_timer.RealTime());
    }

    return rel_diff;
}

class RelativeDifferenceCostFunction : public CAROM::IOptimizable
{
public:
    RelativeDifferenceCostFunction(unsigned int dims) :
        m_dim(dims)
    {

    }
    double EvaluateCost(std::vector<double> & inputs) const override
    {
        f_factor = inputs[0];

        return simulation();
    }

    unsigned int NumberOfParameters() const override
    {
        return m_dim;
    }

    std::vector<Constraints> GetConstraints() const override
    {
        std::fstream fin(io_dir + "/parameters.txt", std::ios_base::in);
        double curr_param;
        bool first_line = true;
        double min_f_factor = 0.0;
        double max_f_factor = 0.0;


        while (fin >> curr_param)
        {
            double curr_f_factor = curr_param;

            if (first_line)
            {
                min_f_factor = curr_f_factor;
                max_f_factor = curr_f_factor;
                first_line = false;
            }
            else
            {
                min_f_factor = min(curr_f_factor, min_f_factor);
                max_f_factor = max(curr_f_factor, max_f_factor);
            }
        }
        fin.close();

        if (de_min_f_factor != -DBL_MAX) min_f_factor = de_min_f_factor;
        if (de_max_f_factor != DBL_MAX) max_f_factor = de_max_f_factor;

        MFEM_VERIFY(min_f_factor <= max_f_factor, "Radius DE range is invalid.");

        if (myid == 0) cout << "DE f_factor range is: " << min_f_factor << " to " <<
                                max_f_factor << endl;

        std::vector<Constraints> constr(NumberOfParameters());
        constr[0] = Constraints(min_f_factor, max_f_factor, true);
        return constr;
    }

private:
    unsigned int m_dim;
};

int main(int argc, char *argv[])
{
    // 1. Initialize MPI.
    mfem::Mpi::Init();
    num_procs = Mpi::WorldSize();
    myid = Mpi::WorldRank();

    cout.precision(precision);
    // 2. Parse command-line options.
    // MFEM parameters.
    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&problem, "-p", "--problem",
                   "Problem setup to use. See options in velocity_function().");
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                   "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                   "Number of times to refine the mesh uniformly in parallel.");
    args.AddOption(&order, "-o", "--order",
                   "Order (degree) of the finite elements.");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                   "--no-partial-assembly", "Enable Partial Assembly.");
    args.AddOption(&ea, "-ea", "--element-assembly", "-no-ea",
                   "--no-element-assembly", "Enable Element Assembly.");
    args.AddOption(&fa, "-fa", "--full-assembly", "-no-fa",
                   "--no-full-assembly", "Enable Full Assembly.");
    args.AddOption(&device_config, "-d", "--device",
                   "Device configuration string, see Device::Configure().");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                   "ODE solver: 1 - Forward Euler,\n\t"
                   "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6,\n\t"
                   "            11 - Backward Euler,\n\t"
                   "            12 - SDIRK23 (L-stable), 13 - SDIRK33,\n\t"
                   "            22 - Implicit Midpoint Method,\n\t"
                   "            23 - SDIRK23 (A-stable), 24 - SDIRK34");
    args.AddOption(&t_final, "-tf", "--t-final",
                   "Final time; start time is 0.");
    args.AddOption(&dt, "-dt", "--time-step",
                   "Time step.");
    args.AddOption((int *)&prec_type, "-pt", "--prec-type", "Preconditioner for "
                   "implicit solves. 0 for ILU, 1 for pAIR-AMG.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                   "--no-visit-datafiles",
                   "Save data files for VisIt (visit.llnl.gov) visualization.");
    args.AddOption(&paraview, "-paraview", "--paraview-datafiles", "-no-paraview",
                   "--no-paraview-datafiles",
                   "Save data files for ParaView (paraview.org) visualization.");
    args.AddOption(&adios2, "-adios2", "--adios2-streams", "-no-adios2",
                   "--no-adios2-streams",
                   "Save data using adios2 streams.");
    args.AddOption(&binary, "-binary", "--binary-datafiles", "-ascii",
                   "--ascii-datafiles",
                   "Use binary (Sidre) or ascii format for VisIt data files.");
    args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                   "Visualize every n-th timestep.");
    // libROM DMD parameters.
    args.AddOption(&ef, "-ef", "--energy_fraction",
                   "Energy fraction for DMD.");
    args.AddOption(&rdim, "-rdim", "--rdim",
                   "Reduced dimension for DMD.");
    args.AddOption(&run_dmd, "-run_dmd", "--run_dmd",
                   "-no-run_dmd", "--no-run_dmd",
                   "Enable or disable the run_dmd phase.");
    args.AddOption(&temp_io_dir, "-io", "--io-dir-name",
                   "Name of the sub-folder to load/dump input/output files within the current directory.");
    // libROM parameterization parameters.
    args.AddOption(&f_factor, "-ff", "--f-factor",
                   "Frequency scalar factor.");
    // libROM greedy parameters.
    args.AddOption(&build_database, "-build_database", "--build_database",
                   "-no-build_database", "--no-build_database",
                   "Enable or disable the build_database phase.");
    args.AddOption(&closest_rbf_val, "-crv", "--crv",
                   "DMD Closest RBF Value.");
    args.AddOption(&greedy_param_space_f_factor_max, "-greedy-param-f-factor-max",
                   "--greedy-param-f-factor-max",
                   "The maximum f_factor value of the parameter point space.");
    args.AddOption(&greedy_param_space_f_factor_min, "-greedy-param-f-factor-min",
                   "--greedy-param-f-factor-min",
                   "The minimum f_factor value of the parameter point space.");
    args.AddOption(&greedy_param_space_size, "-greedy-param-size",
                   "--greedy-param-size",
                   "The number of values to search in the parameter point space.");
    args.AddOption(&greedy_relative_diff_tol, "-greedyreldifftol",
                   "--greedyreldifftol", "The greedy algorithm relative diff tolerance.");
    args.AddOption(&greedy_subset_size, "-greedysubsize", "--greedysubsize",
                   "The greedy algorithm subset size.");
    args.AddOption(&greedy_convergence_subset_size, "-greedyconvsize",
                   "--greedyconvsize", "The greedy algorithm convergence subset size.");
    // Differential Evolution Parameters
    args.AddOption(&de, "-de", "--de",
                   "-no-de", "--no-de",
                   "Enable or disable the differential evolution phase.");
    args.AddOption(&de_min_f_factor, "-de_min_ff", "--de_min_f-factor",
                   "DE min f-factor");
    args.AddOption(&de_max_f_factor, "-de_max_ff", "--de_max_f-factor",
                   "DE max f-factor");
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

    Device device(device_config);
    if (myid == 0) {
        device.Print();
    }

    io_dir = temp_io_dir;
    mkdir(io_dir.c_str(), 0777);

    // 3. Initialize the DMD database that will be built using a greedy algorithm.
    if(de)
    {
        target_f_factor = f_factor;

        simulation();

        // Create relative error cost function in 4 dimensions (radius, alpha, cx, cy)
        RelativeDifferenceCostFunction cost(1);

        // Create Differential Evolution optimizer with population size of de_ps
        // differential weight of de_F, and crossover probability of de_CR
        CAROM::DifferentialEvolution de_opt(cost, de_PS, de_F, de_CR);

        // Optimize for at least de_min_iter iterations, to a maximum of de_max_iter iterations
        // with verbose output. Stop early, after de_min_iter iterations is run, if the minimum
        // cost did not improve by de_ct
        std::vector<double> optimal_parameters = de_opt.Optimize(de_min_iter,
                de_max_iter, de_ct, true);

        f_factor = optimal_parameters[0];

        simulation();

        delete true_solution_u;

        MPI_Finalize();

        return 0;
    }
    else if (build_database)
    {
        MFEM_VERIFY(rdim != -1, "rdim must be set.");
        MFEM_VERIFY(!visit
                    && !visualization,
                    "visit and visualization must be turned off during the build_database phase.")
        std::ifstream infile(io_dir+"/dg_advection_greedy_parametric_data");
        if (infile.good())
        {
            if (myid == 0) std::cout << "The database has already been built. Exiting." <<
                                         std::endl;

            MPI_Finalize();
            return 0;
        }
        infile.close();
        CAROM::Vector greedy_param_space_min(1, false);
        greedy_param_space_min.item(0) = greedy_param_space_f_factor_min;
        CAROM::Vector greedy_param_space_max(1, false);
        greedy_param_space_max.item(0) = greedy_param_space_f_factor_max;
        greedy_sampler = new CAROM::GreedyRandomSampler(greedy_param_space_min,
                greedy_param_space_max,
                greedy_param_space_size, false, greedy_relative_diff_tol, 1.05,
                2.0, greedy_subset_size, greedy_convergence_subset_size,
                true, "dg_advection_greedy_log.txt");
    }

    target_f_factor = f_factor;

    // 4. If in build_database mode, build the database.
    if (build_database)
    {
        // The simulation is wrapped in a do-while statement so that the greedy
        // algorithm (build_database) can run multiple simulations in succession.
        do
        {
            // Set the correct stage of the greedy algorithm (i.e. sampling new point,
            // calculating relative error of the last sampled point, or calculating
            // an error indicator at a new point.) and run the simulation.
            struct CAROM::GreedyErrorIndicatorPoint pointRequiringRelativeError =
                greedy_sampler->getNextPointRequiringRelativeError();
            struct CAROM::GreedyErrorIndicatorPoint pointRequiringErrorIndicator =
                greedy_sampler->getNextPointRequiringErrorIndicator();
            CAROM::Vector* errorIndicatorPointData =
                pointRequiringErrorIndicator.point.get();

            if (errorIndicatorPointData != NULL)
            {
                if (myid == 0) std::cout << "Calculating a error indicator at a new point." <<
                                             std::endl;
                f_factor = pointRequiringErrorIndicator.point.get()->item(0);
                offline = false;
                calc_err_indicator = true;
            }
            else
            {
                std::shared_ptr<CAROM::Vector> nextSampleParameterPoint =
                    greedy_sampler->getNextParameterPoint();
                CAROM::Vector* samplePointData = nextSampleParameterPoint.get();
                if (samplePointData != NULL)
                {
                    if (myid == 0) std::cout << "Sampling a new point." << std::endl;
                    f_factor = samplePointData->item(0);
                    offline = true;
                    calc_err_indicator = false;
                }
                else
                {
                    if (myid == 0) std::cout << "The greedy algorithm has finished." << std::endl;
                    greedy_sampler->save(io_dir + "/dg_advection_greedy_parametric_data");
                    build_database = false;
                    continue;
                }
            }

            simulation();

            delete true_solution_u;
            true_solution_u = NULL;
        } while (build_database);
    }
    else if(run_dmd == true)
    {
        online = true;
        simulation();
    }
    // 4. Else run a single simulation with the target parameters. This is used
    //    to obtain the target solution to compare with during the differential
    //    evolution optimization process.
    else
    {
        simulation();

        delete true_solution_u;
    }

    MPI_Finalize();

    return 0;
}



// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(ParBilinearForm &M_, ParBilinearForm &K_,
                           const Vector &b_, PrecType prec_type)
    : TimeDependentOperator(M_.Height()), b(b_),
      M_solver(M_.ParFESpace()->GetComm()),
      z(M_.Height())
{
    if (M_.GetAssemblyLevel()==AssemblyLevel::LEGACY)
    {
        M.Reset(M_.ParallelAssemble(), true);
        K.Reset(K_.ParallelAssemble(), true);
    }
    else
    {
        M.Reset(&M_, false);
        K.Reset(&K_, false);
    }

    M_solver.SetOperator(*M);

    Array<int> ess_tdof_list;
    if (M_.GetAssemblyLevel()==AssemblyLevel::LEGACY)
    {
        HypreParMatrix &M_mat = *M.As<HypreParMatrix>();
        HypreParMatrix &K_mat = *K.As<HypreParMatrix>();
        HypreSmoother *hypre_prec = new HypreSmoother(M_mat, HypreSmoother::Jacobi);
        M_prec = hypre_prec;

        dg_solver = new DG_Solver(M_mat, K_mat, *M_.FESpace(), prec_type);
    }
    else
    {
        M_prec = new OperatorJacobiSmoother(M_, ess_tdof_list);
        dg_solver = NULL;
    }

    M_solver.SetPreconditioner(*M_prec);
    M_solver.iterative_mode = false;
    M_solver.SetRelTol(1e-9);
    M_solver.SetAbsTol(0.0);
    M_solver.SetMaxIter(100);
    M_solver.SetPrintLevel(0);
}

// Solve the equation:
//    u_t = M^{-1}(Ku + b),
// by solving associated linear system
//    (M - dt*K) d = K*u + b
void FE_Evolution::ImplicitSolve(const double dt, const Vector &x, Vector &k)
{
    K->Mult(x, z);
    z += b;
    dg_solver->SetTimeStep(dt);
    dg_solver->Mult(z, k);
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
    // y = M^{-1} (K x + b)
    K->Mult(x, z);
    z += b;
    M_solver.Mult(z, y);
}

FE_Evolution::~FE_Evolution()
{
    delete M_prec;
    delete dg_solver;
}

// Velocity coefficient
void velocity_function(const Vector &x, Vector &v)
{
    int dim = x.Size();

    // map to the reference [-1,1] domain
    Vector X(dim);
    for (int i = 0; i < dim; i++)
    {
        double center = (bb_min[i] + bb_max[i]) * 0.5;
        X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
    }

    switch (problem)
    {
    case 0:
    {
        // Translations in 1D, 2D, and 3D
        switch (dim)
        {
        case 1:
            v(0) = 1.0;
            break;
        case 2:
            v(0) = sqrt(2./3.);
            v(1) = sqrt(1./3.);
            break;
        case 3:
            v(0) = sqrt(3./6.);
            v(1) = sqrt(2./6.);
            v(2) = sqrt(1./6.);
            break;
        }
        break;
    }
    case 1:
    case 2:
    {
        // Clockwise rotation in 2D around the origin
        const double w = M_PI/2;
        switch (dim)
        {
        case 1:
            v(0) = 1.0;
            break;
        case 2:
            v(0) = w*X(1);
            v(1) = -w*X(0);
            break;
        case 3:
            v(0) = w*X(1);
            v(1) = -w*X(0);
            v(2) = 0.0;
            break;
        }
        break;
    }
    case 3:
    {
        // Clockwise twisting rotation in 2D around the origin
        const double w = M_PI/2;
        double d = max((X(0)+1.)*(1.-X(0)),0.) * max((X(1)+1.)*(1.-X(1)),0.);
        d = d*d;
        switch (dim)
        {
        case 1:
            v(0) = 1.0;
            break;
        case 2:
            v(0) = d*w*X(1);
            v(1) = -d*w*X(0);
            break;
        case 3:
            v(0) = d*w*X(1);
            v(1) = -d*w*X(0);
            v(2) = 0.0;
            break;
        }
        break;
    }
    }
}

// Initial condition
double u0_function(const Vector &x)
{
    int dim = x.Size();

    // map to the reference [-1,1] domain
    Vector X(dim);
    for (int i = 0; i < dim; i++)
    {
        double center = (bb_min[i] + bb_max[i]) * 0.5;
        X(i) = 2 * (x(i) - center) / (bb_max[i] - bb_min[i]);
    }

    switch (problem)
    {
    case 0:
    case 1:
    {
        switch (dim)
        {
        case 1:
            return exp(-40.*pow(X(0)-0.5,2));
        case 2:
        case 3:
        {
            double rx = 0.45, ry = 0.25, cx = 0., cy = -0.2, w = 10.;
            if (dim == 3)
            {
                const double s = (1. + 0.25*cos(2*M_PI*X(2)));
                rx *= s;
                ry *= s;
            }
            return ( erfc(w*(X(0)-cx-rx))*erfc(-w*(X(0)-cx+rx)) *
                     erfc(w*(X(1)-cy-ry))*erfc(-w*(X(1)-cy+ry)) )/16;
        }
        }
    }
    case 2:
    {
        double x_ = X(0), y_ = X(1), rho, phi;
        rho = hypot(x_, y_);
        phi = atan2(y_, x_);
        return pow(sin(M_PI*rho),2)*sin(3*phi);
    }
    case 3:
    {
// This is our parameterization of the problem.  Parameter = f_factor.
        const double f = M_PI*f_factor;
        return sin(f*X(0))*sin(f*X(1));
    }
    }
    return 0.0;
}

// Inflow boundary condition (zero for the problems considered in this example)
double inflow_function(const Vector &x)
{
    switch (problem)
    {
    case 0:
    case 1:
    case 2:
    case 3:
        return 0.0;
    }
    return 0.0;
}
