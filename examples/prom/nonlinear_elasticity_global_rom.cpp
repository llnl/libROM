//               libROM MFEM Example: parametric ROM for nonlinear elasticity problem (adapted from ex10p.cpp)
//
// Compile with: make nonlinear_elasticity_global_rom
//
// Description:  This examples solves a time dependent nonlinear elasticity
//               problem of the form dv/dt = H(x) + S v, dx/dt = v, where H is a
//               hyperelastic model and S is a viscosity operator of Laplacian
//               type. The geometry of the domain is assumed to be as follows:
//
//                                 +---------------------+
//                    boundary --->|                     |
//                    attribute 1  |                     |
//                    (fixed)      +---------------------+
//
//               The example demonstrates the use of hyper reduction to solve a
//               nonlinear elasticity problem. Time integration is done with various
//               explicit time integrator solvers. The basis for the velocity field
//               is either constructed using a separate velocity basis or using the
//               displacement basis. It is possible to set the initial condition in
//               terms of either velocity or deformation. The velocity initial condition
//               works better when both velocity and displacement bases are used. The
//               deformation initial condition is better when only the displacement
//               basis is used. The input flag -sc controls the scaling of the initial
//               condition applied. This is what parameterizes the ROM. If the scaling
//               factor is within the range +-10%, the results are generally accurate.

// =================================================================================
//
// Sample runs and results for parametric ROM using displacement basis, velocity basis
// and nonlinear term basis, with velocity initial condition:
//
// Offline phase:
//      ./nonlinear_elasticity_global_rom -offline -dt 0.01 -tf 5.0 -s 14 -vs 100 -sc 0.9 -id 0
//      ./nonlinear_elasticity_global_rom -offline -dt 0.01 -tf 5.0 -s 14 -vs 100 -sc 1.1 -id 1
//
// Merge phase:
//      ./nonlinear_elasticity_global_rom -merge -ns 2 -dt 0.01 -tf 5.0
//
// Create FOM comparison data:
//      ./nonlinear_elasticity_global_rom -offline -dt 0.01 -tf 5.0 -s 14 -vs 100 -sc 1.0 -id 2
// Output message:
//      Elapsed time for time integration loop 9.17153
//      Elapsed time for entire simulation 10.0001
//
// Online phase with full sampling:
//      ./nonlinear_elasticity_global_rom -online -dt 0.01 -tf 5.0 -s 14 -vs 100 -hyp -hrtype gnat -rvdim 40 -rxdim 10 -hdim 71 -nsr 1170 -sc 1.0
// Output message:
//      Elapsed time for time integration loop 7.03877
//      Relative error of ROM position (x) at t_final: 5 is 0.000231698
//      Relative error of ROM velocity (v) at t_final: 5 is 0.466941
//      Elapsed time for entire simulation 8.98334
//
// Online phase with strong hyper-reduction, using GNAT (over-sampled DEIM):
//      ./nonlinear_elasticity_global_rom -online -dt 0.01 -tf 5.0 -s 14 -vs 100 -hyp -hrtype gnat -rvdim 40 -rxdim 10 -hdim 71 -nsr 100 -sc 1.0
// Output message:
//      Elapsed time for time integration loop 4.18114
//      Relative error of ROM position (x) at t_final: 5 is 0.00209877
//      Relative error of ROM velocity (v) at t_final: 5 is 1.39472
//      Elapsed time for entire simulation 4.52802
//
// Online phase with strong hyper-reduction, using QDEIM:
//      ./nonlinear_elasticity_global_rom -online -dt 0.01 -tf 5.0 -s 14 -vs 100 -hyp -hrtype qdeim -rvdim 40 -rxdim 10 -hdim 71 -nsr 100 -sc 1.0
// Output message:
//      Elapsed time for time integration loop 4.2972
//      Relative error of ROM position (x) at t_final: 5 is 0.00188458
//      Relative error of ROM velocity (v) at t_final: 5 is 0.978726
//      Elapsed time for entire simulation 5.33154
//
// Online phase with EQP hyper-reduction
//      ./nonlinear_elasticity_global_rom -online -dt 0.01 -tf 5.0 -s 14 -vs 100 -eqp -ns 2 -ntw 50 -rvdim 40 -rxdim 10 -hdim 1 -sc 1.0
// Output message:
//      Elapsed time for time integration loop 2.70006
//      Relative error of ROM position (x) at t_final: 5 is 0.00169666
//      Relative error of ROM velocity (v) at t_final: 5 is 17.71
//      Elapsed time for entire simulation 831.216
//
// Online phase with EQP hyper-reduction and subsampling of snapshots
//      ./nonlinear_elasticity_global_rom -online -dt 0.01 -tf 5.0 -s 14 -vs 100 -eqp -ns 2 -ntw 50 -rvdim 40 -rxdim 10 -hdim 1 -sc 1.0 -sbs 10
// Output message:
//      Elapsed time for time integration loop 0.552546
//      Relative error of ROM position (x) at t_final: 5 is 0.00247479
//      Relative error of ROM velocity (v) at t_final: 5 is 1.33349
//      Elapsed time for entire simulation 11.469
//
// =================================================================================
//
// Sample runs and results for parametric ROM using only displacement basis
// and nonlinear term basis:
// Offline phase:
//      ./nonlinear_elasticity_global_rom -offline -dt 0.01 -tf 5.0 -s 14 -vs 100 -sc 0.9 -xbo -def-ic -id 0
//      ./nonlinear_elasticity_global_rom -offline -dt 0.01 -tf 5.0 -s 14 -vs 100 -sc 1.1 -xbo -def-ic -id 1
//
// Merge phase:
//      ./nonlinear_elasticity_global_rom -merge -ns 2 -dt 0.01 -tf 5.0 -xbo
//
// Create FOM comparison data:
//      ./nonlinear_elasticity_global_rom -offline -dt 0.01 -tf 5.0 -s 14 -vs 100 -sc 1.0 -xbo -def-ic -id 2
// Output message:
//      Elapsed time for time integration loop 9.61062
//      Elapsed time for entire simulation 10.3806
//
// Online phase with full sampling:
//      ./nonlinear_elasticity_global_rom -online -dt 0.01 -tf 5.0 -s 14 -vs 100 -hyp -hrtype gnat -rxdim 57 -hdim 183 -nsr 1170 -sc 1.0 -xbo -def-ic
// Output message:
//      Elapsed time for time integration loop 8.43858
//      Relative error of ROM position (x) at t_final: 5 is 7.08272e-05
//      Relative error of ROM velocity (v) at t_final: 5 is 0.00387647
//      Elapsed time for entire simulation 24.7245
//
// Online phase with strong hyper reduction, using GNAT (over-sampled DEIM):
//      ./nonlinear_elasticity_global_rom -online -dt 0.01 -tf 5.0 -s 14 -vs 100 -hyp -hrtype gnat -rxdim 2 -hdim 4 -nsr 10 -sc 1.0 -xbo -def-ic
// Output message:
//      Elapsed time for time integration loop 0.507339
//      Relative error of ROM position (x) at t_final: 5 is 0.0130818
//      Relative error of ROM velocity (v) at t_final: 5 is 0.979978
//      Elapsed time for entire simulation 0.583618
//
// Online phase with strong hyper reduction, using QDEIM:
//      ./nonlinear_elasticity_global_rom -online -dt 0.01 -tf 5.0 -s 14 -vs 100 -hyp -hrtype qdeim -rxdim 2 -hdim 4 -nsr 10 -sc 1.0 -xbo -def-ic
// Output message:
//      Elapsed time for time integration loop 0.504657
//      Relative error of ROM position (x) at t_final: 5 is 0.00883079
//      Relative error of ROM velocity (v) at t_final: 5 is 1.26721
//      Elapsed time for entire simulation 0.581446
//
// Online phase with EQP hyper reduction:
//      ./nonlinear_elasticity_global_rom -online -dt 0.01 -tf 5.0 -s 14 -vs 100 -eqp -ns 2 -rxdim 2 -hdim 1 -ntw 25 -sc 1.00 -xbo -def-ic
// Elapsed time for time integration loop 0.144368
//      Relative error of ROM position (x) at t_final: 5 is 0.0189361
//      Relative error of ROM velocity (v) at t_final: 5 is 0.830724
//      Elapsed time for entire simulation 4.97996
//
// Online phase with EQP hyper reduction and subsampling of snapshots:
//      ./nonlinear_elasticity_global_rom -online -dt 0.01 -tf 5.0 -s 14 -vs 100 -eqp -ns 2 -rxdim 2 -hdim 1 -ntw 25 -sc 1.00 -xbo -def-ic -sbs 10
// Elapsed time for time integration loop 0.0141128
//      Relative error of ROM position (x) at t_final: 5 is 0.0243952
//      Relative error of ROM velocity (v) at t_final: 5 is 0.986434
//      Elapsed time for entire simulation 0.782108
//
// This example runs in parallel with MPI, by using the same number of MPI ranks
// in all phases (offline, merge, online).

#include "mfem.hpp"
#include "linalg/Vector.h"
#include "linalg/BasisGenerator.h"
#include "linalg/BasisReader.h"
#include "linalg/NNLS.h"
#include "hyperreduction/Hyperreduction.h"
#include "mfem/SampleMesh.hpp"
#include "mfem/Utilities.hpp"

#include <memory>
#include <cmath>
#include <limits>

using namespace std;
using namespace mfem;

#include "nonlinear_elasticity_global_rom_eqp.hpp"

class RomOperator : public TimeDependentOperator
{
private:
    int rxdim, rvdim, hdim;
    int nsamp_H;
    double current_dt;
    bool oversampling;
    CAROM::Matrix *V_v_sp, *V_x_sp;
    CAROM::Vector *V_xTv_0_sp;
    CAROM::Matrix *V_xTV_v_sp;
    CAROM::Matrix *V_v_sp_dist;
    CAROM::Vector *psp_librom, *psp_v_librom;
    CAROM::Vector *Vx_librom_temp;
    Vector *psp;
    Vector *psp_x;
    Vector *psp_v;
    Vector *Vx_temp;
    mutable Vector zH;
    mutable CAROM::Vector zX;
    mutable Vector zX_MFEM;
    mutable CAROM::Vector zN;
    const CAROM::Matrix *Hsinv;
    mutable CAROM::Vector *z_librom;
    mutable Vector z;
    mutable Vector z_x;
    mutable Vector z_v;

    CAROM::Vector *v0_fom_librom, *v0_librom, *V_xTv_0;
    CAROM::Matrix *V_xTV_v;

    bool hyperreduce;
    bool x_base_only;

    CAROM::Vector *pfom_librom, *pfom_v_librom;
    Vector *pfom;
    Vector *pfom_x;
    Vector *pfom_v;
    mutable Vector *zfom_x;
    mutable Vector *zfom_v;
    CAROM::Vector *zfom_x_librom;

    CAROM::SampleMeshManager *smm;

    CAROM::Vector *z_v_librom;
    CAROM::Vector *z_x_librom;

    // Data for EQP
    bool eqp;
    const IntegrationRule *ir_eqp;
    std::vector<double> eqp_rw;
    std::vector<int> eqp_qp;
    Vector eqp_coef;
    Vector eqp_DS_coef;
    const bool fastIntegration = true;
    ElemMatrices *em;

    CAROM::Matrix eqp_lifting;
    std::vector<int> eqp_liftDOFs;
    mutable CAROM::Vector eqp_lifted;

    int rank;

    NeoHookeanModel *model;

protected:
    CAROM::Matrix *S_hat;
    CAROM::Vector *S_hat_v0;
    Vector *S_hat_v0_temp;
    CAROM::Vector *S_hat_v0_temp_librom;
    CAROM::Matrix *M_hat;
    CAROM::Matrix *M_hat_inv;

    const std::shared_ptr<CAROM::Matrix> U_H;

    HyperelasticOperator *fomSp;

    CGSolver M_hat_solver;    // Krylov solver for inverting the reduced mass matrix M_hat
    HypreSmoother M_hat_prec; // Preconditioner for the reduced mass matrix M_hat

public:
    HyperelasticOperator *fom;

    RomOperator(HyperelasticOperator *fom_,
                HyperelasticOperator *fomSp_, const int rvdim_, const int rxdim_,
                const int hdim_, CAROM::SampleMeshManager *smm_, const Vector *v0_,
                const Vector *x0_, const Vector v0_fom_,
                const std::shared_ptr<CAROM::Matrix> V_v_,
                const std::shared_ptr<CAROM::Matrix> V_x_,
                const std::shared_ptr<CAROM::Matrix> U_H_,
                const CAROM::Matrix *Hsinv_, const int myid, const bool oversampling_,
                const bool hyperreduce_, const bool x_base_only_, const bool use_eqp,
                CAROM::Vector *eqpSol,
                const IntegrationRule *ir_eqp_, NeoHookeanModel *model_);

    virtual void Mult(const Vector &y, Vector &dy_dt) const;
    void Mult_Hyperreduced(const Vector &y, Vector &dy_dt) const;
    void Mult_FullOrder(const Vector &y, Vector &dy_dt) const;
    void SetEQP(CAROM::Vector *eqpSol);

    CAROM::Matrix V_v, V_x, V_vTU_H;
    const Vector *x0;
    const Vector *v0;
    Vector v0_fom;

    virtual ~RomOperator();
};

/** Function representing the elastic energy density for the given hyperelastic
    model+deformation. Used in HyperelasticOperator::GetElasticEnergyDensity. */
class ElasticEnergyCoefficient : public Coefficient
{
private:
    HyperelasticModel &model;
    const ParGridFunction &x;
    DenseMatrix J;

public:
    ElasticEnergyCoefficient(HyperelasticModel &m, const ParGridFunction &x_)
        : model(m), x(x_) {}
    virtual double Eval(ElementTransformation &T, const IntegrationPoint &ip);
    virtual ~ElasticEnergyCoefficient() {}
};

void InitialDeformationIC1(const Vector &x, Vector &y);

void InitialVelocityIC1(const Vector &x, Vector &v);

void InitialDeformationIC2(const Vector &x, Vector &y);

void InitialVelocityIC2(const Vector &x, Vector &v);

void visualize(ostream &out, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name = NULL,
               bool init_vis = false);

// TODO: move this to the library?
void MergeBasis(const int dimFOM, const int nparam, const int max_num_snapshots,
                std::string name, bool squareSV)
{
    MFEM_VERIFY(nparam > 0, "Must specify a positive number of parameter sets");

    bool update_right_SV = false;
    bool isIncremental = false;

    CAROM::Options options(dimFOM, nparam * max_num_snapshots, 1, update_right_SV);
    CAROM::BasisGenerator generator(options, isIncremental, "basis" + name);

    for (int paramID = 0; paramID < nparam; ++paramID)
    {
        std::string snapshot_filename = "basis" + std::to_string(
                                            paramID) + "_" + name + "_snapshot";
        generator.loadSamples(snapshot_filename, "snapshot");
    }

    generator.endSamples(); // save the merged basis file

    int cutoff = 0;
    generator.finalSummary(1e-7, cutoff, "mergedSV_" + name + ".txt", 0, squareSV);
}

const CAROM::Matrix *GetSnapshotMatrix(const int dimFOM, const int nparam,
                                       const int max_num_snapshots, std::string name)
{
    MFEM_VERIFY(nparam > 0, "Must specify a positive number of parameter sets");

    bool update_right_SV = false;
    bool isIncremental = false;

    CAROM::Options options(dimFOM, nparam * max_num_snapshots, 1, update_right_SV);
    CAROM::BasisGenerator generator(options, isIncremental, "basis" + name);

    for (int paramID = 0; paramID < nparam; ++paramID)
    {
        std::string snapshot_filename = "basis" + std::to_string(
                                            paramID) + "_" + name + "_snapshot";
        generator.loadSamples(snapshot_filename, "snapshot");
    }

    // TODO: this deep copy is inefficient, just to get around generator owning the matrix.
    CAROM::Matrix *s = new CAROM::Matrix(*generator.getSnapshotMatrix());

    return s;
    // return generator.getSnapshotMatrix();  // BUG: the matrix is deleted when generator goes out of scope.
}

// TODO: remove this by making online computation serial?
void BroadcastUndistributedRomVector(CAROM::Vector *v)
{
    const int N = v->dim();

    MFEM_VERIFY(N > 0, "");

    double *d = new double[N];

    MFEM_VERIFY(d != 0, "");

    for (int i = 0; i < N; ++i)
        d[i] = (*v)(i);

    MPI_Bcast(d, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < N; ++i)
        (*v)(i) = d[i];

    delete[] d;
}

// Scaling factor for parameterization
double s = 1.0;

int main(int argc, char *argv[])
{
    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // 2. Parse command-line options.
    const char *mesh_file = "../data/beam-quad.mesh";
    int ser_ref_levels = 2;
    int par_ref_levels = 0;
    int order = 2;
    int ode_solver_type = 14; // RK4
    int vis_steps = 1;
    double t_final = 15.0;
    double dt = 0.03;
    double visc = 1e-2;
    double mu = 0.25;
    double K = 5.0;
    bool adaptive_lin_rtol = true;
    bool visualization = true;
    bool visit = false;
    bool def_ic = false;

    // ROM parameters
    bool offline = false;
    bool merge = false;
    bool online = false;
    bool use_eqp = false;
    bool writeSampleMesh = false;
    bool hyperreduce = true;
    bool x_base_only = false;
    int num_samples_req = -1;
    const char *samplingType = "gnat";
    bool squareSV = true;

    int nsets = 0;
    int id_param = 0;

    // Number of basis vectors to use
    int rxdim = -1;
    int rvdim = -1;
    int hdim = -1;

    bool preconditionNNLS = false;
    double tolNNLS = 1.0e-14;
    int maxNNLSnnz = 0;

    // Number of time windows
    int n_windows = 0;

    // Subsampling steps
    int snap_step = 1;

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                   "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                   "Number of times to refine the mesh uniformly in parallel.");
    args.AddOption(&order, "-o", "--order",
                   "Order (degree) of the finite elements.");
    args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                   "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                   "            11 - Forward Euler, 12 - RK2,\n\t"
                   "            13 - RK3 SSP, 14 - RK4."
                   "            22 - Implicit Midpoint Method,\n\t"
                   "            23 - SDIRK23 (A-stable), 24 - SDIRK34");
    args.AddOption(&t_final, "-tf", "--t-final",
                   "Final time; start time is 0.");
    args.AddOption(&dt, "-dt", "--time-step",
                   "Time step.");
    args.AddOption(&visc, "-v", "--viscosity",
                   "Viscosity coefficient.");
    args.AddOption(&mu, "-mu", "--shear-modulus",
                   "Shear modulus in the Neo-Hookean hyperelastic model.");
    args.AddOption(&K, "-K", "--bulk-modulus",
                   "Bulk modulus in the Neo-Hookean hyperelastic model.");
    args.AddOption(&adaptive_lin_rtol, "-alrtol", "--adaptive-lin-rtol",
                   "-no-alrtol", "--no-adaptive-lin-rtol",
                   "Enable or disable adaptive linear solver rtol.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                   "--no-visit-datafiles",
                   "Save data files for VisIt (visit.llnl.gov) visualization.");
    args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                   "Visualize every n-th timestep.");
    args.AddOption(&nsets, "-ns", "--nset", "Number of parametric snapshot sets");
    args.AddOption(&offline, "-offline", "--offline", "-no-offline", "--no-offline",
                   "Enable or disable the offline phase.");
    args.AddOption(&online, "-online", "--online", "-no-online", "--no-online",
                   "Enable or disable the online phase.");
    args.AddOption(&merge, "-merge", "--merge", "-no-merge", "--no-merge",
                   "Enable or disable the merge phase.");
    args.AddOption(&squareSV, "-sqsv", "--square-sv", "-no-sqsv", "--no-square-sv",
                   "Use singular values squared in energy fraction.");
    args.AddOption(&samplingType, "-hrtype", "--hrsamplingtype",
                   "Sampling type for hyperreduction.");
    args.AddOption(&num_samples_req, "-nsr", "--nsr",
                   "number of samples we want to select for the sampling algorithm.");
    args.AddOption(&rxdim, "-rxdim", "--rxdim",
                   "Basis dimension for displacement solution space.");
    args.AddOption(&rvdim, "-rvdim", "--rvdim",
                   "Basis dimension for velocity solution space.");
    args.AddOption(&hdim, "-hdim", "--hdim",
                   "Basis dimension for the nonlinear term.");
    args.AddOption(&id_param, "-id", "--id", "Parametric index");
    args.AddOption(&hyperreduce, "-hyp", "--hyperreduce", "-no-hyp",
                   "--no-hyperreduce", "Hyper reduce nonlinear term");
    args.AddOption(&x_base_only, "-xbo", "--xbase-only", "-no-xbo",
                   "--not-xbase-only", "Use the displacement (X) basis to approximate velocity.");
    args.AddOption(&def_ic, "-def-ic", "--deformation-ic", "-vel-ic",
                   "--velocity-ic",
                   "Use a deformation-, or velocity initial condition. Default is velocity IC.");
    args.AddOption(&s, "-sc", "--scaling", "Scaling factor for initial condition.");
    args.AddOption(&use_eqp, "-eqp", "--eqp", "-no-eqp", "--no-eqp",
                   "Use EQP instead of DEIM for the hyperreduction.");
    args.AddOption(&writeSampleMesh, "-smesh", "--sample-mesh", "-no-smesh",
                   "--no-sample-mesh", "Write the sample mesh to file.");
    args.AddOption(&preconditionNNLS, "-preceqp", "--preceqp", "-no-preceqp",
                   "--no-preceqp", "Precondition the NNLS system for EQP.");
    args.AddOption(&tolNNLS, "-tolnnls", "--tol-nnls",
                   "Tolerance for NNLS solver.");
    args.AddOption(&maxNNLSnnz, "-maxnnls", "--max-nnls",
                   "Maximum nnz for NNLS");
    args.AddOption(&n_windows, "-ntw", "--n-time-windows",
                   "Number of time windows. Default is 0");
    args.AddOption(&snap_step, "-sbs", "--subsampling-step",
                   "Subsamble every n-th snapshot. Default is 1, meaning no subsampling");

    args.Parse();
    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        MPI_Finalize();
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }

    const bool check = (offline && !merge && !online) || (!offline && merge
                       && !online) || (!offline && !merge && online);
    MFEM_VERIFY(check, "only one of offline, merge, or online must be true!");

    StopWatch solveTimer, totalTimer;
    totalTimer.Start();

    // 3. Read the serial mesh from the given mesh file on all processors. We can
    //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
    //    with the same code.
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // 4. Define the ODE solver used for time integration. Several implicit
    //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
    //    explicit Runge-Kutta methods are available.
    ODESolver *ode_solver;
    switch (ode_solver_type)
    {
    // Implicit L-stable methods
    // To be added...

    // Explicit methods
    case 11:
        ode_solver = new ForwardEulerSolver;
        break;
    case 12:
        ode_solver = new RK2Solver(0.5);
        break; // midpoint method
    case 13:
        ode_solver = new RK3SSPSolver;
        break;
    case 14:
        ode_solver = new RK4Solver;
        break;
    case 15:
        ode_solver = new GeneralizedAlphaSolver(0.5);
        break;
    // Implicit A-stable methods (not L-stable)
    // To be added...
    default:
        if (myid == 0)
        {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
        }
        delete mesh;
        MPI_Finalize();
        return 3;
    }

    // 5. Refine the mesh in serial to increase the resolution. In this example
    //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
    //    a command-line parameter.
    for (int lev = 0; lev < ser_ref_levels; lev++)
    {
        mesh->UniformRefinement();
    }

    // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    for (int lev = 0; lev < par_ref_levels; lev++)
    {
        pmesh->UniformRefinement();
    }

    // 7. Define the parallel vector finite element spaces representing the mesh
    //    deformation x_gf, the velocity v_gf, and the initial configuration,
    //    x_ref. Define also the elastic energy density, w_gf, which is in a
    //    discontinuous higher-order space. Since x and v are integrated in time
    //    as a system, we group them together in block vector vx, on the unique
    //    parallel degrees of freedom, with offsets given by array true_offset.
    H1_FECollection fe_coll(order, dim);
    ParFiniteElementSpace fespace(pmesh, &fe_coll, dim);

    HYPRE_BigInt glob_size = fespace.GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of velocity/deformation unknowns: " << glob_size << endl;
    }
    int true_size = fespace.TrueVSize();
    Array<int> true_offset(3);
    true_offset[0] = 0;
    true_offset[1] = true_size;
    true_offset[2] = 2 * true_size;

    BlockVector vx(true_offset);
    ParGridFunction v_gf, x_gf;
    // Associate a FiniteElementSpace and true-dof data with the GridFunctions.
    v_gf.MakeTRef(&fespace, vx, true_offset[0]);
    x_gf.MakeTRef(&fespace, vx, true_offset[1]);

    ParGridFunction x_ref(&fespace);
    pmesh->GetNodes(x_ref);

    L2_FECollection w_fec(order + 1, dim);
    ParFiniteElementSpace w_fespace(pmesh, &w_fec);
    ParGridFunction w_gf(&w_fespace);

    // Basis params
    bool update_right_SV = false;
    bool isIncremental = false;
    const std::string basisFileName = "basis" + std::to_string(id_param);
    int max_num_snapshots = int(t_final / dt) + 2;

    // The merge phase
    if (merge)
    {
        totalTimer.Clear();
        totalTimer.Start();

        // Merge bases
        if (x_base_only == false)
        {
            MergeBasis(true_size, nsets, max_num_snapshots, "V", squareSV);
        }

        MergeBasis(true_size, nsets, max_num_snapshots, "X", squareSV);
        MergeBasis(true_size, nsets, max_num_snapshots, "H", squareSV);

        totalTimer.Stop();
        if (myid == 0)
        {
            printf("Elapsed time for merging and building ROM basis: %e second\n",
                   totalTimer.RealTime());
        }
        MPI_Finalize();
        return 0;
    }

    // 8. Set the initial conditions for v_gf, x_gf and vx, and define the
    //    boundary conditions on a beam-like mesh (see description above).
    VectorFunctionCoefficient *velo = 0;
    VectorFunctionCoefficient *deform = 0;

    if (def_ic)
    {
        velo = new VectorFunctionCoefficient(dim, InitialVelocityIC2);
    }
    else
    {
        velo = new VectorFunctionCoefficient(dim, InitialVelocityIC1);
    }

    v_gf.ProjectCoefficient(*velo);
    v_gf.SetTrueVector();

    if (def_ic)
    {
        deform = new VectorFunctionCoefficient(dim, InitialDeformationIC2);
    }
    else
    {
        deform = new VectorFunctionCoefficient(dim, InitialDeformationIC1);
    }

    x_gf.ProjectCoefficient(*deform);
    x_gf.SetTrueVector();

    v_gf.SetFromTrueVector();
    x_gf.SetFromTrueVector();

    Array<int> ess_bdr(fespace.GetMesh()->bdr_attributes.Max());
    ess_bdr = 0;
    ess_bdr[0] = 1; // boundary attribute 1 (index 0) is fixed

    Array<int> ess_tdof_list;
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    // Store initial vx
    BlockVector vx0 = BlockVector(vx);
    BlockVector vx_diff = BlockVector(vx);

    // Reduced order solution
    Vector *wMFEM = 0;
    CAROM::Vector *w = 0;
    CAROM::Vector *w_v = 0;
    CAROM::Vector *w_x = 0;

    // Initialize reconstructed solution
    Vector *v_rec = new Vector(v_gf.GetTrueVector());
    Vector *x_rec = new Vector(x_gf.GetTrueVector());

    CAROM::Vector *v_rec_librom = new CAROM::Vector(v_rec->GetData(), v_rec->Size(),
            true, false);
    CAROM::Vector *x_rec_librom = new CAROM::Vector(x_rec->GetData(), x_rec->Size(),
            true, false);

    // 9. Initialize the hyperelastic operator, the GLVis visualization and print
    //    the initial energies.
    HyperelasticOperator oper(fespace, ess_tdof_list, visc, mu, K);
    NeoHookeanModel *model = new NeoHookeanModel(mu, K);
    HyperelasticOperator *soper = 0;

    // Fill dvdt and dxdt
    Vector dvxdt(true_size * 2);
    Vector dvdt(dvxdt.GetData() + 0, true_size);
    Vector dxdt(dvxdt.GetData() + true_size, true_size);

    socketstream vis_v, vis_w;
    if (visualization)
    {
        char vishost[] = "localhost";
        int visport = 19916;
        vis_v.open(vishost, visport);
        vis_v.precision(8);
        visualize(vis_v, pmesh, &x_gf, &v_gf, "Velocity", true);
        // Make sure all ranks have sent their 'v' solution before initiating
        // another set of GLVis connections (one from each rank)
        MPI_Barrier(pmesh->GetComm());
        vis_w.open(vishost, visport);
        if (vis_w)
        {
            oper.GetElasticEnergyDensity(x_gf, w_gf);
            vis_w.precision(8);
            visualize(vis_w, pmesh, &x_gf, &w_gf, "Elastic energy density", true);
        }
    }

    // Create data collection for solution output: either VisItDataCollection for
    // ascii data files, or SidreDataCollection for binary data files.
    DataCollection *dc = NULL;
    if (visit)
    {
        if (offline)
            dc = new VisItDataCollection("nlelast-fom", pmesh);
        else
            dc = new VisItDataCollection("nlelast-rom", pmesh);

        dc->SetPrecision(8);
        // To save the mesh using MFEM's parallel mesh format:
        // dc->SetFormat(DataCollection::PARALLEL_FORMAT);
        dc->RegisterField("x", &x_gf);
        dc->RegisterField("v", &v_gf);
        dc->SetCycle(0);
        dc->SetTime(0.0);
        dc->Save();
    }

    double ee0 = oper.ElasticEnergy(x_gf);
    double ke0 = oper.KineticEnergy(v_gf);

    if (myid == 0)
    {
        cout << "initial elastic energy (EE) = " << ee0 << endl;
        cout << "initial kinetic energy (KE) = " << ke0 << endl;
        cout << "initial   total energy (TE) = " << (ee0 + ke0) << endl;
    }

    // 10. Create pROM object.
    CAROM::BasisGenerator *basis_generator_v = 0;
    CAROM::BasisGenerator *basis_generator_x = 0;
    CAROM::BasisGenerator *basis_generator_H = 0;

    if (offline)
    {
        CAROM::Options options(fespace.GetTrueVSize(), max_num_snapshots, 1,
                               update_right_SV);

        if (x_base_only == false)
        {
            basis_generator_v = new CAROM::BasisGenerator(options, isIncremental,
                    basisFileName + "_V");
        }

        basis_generator_x = new CAROM::BasisGenerator(options, isIncremental,
                basisFileName + "_X");

        basis_generator_H = new CAROM::BasisGenerator(options, isIncremental,
                basisFileName + "_H");
    }

    RomOperator *romop = 0;

    std::shared_ptr<CAROM::Matrix> BV_librom;
    std::shared_ptr<CAROM::Matrix> BX_librom;
    std::shared_ptr<CAROM::Matrix> H_librom;
    const CAROM::Matrix *Hsinv = 0;

    int nsamp_H = -1;

    CAROM::SampleMeshManager *smm = nullptr;

    CAROM::Vector *eqpSol = nullptr;
    CAROM::Vector *eqpSol_S = nullptr;

    CAROM::Vector *window_ids = nullptr;
    CAROM::Vector *load_eqpsol = new CAROM::Vector(1,
            false); // Will be resized later

    // 11. Initialize ROM operator
    // I guess most of this should be done on id =0
    if (online)
    {
        // Read bases
        CAROM::BasisReader *readerV = 0;

        if (x_base_only)
        {
            // The basis for v uses the x basis instead.
            readerV = new CAROM::BasisReader("basisX");
            rvdim = rxdim;
        }
        else
            readerV = new CAROM::BasisReader("basisV");

        BV_librom = readerV->getSpatialBasis();

        if (rvdim == -1) // Change rvdim
            rvdim = BV_librom->numColumns();
        else
            BV_librom = BV_librom->getFirstNColumns(rvdim);

        MFEM_VERIFY(BV_librom->numRows() == true_size, "");

        if (myid == 0)
            printf("reduced V dim = %d\n", rvdim);

        CAROM::BasisReader readerX("basisX");
        BX_librom = readerX.getSpatialBasis();

        if (rxdim == -1) // Change rxdim
            rxdim = BX_librom->numColumns();
        else
            BX_librom = BX_librom->getFirstNColumns(rxdim);

        MFEM_VERIFY(BX_librom->numRows() == true_size, "");

        if (myid == 0)
            printf("reduced X dim = %d\n", rxdim);

        // Hyper reduce H
        CAROM::BasisReader readerH("basisH");
        H_librom = readerH.getSpatialBasis();

        // Compute sample points
        if (hdim == -1)
        {
            hdim = H_librom->numColumns();
        }
        CAROM::Matrix *Hsinv = new CAROM::Matrix(hdim, hdim, false);
        MFEM_VERIFY(H_librom->numColumns() >= hdim, "");

        if (H_librom->numColumns() > hdim)
            H_librom = H_librom->getFirstNColumns(hdim);

        if (myid == 0)
            printf("reduced H dim = %d\n", hdim);

        // Setup hyperreduction, using either EQP or sampled DOFs and a sample mesh.
        ParFiniteElementSpace *sp_XV_space;
        const IntegrationRule *ir0 = NULL;

        if (ir0 == NULL)
        {
            const FiniteElement &fe = *fespace.GetFE(0);
            ir0 = &(IntRules.Get(fe.GetGeomType(), 2 * fe.GetOrder() + 3));
        }

        // Timewindowing setup for EQP
        int n_step = int(t_final / dt);

        if (n_windows > 1)
        {
            window_ids = new CAROM::Vector(n_windows + 1, false);
            get_window_ids(n_step, n_windows, window_ids);
        }

        if (use_eqp)
        {
            // EQP setup
            eqpSol = new CAROM::Vector(ir0->GetNPoints() * fespace.GetNE(), true);
            SetupEQP_snapshots(ir0, myid, &fespace, nsets, BV_librom.get(),
                               GetSnapshotMatrix(fespace.GetTrueVSize(), nsets, max_num_snapshots, "X"),
                               vx0.GetBlock(0),
                               preconditionNNLS, tolNNLS, maxNNLSnnz, model, *eqpSol, window_ids, snap_step);

            if (writeSampleMesh)
                WriteMeshEQP(pmesh, myid, ir0->GetNPoints(), *eqpSol);
        }
        else
        {
            vector<int> num_sample_dofs_per_proc(num_procs);

            if (num_samples_req != -1)
            {
                nsamp_H = num_samples_req;
            }
            else
            {
                nsamp_H = hdim;
            }

            Hsinv->setSize(nsamp_H, hdim);
            vector<int> sample_dofs(nsamp_H);

            // Setup hyperreduction using DEIM, GNAT, or S-OPT
            CAROM::Hyperreduction hr(samplingType);
            hr.ComputeSamples(H_librom,
                              hdim,
                              sample_dofs,
                              num_sample_dofs_per_proc,
                              *Hsinv,
                              myid,
                              num_procs,
                              nsamp_H);

            // Construct sample mesh
            const int nspaces = 1;
            std::vector<ParFiniteElementSpace *> spfespace(nspaces);
            spfespace[0] = &fespace;

            smm = new CAROM::SampleMeshManager(spfespace);

            vector<int> sample_dofs_empty;
            vector<int> num_sample_dofs_per_proc_empty;
            num_sample_dofs_per_proc_empty.assign(num_procs, 0);

            smm->RegisterSampledVariable("V", 0, sample_dofs,
                                         num_sample_dofs_per_proc);
            smm->RegisterSampledVariable("X", 0, sample_dofs,
                                         num_sample_dofs_per_proc);
            smm->RegisterSampledVariable("H", 0, sample_dofs,
                                         num_sample_dofs_per_proc);

            smm->ConstructSampleMesh();
        }

        w = new CAROM::Vector(rxdim + rvdim, false);
        w_v = new CAROM::Vector(rvdim, false);
        w_x = new CAROM::Vector(rxdim, false);
        *w = 0.0;

        // Note that some of this could be done only on the ROM solver process,
        // but it is tricky, since RomOperator assembles Bsp in parallel.
        wMFEM = new Vector(&((*w)(0)), rxdim + rvdim);

        // Initial conditions
        Vector *w_v0 = 0;
        Vector *w_x0 = 0;

        int sp_size = 0;
        Array<int> ess_tdof_list_sp; // Initialize sample essential boundary conditions

        if (myid == 0 && !use_eqp)
        {
            sp_XV_space = smm->GetSampleFESpace(0);

            sp_size = sp_XV_space->TrueVSize();
            Array<int> sp_offset(3);
            sp_offset[0] = 0;
            sp_offset[1] = sp_size;
            sp_offset[2] = 2 * sp_size;

            // Initialize sp_p with initial conditions.
            BlockVector sp_vx(sp_offset);
            ParGridFunction sp_v_gf, sp_x_gf;

            // 12. Set the initial conditions for v_gf, x_gf and vx, and define the
            //    boundary conditions on a beam-like mesh (see description above).

            // Associate a FiniteElementSpace and true-dof data with the GridFunctions.
            sp_v_gf.MakeTRef(sp_XV_space, sp_vx, sp_offset[0]);
            sp_x_gf.MakeTRef(sp_XV_space, sp_vx, sp_offset[1]);

            VectorFunctionCoefficient *velo = 0;
            VectorFunctionCoefficient *deform = 0;

            if (def_ic)
            {
                velo = new VectorFunctionCoefficient(dim, InitialVelocityIC2);
            }
            else
            {
                velo = new VectorFunctionCoefficient(dim, InitialVelocityIC1);
            }
            sp_v_gf.ProjectCoefficient(*velo);
            sp_v_gf.SetTrueVector();

            if (def_ic)
            {
                deform = new VectorFunctionCoefficient(dim, InitialDeformationIC2);
            }
            else
            {
                deform = new VectorFunctionCoefficient(dim, InitialDeformationIC1);
            }
            sp_x_gf.ProjectCoefficient(*deform);
            sp_x_gf.SetTrueVector();

            sp_v_gf.SetFromTrueVector();
            sp_x_gf.SetFromTrueVector();

            // Get initial conditions
            w_v0 = new Vector(sp_v_gf.GetTrueVector());
            w_x0 = new Vector(sp_x_gf.GetTrueVector());
        }

        if (!use_eqp)
        {
            // Convert essential boundary list from FOM mesh to sample mesh
            // Create binary list where 1 means essential boundary element, 0 means nonessential.
            CAROM::Matrix Ess_mat(true_size, 1, true);
            for (size_t i = 0; i < true_size; i++)
            {
                Ess_mat(i, 0) = 0;
                for (size_t j = 0; j < ess_tdof_list.Size(); j++)
                {
                    if (ess_tdof_list[j] == i)
                    {
                        Ess_mat(i, 0) = 1;
                    }
                }
            }

            // Project binary FOM list onto sampling space
            MPI_Bcast(&sp_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            CAROM::Matrix Ess_mat_sp(sp_size, 1, false);
            smm->GatherDistributedMatrixRows("X", Ess_mat, 1, Ess_mat_sp);

            // Count number of true elements in new matrix
            int num_ess_sp = 0;

            for (size_t i = 0; i < sp_size; i++)
            {
                if (Ess_mat_sp(i, 0) == 1)
                {
                    num_ess_sp++;
                }
            }

            // Set essential dof list in sampling space size
            ess_tdof_list_sp.SetSize(num_ess_sp);

            // Add indices to list
            int ctr = 0;
            for (size_t i = 0; i < sp_size; i++)
            {
                if (Ess_mat_sp(i, 0) == 1)
                {
                    ess_tdof_list_sp[ctr] = i;
                    ctr++;
                }
            }
        }

        if (myid == 0)
        {
            if (!use_eqp)
            {
                // Define operator in sample space
                soper = new HyperelasticOperator(*sp_XV_space, ess_tdof_list_sp, visc, mu, K);
            }
            else
            {
                soper = new HyperelasticOperator(fespace, ess_tdof_list, visc, mu, K);
            }
        }

        if (!use_eqp)
        {
            romop = new RomOperator(&oper, soper, rvdim, rxdim, hdim, smm, w_v0, w_x0,
                                    vx0.GetBlock(0), BV_librom, BX_librom, H_librom, Hsinv, myid,
                                    num_samples_req != -1, hyperreduce, x_base_only, use_eqp, eqpSol, ir0, model);
        }
        else
        {
            romop = new RomOperator(&oper, soper, rvdim, rxdim, hdim, smm,
                                    &(vx0.GetBlock(0)),
                                    &(vx0.GetBlock(1)), vx0.GetBlock(0), BV_librom, BX_librom, H_librom, Hsinv,
                                    myid,
                                    num_samples_req != -1, hyperreduce, x_base_only, use_eqp, eqpSol, ir0, model);
        }

        // Print lifted initial energies
        BroadcastUndistributedRomVector(w);

        for (int i = 0; i < rvdim; ++i)
            (*w_v)(i) = (*w)(i);

        for (int i = 0; i < rxdim; ++i)
            (*w_x)(i) = (*w)(rvdim + i);

        romop->V_v.mult(*w_v, *v_rec_librom);
        romop->V_x.mult(*w_x, *x_rec_librom);

        *v_rec += vx0.GetBlock(0);
        *x_rec += vx0.GetBlock(1);

        v_gf.SetFromTrueDofs(*v_rec);
        x_gf.SetFromTrueDofs(*x_rec);

        double ee = oper.ElasticEnergy(x_gf);
        double ke = oper.KineticEnergy(v_gf);

        if (myid == 0)
        {
            cout << "Lifted initial energies, EE = " << ee
                 << ", KE = " << ke << ", ΔTE = " << (ee + ke) - (ee0 + ke0) << endl;
        }

        ode_solver->Init(*romop);
    }
    else
    {
        // fom
        ode_solver->Init(oper);
    }

    // 13. Perform time-integration
    //     (looping over the time iterations, ti, with a time-step dt).
    //     (taking samples and storing it into the pROM object)

    double t = 0.0;
    vector<double> ts;
    oper.SetTime(t);

    bool last_step = false;

    int current_window = 0;

    for (int ti = 1; !last_step; ti++)
    {
        double dt_real = min(dt, t_final - t);

        if (online)
        {
            if (myid == 0)
            {
                if (use_eqp && window_ids && current_window < n_windows
                        && ti == window_ids->item(current_window))
                {
                    // Load eqp and reinitialize ROM operator
                    cout << "Time window start at " << ti << endl;
                    get_EQPsol(current_window, load_eqpsol);
                    romop->SetEQP(load_eqpsol);
                    ode_solver->Init(*romop);
                    current_window++;
                }
                solveTimer.Start();
                ode_solver->Step(*wMFEM, t, dt_real);
                solveTimer.Stop();
            }

            MPI_Bcast(&t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        else
        {
            solveTimer.Start();
            ode_solver->Step(vx, t, dt_real);
            solveTimer.Stop();
        }

        last_step = (t >= t_final - 1e-8 * dt);

        if (offline)
        {
            if (basis_generator_x->isNextSample(t) || x_base_only == false
                    && basis_generator_v->isNextSample(t))
            {
                dvxdt = oper.dvxdt_sp.GetData();
                vx_diff = BlockVector(vx);
                vx_diff -= vx0;
            }

            // Take samples
            // NOTE(kevin): I don't know why this example checks next sample time.
            // IncrementalSVD is never turned on in this example and isNextSample is always true.
            if (x_base_only == false && basis_generator_v->isNextSample(t))
            {
                basis_generator_v->takeSample(vx_diff.GetBlock(0).GetData());
                basis_generator_v->computeNextSampleTime(vx_diff.GetBlock(0).GetData(),
                        dvdt.GetData(), t);
                basis_generator_H->takeSample(oper.H_sp.GetData());
            }

            if (basis_generator_x->isNextSample(t))
            {
                basis_generator_x->takeSample(vx_diff.GetBlock(1).GetData());
                basis_generator_x->computeNextSampleTime(vx_diff.GetBlock(1).GetData(),
                        dxdt.GetData(), t);

                if (x_base_only == true)
                {
                    basis_generator_H->takeSample(oper.H_sp.GetData());
                }
            }
        }

        if (last_step || (ti % vis_steps) == 0)
        {
            if (online)
            {
                BroadcastUndistributedRomVector(w);

                for (int i = 0; i < rvdim; ++i)
                    (*w_v)(i) = (*w)(i);

                for (int i = 0; i < rxdim; ++i)
                    (*w_x)(i) = (*w)(rvdim + i);

                romop->V_v.mult(*w_v, *v_rec_librom);
                romop->V_x.mult(*w_x, *x_rec_librom);

                *v_rec += vx0.GetBlock(0);
                *x_rec += vx0.GetBlock(1);

                v_gf.SetFromTrueDofs(*v_rec);
                x_gf.SetFromTrueDofs(*x_rec);
            }
            else
            {
                v_gf.SetFromTrueVector();
                x_gf.SetFromTrueVector();
            }

            double ee = oper.ElasticEnergy(x_gf);
            double ke = oper.KineticEnergy(v_gf);

            if (myid == 0)
            {
                cout << "step " << ti << ", t = " << t << ", EE = " << ee
                     << ", KE = " << ke << ", ΔTE = " << (ee + ke) - (ee0 + ke0) << endl;
            }

            if (visualization)
            {
                visualize(vis_v, pmesh, &x_gf, &v_gf);
                if (vis_w)
                {
                    oper.GetElasticEnergyDensity(x_gf, w_gf);
                    visualize(vis_w, pmesh, &x_gf, &w_gf);
                }
            }

            if (visit)
            {
                GridFunction *nodes = &x_gf;
                int owns_nodes = 0;
                pmesh->SwapNodes(nodes, owns_nodes);

                dc->SetCycle(ti);
                dc->SetTime(t);
                dc->Save();
            }
        }

    } // timestep loop

    if (myid == 0)
        cout << "Elapsed time for time integration loop " << solveTimer.RealTime() <<
             endl;

    ostringstream velo_name, pos_name;

    velo_name << "velocity_s" << s << "." << setfill('0') << setw(6) << myid;
    pos_name << "position_s" << s << "." << setfill('0') << setw(6) << myid;

    if (offline)
    {
        // Sample final solution, to prevent extrapolation in ROM between the
        // last sample and the end of the simulation.
        dvxdt = oper.dvxdt_sp.GetData();
        vx_diff = BlockVector(vx);
        vx_diff -= vx0;

        // Take samples
        if (x_base_only == false)
        {
            basis_generator_v->takeSample(vx_diff.GetBlock(0).GetData());
            basis_generator_v->writeSnapshot();
            delete basis_generator_v;
        }

        basis_generator_H->takeSample(oper.H_sp.GetData());
        basis_generator_H->writeSnapshot();
        delete basis_generator_H;

        basis_generator_x->takeSample(vx_diff.GetBlock(1).GetData());
        basis_generator_x->writeSnapshot();
        delete basis_generator_x;

        // 14. Save the displaced mesh, the velocity and elastic energy.
        GridFunction *nodes = &x_gf;
        int owns_nodes = 0;
        pmesh->SwapNodes(nodes, owns_nodes);

        ostringstream mesh_name, ee_name;
        mesh_name << "deformed." << setfill('0') << setw(6) << myid;
        ee_name << "elastic_energy." << setfill('0') << setw(6) << myid;

        ofstream mesh_ofs(mesh_name.str().c_str());
        mesh_ofs.precision(8);
        pmesh->Print(mesh_ofs);
        pmesh->SwapNodes(nodes, owns_nodes);
        ofstream velo_ofs(velo_name.str().c_str());
        velo_ofs.precision(16);

        Vector v_final(vx.GetBlock(0));
        for (int i = 0; i < v_final.Size(); ++i)
        {
            velo_ofs << v_final[i] << std::endl;
        }

        ofstream pos_ofs(pos_name.str().c_str());
        pos_ofs.precision(16);

        Vector x_final(vx.GetBlock(1));
        for (int i = 0; i < x_final.Size(); ++i)
        {
            pos_ofs << x_final[i] << std::endl;
        }

        ofstream ee_ofs(ee_name.str().c_str());
        ee_ofs.precision(8);
        oper.GetElasticEnergyDensity(x_gf, w_gf);
        w_gf.Save(ee_ofs);
    }

    // 15. Calculate the relative error between the ROM final solution and the true solution.
    if (online)
    {
        // Initialize FOM solution
        Vector v_fom(v_rec->Size());
        Vector x_fom(x_rec->Size());

        ifstream fom_v_file, fom_x_file;

        // Open and load file
        fom_v_file.open(velo_name.str().c_str());
        fom_x_file.open(pos_name.str().c_str());

        v_fom.Load(fom_v_file, v_rec->Size());
        x_fom.Load(fom_x_file, x_rec->Size());

        fom_v_file.close();
        fom_x_file.close();

        // Get difference vector
        Vector diff_v(v_rec->Size());
        Vector diff_x(x_rec->Size());

        subtract(*v_rec, v_fom, diff_v);
        subtract(*x_rec, x_fom, diff_x);

        // Get norms
        double tot_diff_norm_v = sqrt(InnerProduct(MPI_COMM_WORLD, diff_v, diff_v));
        double tot_diff_norm_x = sqrt(InnerProduct(MPI_COMM_WORLD, diff_x, diff_x));

        double tot_v_fom_norm = sqrt(InnerProduct(MPI_COMM_WORLD,
                                     v_fom, v_fom));
        double tot_x_fom_norm = sqrt(InnerProduct(MPI_COMM_WORLD,
                                     x_fom, x_fom));

        if (myid == 0)
        {
            cout << "Relative error of ROM position (x) at t_final: " << t_final
                 << " is " << tot_diff_norm_x / tot_x_fom_norm << endl;
            cout << "Relative error of ROM velocity (v) at t_final: " << t_final
                 << " is " << tot_diff_norm_v / tot_v_fom_norm << endl;
        }
    }

    // 16. Free the used memory.
    delete ode_solver;
    delete pmesh;
    delete romop;
    delete soper;
    delete Hsinv;
    delete smm;
    delete eqpSol;
    delete eqpSol_S;
    delete window_ids;
    delete load_eqpsol;

    totalTimer.Stop();
    if (myid == 0)
        cout << "Elapsed time for entire simulation " << totalTimer.RealTime() << endl;

    MPI_Finalize();
    return 0;
}

void visualize(ostream &out, ParMesh *mesh, ParGridFunction *deformed_nodes,
               ParGridFunction *field, const char *field_name, bool init_vis)
{
    if (!out)
    {
        return;
    }

    GridFunction *nodes = deformed_nodes;
    int owns_nodes = 0;

    mesh->SwapNodes(nodes, owns_nodes);

    out << "parallel " << mesh->GetNRanks() << " " << mesh->GetMyRank() << "\n";
    out << "solution\n"
        << *mesh << *field;

    mesh->SwapNodes(nodes, owns_nodes);

    if (init_vis)
    {
        out << "window_size 800 800\n";
        out << "window_title '" << field_name << "'\n";
        if (mesh->SpaceDimension() == 2)
        {
            out << "view 0 0\n"; // view from top
            out << "keys jl\n";  // turn off perspective and light
        }
        out << "keys cm\n";         // show colorbar and mesh
        out << "autoscale value\n"; // update value-range; keep mesh-extents fixed
    }
    out << flush;
}

HyperelasticOperator::HyperelasticOperator(ParFiniteElementSpace &f,
        Array<int> &ess_tdof_list_, double visc,
        double mu, double K)
    : TimeDependentOperator(2 * f.TrueVSize(), 0.0), fespace(f),
      ess_tdof_list(ess_tdof_list_),
      M(NULL), S(NULL), H(NULL),
      viscosity(visc), M_solver(f.GetComm()),
      z(height / 2), z2(height / 2), H_sp(height / 2), dvxdt_sp(height / 2)
{
    const double rel_tol = 1e-8;
    const int skip_zero_entries = 0;

    const double ref_density = 1.0; // density in the reference configuration
    ConstantCoefficient rho0(ref_density);

    M = new ParBilinearForm(&fespace);
    M->AddDomainIntegrator(new VectorMassIntegrator(rho0));
    M->Assemble(skip_zero_entries);
    M->Finalize(skip_zero_entries);
    Mmat = M->ParallelAssemble();
    HypreParMatrix *Me = Mmat->EliminateRowsCols(ess_tdof_list);
    delete Me;

    M_solver.iterative_mode = false;
    M_solver.SetRelTol(rel_tol);
    M_solver.SetAbsTol(0.0);
    M_solver.SetMaxIter(1000);
    M_solver.SetPrintLevel(0);
    M_prec.SetType(HypreSmoother::Jacobi);
    M_solver.SetPreconditioner(M_prec);
    M_solver.SetOperator(*Mmat);

    H = new ParNonlinearForm(&fespace);
    model = new NeoHookeanModel(mu, K);
    H->AddDomainIntegrator(new HyperelasticNLFIntegrator(model));
    H->SetEssentialTrueDofs(ess_tdof_list);

    ConstantCoefficient visc_coeff(viscosity);
    S = new ParBilinearForm(&fespace);
    S->AddDomainIntegrator(new VectorDiffusionIntegrator(visc_coeff));
    S->Assemble(skip_zero_entries);
    S->Finalize(skip_zero_entries);
    S->FormSystemMatrix(ess_tdof_list, Smat);
}

void HyperelasticOperator::Mult(const Vector &vx, Vector &dvx_dt) const
{
    // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
    int sc = height / 2;
    Vector v(vx.GetData() + 0, sc);
    Vector x(vx.GetData() + sc, sc);
    Vector dv_dt(dvx_dt.GetData() + 0, sc);
    Vector dx_dt(dvx_dt.GetData() + sc, sc);

    H->Mult(x, z);
    H_sp = z;

    if (viscosity != 0.0)
    {
        Smat.Mult(v, z2);
        z += z2;
    }
    z.Neg(); // z = -z
    M_solver.Mult(z, dv_dt);

    dx_dt = v;

    dvxdt_sp = dvx_dt;
}

double HyperelasticOperator::ElasticEnergy(const ParGridFunction &x) const
{
    return H->GetEnergy(x);
}

double HyperelasticOperator::KineticEnergy(const ParGridFunction &v) const
{
    double loc_energy = 0.5 * M->InnerProduct(v, v);
    double energy;
    MPI_Allreduce(&loc_energy, &energy, 1, MPI_DOUBLE, MPI_SUM,
                  fespace.GetComm());
    return energy;
}

void HyperelasticOperator::GetElasticEnergyDensity(
    const ParGridFunction &x, ParGridFunction &w) const
{
    ElasticEnergyCoefficient w_coeff(*model, x);
    w.ProjectCoefficient(w_coeff);
}

HyperelasticOperator::~HyperelasticOperator()
{
    delete model;
    delete Mmat;
    delete M;
    delete S;
    delete H;
}

double ElasticEnergyCoefficient::Eval(ElementTransformation &T,
                                      const IntegrationPoint &ip)
{
    model.SetTransformation(T);
    x.GetVectorGradient(T, J);
    // return model.EvalW(J);  // in reference configuration
    return model.EvalW(J) / J.Det(); // in deformed configuration
}

void InitialDeformationIC1(const Vector &x, Vector &y)
{
    y = x;
}

void InitialVelocityIC1(const Vector &x, Vector &v)
{
    const int dim = x.Size();
    const double s_eff = s / 80.0;

    v = 0.0;
    v(dim - 1) = -s_eff * sin(s * x(0));
}

void InitialDeformationIC2(const Vector &x, Vector &y) // See MFEM ex19
{
    // Set the initial configuration. Having this different from the reference
    // configuration can help convergence
    const int dim = x.Size();
    const double s_eff = s;
    y = x;
    y(dim - 1) = x(dim - 1) + 0.25 * x(0) * s_eff;
}

void InitialVelocityIC2(const Vector &x, Vector &v)
{
    v = 0.0;
}

RomOperator::RomOperator(HyperelasticOperator *fom_,
                         HyperelasticOperator *fomSp_, const int rvdim_, const int rxdim_,
                         const int hdim_, CAROM::SampleMeshManager *smm_, const Vector *v0_,
                         const Vector *x0_, const Vector v0_fom_,
                         const std::shared_ptr<CAROM::Matrix> V_v_,
                         const std::shared_ptr<CAROM::Matrix> V_x_,
                         const std::shared_ptr<CAROM::Matrix> U_H_,
                         const CAROM::Matrix *Hsinv_, const int myid, const bool oversampling_,
                         const bool hyperreduce_, const bool x_base_only_, const bool use_eqp,
                         CAROM::Vector *eqpSol, const IntegrationRule *ir_eqp_, NeoHookeanModel *model_)
    : TimeDependentOperator(rxdim_ + rvdim_, 0.0), fom(fom_), fomSp(fomSp_),
      rxdim(rxdim_), rvdim(rvdim_), hdim(hdim_), x0(x0_), v0(v0_), v0_fom(v0_fom_),
      smm(smm_), V_x(*V_x_), V_v(*V_v_), U_H(U_H_), Hsinv(Hsinv_),
      M_hat_solver(fom_->fespace.GetComm()), oversampling(oversampling_),
      z(height / 2), hyperreduce(hyperreduce_), x_base_only(x_base_only_),
      eqp(use_eqp),
      ir_eqp(ir_eqp_), model(model_), rank(myid)
{
    if (!eqp)
    {
        nsamp_H = smm_->GetNumVarSamples("H");
        zN = CAROM::Vector(std::max(nsamp_H, 1), false);
        zX = CAROM::Vector(std::max(nsamp_H, 1), false);
    }

    if (myid == 0 && !eqp)
    {
        V_v_sp = new CAROM::Matrix(fomSp->Height() / 2, rvdim, false);
        V_x_sp = new CAROM::Matrix(fomSp->Height() / 2, rxdim, false);
    }

    // Gather distributed vectors
    if (!eqp)
    {
        if (x_base_only)
        {
            smm->GatherDistributedMatrixRows("X", V_v, rvdim, *V_v_sp);
        }
        else
        {
            smm->GatherDistributedMatrixRows("V", V_v, rvdim, *V_v_sp);
        }

        smm->GatherDistributedMatrixRows("X", V_x, rxdim, *V_x_sp);
        // Create V_vTU_H, for hyperreduction
        V_v.transposeMult(*U_H, V_vTU_H);
    }

    S_hat = new CAROM::Matrix(rvdim, rvdim, false);
    S_hat_v0 = new CAROM::Vector(rvdim, false);
    S_hat_v0_temp = new Vector(v0_fom.Size());
    S_hat_v0_temp_librom = new CAROM::Vector(S_hat_v0_temp->GetData(),
            S_hat_v0_temp->Size(), true, false);
    M_hat = new CAROM::Matrix(rvdim, rvdim, false);
    M_hat_inv = new CAROM::Matrix(rvdim, rvdim, false);

    // Set the max iterations for the mass matrix solver
    M_hat_solver.SetMaxIter(1000);

    // Create S_hat
    ComputeCtAB(fom->Smat, V_v, V_v, *S_hat);
    // Apply S_hat to the initial velocity and store
    fom->Smat.Mult(v0_fom, *S_hat_v0_temp);
    V_v.transposeMult(*S_hat_v0_temp_librom, *S_hat_v0);
    // Create M_hat
    ComputeCtAB(*(fom->Mmat), V_v, V_v, *M_hat);

    // Invert M_hat and store
    M_hat->inverse(*M_hat_inv);

    if (myid == 0 && hyperreduce)
    {
        if (!eqp)
        {
            const int spdim = fomSp->Height(); // Reduced height

            // Allocate auxillary vectors
            z.SetSize(spdim / 2);
            z_v.SetSize(spdim / 2);
            z_x.SetSize(spdim / 2);
            zH.SetSize(spdim / 2); // Samples of H
            z_librom = new CAROM::Vector(z.GetData(), z.Size(), false, false);
            z_v_librom = new CAROM::Vector(z_v.GetData(), z_v.Size(), false, false);
            z_x_librom = new CAROM::Vector(z_x.GetData(), z_x.Size(), false, false);

            v0_librom = new CAROM::Vector(v0->GetData(), v0->Size(), false, false);
            V_xTV_v_sp = new CAROM::Matrix(rxdim, rvdim, false);
            V_xTv_0_sp = new CAROM::Vector(spdim / 2, false);
            V_x_sp->transposeMult(*V_v_sp, *V_xTV_v_sp);
            V_x_sp->transposeMult(*v0_librom, *V_xTv_0_sp);

            // This is for saving the recreated predictions
            psp_librom = new CAROM::Vector(spdim, false);
            psp = new Vector(&((*psp_librom)(0)), spdim);

            // Define sub-vectors of psp.
            psp_x = new Vector(psp->GetData(), spdim / 2);
            psp_v = new Vector(psp->GetData() + spdim / 2, spdim / 2);

            psp_v_librom = new CAROM::Vector(psp_v->GetData(), psp_v->Size(), false, false);
        }
        else
        {
            z.SetSize(rvdim);
            z_librom = new CAROM::Vector(z.GetData(), z.Size(), false, false);
        }
    }

    const int fdim = fom->Height(); // Unreduced height
    if (!hyperreduce || (eqp && myid == 0))
    {
        // This is for saving the recreated predictions
        pfom_librom = new CAROM::Vector(fdim, false);
        pfom = new Vector(&((*pfom_librom)(0)), fdim);
        // Define sub-vectors of pfom.
        pfom_x = new Vector(pfom->GetData(), fdim / 2);
        pfom_v = new Vector(pfom->GetData() + fdim / 2, fdim / 2);
        zfom_x = new Vector(fdim / 2);
        zfom_x_librom = new CAROM::Vector(zfom_x->GetData(), zfom_x->Size(), true,
                                          false);

        pfom_v_librom = new CAROM::Vector(pfom_v->GetData(), pfom_v->Size(), false,
                                          false);

        // Auxiliary vectors
        z_v.SetSize(fdim / 2);
        z_v_librom = new CAROM::Vector(z_v.GetData(), z_v.Size(), true, false);
        v0_fom_librom = new CAROM::Vector(v0_fom.GetData(), v0_fom.Size(), true, false);
        V_xTV_v = new CAROM::Matrix(rxdim, rvdim, false);
        V_xTv_0 = new CAROM::Vector(fdim / 2, false);
        V_x.transposeMult(V_v, *V_xTV_v);
        V_x.transposeMult(*v0_fom_librom, *V_xTv_0);
        Vx_librom_temp = new CAROM::Vector(fdim / 2, true);
        Vx_temp = new Vector(&((*Vx_librom_temp)(0)), fdim / 2);
    }

    if (!hyperreduce)
    {
        z.SetSize(fdim / 2);

        z_x.SetSize(fdim / 2);
        z_librom = new CAROM::Vector(z.GetData(), z.Size(), false, false);

        z_x_librom = new CAROM::Vector(z_x.GetData(), z_x.Size(), true, false);
    }

    if (eqp)
    {
        std::set<int> elements;
        const int nqe = ir_eqp->GetWeights().Size();

        for (int i = 0; i < eqpSol->dim(); ++i)
        {
            if ((*eqpSol)(i) > 1.0e-12)
            {
                const int e = i / nqe; // Element index
                elements.insert(e);
                eqp_rw.push_back((*eqpSol)(i));
                eqp_qp.push_back(i);
            }
        }

        cout << myid << ": EQP using " << elements.size() << " elements out of "
             << fom->fespace.GetNE() << endl;
        const FiniteElement &fe1 = *(fom->fespace).GetFE(0);
        const int dof = fe1.GetDof();
        const int dim = fe1.GetDim();
        em = new ElemMatrices(dof, dim);

        GetEQPCoefficients_HyperelasticNLFIntegrator(&(fom->fespace), eqp_rw, eqp_qp,
                ir_eqp, model, V_v, rank, eqp_coef, eqp_DS_coef, em);

        // Set the matrix of the linear map from the ROM X space to all DOFs on
        // sampled elements.
        Array<int> vdofs;
        int numSampledDofs = 0;
        for (auto e : elements)
        {
            fom->fespace.GetElementVDofs(e, vdofs);
            numSampledDofs += vdofs.Size();
        }

        eqp_lifting.setSize(numSampledDofs, rxdim);
        eqp_liftDOFs.resize(numSampledDofs);
        eqp_lifted.setSize(numSampledDofs);

        int cnt = 0;
        for (auto e : elements)
        {
            fom->fespace.GetElementVDofs(e, vdofs);
            for (int i = 0; i < vdofs.Size(); ++i, ++cnt)
                eqp_liftDOFs[cnt] =
                    vdofs[i]; // eqp_liftDOFs maps an index of eqp_lifted to the FOM dof index
        }

        MFEM_VERIFY(cnt == numSampledDofs, "");

        CAROM::Vector ej(rxdim, false);

        for (int j = 0; j < rxdim; ++j)
        {
            ej = 0.0;
            ej(j) = 1.0;
            V_x.mult(ej, *zfom_x_librom);

            for (int i = 0; i < numSampledDofs; ++i)
            {
                eqp_lifting(i, j) = (*zfom_x)[eqp_liftDOFs[i]];
            }
        }
    }
}

RomOperator::~RomOperator()
{
    delete S_hat;
    delete M_hat;
    delete M_hat_inv;
}

void RomOperator::Mult_Hyperreduced(const Vector &vx, Vector &dvx_dt) const
{
    // Check that the sizes match
    MFEM_VERIFY(vx.Size() == rvdim + rxdim && dvx_dt.Size() == rvdim + rxdim, "");

    // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
    Vector v(vx.GetData() + 0, rvdim);
    CAROM::Vector v_librom(vx.GetData(), rvdim, false, false);
    CAROM::Vector x_librom(vx.GetData() + rvdim, rxdim, false, false);
    Vector dv_dt(dvx_dt.GetData() + 0, rvdim);
    Vector dx_dt(dvx_dt.GetData() + rvdim, rxdim);
    CAROM::Vector dv_dt_librom(dv_dt.GetData(), rvdim, false, false);
    CAROM::Vector dx_dt_librom(dx_dt.GetData(), rxdim, false, false);

    if (eqp)
    {   // Lift v-vector and save
        V_xTV_v->mult(v_librom, *pfom_v_librom);
        pfom_v_librom->plus(*V_xTv_0, dx_dt_librom);
        Vector resEQP;
        if (fastIntegration)
        {
            HyperelasticNLFIntegrator_ComputeReducedEQP_Fast(&(fom->fespace), eqp_rw,
                    eqp_qp, ir_eqp, model, x0, rvdim, x_librom,eqp_coef, eqp_DS_coef, rank, resEQP,
                    em, eqp_lifting, eqp_liftDOFs, eqp_lifted);
        }
        else
            HyperelasticNLFIntegrator_ComputeReducedEQP(&(fom->fespace), eqp_rw, eqp_qp,
                    ir_eqp, model, x0, V_v, x_librom, rank, resEQP, em, eqp_lifting, eqp_liftDOFs,
                    eqp_lifted);
        Vector recv(resEQP);
        MPI_Allreduce(resEQP.GetData(), recv.GetData(), resEQP.Size(), MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        resEQP = recv;

        // NOTE: in the hyperreduction case, the residual is of dimension nldim,
        // which is the dimension of the ROM space for the nonlinear term.
        // In the EQP case, there is no use of a ROM space for the nonlinear
        // term. Instead, the FOM computation of the nonlinear term is
        // approximated by the reduced quadrature rule in the FOM space.
        // Therefore, the residual here is of dimension rxdim.
        z = 0.0;
        MFEM_VERIFY(resEQP.Size() == rvdim, "");
        for (int i = 0; i < rvdim; ++i)
            z[i] += resEQP[i];
    }
    else
    {
        // Lift x- and v-vector
        // I.e. perform v = v0 + V_v v^, where v^ is the input
        V_xTV_v_sp->mult(v_librom, *z_v_librom);
        z_v_librom->plus(*V_xTv_0_sp, dx_dt_librom); // Store dx/dt

        V_x_sp->mult(x_librom, *z_x_librom);

        // Store liftings
        add(z_x, *x0, *psp_x);

        // Hyperreduce H
        // Apply H to x to get zH
        fomSp->H->Mult(*psp_x, zH);

        // Sample the values from zH
        smm->GetSampledValues("H", zH, zN);

        // Apply inverse H-basis
        if (oversampling)
        {
            Hsinv->transposeMult(zN, zX);
        }
        else
        {
            Hsinv->mult(zN, zX);
        }

        // Multiply by V_v^T * U_H
        V_vTU_H.mult(zX, *z_librom);
    }

    if (fomSp->viscosity != 0.0)
    {
        // Apply S^, the reduced S operator, to v
        S_hat->multPlus(*z_librom, v_librom, 1.0);
        *z_librom += *S_hat_v0;
    }
    z.Neg(); // z = -z
    M_hat_inv->mult(*z_librom,
                    dv_dt_librom); // to invert reduced mass matrix operator.
}

void RomOperator::Mult_FullOrder(const Vector &vx, Vector &dvx_dt) const
{
    // Check that the sizes match
    MFEM_VERIFY(vx.Size() == rvdim + rxdim && dvx_dt.Size() == rvdim + rxdim, "");

    // Create views to the sub-vectors v, x of vx, and dv_dt, dx_dt of dvx_dt
    Vector v(vx.GetData() + 0, rvdim);
    CAROM::Vector v_librom(vx.GetData(), rvdim, false, false);
    CAROM::Vector x_librom(vx.GetData() + rvdim, rxdim, false, false);
    Vector dv_dt(dvx_dt.GetData() + 0, rvdim);
    Vector dx_dt(dvx_dt.GetData() + rvdim, rxdim);
    CAROM::Vector dv_dt_librom(dv_dt.GetData(), rvdim, false, false);
    CAROM::Vector dx_dt_librom(dx_dt.GetData(), rxdim, false, false);

    // Lift the x-, and v-vectors
    // I.e. perform v = v0 + V_v v^, where v^ is the input

    V_xTV_v->mult(v_librom, *pfom_v_librom);
    pfom_v_librom->plus(*V_xTv_0, dx_dt_librom);

    V_x.mult(x_librom, *z_x_librom);

    add(z_x, *x0, *pfom_x); // Store liftings

    // Apply H to x to get z
    fom->H->Mult(*pfom_x, *zfom_x);

    V_v.transposeMult(*zfom_x_librom, *z_librom);

    if (fom->viscosity != 0.0)
    {
        // Apply S^, the reduced S operator, to v
        S_hat->multPlus(*z_librom, v_librom, 1.0);
        *z_librom += *S_hat_v0;
    }

    z.Neg(); // z = -z
    M_hat_inv->mult(*z_librom,
                    dv_dt_librom); // to invert reduced mass matrix operator.
}

void RomOperator::Mult(const Vector &vx, Vector &dvx_dt) const
{
    if (hyperreduce)
        Mult_Hyperreduced(vx, dvx_dt);
    else
        Mult_FullOrder(vx, dvx_dt);
}

void RomOperator::SetEQP(CAROM::Vector *eqpSol)
{
    std::set<int> elements;
    const int nqe = ir_eqp->GetWeights().Size();
    eqp_rw.clear();
    eqp_qp.clear();

    for (int i = 0; i < eqpSol->dim(); ++i)
    {
        if ((*eqpSol)(i) > 1.0e-12)
        {
            const int e = i / nqe; // Element index
            elements.insert(e);
            eqp_rw.push_back((*eqpSol)(i));
            eqp_qp.push_back(i);
        }
    }

    cout << rank << ": EQP using " << elements.size() << " elements out of "
         << fom->fespace.GetNE() << endl;

    GetEQPCoefficients_HyperelasticNLFIntegrator(&(fom->fespace), eqp_rw, eqp_qp,
            ir_eqp, model, V_v, rank, eqp_coef, eqp_DS_coef, em);

    // Set the matrix of the linear map from the ROM X space to all DOFs on
    // sampled elements.
    Array<int> vdofs;
    int numSampledDofs = 0;
    for (auto e : elements)
    {
        fom->fespace.GetElementVDofs(e, vdofs);
        numSampledDofs += vdofs.Size();
    }

    eqp_lifting.setSize(numSampledDofs, rxdim);
    eqp_liftDOFs.resize(numSampledDofs);
    eqp_lifted.setSize(numSampledDofs);

    int cnt = 0;
    for (auto e : elements)
    {
        fom->fespace.GetElementVDofs(e, vdofs);
        for (int i = 0; i < vdofs.Size(); ++i, ++cnt)
            eqp_liftDOFs[cnt] =
                vdofs[i]; // eqp_liftDOFs maps an index of eqp_lifted to the FOM dof index
    }

    MFEM_VERIFY(cnt == numSampledDofs, "");

    CAROM::Vector ej(rxdim, false);

    for (int j = 0; j < rxdim; ++j)
    {
        ej = 0.0;
        ej(j) = 1.0;
        V_x.mult(ej, *zfom_x_librom);

        for (int i = 0; i < numSampledDofs; ++i)
        {
            eqp_lifting(i, j) = (*zfom_x)[eqp_liftDOFs[i]];
        }
    }
}

// Functions for EQP functionality

// Compute coefficients of the reduced integrator with respect to inputs Q and x
// in HyperelasticNLFIntegrator_ComputeReducedEQP.
void GetEQPCoefficients_HyperelasticNLFIntegrator(ParFiniteElementSpace *fesR,
        std::vector<double> const &rw, std::vector<int> const &qp,
        const IntegrationRule *ir, NeoHookeanModel *model,
        CAROM::Matrix const &V_v, const int rank, Vector &coef, Vector &DS_coef,
        ElemMatrices *em)
{
    const int rvdim = V_v.numColumns();
    const int fomdim = V_v.numRows();
    MFEM_VERIFY(rw.size() == qp.size(), "");
    MFEM_VERIFY(fomdim == fesR->GetTrueVSize(), "");

    MFEM_VERIFY(rank == 0,
                "TODO: generalize to parallel. This uses full dofs in V, which has true dofs");

    const int nqe = ir->GetWeights().Size();

    DofTransformation *doftrans;
    ElementTransformation *eltrans;
    Array<int> vdofs;

    int eprev = -1;
    int dof = 0;
    int dim = 0;

    // Get prolongation matrix
    const Operator *P = fesR->GetProlongationMatrix();

    // Vector to be prolongated
    Vector vj(fomdim);

    // Prolongated vector
    Vector p_vj(P->Height());

    // Element vector
    Vector vj_e;
    // Get the element vector size
    doftrans = fesR->GetElementVDofs(0, vdofs);
    int elvect_size = vdofs.Size();

    // Coefficient vector
    coef.SetSize(elvect_size * rw.size() * rvdim);
    coef = 0.0;

    // Vector for storing DS
    const FiniteElement *fe = fesR->GetFE(0);
    dof = fe->GetDof();
    dim = fe->GetDim();
    DS_coef.SetSize(dof * dim * rw.size());
    DS_coef = 0.0;
    int index = 0;
    eprev = -1;
    // For every quadrature weight
    for (int i = 0; i < rw.size(); ++i)
    {
        const int e = qp[i] / nqe; // Element index
        // Local (element) index of the quadrature point
        const int qpi = qp[i] - (e * nqe);
        const IntegrationPoint &ip = ir->IntPoint(qpi);

        if (e != eprev) // Update element transformation
        {
            doftrans = fesR->GetElementVDofs(e, vdofs);
            eltrans = fesR->GetElementTransformation(e);

            if (doftrans)
            {
                MFEM_ABORT("TODO: Implementation for when `doftrans` is not NULL");
            }
            eprev = e;
        }

        // Set integration point in the element transformation
        eltrans->SetIntPoint(&ip);
        model->SetTransformation(*eltrans);
        // Get the transformation weight
        const double t = eltrans->Weight();

        // Calculate DS and store
        CalcInverse(eltrans->Jacobian(), em->Jrt);
        fe->CalcDShape(ip, em->DSh);
        Mult(em->DSh, em->Jrt, em->DS);
        for (int ii = 0; ii < dof; ++ii)
        {
            for (int jj = 0; jj < dim; ++jj)
            {
                index = jj + ii * dim;
                DS_coef[index + (i * dof * dim)] = em->DS.Elem(ii, jj);
            }
        }
        // For every basis vector
        for (int j = 0; j < rvdim; ++j)
        {
            // Get basis vector and prolongate
            for (int k = 0; k < V_v.numRows(); ++k)
                vj[k] = V_v(k, j);
            P->Mult(vj, p_vj);

            // Get element vectors
            p_vj.GetSubVector(vdofs, vj_e);

            // Calculate coeff
            for (int k = 0; k < elvect_size; k++)
            {
                coef[k + (i * elvect_size) + (j * rw.size() * elvect_size)] = vj_e[k] * rw[i] *
                        t;
            }
        }
    }
}

void HyperelasticNLFIntegrator_ComputeReducedEQP(ParFiniteElementSpace *fesR,
        std::vector<double> const &rw, std::vector<int> const &qp,
        const IntegrationRule *ir, NeoHookeanModel *model, const Vector *x0,
        CAROM::Matrix const &V_v, CAROM::Vector const &x, const int rank, Vector &res,
        ElemMatrices *em, const CAROM::Matrix eqp_lifting,
        const std::vector<int> eqp_liftDOFs,CAROM::Vector eqp_lifted)

{
    const int rvdim = V_v.numColumns();
    const int fomdim = V_v.numRows();
    MFEM_VERIFY(rw.size() == qp.size(), "");

    MFEM_VERIFY(rank == 0,
                "TODO: generalize to parallel. This uses full dofs in V, which has true dofs");

    const int nqe = ir->GetWeights().Size();

    ElementTransformation *eltrans;
    DofTransformation *doftrans;
    const FiniteElement *fe = NULL;
    Array<int> vdofs;

    res.SetSize(rvdim);
    res = 0.0;

    int eprev = -1;
    int dof = 0;
    int dim = 0;

    // Basis vector
    Vector vj(fomdim);

    // Element vectors
    Vector Vx_e;
    Vector vj_e;

    // Lift x, add x0
    eqp_lifting.mult(x, eqp_lifted);

    for (int i = 0; i < eqp_liftDOFs.size(); ++i)
        eqp_lifted(i) += x0->Elem(eqp_liftDOFs[i]);

    // Initialize nonlinear operator storage
    // Assuming all elements have the same dim and n_dof
    fe = fesR->GetFE(0);
    dof = fe->GetDof();
    dim = fe->GetDim();
    Vx_e.SetSize(dof * dim);

    eprev = -1;
    double temp = 0.0;
    int eqp_ctr = 0;
    // For every quadrature weight
    for (int i = 0; i < rw.size(); ++i)
    {
        const int e = qp[i] / nqe; // Element index
        // Local (element) index of the quadrature point
        const int qpi = qp[i] - (e * nqe);
        const IntegrationPoint &ip = ir->IntPoint(qpi);

        if (e != eprev) // Update element transformation
        {
            doftrans = fesR->GetElementVDofs(e, vdofs);
            fe = fesR->GetFE(e);
            eltrans = fesR->GetElementTransformation(e);

            dof = fe->GetDof(); // Get number of dofs in element
            dim = fe->GetDim();

            if (doftrans)
            {
                MFEM_ABORT("TODO: Implementation for when `doftrans` is not NULL");
            }

            // Get element vectors
            for (int i = 0; i < dof * dim; ++i)
            {
                Vx_e.Elem(i) = eqp_lifted(eqp_ctr * dof * dim + i);
            }
            eqp_ctr++;
            eprev = e;
        }

        // Integration at ip
        em->PMatI.UseExternalData(Vx_e.GetData(), dof, dim);

        em->elvect = 0.0;

        // Set integration point in the element transformation
        eltrans->SetIntPoint(&ip);
        model->SetTransformation(*eltrans);

        // Get the transformation weight
        double t = eltrans->Weight();

        // Compute action of nonlinear operator
        CalcInverse(eltrans->Jacobian(), em->Jrt);
        fe->CalcDShape(ip, em->DSh);
        Mult(em->DSh, em->Jrt, em->DS);
        MultAtB(em->PMatI, em->DS, em->Jpt);
        model->EvalP(em->Jpt, em->P_f);
        em->P_f *= (t * rw[i]); // NB: Not by ip.weight
        AddMultABt(em->DS, em->P_f, em->PMatO);

        // For every basis vector
        for (int j = 0; j < rvdim; ++j)
        {
            // Get basis vector and prolongate
            for (int k = 0; k < V_v.numRows(); ++k)
                vj[k] = V_v(k, j);

            vj.GetSubVector(vdofs, vj_e);

            temp = 0.0;

            // Calculate r[i] = ve_j^T * elvect
            for (int k = 0; k < em->elvect.Size(); k++)
            {
                temp += vj_e[k] * em->elvect[k];
            }
            res[j] += temp;
        }
    }
}

void HyperelasticNLFIntegrator_ComputeReducedEQP_Fast(ParFiniteElementSpace
        *fesR,std::vector<double> const &rw, std::vector<int> const &qp,
        const IntegrationRule *ir, NeoHookeanModel *model, const Vector *x0,
        const int rvdim,
        CAROM::Vector const &x, Vector const &coef, Vector const &DS_coef,
        const int rank, Vector &res,
        ElemMatrices *em, const CAROM::Matrix eqp_lifting,
        const std::vector<int> eqp_liftDOFs,CAROM::Vector eqp_lifted)
{

    MFEM_VERIFY(rank == 0,
                "TODO: generalize to parallel. This uses full dofs in X, which has true dofs");

    const int nqe = ir->GetWeights().Size();

    ElementTransformation *eltrans;
    DofTransformation *doftrans;
    const FiniteElement *fe = NULL;
    Array<int> vdofs;

    res.SetSize(rvdim);
    res = 0.0;

    int eprev = -1;
    int dof = 0;
    int dim = 0;

    // Element vectors
    Vector Vx_e;

    // Lift x, add x0
    eqp_lifting.mult(x, eqp_lifted);

    for (int i = 0; i < eqp_liftDOFs.size(); ++i)
        eqp_lifted(i) += x0->Elem(eqp_liftDOFs[i]);

    // Initialize nonlinear operator storage
    // Assuming all elements have the same dim and n_dof
    fe = fesR->GetFE(0);
    dof = fe->GetDof();
    dim = fe->GetDim();
    Vx_e.SetSize(dof * dim);
    int index = 0;

    eprev = -1;
    double temp = 0.0;
    int eqp_ctr = 0;

    // For every quadrature weight
    for (int i = 0; i < qp.size(); ++i)
    {
        const int e = qp[i] / nqe; // Element index
        // Local (element) index of the quadrature point
        const int qpi = qp[i] - (e * nqe);
        const IntegrationPoint &ip = ir->IntPoint(qpi);

        if (e != eprev) // Update element transformation
        {
            doftrans = fesR->GetElementVDofs(e, vdofs);
            fe = fesR->GetFE(e);
            eltrans = fesR->GetElementTransformation(e);

            dof = fe->GetDof(); // Get number of dofs in element
            dim = fe->GetDim();

            if (doftrans)
            {
                MFEM_ABORT("TODO: Implementation for when `doftrans` is not NULL");
            }

            // Get element vectors
            for (int i = 0; i < dof * dim; ++i)
            {
                Vx_e.Elem(i) = eqp_lifted(eqp_ctr * dof * dim + i);
            }
            eqp_ctr++;

            eprev = e;
        }

        // Integration at ip
        em->elvect = 0.0;
        em->PMatI.UseExternalData(Vx_e.GetData(), dof, dim);

        // Set integration point in the element transformation
        eltrans->SetIntPoint(&ip);
        model->SetTransformation(*eltrans);

        const int elvec_size = em->elvect.Size();
        for (int ii = 0; ii < dof; ++ii)
        {
            for (int jj = 0; jj < dim; ++jj)
            {
                index = jj + ii * dim;
                em->DS.Elem(ii, jj) = DS_coef[index + (i * elvec_size)];
            }
        }

        MultAtB(em->PMatI, em->DS, em->Jpt);
        model->EvalP(em->Jpt, em->P_f);
        AddMultABt(em->DS, em->P_f, em->PMatO);

        // Calculate r[i] = ve_j^T * elvect
        // coef is size len(vdofs) * rvdim * rw.size
        const int qp_size = qp.size();
        for (int j = 0; j < rvdim; ++j)
        {
            temp = 0.0;
            for (int k = 0; k < elvec_size; k++)
            {
                temp += coef[k + (i * elvec_size) + (j * qp_size * elvec_size)] *
                        em->elvect[k];
            }
            res[j] += temp;
        }
    }
}

void ComputeElementRowOfG(const IntegrationRule *ir, Array<int> const &vdofs,
                          Vector const &ve_j, NeoHookeanModel *model, Vector const &elfun,
                          FiniteElement const &fe, ElementTransformation &Trans, Vector &r, const int dof,
                          const int dim,
                          ElemMatrices &em)
{
    MFEM_VERIFY(r.Size() == ir->GetNPoints(), "");

    em.PMatI.UseExternalData(elfun.GetData(), dof, dim);
    model->SetTransformation(Trans);

    // For each integration point
    for (int i = 0; i < ir->GetNPoints(); i++)
    {
        em.elvect = 0.0;
        // Get integration point
        const IntegrationPoint &ip = ir->IntPoint(i);

        // Set integration point in the element transformation
        Trans.SetIntPoint(&ip);

        // Get the transformation weight
        double t = Trans.Weight();

        // Compute action of nonlinear operator
        CalcInverse(Trans.Jacobian(), em.Jrt);
        fe.CalcDShape(ip, em.DSh);
        Mult(em.DSh, em.Jrt, em.DS);
        MultAtB(em.PMatI, em.DS, em.Jpt);
        model->EvalP(em.Jpt, em.P);
        em.P *= t; // NB: Not by ip.weight
        AddMultABt(em.DS, em.P, em.PMatO);

        r[i] = 0.0;

        // Calculate r[i] = ve_j^T * elvect
        for (int k = 0; k < em.elvect.Size(); k++)
        {
            r[i] += ve_j[k] * em.elvect[k];
        }
    }
}

void SolveNNLS(const int rank, const double nnls_tol, const int maxNNLSnnz,
               CAROM::Vector const &w, CAROM::Matrix &Gt,
               CAROM::Vector &sol)
{
    CAROM::NNLSSolver nnls(nnls_tol, 0, maxNNLSnnz, 2);

    CAROM::Vector rhs_ub(Gt.numColumns(), false);
    // G.mult(w, rhs_ub);  // rhs = Gw
    // rhs = Gw. Note that by using Gt and multTranspose, we do parallel communication.
    Gt.transposeMult(w, rhs_ub);

    CAROM::Vector rhs_lb(rhs_ub);
    CAROM::Vector rhs_Gw(rhs_ub);

    const double delta = 1.0e-11;
    for (int i = 0; i < rhs_ub.dim(); ++i)
    {
        rhs_lb(i) -= delta;
        rhs_ub(i) += delta;
    }

    // nnls.normalize_constraints(Gt, rhs_lb, rhs_ub);
    nnls.solve_parallel_with_scalapack(Gt, rhs_lb, rhs_ub, sol);

    int nnz = 0;
    for (int i = 0; i < sol.dim(); ++i)
    {
        if (sol(i) != 0.0)
        {
            nnz++;
        }
    }

    cout << rank << ": Number of nonzeros in NNLS solution: " << nnz
         << ", out of " << sol.dim() << endl;

    MPI_Allreduce(MPI_IN_PLACE, &nnz, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (rank == 0)
        cout << "Global number of nonzeros in NNLS solution: " << nnz << endl;

    // Check residual of NNLS solution
    CAROM::Vector res(Gt.numColumns(), false);
    Gt.transposeMult(sol, res);

    const double normGsol = res.norm();
    const double normRHS = rhs_Gw.norm();

    res -= rhs_Gw;
    const double relNorm = res.norm() / std::max(normGsol, normRHS);
    cout << rank << ": relative residual norm for NNLS solution of Gs = Gw: " <<
         relNorm << endl;
}

int getSteps(const int nsnap, const int snap_step)
{
    if (nsnap % snap_step != 0.0)
    {
        return int(nsnap / snap_step) + 1;
    }
    else
    {
        return nsnap / snap_step;
    }
}

// Compute EQP solution from constraints on snapshots.
void SetupEQP_snapshots(const IntegrationRule *ir0, const int rank,
                        ParFiniteElementSpace *fespace_X,
                        const int nsets, const CAROM::Matrix *BV,
                        const CAROM::Matrix *BX_snapshots, const Vector x0,
                        const bool precondition, const double nnls_tol,
                        const int maxNNLSnnz, NeoHookeanModel *model,
                        CAROM::Vector &sol, CAROM::Vector *window_ids, const int snap_step)
{
    const int nqe = ir0->GetNPoints();
    const int ne = fespace_X->GetNE();
    const int NB = BV->numColumns();
    const int NQ = ne * nqe;
    const int nsnap = BX_snapshots->numColumns();

    MFEM_VERIFY(nsnap == BX_snapshots->numColumns() ||
                nsnap + nsets == BX_snapshots->numColumns(),
                "");
    MFEM_VERIFY(BV->numRows() == BX_snapshots->numRows(), "");
    MFEM_VERIFY(BV->numRows() == fespace_X->GetTrueVSize(), "");

    const bool skipFirstW = (nsnap + nsets == BX_snapshots->numColumns());

    // Compute G of size (NB * nsnap) x NQ, but only store its transpose Gt.
    CAROM::Matrix Gt(NQ, NB * getSteps(nsnap, snap_step), true);
    int current_size = 0;
    int i_start = 0;
    int i_end = 0;
    int outer_loop_length = 0;
    if (!window_ids)
    {
        current_size = nsnap;
        outer_loop_length = 1;
        i_end = nsnap;
    }
    else
    {
        outer_loop_length = window_ids->dim() - 1;
    }

    // For 0 <= j < NB, 0 <= i < nsnap, 0 <= e < ne, 0 <= m < nqe,
    // G(j + (i*NB), (e*nqe) + m)
    // is the coefficient of v_j^T M(p_i) V v_i at point m of element e,
    // with respect to the integration rule weight at that point,
    // where the "exact" quadrature solution is ir0->GetWeights().

    Vector x_i(BX_snapshots->numRows());
    Vector v_j(BV->numRows());

    Vector r(nqe);

    int skip = 0;
    const int nsnapPerSet = nsnap / nsets;
    if (skipFirstW)
    {
        MFEM_VERIFY(nsets * nsnapPerSet == nsnap, "");
        skip = 1;
    }

    // Get prolongation matrix
    const Operator *P = fespace_X->GetProlongationMatrix();
    if (!P)
    {
        MFEM_ABORT("P is null, generalize to serial case")
    }

    // Outer loop for time windowing
    // NB: Currently time windowing is done using the same velocity basis for every time window.
    // A possible next step is to also do time windowing on the velocity basis, which should lead to smaller EQP systems.
    for (int oi = 0; oi < outer_loop_length; ++oi)
    {
        if (window_ids)
        {
            i_start = window_ids->item(oi) - 1;
            i_end = window_ids->item(oi + 1) - 1;
            current_size = i_end - i_start;
            cout << "i_start = " << i_start << endl;
            cout << "Number of NNLS equations: " << NB * current_size << endl;
            Gt.setSize(NQ, NB * getSteps(current_size, snap_step));
        }

        // Assuming dof and dim are constant, initialize element matrices once
        const FiniteElement &fe1 = *fespace_X->GetFE(0);
        const int dof = fe1.GetDof();
        const int dim = fe1.GetDim();
        ElemMatrices em(dof, dim);

        // For every snapshot in batch
        for (int i = i_start; i < i_end; i += snap_step)
        {
            // Set the sampled dofs from the snapshot matrix
            for (int j = 0; j < BX_snapshots->numRows(); ++j)
                x_i[j] = (*BX_snapshots)(j, i) + x0[j];

            // Get prolongated dofs
            Vector px_i;
            Vector elfun;

            px_i.SetSize(P->Height());
            P->Mult(x_i, px_i);

            if (skipFirstW && i > 0 && i % nsnapPerSet == 0)
                skip++;

            // TODO: is it better to make the element loop the outer loop?
            // For each element
            for (int e = 0; e < ne; ++e)
            {
                // Get element and its dofs and transformation.
                Array<int> vdofs;
                DofTransformation *doftrans = fespace_X->GetElementVDofs(e, vdofs);
                const FiniteElement &fe = *fespace_X->GetFE(e);

                ElementTransformation *eltrans = fespace_X->GetElementTransformation(e);
                px_i.GetSubVector(vdofs, elfun);

                if (doftrans)
                {
                    MFEM_ABORT("Doftrans is true, make corresponding edits")
                }

                // For each basis vector
                for (int j = 0; j < NB; ++j)
                {
                    // Get basis vector
                    for (int k = 0; k < BV->numRows(); ++k)
                        v_j[k] = (*BV)(k, j);

                    // Get prolongated dofs
                    Vector pv_j;
                    Vector ve_j;

                    pv_j.SetSize(P->Height());
                    P->Mult(v_j, pv_j);
                    pv_j.GetSubVector(vdofs, ve_j);
                    // Compute the row of G corresponding to element e, store in r
                    ComputeElementRowOfG(ir0, vdofs, ve_j, model, elfun, fe, *eltrans, r, dof, dim,
                                         em);

                    for (int m = 0; m < nqe; ++m)
                        Gt((e * nqe) + m, j + (i * NB)) = r[m];
                }
            }

            if (precondition)
            {
                MFEM_ABORT("TODO: Implement preconditioned NNLS for this example");
            }
        } // Loop (i) over snapshots

        // Rescale every Gt column (NNLS equation) by its max absolute value.
        Gt.rescale_cols_max();

        Array<double> const &w_el = ir0->GetWeights();
        MFEM_VERIFY(w_el.Size() == nqe, "");

        CAROM::Vector w(ne * nqe, true);

        for (int i = 0; i < ne; ++i)
        {
            for (int j = 0; j < nqe; ++j)
                w((i * nqe) + j) = w_el[j];
        }

        SolveNNLS(rank, nnls_tol, maxNNLSnnz, w, Gt, sol);
        if (window_ids)
        {
            sol.write("sol_" + std::to_string(oi));
        }
    }
}

// Utility functions
void WriteMeshEQP(ParMesh *pmesh, const int myid, const int nqe,
                  CAROM::Vector const &eqpSol)
{
    // Find the elements with quadrature points in eqpSol.
    std::set<int> elements;
    Vector elemCount(pmesh->GetNE());
    elemCount = 0.0;

    for (int i = 0; i < eqpSol.dim(); ++i)
    {
        if (eqpSol(i) > 1.0e-12)
        {
            const int e = i / nqe; // Element index
            elements.insert(e);
            elemCount[e] += 1.0;
        }
    }

    // Empty sets, since EQP on samples inside elements.
    std::set<int> faces, edges, vertices;
    CAROM::SampleVisualization(pmesh, elements, elements, faces, edges,
                               vertices, "EQPvis", &elemCount);
}

// Function to compute the indices at which to switch time windows
void get_window_ids(int n_step, int n_window, CAROM::Vector *ids)
{
    int window_size = std::round(n_step / n_window);

    int res = n_step - (n_window * (window_size - 1) + 1);
    int res_up = res - n_window;
    int ctr = -1;

    for (int i = 1; i <= n_window + 1; ++i)
    {
        ids->item(i - 1) = (i - 1) * window_size + 1 - (i - 1);
        if (res_up > 0)
        {
            if (i <= res - res_up)
            {
                ctr++;
            }
            if (i > 2 && i <= res_up + 1)
            {
                ctr++;
            }
        }
        else
        {
            if (i <= res + 1)
            {
                ctr++;
            }
        }
        ids->item(i - 1) += ctr;
    }
    ids->item(n_window) = n_step + 1;
}

bool fileExists(const std::string &filename)
{
    std::ifstream file(filename);
    return file.good();
}

void get_EQPsol(const int current_window, CAROM::Vector *load_eqpsol)
{
    string filename = "sol_" + std::to_string(current_window);
    if (fileExists(filename + ".000000"))
    {
        std::cout << "The length of the vector is: " << load_eqpsol->dim() << std::endl;
        load_eqpsol->read(filename);
    }
}
