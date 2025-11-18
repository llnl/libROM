/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

//                       libROM MFEM Example: Elliptic Eigenproblem (adapted from ex11p.cpp)
//
// =================================================================================
//
// Description:  This example code demonstrates the use of MFEM and libROM to
//               define a simple projection-based reduced order model of the
//               eigenvalue problem Lu = lambda u with homogeneous
//               Dirichlet boundary conditions.
//
//               We compute a number of the lowest eigenmodes by discretizing
//               the Laplacian and Mass operators using a FE space of the
//               specified order, or an isoparametric/isogeometric space if
//               order < 1 (quadratic for quadratic curvilinear mesh, NURBS for
//               NURBS mesh, etc.)
//
// Offline phase: elliptic_eigenproblem_global_rom -offline -p 2 -rs 2 -n 4 -id 0 -a 0
//                elliptic_eigenproblem_global_rom -offline -p 2 -rs 2 -n 4 -id 1 -a 1
//
// Merge phase:   elliptic_eigenproblem_global_rom -merge -p 2 -rs 2 -n 4 -ns 2
//
// FOM run (for error calculation):
//                elliptic_eigenproblem_global_rom -fom -p 2 -rs 2 -n 4 -a 0.5
//
// Online phase:  elliptic_eigenproblem_global_rom -online -p 2 -rs 2 -n 4 -a 0.5 -ef 1.0
//
// Example output:
//   FOM solution for eigenvalue 0 = 19.878564
//   ROM solution for eigenvalue 0 = 19.878631
//   Absolute error of ROM solution for eigenvalue 0 = 6.6618756e-05
//   Relative error of ROM solution for eigenvalue 0 = 3.3512861e-06
//   FOM solution for eigenvalue 1 = 53.177886
//   ROM solution for eigenvalue 1 = 53.179662
//   Absolute error of ROM solution for eigenvalue 1 = 0.0017768914
//   Relative error of ROM solution for eigenvalue 1 = 3.3414104e-05
//   FOM solution for eigenvalue 2 = 53.177886
//   ROM solution for eigenvalue 2 = 53.179662
//   Absolute error of ROM solution for eigenvalue 2 = 0.0017769039
//   Relative error of ROM solution for eigenvalue 2 = 3.3414339e-05
//   FOM solution for eigenvalue 3 = 81.245811
//   ROM solution for eigenvalue 3 = 81.246105
//   Absolute error of ROM solution for eigenvalue 3 = 0.00029387266
//   Relative error of ROM solution for eigenvalue 3 = 3.6170808e-06
//   Relative l2 error of ROM eigenvector 0 = 0.0023815683
//   Relative l2 error of ROM eigenvector 1 = 0.0097810599
//   Relative l2 error of ROM eigenvector 2 = 0.0097816076
//   Relative l2 error of ROM eigenvector 3 = 0.0043291658

#include "mfem.hpp"
#include "linalg/BasisGenerator.h"
#include "linalg/BasisReader.h"
#include "linalg/Vector.h"
#include "mfem/Utilities.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

double Conductivity(const Vector &x);
double Potential(const Vector &x);
int problem = 1;

double amplitude = -800.0; // amplitude of coefficient fields
double pseudo_time = 0.0; // potential parametetr
Vector bb_min, bb_max; // bounding box
double h_min, h_max, k_min, k_max; // mesh size
double L = 1.0; // scaling factor
int potential_well_switch = 0; // switch for separate training

int main(int argc, char *argv[])
{
    // 1. Initialize MPI.
    int num_procs, myid;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // 2. Parse command-line options.
    const char *mesh_file = "";
    int ser_ref_levels = 2;
    int par_ref_levels = 1;
    bool dirichlet = true;
    int order = 1;
    int nev = 4;
    int nev_os = 0;
    int seed = 75;
    bool prescribe_init = false;
    int lobpcg_niter = 200;
    double lobpcg_tol = 1e-8;
    double eig_tol = 1e-6;
    bool slu_solver = false;
    bool sp_solver = false;
    bool cpardiso_solver = false;
    int verbose_level = 0;

    bool visualization = true;
    bool visit = false;
    int vis_steps = 5;
    bool fom = false;
    bool offline = false;
    bool merge = false;
    bool online = false;
    int id = 0;
    int nsets = 0;
    double ef = 1.0;
    int rdim = -1;

    int precision = 8;
    cout.precision(precision);

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                   "Mesh file to use.");
    args.AddOption(&problem, "-p", "--problem",
                   "Problem setup to use. See options in Conductivity and Potential functions.");
    args.AddOption(&dirichlet, "-dir", "--dirichlet", "-neu", "--neumann",
                   "BC switch.");
    args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                   "Number of times to refine the mesh uniformly in serial.");
    args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                   "Number of times to refine the mesh uniformly in parallel.");
    args.AddOption(&order, "-o", "--order",
                   "Order (degree) of the finite elements.");
    args.AddOption(&nev, "-n", "--num-eigs",
                   "Number of desired eigenmodes.");
    args.AddOption(&nev_os, "-nos", "--num-eigs-os",
                   "Number of oversampled eigenmodes.");
    args.AddOption(&seed, "-s", "--seed",
                   "Random seed used to initialize LOBPCG.");
    args.AddOption(&amplitude, "-a", "--amplitude",
                   "Amplitude of coefficient fields.");
    args.AddOption(&pseudo_time, "-t", "--pseudo-time",
                   "Pseudo-time of the motion of the center.");
    args.AddOption(&potential_well_switch, "-pw", "--potential-well",
                   "Customized which potential well is turned off in problem 4.\n\t"
                   "Available options: 0 (default, no wells off), 1, 2.");
    args.AddOption(&id, "-id", "--id", "Parametric id");
    args.AddOption(&nsets, "-ns", "--nset", "Number of parametric snapshot sets");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                   "--no-visit-datafiles",
                   "Save data files for VisIt (visit.llnl.gov) visualization.");
    args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                   "Visualize every n-th timestep.");
    args.AddOption(&fom, "-fom", "--fom", "-no-fom", "--no-fom",
                   "Enable or disable the fom phase.");
    args.AddOption(&offline, "-offline", "--offline", "-no-offline", "--no-offline",
                   "Enable or disable the offline phase.");
    args.AddOption(&online, "-online", "--online", "-no-online", "--no-online",
                   "Enable or disable the online phase.");
    args.AddOption(&merge, "-merge", "--merge", "-no-merge", "--no-merge",
                   "Enable or disable the merge phase.");
    args.AddOption(&ef, "-ef", "--energy_fraction",
                   "Energy fraction.");
    args.AddOption(&rdim, "-rdim", "--rdim",
                   "Reduced dimension.");
    args.AddOption(&verbose_level, "-v", "--verbose",
                   "Set the verbosity level of the LOBPCG solver and preconditioner. 0 is off.");
    args.AddOption(&prescribe_init, "-init", "--init", "-no-init", "--no-init",
                   "Prescribe custom Initialization of LOBPCG.");
    args.AddOption(&lobpcg_niter, "-fi", "--fom-iter",
                   "Number of iterations for the LOBPCG solver.");
    args.AddOption(&lobpcg_tol, "-tol", "--fom-tol",
                   "Tolerance for the LOBPCG solver.");
    args.AddOption(&eig_tol, "-eig-tol", "--eig-tol",
                   "Tolerance for eigenvalues to be considered equal.");
#ifdef MFEM_USE_SUPERLU
    args.AddOption(&slu_solver, "-slu", "--superlu", "-no-slu",
                   "--no-superlu", "Use the SuperLU Solver.");
#endif
#ifdef MFEM_USE_STRUMPACK
    args.AddOption(&sp_solver, "-sp", "--strumpack", "-no-sp",
                   "--no-strumpack", "Use the STRUMPACK Solver.");
#endif
#ifdef MFEM_USE_MKL_CPARDISO
    args.AddOption(&cpardiso_solver, "-cpardiso", "--cpardiso", "-no-cpardiso",
                   "--no-cpardiso", "Use the MKL CPardiso Solver.");
#endif
    args.Parse();
    if (slu_solver && sp_solver)
    {
        if (myid == 0)
            std::cout << "WARNING: Both SuperLU and STRUMPACK have been selected,"
                      << " please choose either one." << std::endl
                      << "         Defaulting to SuperLU." << std::endl;
        sp_solver = false;
    }
    // The command line options are also passed to the STRUMPACK
    // solver. So do not exit if some options are not recognized.
    if (!sp_solver)
    {
        if (!args.Good())
        {
            if (myid == 0)
            {
                args.PrintUsage(cout);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (myid == 0)
    {
        args.PrintOptions(cout);
    }

    if (fom)
    {
        MFEM_VERIFY(fom && !offline && !online
                    && !merge, "everything must be turned off if fom is used.");
    }
    else
    {
        bool check = (offline && !merge && !online) || (!offline && merge && !online)
                     || (!offline && !merge && online);
        MFEM_VERIFY(check, "only one of offline, merge, or online must be true!");
    }

    // 3. Read the serial mesh from the given mesh file on all processors. We can
    //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
    //    with the same code.
    Mesh *mesh;
    if (strcmp(mesh_file, "") == 0 )
    {
        mesh = new Mesh(Mesh::MakeCartesian2D(2, 2, Element::QUADRILATERAL));
    }
    else
    {
        mesh = new Mesh(mesh_file, 1, 1);
    }
    int dim = mesh->Dimension();
    mesh->GetBoundingBox(bb_min, bb_max, max(order, 1));
    mesh->GetCharacteristics(h_min, h_max, k_min, k_max);
    h_max *= pow(0.5, ser_ref_levels + par_ref_levels);

    // 4. Refine the mesh in serial to increase the resolution. In this example
    //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
    //    a command-line parameter.
    for (int lev = 0; lev < ser_ref_levels; lev++)
    {
        mesh->UniformRefinement();
    }

    // 5. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    for (int lev = 0; lev < par_ref_levels; lev++)
    {
        pmesh->UniformRefinement();
    }

    // 6. Define a parallel finite element space on the parallel mesh. Here we
    //    use continuous Lagrange finite elements of the specified order. If
    //    order < 1, we instead use an isoparametric/isogeometric space.
    FiniteElementCollection *fec;
    if (order > 0)
    {
        fec = new H1_FECollection(order, dim);
    }
    else if (pmesh->GetNodes())
    {
        fec = pmesh->GetNodes()->OwnFEC();
    }
    else
    {
        fec = new H1_FECollection(order = 1, dim);
    }
    ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
    HYPRE_BigInt size = fespace->GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of unknowns: " << size << endl;
    }

    // 7. Initiate ROM related variables
    int max_num_snapshots = 100;
    bool update_right_SV = false;
    bool isIncremental = false;
    const std::string baseName = "elliptic_eigenproblem_";
    const std::string basis_filename = baseName + "basis";
    CAROM::Options* options;
    CAROM::BasisGenerator *generator;
    StopWatch solveTimer, assembleTimer, mergeTimer;

    // 8. Set BasisGenerator if offline
    if (offline)
    {
        options = new CAROM::Options(fespace->GetTrueVSize(), nev,
                                     update_right_SV);
        std::string snapshot_basename = baseName + "par" + std::to_string(id);
        generator = new CAROM::BasisGenerator(*options, isIncremental,
                                              snapshot_basename);
    }

    // 9. The merge phase
    if (merge)
    {
        mergeTimer.Start();
        options = new CAROM::Options(fespace->GetTrueVSize(), max_num_snapshots,
                                     update_right_SV);
        generator = new CAROM::BasisGenerator(*options, isIncremental, basis_filename);
        for (int paramID=0; paramID<nsets; ++paramID)
        {
            std::string snapshot_filename = baseName + "par" + std::to_string(
                                                paramID) + "_snapshot";
            generator->loadSamples(snapshot_filename, "snapshot");
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
        MPI_Finalize();
        return 0;
    }

    // 10. Set up the parallel bilinear forms a(.,.) and m(.,.) on the finite
    //     element space. The first corresponds to the Laplacian operator -Delta,
    //     while the second is a simple mass matrix needed on the right hand side
    //     of the generalized eigenvalue problem below. The boundary conditions
    //     are implemented by elimination with special values on the diagonal to
    //     shift the Dirichlet eigenvalues out of the computational range. After
    //     serial and parallel assembly we extract the corresponding parallel
    //     matrices A and M.
    assembleTimer.Start();
    ConstantCoefficient one(1.0);

    FunctionCoefficient kappa_0(Conductivity);
    FunctionCoefficient v_0(Potential);

    // Project conductivity and potential for visualization
    ParGridFunction c_gf(fespace);
    c_gf.ProjectCoefficient(kappa_0);

    ParGridFunction p_gf(fespace);
    p_gf.ProjectCoefficient(v_0);

    Array<int> ess_bdr;
    if (pmesh->bdr_attributes.Size())
    {
        ess_bdr.SetSize(pmesh->bdr_attributes.Max());
        ess_bdr = (dirichlet) ? 1 : 0;
    }

    ParBilinearForm *a = new ParBilinearForm(fespace);
    a->AddDomainIntegrator(new DiffusionIntegrator(kappa_0));
    a->AddDomainIntegrator(new MassIntegrator(v_0));
    a->Assemble();
    a->EliminateEssentialBCDiag(ess_bdr, 1.0);
    a->Finalize();

    ParBilinearForm *m = new ParBilinearForm(fespace);
    m->AddDomainIntegrator(new MassIntegrator(one));
    m->Assemble();
    // shift the eigenvalue corresponding to eliminated dofs to a large value
    m->EliminateEssentialBCDiag(ess_bdr, numeric_limits<double>::min());
    m->Finalize();

    HypreParMatrix *A = a->ParallelAssemble();
    HypreParMatrix *M = m->ParallelAssemble();

#if defined(MFEM_USE_SUPERLU) || defined(MFEM_USE_STRUMPACK)
    Operator * Arow = NULL;
#endif
#ifdef MFEM_USE_SUPERLU
    if (slu_solver)
    {
        Arow = new SuperLURowLocMatrix(*A);
    }
#endif
#ifdef MFEM_USE_STRUMPACK
    if (sp_solver)
    {
        Arow = new STRUMPACKRowLocMatrix(*A);
    }
#endif

    assembleTimer.Stop();

    delete a;
    delete m;

    ParGridFunction eigenfunction_i(fespace);
    Vector eigenvector_i(size);
    HypreLOBPCG *lobpcg;
    Array<double> eigenvalues;
    std::vector<Vector> eigenvectors;

    // 11. The offline phase
    if (fom || offline)
    {
        // 12. Define and configure the LOBPCG eigensolver and the BoomerAMG
        //     preconditioner for A to be used within the solver. Set the matrices
        //     which define the generalized eigenproblem A x = lambda M x.
        Solver * precond = NULL;
        if (!slu_solver && !sp_solver && !cpardiso_solver)
        {
            HypreBoomerAMG * amg = new HypreBoomerAMG(*A);
            amg->SetPrintLevel(verbose_level);
            precond = amg;
        }
        else
        {
#ifdef MFEM_USE_SUPERLU
            if (slu_solver)
            {
                SuperLUSolver * superlu = new SuperLUSolver(MPI_COMM_WORLD);
                superlu->SetPrintStatistics(verbose_level > 0 ? true : false);
                superlu->SetSymmetricPattern(true);
                superlu->SetColumnPermutation(superlu::PARMETIS);
                superlu->SetOperator(*Arow);
                precond = superlu;
            }
#endif
#ifdef MFEM_USE_STRUMPACK
            if (sp_solver)
            {
                STRUMPACKSolver * strumpack = new STRUMPACKSolver(MPI_COMM_WORLD, argc, argv);
                strumpack->SetPrintFactorStatistics(true);
                strumpack->SetPrintSolveStatistics(verbose_level > 0 ? true : false);
                strumpack->SetKrylovSolver(strumpack::KrylovSolver::DIRECT);
                strumpack->SetReorderingStrategy(strumpack::ReorderingStrategy::METIS);
                strumpack->SetMatching(strumpack::MatchingJob::NONE);
                strumpack->SetCompression(strumpack::CompressionType::NONE);
                strumpack->SetOperator(*Arow);
                strumpack->SetFromCommandLine();
                precond = strumpack;
            }
#endif
#ifdef MFEM_USE_MKL_CPARDISO
            if (cpardiso_solver)
            {
                auto cpardiso = new CPardisoSolver(A->GetComm());
                cpardiso->SetMatrixType(CPardisoSolver::MatType::REAL_STRUCTURE_SYMMETRIC);
                cpardiso->SetPrintLevel(verbose_level);
                cpardiso->SetOperator(*A);
                precond = cpardiso;
            }
#endif
        }

        lobpcg = new HypreLOBPCG(MPI_COMM_WORLD);
        lobpcg->SetNumModes(nev + nev_os);
        lobpcg->SetRandomSeed(seed);
        lobpcg->SetPreconditioner(*precond);
        lobpcg->SetMaxIter(lobpcg_niter);
        lobpcg->SetTol(lobpcg_tol);
        lobpcg->SetPrecondUsageMode(1);
        lobpcg->SetPrintLevel(verbose_level);
        lobpcg->SetMassMatrix(*M);
        lobpcg->SetOperator(*A);

        if (prescribe_init && (fom || (offline && id > 0)))
        {
            HypreParVector** snapshot_vecs = new HypreParVector*[nev + nev_os];
            for (int i = 0; i < nev + nev_os; i++)
            {
                std::string snapshot_filename = baseName + "ref_snapshot_" + std::to_string(i);
                std::ifstream snapshot_infile(snapshot_filename + "." + std::to_string(myid));
                MFEM_VERIFY(snapshot_infile.good(), "Failed to load vector for initialization");
                snapshot_vecs[i] = new HypreParVector();
                snapshot_vecs[i]->Read(MPI_COMM_WORLD, snapshot_filename.c_str());
                if (myid == 0) std::cout << "Loaded " << snapshot_filename << std::endl;
            }
            lobpcg->SetInitialVectors(nev + nev_os, snapshot_vecs);
            if (myid == 0) std::cout << "LOBPCG initial vectors set" << std::endl;
        }

        // 13. Compute the eigenmodes and extract the array of eigenvalues. Define a
        //     parallel grid function to represent each of the eigenmodes returned by
        //     the solver.
        solveTimer.Start();
        lobpcg->Solve();
        solveTimer.Stop();
        lobpcg->GetEigenvalues(eigenvalues);

        // 14. take and write snapshots for ROM
        for (int i = 0; i < nev; i++)
        {
            if (myid == 0)
            {
                std::cout << "Eigenvalue " << i << ": " << eigenvalues[i] << "\n";
            }
            if (offline)
            {
                eigenfunction_i = lobpcg->GetEigenvector(i);
                eigenfunction_i /= sqrt(InnerProduct(eigenfunction_i, eigenfunction_i));
                generator->takeSample(eigenfunction_i.GetData());
            }
        }

        if (offline)
        {
            if (prescribe_init && id == 0)
            {
                for (int i = 0; i < nev + nev_os; i++)
                {
                    std::string snapshot_filename = baseName + "ref_snapshot_" + std::to_string(i);
                    const HypreParVector snapshot_vec = lobpcg->GetEigenvector(i);
                    snapshot_vec.Print(snapshot_filename.c_str());
                    if (myid == 0) std::cout << "Saved " << snapshot_filename <<
                                                 " for LOBPCG initialization" << std::endl;
                }
            }
            generator->writeSnapshot();
            delete generator;
            delete options;
        }

        delete precond;
    }

    // 15. The online phase
    if (online)
    {
        // 16. read the reduced basis
        assembleTimer.Start();
        CAROM::BasisReader reader(basis_filename);

        std::unique_ptr<const CAROM::Matrix> spatialbasis;
        if (rdim != -1)
            spatialbasis = reader.getSpatialBasis(rdim);
        else
            spatialbasis = reader.getSpatialBasis(ef);

        const int numRowRB = spatialbasis->numRows();
        const int numColumnRB = spatialbasis->numColumns();
        if (myid == 0) printf("spatial basis dimension is %d x %d\n", numRowRB,
                                  numColumnRB);
        MFEM_VERIFY(numColumnRB >= nev, "basis dimension >= number of eigenvectors");

        // 17. form ROM operator
        CAROM::Matrix ReducedA;
        ComputeCtAB(*A, *spatialbasis, *spatialbasis, ReducedA);
        DenseMatrix *A_mat = new DenseMatrix(ReducedA.numRows(),
                                             ReducedA.numColumns());
        A_mat->Set(1, ReducedA.getData());

        CAROM::Matrix ReducedM;
        ComputeCtAB(*M, *spatialbasis, *spatialbasis, ReducedM);
        DenseMatrix *M_mat = new DenseMatrix(ReducedM.numRows(),
                                             ReducedM.numColumns());
        M_mat->Set(1, ReducedM.getData());

        assembleTimer.Stop();

        Vector eigenvalues_rom;
        DenseMatrix reduced_eigenvectors;
        // 18. solve ROM
        solveTimer.Start();
        // (Q^T A Q) c = \lambda (Q^T M Q) c
        A_mat->Eigenvalues(*M_mat, eigenvalues_rom, reduced_eigenvectors);
        solveTimer.Stop();

        // Postprocessing
        eigenvalues = Array<double>(eigenvalues_rom.GetData(), eigenvalues_rom.Size());
        Vector mode_rom(numRowRB);
        std::vector<Vector> modes_rom;
        for (int j = 0; j < nev; j++)
        {
            Vector reduced_eigenvector_j;
            reduced_eigenvectors.GetColumn(j, reduced_eigenvector_j);
            CAROM::Vector reduced_eigenvector_j_carom(reduced_eigenvector_j.GetData(),
                    reduced_eigenvector_j.Size(), false, false);
            std::unique_ptr<CAROM::Vector> eigenvector_j_carom = spatialbasis->mult(
                        reduced_eigenvector_j_carom);
            mode_rom.SetData(eigenvector_j_carom->getData());
            mode_rom /= sqrt(InnerProduct(mode_rom, mode_rom));
            modes_rom.push_back(mode_rom);
        }

        // Calculate errors of eigenvalues
        ostringstream sol_eigenvalue_name_fom;
        sol_eigenvalue_name_fom << "sol_eigenvalues_fom." << setfill('0')
                                << setw(6) << myid;
        Vector eigenvalues_fom(nev);
        ifstream fom_file;
        fom_file.open(sol_eigenvalue_name_fom.str().c_str());
        eigenvalues_fom.Load(fom_file, eigenvalues_fom.Size());
        fom_file.close();

        Vector diff_eigenvalues(nev);
        for (int i = 0; i < nev; i++)
        {
            diff_eigenvalues[i] = eigenvalues_fom[i] - eigenvalues_rom[i];
            if (myid == 0)
            {
                std::cout << "FOM solution for eigenvalue " << i << " = " <<
                          eigenvalues_fom[i] << std::endl;
                std::cout << "ROM solution for eigenvalue " << i << " = " <<
                          eigenvalues_rom[i] << std::endl;
                std::cout << "Absolute error of ROM solution for eigenvalue " << i << " = " <<
                          abs(diff_eigenvalues[i]) << std::endl;
                std::cout << "Relative error of ROM solution for eigenvalue " << i << " = " <<
                          abs(diff_eigenvalues[i]) / abs(eigenvalues_fom[i]) << std::endl;
            }
        }

        // Calculate errors of eigenvectors
        ostringstream mode_name_fom;
        Vector mode_fom(numRowRB);
        for (int i = 0; i < nev; i++)
        {
            mode_name_fom << "eigenfunction_fom_" << setfill('0') << setw(2) << i << "."
                          << setfill('0') << setw(6) << myid;

            ifstream mode_fom_ifs(mode_name_fom.str().c_str());
            mode_fom_ifs.precision(16);
            mode_fom.Load(mode_fom_ifs, numRowRB);
            mode_fom_ifs.close();

            eigenvector_i = 0.0;
            for (int j = 0; j < nev; j++)
            {
                if (myid == 0)
                {
                    std::cout << "correlation_matrix(" << j+1 << "," << i+1 << ") = " <<
                              InnerProduct(mode_fom, modes_rom[j]) << ";" << std::endl;
                }
                if (abs(eigenvalues_fom[j] - eigenvalues_fom[i]) < eig_tol)
                {
                    eigenvector_i.Add(InnerProduct(mode_fom, modes_rom[j]), modes_rom[j]);
                }
            }
            eigenvectors.push_back(eigenvector_i);
            mode_fom.Add(-1.0, eigenvector_i);
            const double diffNorm = sqrt(InnerProduct(mode_fom, mode_fom));
            if (myid == 0) std::cout << "Relative l2 error of ROM eigenvector " << i <<
                                         " = " << diffNorm << std::endl;

            mode_name_fom.str("");
        }

        delete A_mat;
        delete M_mat;
    }

    // 19. Save the refined mesh and the modes in parallel. This output can be
    //     viewed later using GLVis: "glvis -np <np> -m mesh -g mode".
    {
        ostringstream sol_eigenvalue_name;
        if (fom || offline)
        {
            sol_eigenvalue_name << "sol_eigenvalues_fom." << setfill('0') << setw(
                                    6) << myid;
        }
        if (online)
        {
            sol_eigenvalue_name << "sol_eigenvalues." << setfill('0') << setw(6) << myid;
        }

        ofstream sol_eigenvalue_ofs(sol_eigenvalue_name.str().c_str());
        sol_eigenvalue_ofs.precision(16);
        for (int i = 0; i < nev; ++i)
        {
            sol_eigenvalue_ofs << eigenvalues[i] << std::endl;
        }

        ostringstream mesh_name, mode_name;
        mesh_name << "elliptic_eigenproblem-mesh." << setfill('0') << setw(6) << myid;

        ofstream mesh_ofs(mesh_name.str().c_str());
        mesh_ofs.precision(8);
        pmesh->Print(mesh_ofs);

        std::string mode_prefix = "eigenfunction_";
        if (fom || offline)
        {
            mode_prefix += "fom_";
        }
        else if (online)
        {
            mode_prefix += "rom_";
        }

        for (int i = 0; i < nev; i++)
        {
            if (fom || offline)
            {
                eigenfunction_i = lobpcg->GetEigenvector(i);
                eigenfunction_i /= sqrt(InnerProduct(eigenfunction_i, eigenfunction_i));
            }
            else
            {
                // for online, eigenvectors are stored in eigenvectors matrix
                // convert eigenvector from HypreParVector to ParGridFunction
                eigenfunction_i = eigenvectors[i];
            }

            mode_name << mode_prefix << setfill('0') << setw(2) << i << "."
                      << setfill('0') << setw(6) << myid;

            ofstream mode_ofs(mode_name.str().c_str());
            mode_ofs.precision(16);
            // TODO: issue using .Load() function if file written with .Save()?
            //eigenfunction_i.Save(mode_ofs);
            for (int j = 0; j < eigenfunction_i.Size(); j++)
            {
                mode_ofs << eigenfunction_i[j] << "\n";
            }
            mode_ofs.close();
            mode_name.str("");
        }
    }

    VisItDataCollection *visit_dc = NULL;
    if (visit)
    {
        if (offline) visit_dc = new VisItDataCollection(baseName + "offline_par" +
                    std::to_string(id), pmesh);
        else if (fom) visit_dc = new VisItDataCollection(baseName + "fom", pmesh);
        else if (online) visit_dc = new VisItDataCollection(baseName + "rom", pmesh);
        visit_dc->RegisterField("Conductivity", &c_gf);
        visit_dc->RegisterField("Potential", &p_gf);
        std::vector<ParGridFunction*> visit_eigenvectors;
        for (int i = 0; i < nev; i++)
        {
            if (fom || offline)
            {
                eigenfunction_i = lobpcg->GetEigenvector(i);
                eigenfunction_i /= sqrt(InnerProduct(eigenfunction_i, eigenfunction_i));
            }
            else
            {
                // for online, eigenvectors are stored in eigvenvectors matrix
                // convert eigenvector from HypreParVector to ParGridFunction
                eigenfunction_i = eigenvectors[i];
            }

            visit_eigenvectors.push_back(new ParGridFunction(eigenfunction_i));
            visit_dc->RegisterField("Eigenmode_" + std::to_string(i),
                                    visit_eigenvectors.back());
        }
        visit_dc->SetCycle(0);
        visit_dc->SetTime(0.0);
        visit_dc->Save();
        for (size_t i = 0; i < visit_eigenvectors.size(); i++)
        {
            delete visit_eigenvectors[i];
        }
    }

    socketstream sout;
    if (visualization)
    {
        char vishost[] = "localhost";
        int  visport   = 19916;
        sout.open(vishost, visport);
        sout << "parallel " << num_procs << " " << myid << endl;
        int good = sout.good(), all_good;
        MPI_Allreduce(&good, &all_good, 1, MPI_INT, MPI_MIN, pmesh->GetComm());
        if (!all_good)
        {
            sout.close();
            visualization = false;
            if (myid == 0)
            {
                cout << "Unable to connect to GLVis server at "
                     << vishost << ':' << visport << endl;
                cout << "GLVis visualization disabled.\n";
            }
        }
        else
        {
            for (int i = 0; i < nev; i++)
            {
                if ( myid == 0 )
                {
                    cout << "Eigenmode " << i+1 << '/' << nev
                         << ", Lambda = " << eigenvalues[i] << endl;
                }

                if (fom || offline)
                {
                    eigenfunction_i = lobpcg->GetEigenvector(i);
                    eigenfunction_i /= sqrt(InnerProduct(eigenfunction_i, eigenfunction_i));
                }
                else
                {
                    // for online, eigenvectors are stored in eigvenvectors matrix
                    // convert eigenvector from HypreParVector to ParGridFunction
                    eigenfunction_i = eigenvectors[i];
                }

                sout << "parallel " << num_procs << " " << myid << "\n"
                     << "solution\n" << *pmesh << eigenfunction_i << flush
                     << "window_title 'Eigenmode " << i+1 << '/' << nev
                     << ", Lambda = " << eigenvalues[i] << "'" << endl;

                char c;
                if (myid == 0)
                {
                    cout << "press (q)uit or (c)ontinue --> " << flush;
                    cin >> c;
                }
                MPI_Bcast(&c, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

                if (c != 'c')
                {
                    break;
                }
            }
            sout.close();
        }
    }

    // 20. print timing info
    if (myid == 0)
    {
        if (fom || offline)
        {
            printf("Elapsed time for assembling FOM: %e second\n",
                   assembleTimer.RealTime());
            printf("Elapsed time for solving FOM: %e second\n", solveTimer.RealTime());
        }
        if (online)
        {
            printf("Elapsed time for assembling ROM: %e second\n",
                   assembleTimer.RealTime());
            printf("Elapsed time for solving ROM: %e second\n", solveTimer.RealTime());
        }
    }

    // 21. Free the used memory.
    if (fom || offline)
    {
        delete lobpcg;
    }
    delete M;
    delete A;

    delete fespace;
    if (order > 0)
    {
        delete fec;
    }
    delete pmesh;

#if defined(MFEM_USE_SUPERLU) || defined(MFEM_USE_STRUMPACK)
    delete Arow;
#endif

    MPI_Finalize();

    return 0;
}

double Conductivity(const Vector &x)
{
    switch (problem)
    {
    case 1:
        return 1.0;
    case 2:
        for (int i = 0; i < x.Size(); ++i)
            if (8 * abs(x(i) - 0.5 * (bb_max[i] + bb_min[i])) > bb_max[i] - bb_min[i])
                return 1.0;
        return 1.0 + amplitude;
    case 3:
    case 4:
        return 1.0;
    case 5:
    case 6:
        L = (bb_max[0] - bb_min[0]) / 18.0;
        return pow(L, 2.0);
    }
    return 1.0;
}

double Potential(const Vector &x)
{
    Vector center(x.Size());
    double rho = 0.0;
    double rho_bg = -5.0;
    double d_sq, radius_sq;

    switch (problem)
    {
    case 1:
    case 2:
        return 0.0;
    case 3:
        radius_sq = pow(4.0 * h_max, 2.0);
        // center = (0.5 + t*h, 0.5)
        center(0) = 0.5 * (bb_min[0] + bb_max[0]) + pseudo_time * h_max;
        for (int i = 1; i < x.Size(); i++)
            center(i) = 0.5 * (bb_min[i] + bb_max[i]);
        d_sq = x.DistanceSquaredTo(center);
        return amplitude * std::exp(-d_sq / (2 * radius_sq));
    case 4:
        radius_sq = pow(4.0 * h_max, 2.0);
        if (potential_well_switch == 0 || potential_well_switch == 1)
        {
            // add well with first center
            center(0) = (0.25 + 0.02 * cos(2 * M_PI * pseudo_time)) *
                        (bb_min[0] + bb_max[0]);
            center(1) = (0.25 + 0.02 * sin(2 * M_PI * pseudo_time)) *
                        (bb_min[1] + bb_max[1]);
            for (int i = 2; i < x.Size(); i++)
                center(i) = 0.25 * (bb_min[i] + bb_max[i]);
            d_sq = x.DistanceSquaredTo(center);
            rho += amplitude * std::exp(-d_sq / (2 * radius_sq));
        }
        if (potential_well_switch == 0 || potential_well_switch == 2)
        {
            // add well with second center
            center(0) = (0.75 - 0.02 * cos(2 * M_PI * pseudo_time)) *
                        (bb_min[0] + bb_max[0]);
            center(1) = (0.75 - 0.02 * sin(2 * M_PI * pseudo_time)) *
                        (bb_min[1] + bb_max[1]);
            for (int i = 2; i < x.Size(); i++)
                center(i) = 0.75 * (bb_min[i] + bb_max[i]);
            d_sq = x.DistanceSquaredTo(center);
            rho += amplitude * std::exp(-d_sq / (2 * radius_sq));
        }
        return rho;
    case 5:
        L = (bb_max[0] - bb_min[0]) / 18.0;
        for (int i = 1; i < x.Size(); i++)
            center(i) = (bb_min[i] + bb_max[i]) / 2;
        if (potential_well_switch == 0 || potential_well_switch == 1)
        {
            // add well with first center
            center(0) = (23 * bb_min[0] + 13 * bb_max[0]) / 36 - h_max * pseudo_time;
            d_sq = x.DistanceSquaredTo(center);
            radius_sq = pow(0.28 * L, 2.0);
            rho += -28.9 * std::exp(-d_sq / (2 * radius_sq));
            radius_sq = pow(1.08 * L, 2.0);
            rho += -3.6 * std::exp(-d_sq / (2 * radius_sq));
        }
        if (potential_well_switch == 0 || potential_well_switch == 2)
        {
            // add well with second center
            center(0) = (13 * bb_min[0] + 23 * bb_max[0]) / 36 + h_max * pseudo_time;
            d_sq = x.DistanceSquaredTo(center);
            radius_sq = pow(0.28 * L, 2.0);
            rho += -28.9 * std::exp(-d_sq / (2 * radius_sq));
            radius_sq = pow(1.08 * L, 2.0);
            rho += -3.6 * std::exp(-d_sq / (2 * radius_sq));
        }
        return rho;
    case 6:
        L = (bb_max[0] - bb_min[0]) / 18.0;
        for (int i = 1; i < x.Size(); i++)
            center(i) = (bb_min[i] + bb_max[i]) / 2;

        // coefficients for H
        double rloc = 0.4;
        double c1 = -14.081155;
        double c2 = 9.626220;
        double c3 = -1.783616;
        double c4 = 0.085152;
        double zion = 3.0;

        double alpha;
        if (potential_well_switch == 0 || potential_well_switch == 1)
        {
            // add well with first center
            center(0) = (23 * bb_min[0] + 13 * bb_max[0]) / 36 - h_max * pseudo_time;
            d_sq = x.DistanceSquaredTo(center);
            alpha = d_sq / pow(L * rloc, 2.0);
            rho += exp(-0.5 * alpha) * (c1 + c2 * alpha + c3 * alpha * alpha + c4 * alpha *
                                        alpha * alpha);
            if (d_sq > 1.e-16)
            {
                rho -= zion * erf(sqrt(d_sq) / (sqrt(2.0) * rloc)) / sqrt(d_sq);
            }
            else
            {
                rho -= zion * sqrt(2.0) / (sqrt(M_PI) * rloc);
            }
        }
        if (potential_well_switch == 0 || potential_well_switch == 2)
        {
            // add well with second center
            center(0) = (13 * bb_min[0] + 23 * bb_max[0]) / 36 + h_max * pseudo_time;
            d_sq = x.DistanceSquaredTo(center);
            alpha = d_sq / pow(L * rloc, 2.0);
            rho += exp(-0.5 * alpha) * (c1 + c2 * alpha + c3 * alpha * alpha + c4 * alpha *
                                        alpha * alpha);
            if (d_sq > 1.e-16)
            {
                rho -= zion * erf(sqrt(d_sq) / (sqrt(2.0) * rloc)) / sqrt(d_sq);
            }
            else
            {
                rho -= zion * sqrt(2.0) / (sqrt(M_PI) * rloc);
            }
        }
        return rho;
    }
    return 0.0;
}
