/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

//                       libROM MFEM Example: Greedy Parametric_Heat_Conduction with Differential Evolution (adapted from ex16p.cpp)
//
// Compile with: make de_parametric_heat_conduction_greedy
//
// =================================================================================
//
// In these examples, the radius of the interface between different initial temperatures, the
// alpha coefficient, and two center location variables are modified. First, a database of local
// DMDs is built in a greedy fashion within a certain radius, alpha, and center location parameter
// range. Then, a target FOM is built at a certain parameter point. Finally, this parameter point
// is treated as unknown and differential evolution (an iterative optimization algorithm that minimizes
// a cost function without the use of gradients) is run within some parameter range and automatically
// converges to the same parameter point where the target FOM was built.
//
// In this example, the optimization objective of the differential evolution (DE) algorithm is to minimize
// the relative difference between the final solution of the parametric DMD and the target FOM solution.
// The relative difference is minimized by adjusting the parameter point (radius, alpha, and center
// location) where the parametric DMD is built until some parameter point is found where the relative
// difference is at a global minimum. DE optimizes this problem by maintaining a population of
// candidate parameters and iteratively creating new candidate parameters by combining existing ones and
// storing whichever parameter set's final solution has the smallest relative difference. Thus, given an
// unknown solution and a database of local DMDs, DE is able to estimate the parameters of this unknown
// solution.
//
// For Parametric DMD with differential evolution (radius & alpha & cx & cy):
//   rm -rf parameters.txt
//   rm -rf de_parametric_heat_conduction_greedy_*
//   mpirun -np 8 de_parametric_heat_conduction_greedy -build_database -rdim 16 -greedy-param-size 5 -greedysubsize 2 -greedyconvsize 3 -greedyreldifftol 0.01 (Create DMDs in a greedy fashion at different training points)
//   mpirun -np 8 de_parametric_heat_conduction_greedy -r 0.2 -cx 0.2 -cy 0.2 -visit (Compute target FOM)
//   mpirun -np 8 de_parametric_heat_conduction_greedy -r 0.2 -cx 0.2 -cy 0.2 -visit -de -de_f 0.9 -de_cr 0.9 -de_ps 50 -de_min_iter 10 -de_max_iter 100 -de_ct 0.001 (Run interpolative differential evolution to see if target FOM can be matched)
//
// =================================================================================
//
// Description:  This example solves a time dependent nonlinear heat equation
//               problem of the form du/dt = C(u), with a non-linear diffusion
//               operator C(u) = \nabla \cdot (\kappa + \alpha u) \nabla u.
//
//               The example demonstrates the use of nonlinear operators (the
//               class ConductionOperator defining C(u)), as well as their
//               implicit time integration. Note that implementing the method
//               ConductionOperator::ImplicitSolve is the only requirement for
//               high-order implicit (SDIRK) time integration. Optional saving
//               with ADIOS2 (adios2.readthedocs.io) is also illustrated.

#include "mfem.hpp"
#include "algo/DMD.h"
#include "algo/DifferentialEvolution.h"
#include "algo/greedy/GreedyRandomSampler.h"
#include "linalg/Vector.h"
#include <cmath>
#include <cfloat>
#include <fstream>
#include <iostream>
#include "utils/CSVDatabase.h"

#ifndef _WIN32
#include <sys/stat.h>  // mkdir
#else
#include <direct.h>    // _mkdir
#define mkdir(dir, mode) _mkdir(dir)
#endif

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

/** After spatial discretization, the conduction model can be written as:
 *
 *     du/dt = M^{-1}(-Ku)
 *
 *  where u is the vector representing the temperature, M is the mass matrix,
 *  and K is the diffusion operator with diffusivity depending on u:
 *  (\kappa + \alpha u).
 *
 *  Class ConductionOperator represents the right-hand side of the above ODE.
 */
class ConductionOperator : public TimeDependentOperator
{
protected:
    ParFiniteElementSpace &fespace;
    Array<int> ess_tdof_list; // this list remains empty for pure Neumann b.c.

    ParBilinearForm *M;
    ParBilinearForm *K;

    HypreParMatrix Mmat;
    HypreParMatrix Kmat;
    HypreParMatrix *T; // T = M + dt K
    double current_dt;

    CGSolver M_solver;    // Krylov solver for inverting the mass matrix M
    HypreSmoother M_prec; // Preconditioner for the mass matrix M

    CGSolver T_solver;    // Implicit solver for T = M + dt K
    HypreSmoother T_prec; // Preconditioner for the implicit solver

    double alpha, kappa;

    mutable Vector z; // auxiliary vector

public:
    ConductionOperator(ParFiniteElementSpace &f, double alpha, double kappa,
                       const Vector &u);

    virtual void Mult(const Vector &u, Vector &du_dt) const;
    /** Solve the Backward-Euler equation: k = f(u + dt*k, t), for the unknown k.
        This is the only requirement for high-order SDIRK implicit integration.*/
    virtual void ImplicitSolve(const double dt, const Vector &u, Vector &k);

    /// Update the diffusion BilinearForm K using the given true-dof vector `u`.
    void SetParameters(const Vector &u);

    virtual ~ConductionOperator();
};

double InitialTemperature(const Vector &x);

int num_procs, myid;
const char *mesh_file = "../data/star.mesh";
int ser_ref_levels = 2;
int par_ref_levels = 1;
int order = 2;
int ode_solver_type = 3;
double t_final = 0.5;
double dt = 1.0e-2;
double alpha = 1.0e-2;
double kappa = 0.5;
double closest_rbf_val = 0.9;
int rdim = -1;
bool offline = false;
bool online = false;
bool build_database = false;
bool calc_err_indicator = false;
bool de = false;
double greedy_param_space_radius_min = 0.1;
double greedy_param_space_radius_max = 0.3;
double greedy_param_space_alpha_min = 0.1;
double greedy_param_space_alpha_max = 0.1;
double greedy_param_space_cx_min = 0.1;
double greedy_param_space_cx_max = 0.3;
double greedy_param_space_cy_min = 0.1;
double greedy_param_space_cy_max = 0.3;
int greedy_param_space_size = 8;
double greedy_relative_diff_tol = 0.01;
int greedy_subset_size = 2;
int greedy_convergence_subset_size = 3;
bool visualization = false;
bool visit = false;
int vis_steps = 5;
bool adios2 = false;
const char *baseoutputname = "";
int precision = 16;
double radius = 0.5;
double cx = 0.0;
double cy = 0.0;
double de_min_radius = -DBL_MAX;
double de_min_alpha = -DBL_MAX;
double de_min_cx = -DBL_MAX;
double de_min_cy = -DBL_MAX;
double de_max_radius = DBL_MAX;
double de_max_alpha = DBL_MAX;
double de_max_cx = DBL_MAX;
double de_max_cy = DBL_MAX;
double target_radius = radius;
double target_alpha = alpha;
double target_cx = cx;
double target_cy = cy;
double de_F = 0.8;
double de_CR = 0.9;
int de_PS = 50;
int de_min_iter = 10;
int de_max_iter = 100;
double de_ct = 0.001;
CAROM::GreedySampler* greedy_sampler = NULL;

Vector* true_solution_u = NULL;
double tot_true_solution_u_norm = 0.0;

double simulation()
{
    // 6. Read the serial mesh from the given mesh file on all processors. We can
    //    handle triangular, quadrilateral, tetrahedral and hexahedral meshes
    //    with the same code.
    Mesh *mesh = new Mesh(mesh_file, 1, 1);
    int dim = mesh->Dimension();

    // 7. Define the ODE solver used for time integration. Several implicit
    //    singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
    //    explicit Runge-Kutta methods are available.
    ODESolver *ode_solver;
    switch (ode_solver_type)
    {
    // Implicit L-stable methods
    case 1:
        ode_solver = new BackwardEulerSolver;
        break;
    case 2:
        ode_solver = new SDIRK23Solver(2);
        break;
    case 3:
        ode_solver = new SDIRK33Solver;
        break;
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
    case 22:
        ode_solver = new ImplicitMidpointSolver;
        break;
    case 23:
        ode_solver = new SDIRK23Solver;
        break;
    case 24:
        ode_solver = new SDIRK34Solver;
        break;
    default:
        cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
        delete mesh;
        return 3;
    }

    // 8. Refine the mesh in serial to increase the resolution. In this example
    //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
    //    a command-line parameter.
    for (int lev = 0; lev < ser_ref_levels; lev++)
    {
        mesh->UniformRefinement();
    }

    // 9. Define a parallel mesh by a partitioning of the serial mesh. Refine
    //    this mesh further in parallel to increase the resolution. Once the
    //    parallel mesh is defined, the serial mesh can be deleted.
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    for (int lev = 0; lev < par_ref_levels; lev++)
    {
        pmesh->UniformRefinement();
    }

    // 10. Define the vector finite element space representing the current and the
    //    initial temperature, u_ref.
    H1_FECollection fe_coll(order, dim);
    ParFiniteElementSpace fespace(pmesh, &fe_coll);

    int fe_size = fespace.GlobalTrueVSize();
    if (!de && !online && myid == 0)
    {
        cout << "Number of temperature unknowns: " << fe_size << endl;
    }

    ParGridFunction u_gf(&fespace);

    // 11. Set the initial conditions for u. All boundaries are considered
    //     natural.
    FunctionCoefficient u_0(InitialTemperature);
    u_gf.ProjectCoefficient(u_0);
    Vector u;
    u_gf.GetTrueDofs(u);

    // 12. Initialize the conduction operator and the VisIt visualization.
    ConductionOperator oper(fespace, alpha, kappa, u);

    u_gf.SetFromTrueDofs(u);
    if (!de && !online)
    {
        ostringstream mesh_name, sol_name;
        mesh_name << "de_parametric_heat_conduction_greedy_" << to_string(
                      radius) << "_"
                  << to_string(alpha) << "_" << to_string(cx) << "_" << to_string(cy)
                  << "-mesh." << setfill('0') << setw(6) << myid;
        sol_name << "de_parametric_heat_conduction_greedy_" << to_string(
                     radius) << "_"
                 << to_string(alpha) << "_" << to_string(cx) << "_" << to_string(cy)
                 << "-init." << setfill('0') << setw(6) << myid;
        ofstream omesh(mesh_name.str().c_str());
        omesh.precision(precision);
        pmesh->Print(omesh);
        ofstream osol(sol_name.str().c_str());
        osol.precision(precision);
        u_gf.Save(osol);
    }

    VisItDataCollection visit_dc("DE_Parametric_Heat_Conduction_Greedy_" +
                                 to_string(radius) + "_" + to_string(alpha) + "_" + to_string(cx) + "_" +
                                 to_string(cy), pmesh);
    visit_dc.RegisterField("temperature", &u_gf);
    if (!de && !online && visit)
    {
        visit_dc.SetCycle(0);
        visit_dc.SetTime(0.0);
        visit_dc.Save();
    }

    // Optionally output a BP (binary pack) file using ADIOS2. This can be
    // visualized with the ParaView VTX reader.
#ifdef MFEM_USE_ADIOS2
    ADIOS2DataCollection* adios2_dc = NULL;
    if (adios2)
    {
        std::string postfix(mesh_file);
        postfix.erase(0, std::string("../data/").size() );
        postfix += "_o" + std::to_string(order);
        postfix += "_solver" + std::to_string(ode_solver_type);
        const std::string collection_name = "de_parametric_heat_conduction_greedy-p-" +
                                            postfix + ".bp";

        adios2_dc = new ADIOS2DataCollection(MPI_COMM_WORLD, collection_name, pmesh);
        adios2_dc->SetParameter("SubStreams", std::to_string(num_procs/2) );
        adios2_dc->RegisterField("temperature", &u_gf);
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
            sout.precision(precision);
            sout << "solution\n" << *pmesh << u_gf;
            sout << flush;
        }
    }

    StopWatch fom_timer, dmd_training_timer, dmd_prediction_timer;

    fom_timer.Start();

    // 13. Perform time-integration (looping over the time iterations, ti, with a
    //     time-step dt).
    ode_solver->Init(oper);
    double t = 0.0;
    vector<double> ts;
    CAROM::Vector* init = NULL;

    CAROM::CSVDatabase csv_db;

    fom_timer.Stop();

    std::unique_ptr<CAROM::DMD> dmd_u;

    // 14. If in offline mode, create DMD object and take initial sample.
    if (offline)
    {
        dmd_training_timer.Start();

        u_gf.SetFromTrueDofs(u);
        dmd_u.reset(new CAROM::DMD(u.Size(), dt));
        dmd_u->takeSample(u.GetData(), t);

        if (myid == 0)
        {
            std::cout << "Taking snapshot at: " << t << std::endl;
        }

        dmd_training_timer.Stop();
    }

    // 14. If in de or online mode, save the initial vector.
    if (de || online)
    {
        u_gf.SetFromTrueDofs(u);
        init = new CAROM::Vector(u.GetData(), u.Size(), true);
    }
    else
    {
        // 14. If in calc_err_indicator mode, load the current DMD database,
        //     and create a parametric DMD object at the current set of parameters.
        if (calc_err_indicator)
        {
            init = new CAROM::Vector(u.GetData(), u.Size(), true);

            std::fstream fin("parameters.txt", std::ios_base::in);
            double curr_param;
            std::vector<std::string> dmd_paths;
            std::vector<CAROM::Vector> param_vectors;

            while (fin >> curr_param)
            {
                double curr_radius = curr_param;
                fin >> curr_param;
                double curr_alpha = curr_param;
                fin >> curr_param;
                double curr_cx = curr_param;
                fin >> curr_param;
                double curr_cy = curr_param;

                dmd_paths.push_back(to_string(curr_radius) + "_" +
                                    to_string(curr_alpha) + "_" + to_string(curr_cx) + "_" +
                                    to_string(curr_cy));
                CAROM::Vector param_vector(4, false);
                param_vector(0) = curr_radius;
                param_vector(1) = curr_alpha;
                param_vector(2) = curr_cx;
                param_vector(3) = curr_cy;
                param_vectors.push_back(param_vector);
            }
            fin.close();

            if (dmd_paths.size() > 1)
            {
                CAROM::Vector desired_param(4, false);
                desired_param(0) = radius;
                desired_param(1) = alpha;
                desired_param(2) = cx;
                desired_param(3) = cy;

                dmd_training_timer.Start();

                CAROM::getParametricDMD(dmd_u, param_vectors, dmd_paths, desired_param,
                                        "G", "LS", closest_rbf_val);
            }
            else
            {
                dmd_u.reset(new CAROM::DMD(dmd_paths[0]));
            }

            dmd_u->projectInitialCondition(*init);

            dmd_training_timer.Stop();

            // For the error indicator, load in the DMD predicted solution 10
            // steps before t_final, then run the FOM for the last 10 steps and
            // compare the final FOM solution to the DMD predicted solution.
            t = t_final - 10.0 * dt;

            std::shared_ptr<CAROM::Vector> carom_tf_u_minus_some = dmd_u->predict(t);

            Vector tf_u_minus_some(carom_tf_u_minus_some->getData(),
                                   carom_tf_u_minus_some->dim());

            u = tf_u_minus_some;
            u_gf.SetFromTrueDofs(u);
        }

        ts.push_back(t);

        // 15. Iterate through the time loop.
        bool last_step = false;
        for (int ti = 1; !last_step; ti++)
        {
            fom_timer.Start();

            if (t + dt >= t_final - dt/2)
            {
                last_step = true;
            }

            ode_solver->Step(u, t, dt);

            fom_timer.Stop();

            if (offline)
            {
                dmd_training_timer.Start();

                u_gf.SetFromTrueDofs(u);
                dmd_u->takeSample(u.GetData(), t);

                if (myid == 0)
                {
                    std::cout << "Taking snapshot at: " << t << std::endl;
                }

                dmd_training_timer.Stop();
            }

            ts.push_back(t);

            if (last_step || (ti % vis_steps) == 0)
            {
                if (myid == 0)
                {
                    cout << "step " << ti << ", t = " << t << endl;
                }

                u_gf.SetFromTrueDofs(u);
                if (visualization)
                {
                    sout << "parallel " << num_procs << " " << myid << "\n";
                    sout << "solution\n" << *pmesh << u_gf << flush;
                }

                if (!de && !online && visit)
                {
                    visit_dc.SetCycle(ti);
                    visit_dc.SetTime(t);
                    visit_dc.Save();
                }

#ifdef MFEM_USE_ADIOS2
                if (adios2)
                {
                    adios2_dc->SetCycle(ti);
                    adios2_dc->SetTime(t);
                    adios2_dc->Save();
                }
#endif
            }
            oper.SetParameters(u);
        }

        if (!build_database && myid == 0)
        {
            std::ofstream outFile("ts.txt");
            for (int i = 0; i < ts.size(); i++)
            {
                outFile << ts[i] << "\n";
            }
        }

#ifdef MFEM_USE_ADIOS2
        if (adios2)
        {
            delete adios2_dc;
        }
#endif

        // 16. Save the final solution in parallel. This output can be viewed later
        //     using GLVis: "glvis -np <np> -m de_parametric_heat_conduction_greedy-mesh -g de_parametric_heat_conduction_greedy-final".
        {
            ostringstream sol_name;
            sol_name << "de_parametric_heat_conduction_greedy_" << to_string(
                         radius) << "_"
                     << to_string(alpha) << "_" << to_string(cx) << "_" << to_string(cy)
                     << "-final." << setfill('0') << setw(6) << myid;
            ofstream osol(sol_name.str().c_str());
            osol.precision(precision);
            u.Print(osol, 1);
        }
    }

    double rel_diff = 0.0;

    // 17. If in de or online mode, create the parametric DMD.
    if (de || online)
    {
        std::fstream fin("parameters.txt", std::ios_base::in);
        double curr_param;
        std::vector<std::string> dmd_paths;
        std::vector<CAROM::Vector> param_vectors;

        while (fin >> curr_param)
        {
            double curr_radius = curr_param;
            fin >> curr_param;
            double curr_alpha = curr_param;
            fin >> curr_param;
            double curr_cx = curr_param;
            fin >> curr_param;
            double curr_cy = curr_param;

            dmd_paths.push_back(to_string(curr_radius) + "_" +
                                to_string(curr_alpha) + "_" + to_string(curr_cx) + "_" +
                                to_string(curr_cy));
            CAROM::Vector param_vector(4, false);
            param_vector(0) = curr_radius;
            param_vector(1) = curr_alpha;
            param_vector(2) = curr_cx;
            param_vector(3) = curr_cy;
            param_vectors.push_back(param_vector);
        }
        fin.close();

        CAROM::Vector desired_param(4, false);
        desired_param(0) = radius;
        desired_param(1) = alpha;
        desired_param(2) = cx;
        desired_param(3) = cy;

        dmd_training_timer.Start();

        CAROM::getParametricDMD(dmd_u, param_vectors, dmd_paths, desired_param,
                                "G", "LS", closest_rbf_val);

        dmd_u->projectInitialCondition(*init);

        dmd_training_timer.Stop();
    }

    if (offline || de || calc_err_indicator)
    {
        // 17. If in offline mode, save the DMD object.
        if (offline)
        {
            if (myid == 0)
            {
                std::cout << "Creating DMD with rdim: " << rdim << std::endl;
            }

            dmd_training_timer.Start();

            dmd_u->train(rdim);

            dmd_training_timer.Stop();

            dmd_u->save(to_string(radius) + "_" + to_string(alpha) + "_" +
                        to_string(cx) + "_" + to_string(cy));

            if (myid == 0)
            {
                std::ofstream fout;
                fout.open("parameters.txt", std::ios::app);
                fout << radius << " " << alpha << " " << cx << " " << cy << std::endl;
                fout.close();
            }
        }

        // 18. Compare the DMD solution to the FOM solution.
        if (de)
        {
            if (true_solution_u == NULL)
            {
                ifstream solution_file;
                ostringstream sol_name;
                ostringstream target_name;
                sol_name << "de_parametric_heat_conduction_greedy_" << to_string(
                             target_radius) << "_"
                         << to_string(target_alpha) << "_" << to_string(target_cx) << "_" << to_string(
                             target_cy)
                         << "-final." << setfill('0') << setw(6) << myid;
                solution_file.open(sol_name.str().c_str());
                true_solution_u = new Vector(u.Size());
                true_solution_u->Load(solution_file, u.Size());
                solution_file.close();
                tot_true_solution_u_norm = sqrt(InnerProduct(MPI_COMM_WORLD,
                                                *true_solution_u, *true_solution_u));
            }
        }
        else
        {
            if (true_solution_u == NULL)
            {
                true_solution_u = new Vector(u.Size());
                *true_solution_u = u.GetData();
                tot_true_solution_u_norm = sqrt(InnerProduct(MPI_COMM_WORLD,
                                                *true_solution_u, *true_solution_u));
            }
        }

        std::shared_ptr<CAROM::Vector> result_u = dmd_u->predict(t_final);

        Vector dmd_solution_u(result_u->getData(), result_u->dim());
        Vector diff_u(true_solution_u->Size());
        subtract(dmd_solution_u, *true_solution_u, diff_u);

        double tot_diff_norm_u = sqrt(InnerProduct(MPI_COMM_WORLD, diff_u, diff_u));
        rel_diff = tot_diff_norm_u / tot_true_solution_u_norm;

        if (myid == 0)
        {
            std::cout << "Rel. diff. of DMD temp. (u) at t_final at radius " << radius <<
                      ", alpha " << alpha << ", cx " << cx << ", cy " << cy << ": "
                      << rel_diff << std::endl;
        }

        if (!de && myid == 0)
        {
            printf("Elapsed time for training DMD: %e second\n",
                   dmd_training_timer.RealTime());
        }
    }
    else if (online)
    {
        std::ifstream infile("ts.txt");

        std::string str;
        while (std::getline(infile, str))
        {
            // Line contains string of length > 0 then save it in vector
            if(str.size() > 0)
            {
                ts.push_back(std::stod(str));
            }
        }
        infile.close();

        dmd_prediction_timer.Start();

        // 14. Predict the state at t_final using DMD.
        if (myid == 0)
        {
            std::cout << "Predicting temperature using DMD at: " << ts[0] << std::endl;
        }

        std::shared_ptr<CAROM::Vector> result_u = dmd_u->predict(ts[0]);
        Vector initial_dmd_solution_u(result_u->getData(), result_u->dim());
        u_gf.SetFromTrueDofs(initial_dmd_solution_u);

        VisItDataCollection dmd_visit_dc("DMD_DE_Parametric_Heat_Conduction_Greedy_" +
                                         to_string(radius) + "_" + to_string(alpha) + "_" +
                                         to_string(cx) + "_" + to_string(cy), pmesh);
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
                    result_u = dmd_u->predict(ts[i]);
                    if (myid == 0)
                    {
                        std::cout << "Predicting temperature using DMD at: " << ts[i] << std::endl;
                    }

                    Vector dmd_solution_u(result_u->getData(), result_u->dim());
                    u_gf.SetFromTrueDofs(dmd_solution_u);

                    dmd_visit_dc.SetCycle(i);
                    dmd_visit_dc.SetTime(ts[i]);
                    dmd_visit_dc.Save();
                }
            }
        }

        dmd_prediction_timer.Stop();

        result_u = dmd_u->predict(t_final);

        // 15. Calculate the relative error between the DMD final solution and the true solution.
        Vector dmd_solution_u(result_u->getData(), result_u->dim());
        Vector diff_u(true_solution_u->Size());
        subtract(dmd_solution_u, *true_solution_u, diff_u);

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
    }

    // 19. Calculate the relative error as commanded by the greedy algorithm.
    if (offline)
    {
        if (myid == 0) std::cout << "The relative error is: " << rel_diff <<
                                     std::endl;
        greedy_sampler->setPointRelativeError(rel_diff);
    }
    // 20. Or calculate the error indicator as commanded by the greedy algorithm.
    else if (calc_err_indicator)
    {
        if (myid == 0) std::cout << "The error indicator is: " << rel_diff << std::endl;
        greedy_sampler->setPointErrorIndicator(rel_diff, 1);
    }

    if (!de && !online && myid == 0)
    {
        printf("Elapsed time for solving FOM: %e second\n", fom_timer.RealTime());
    }

    // 21. Free the used memory.
    delete ode_solver;
    delete pmesh;

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
        radius = inputs[0];
        alpha = inputs[1];
        cx = inputs[2];
        cy = inputs[3];

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
        double min_radius = 0.0;
        double max_radius = 0.0;
        double min_alpha = 0.0;
        double max_alpha = 0.0;
        double min_cx = 0.0;
        double max_cx = 0.0;
        double min_cy = 0.0;
        double max_cy = 0.0;

        while (fin >> curr_param)
        {
            double curr_radius = curr_param;
            fin >> curr_param;
            double curr_alpha = curr_param;
            fin >> curr_param;
            double curr_cx = curr_param;
            fin >> curr_param;
            double curr_cy = curr_param;

            if (first_line)
            {
                min_radius = curr_radius;
                max_radius = curr_radius;
                min_alpha = curr_alpha;
                max_alpha = curr_alpha;
                min_cx = curr_cx;
                max_cx = curr_cx;
                min_cy = curr_cy;
                max_cy = curr_cy;
                first_line = false;
            }
            else
            {
                min_radius = min(curr_radius, min_radius);
                max_radius = max(curr_radius, max_radius);
                min_alpha = min(curr_alpha, min_alpha);
                max_alpha = max(curr_alpha, max_alpha);
                min_cx = min(curr_cx, min_cx);
                max_cx = max(curr_cx, max_cx);
                min_cy = min(curr_cy, min_cy);
                max_cy = max(curr_cy, max_cy);
            }
        }
        fin.close();

        if (de_min_radius != -DBL_MAX) min_radius = de_min_radius;
        if (de_min_alpha != -DBL_MAX) min_alpha = de_min_alpha;
        if (de_min_cx != -DBL_MAX) min_cx = de_min_cx;
        if (de_min_cy != -DBL_MAX) min_cy = de_min_cy;
        if (de_max_radius != DBL_MAX) max_radius = de_max_radius;
        if (de_max_alpha != DBL_MAX) max_alpha = de_max_alpha;
        if (de_max_cx != DBL_MAX) max_cx = de_max_cx;
        if (de_max_cy != DBL_MAX) max_cy = de_max_cy;

        MFEM_VERIFY(min_radius <= max_radius, "Radius DE range is invalid.");
        MFEM_VERIFY(min_alpha <= max_alpha, "Alpha DE range is invalid.");
        MFEM_VERIFY(min_cx <= max_cx, "cx DE range is invalid.");
        MFEM_VERIFY(min_cy <= max_cy, "cy DE range is invalid.");

        if (myid == 0) cout << "DE radius range is: " << min_radius << " to " <<
                                max_radius << endl;
        if (myid == 0) cout << "DE alpha range is: " << min_alpha << " to " << max_alpha
                                << endl;
        if (myid == 0) cout << "DE cx range is: " << min_cx << " to " << max_cx << endl;
        if (myid == 0) cout << "DE cy range is: " << min_cy << " to " << max_cy << endl;

        std::vector<Constraints> constr(NumberOfParameters());
        constr[0] = Constraints(min_radius, max_radius, true);
        constr[1] = Constraints(min_alpha, max_alpha, true);
        constr[2] = Constraints(min_cx, max_cx, true);
        constr[3] = Constraints(min_cy, max_cy, true);
        return constr;
    }

private:
    unsigned int m_dim;
};

int main(int argc, char *argv[])
{
    // 1. Initialize MPI.
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    cout.precision(precision);

    // 2. Parse command-line options.
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
                   "\t   11 - Forward Euler, 12 - RK2, 13 - RK3 SSP, 14 - RK4.");
    args.AddOption(&t_final, "-tf", "--t-final",
                   "Final time; start time is 0.");
    args.AddOption(&dt, "-dt", "--time-step",
                   "Time step.");
    args.AddOption(&alpha, "-a", "--alpha",
                   "Alpha coefficient.");
    args.AddOption(&kappa, "-k", "--kappa",
                   "Kappa coefficient offset.");
    args.AddOption(&radius, "-r", "--radius",
                   "Radius of the interface of initial temperature.");
    args.AddOption(&cx, "-cx", "--cx",
                   "Center offset in the x direction.");
    args.AddOption(&cy, "-cy", "--cy",
                   "Center offset in the y direction.");
    args.AddOption(&de_min_radius, "-de_min_r", "--de_min_radius",
                   "DE min radius.");
    args.AddOption(&de_max_radius, "-de_max_r", "--de_max_radius",
                   "DE max radius.");
    args.AddOption(&de_min_alpha, "-de_min_a", "--de_min_alpha",
                   "DE min alpha.");
    args.AddOption(&de_max_alpha, "-de_max_a", "--de_max_alpha",
                   "DE max alpha.");
    args.AddOption(&de_min_cx, "-de_min_cx", "--de_min_cx",
                   "DE min cx.");
    args.AddOption(&de_max_cx, "-de_max_cx", "--de_max_cx",
                   "DE max cx.");
    args.AddOption(&de_min_cy, "-de_min_cy", "--de_min_cy",
                   "DE min cy.");
    args.AddOption(&de_max_cy, "-de_max_cy", "--de_max_cy",
                   "DE max cy.");
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
    args.AddOption(&greedy_param_space_radius_min, "-greedy-param-radius-min",
                   "--greedy-param-radius-min",
                   "The minimum radius value of the parameter point space.");
    args.AddOption(&greedy_param_space_radius_max, "-greedy-param-radius-max",
                   "--greedy-param-radius-max",
                   "The maximum radius value of the parameter point space.");
    args.AddOption(&greedy_param_space_alpha_min, "-greedy-param-alpha-min",
                   "--greedy-param-alpha-min",
                   "The minimum alpha value of the parameter point space.");
    args.AddOption(&greedy_param_space_alpha_max, "-greedy-param-alpha-max",
                   "--greedy-param-alpha-max",
                   "The maximum alpha value of the parameter point space.");
    args.AddOption(&greedy_param_space_cx_min, "-greedy-param-cx-min",
                   "--greedy-param-cx-min",
                   "The minimum cx value of the parameter point space.");
    args.AddOption(&greedy_param_space_cx_max, "-greedy-param-cx-max",
                   "--greedy-param-cx-max",
                   "The maximum cx value of the parameter point space.");
    args.AddOption(&greedy_param_space_cy_min, "-greedy-param-cy-min",
                   "--greedy-param-cy-min",
                   "The minimum cy value of the parameter point space.");
    args.AddOption(&greedy_param_space_cy_max, "-greedy-param-cy-max",
                   "--greedy-param-cy-max",
                   "The maximum cy value of the parameter point space.");
    args.AddOption(&greedy_param_space_size, "-greedy-param-size",
                   "--greedy-param-size",
                   "The number of values to search in the parameter point space.");
    args.AddOption(&greedy_relative_diff_tol, "-greedyreldifftol",
                   "--greedyreldifftol", "The greedy algorithm relative diff tolerance.");
    args.AddOption(&greedy_subset_size, "-greedysubsize", "--greedysubsize",
                   "The greedy algorithm subset size.");
    args.AddOption(&greedy_convergence_subset_size, "-greedyconvsize",
                   "--greedyconvsize", "The greedy algorithm convergence subset size.");
    args.AddOption(&closest_rbf_val, "-crv", "--crv",
                   "DMD Closest RBF Value.");
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                   "--no-visit-datafiles",
                   "Save data files for VisIt (visit.llnl.gov) visualization.");
    args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                   "Visualize every n-th timestep.");
    args.AddOption(&rdim, "-rdim", "--rdim",
                   "Reduced dimension for DMD.");
    args.AddOption(&build_database, "-build_database", "--build_database",
                   "-no-build_database", "--no-build_database",
                   "Enable or disable the build_database phase.");
    args.AddOption(&de, "-de", "--de", "-no-de", "--no-de",
                   "Enable or disable the differential evolution phase.");
    args.AddOption(&adios2, "-adios2", "--adios2-streams", "-no-adios2",
                   "--no-adios2-streams",
                   "Save data using adios2 streams.");
    args.AddOption(&baseoutputname, "-out", "--outputfile-name",
                   "Name of the sub-folder to dump files within the run directory.");
    args.Parse();
    if (!args.Good())
    {
        args.PrintUsage(cout);
        MPI_Finalize();
        return 1;
    }

    if (myid == 0)
    {
        args.PrintOptions(cout);
    }

    MFEM_VERIFY(!(build_database
                  && de), "both build_database and de can not be true!");

    // 3. Initialize the DMD database that will be built using a greedy algorithm.
    if (build_database)
    {
        MFEM_VERIFY(rdim != -1, "rdim must be set.");
        MFEM_VERIFY(!visit
                    && !visualization,
                    "visit and visualization must be turned off during the build_database phase.")
        std::ifstream infile("de_parametric_heat_conduction_greedy_data");
        if (infile.good())
        {
            if (myid == 0) std::cout << "The database has already been built. Exiting." <<
                                         std::endl;
            return 0;
        }
        infile.close();
        CAROM::Vector greedy_param_space_min(4, false);
        greedy_param_space_min.item(0) = greedy_param_space_radius_min;
        greedy_param_space_min.item(1) = greedy_param_space_alpha_min;
        greedy_param_space_min.item(2) = greedy_param_space_cx_min;
        greedy_param_space_min.item(3) = greedy_param_space_cy_min;
        CAROM::Vector greedy_param_space_max(4, false);
        greedy_param_space_max.item(0) = greedy_param_space_radius_max;
        greedy_param_space_max.item(1) = greedy_param_space_alpha_max;
        greedy_param_space_max.item(2) = greedy_param_space_cx_max;
        greedy_param_space_max.item(3) = greedy_param_space_cy_max;
        greedy_sampler = new CAROM::GreedyRandomSampler(greedy_param_space_min,
                greedy_param_space_max,
                greedy_param_space_size, false, greedy_relative_diff_tol, 1.05,
                2.0, greedy_subset_size, greedy_convergence_subset_size,
                true, "de_parametric_heat_conduction_greedy_log.txt");
    }

    target_radius = radius;
    target_alpha = alpha;
    target_cx = cx;
    target_cy = cy;

    // 4. If in differential evolution mode, run the DE algorithm using the
    //    stored DMD database.
    if (de)
    {
        std::ifstream infile("de_parametric_heat_conduction_greedy_data");
        if (!infile.good())
        {
            if (myid == 0) std::cout << "The database has not been built. Exiting." <<
                                         std::endl;
            return 0;
        }
        infile.close();

        // Create relative error cost function in 4 dimensions (radius, alpha, cx, cy)
        RelativeDifferenceCostFunction cost(4);

        // Create Differential Evolution optimizer with population size of de_ps
        // differential weight of de_F, and crossover probability of de_CR
        CAROM::DifferentialEvolution de_opt(cost, de_PS, de_F, de_CR);

        // Optimize for at least de_min_iter iterations, to a maximum of de_max_iter iterations with verbose output.
        // Stop early, after de_min_iter iterations is run, if the minimum cost did not improve by de_ct
        std::vector<double> optimal_parameters = de_opt.Optimize(de_min_iter,
                de_max_iter, de_ct, true);

        radius = optimal_parameters[0];
        alpha = optimal_parameters[1];
        cx = optimal_parameters[2];
        cy = optimal_parameters[3];

        online = true;
        de = false;
        simulation();

        delete true_solution_u;
    }
    // 4. If in build_database mode, build the database.
    else if (build_database)
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
            CAROM::Vector* relativeErrorPointData = pointRequiringRelativeError.point.get();
            struct CAROM::GreedyErrorIndicatorPoint pointRequiringErrorIndicator =
                greedy_sampler->getNextPointRequiringErrorIndicator();
            CAROM::Vector* errorIndicatorPointData =
                pointRequiringErrorIndicator.point.get();

            if (errorIndicatorPointData != NULL)
            {
                if (myid == 0) std::cout << "Calculating a error indicator at a new point." <<
                                             std::endl;
                radius = pointRequiringErrorIndicator.point.get()->item(0);
                alpha = pointRequiringErrorIndicator.point.get()->item(1);
                cx = pointRequiringErrorIndicator.point.get()->item(2);
                cy = pointRequiringErrorIndicator.point.get()->item(3);
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
                    radius = samplePointData->item(0);
                    alpha = samplePointData->item(1);
                    cx = samplePointData->item(2);
                    cy = samplePointData->item(3);
                    offline = true;
                    calc_err_indicator = false;
                }
                else
                {
                    if (myid == 0) std::cout << "The greedy algorithm has finished." << std::endl;
                    greedy_sampler->save("de_parametric_heat_conduction_greedy_data");
                    build_database = false;
                    continue;
                }
            }

            simulation();

            delete true_solution_u;
            true_solution_u = NULL;
        } while (build_database);
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

ConductionOperator::ConductionOperator(ParFiniteElementSpace &f, double al,
                                       double kap, const Vector &u)
    : TimeDependentOperator(f.GetTrueVSize(), 0.0), fespace(f), M(NULL), K(NULL),
      T(NULL), current_dt(0.0),
      M_solver(f.GetComm()), T_solver(f.GetComm()), z(height)
{
    const double rel_tol = 1e-8;

    M = new ParBilinearForm(&fespace);
    M->AddDomainIntegrator(new MassIntegrator());
    M->Assemble(0); // keep sparsity pattern of M and K the same
    M->FormSystemMatrix(ess_tdof_list, Mmat);

    M_solver.iterative_mode = false;
    M_solver.SetRelTol(rel_tol);
    M_solver.SetAbsTol(0.0);
    M_solver.SetMaxIter(100);
    M_solver.SetPrintLevel(0);
    M_prec.SetType(HypreSmoother::Jacobi);
    M_solver.SetPreconditioner(M_prec);
    M_solver.SetOperator(Mmat);

    alpha = al;
    kappa = kap;

    T_solver.iterative_mode = false;
    T_solver.SetRelTol(rel_tol);
    T_solver.SetAbsTol(0.0);
    T_solver.SetMaxIter(100);
    T_solver.SetPrintLevel(0);
    T_solver.SetPreconditioner(T_prec);

    SetParameters(u);
}

void ConductionOperator::Mult(const Vector &u, Vector &du_dt) const
{
    // Compute:
    //    du_dt = M^{-1}*-K(u)
    // for du_dt
    Kmat.Mult(u, z);
    z.Neg(); // z = -z
    M_solver.Mult(z, du_dt);
}

void ConductionOperator::ImplicitSolve(const double dt,
                                       const Vector &u, Vector &du_dt)
{
    // Solve the equation:
    //    du_dt = M^{-1}*[-K(u + dt*du_dt)]
    // for du_dt
    if (!T)
    {
        T = Add(1.0, Mmat, dt, Kmat);
        current_dt = dt;
        T_solver.SetOperator(*T);
    }
    MFEM_VERIFY(dt == current_dt, ""); // SDIRK methods use the same dt
    Kmat.Mult(u, z);
    z.Neg();
    T_solver.Mult(z, du_dt);
}

void ConductionOperator::SetParameters(const Vector &u)
{
    ParGridFunction u_alpha_gf(&fespace);
    u_alpha_gf.SetFromTrueDofs(u);
    for (int i = 0; i < u_alpha_gf.Size(); i++)
    {
        u_alpha_gf(i) = kappa + alpha*u_alpha_gf(i);
    }

    delete K;
    K = new ParBilinearForm(&fespace);

    GridFunctionCoefficient u_coeff(&u_alpha_gf);

    K->AddDomainIntegrator(new DiffusionIntegrator(u_coeff));
    K->Assemble(0); // keep sparsity pattern of M and K the same
    K->FormSystemMatrix(ess_tdof_list, Kmat);
    delete T;
    T = NULL; // re-compute T on the next ImplicitSolve
}

ConductionOperator::~ConductionOperator()
{
    delete T;
    delete M;
    delete K;
}

double InitialTemperature(const Vector &x)
{
    Vector y(x);
    Vector c(2);
    c.Elem(0) = cx;
    c.Elem(1) = cy;
    y -= c;
    if (y.Norml2() < radius)
    {
        return 2.0;
    }
    else
    {
        return 1.0;
    }
}
