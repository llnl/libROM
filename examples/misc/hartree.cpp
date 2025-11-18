#include "linalg/BasisGenerator.h"
#include "linalg/scalapack_wrapper.h"
#include "linalg/BasisReader.h"
#include "linalg/BasisWriter.h"
#include <vector>
#include "mpi.h"
#include <string>
#include <stdio.h>
#include <iostream>
#include <cmath>

// Preconditioned Conjugate Gradient Solver
void preconditionedConjugateGradient(const CAROM::Matrix& A, const CAROM::Matrix& b, CAROM::Matrix& x, const CAROM::Matrix& M, int maxIter, double tol)
{
    int n = b.numRows();
    
    // Initial guess for x (zero vector)
    x = CAROM::Matrix(n, n, true);  // Initialize x with zeros
    
    // Residual: r = b - A * x
    CAROM::Matrix r = b; // r = b - A * x, x is initially zero
    CAROM::Matrix Ax = CAROM::Matrix(n, n, true);  // A * x
    A.multiply(x, Ax);  // Multiply A with x
    r.subtract(Ax, r);  // r = b - A * x
    
    // Preconditioned residual: z = M^-1 * r
    CAROM::Matrix z = M; // Assume M is an identity matrix for simplicity
    r.copy(z);  // z = r (if M is identity)
    
    // Initial search direction: p = z
    CAROM::Matrix p = z;
    
    double rsOld = r.dotProduct(r);  // Initial residual norm squared
    
    for (int iter = 0; iter < maxIter; ++iter) {
        // A * p
        CAROM::Matrix Ap = CAROM::Matrix(n, n, true);
        A.multiply(p, Ap);
        
        // Step size alpha: alpha = rsOld / (p^T * A * p)
        double alpha = rsOld / p.dotProduct(Ap);
        
        // Update solution: x = x + alpha * p
        CAROM::Matrix alphaP = p; 
        alphaP.scale(alpha);
        x.add(alphaP, x);
        
        // Update residual: r = r - alpha * A * p
        CAROM::Matrix alphaAp = Ap;
        alphaAp.scale(alpha);
        r.subtract(alphaAp, r);
        
        double rsNew = r.dotProduct(r);  // New residual norm squared
        
        // Check convergence
        if (sqrt(rsNew) < tol) {
            std::cout << "Converged after " << iter + 1 << " iterations" << std::endl;
            break;
        }
        
        // Preconditioned residual: z = M^-1 * r
        r.copy(z);  // z = r (if M is identity)
        
        // Update search direction: p = z + (rsNew / rsOld) * p
        double beta = rsNew / rsOld;
        p.scale(beta);
        p.add(z, p);
        
        rsOld = rsNew;  // Update old residual norm squared
    }
}

// Function to solve the Poisson equation (Delta u = source) using PCG
void solvePoisson(const CAROM::Matrix& source, CAROM::Matrix& solution) {
    int maxIter = 1000;  // Maximum iterations for PCG
    double tol = 1e-6;   // Tolerance for convergence
    
    // Set up the Poisson operator (Delta), which could be represented by a matrix A
    CAROM::Matrix A(source.numRows(), source.numColumns(), true);  // Placeholder for the Poisson operator
    // Here you would build your Poisson matrix A (e.g., using finite differences)

    // Preconditioner M (we'll use the identity matrix as a placeholder)
    CAROM::Matrix M(source.numRows(), source.numColumns(), true);  // Identity matrix
    M.setIdentity();  // For now, use the identity matrix as the preconditioner
    
    // Solve Poisson equation using PCG
    preconditionedConjugateGradient(A, source, solution, M, maxIter, tol);
}

// Function to compute u_{ij} = (Delta)^{-1} (q_i * q_j)
void computeUij(int r, const std::vector<CAROM::Matrix>& q, int rank)
{
    // Iterate over all pairs (i, j) with 1 <= i <= j <= r
    for (int i = 0; i < r; ++i) {
        for (int j = i; j < r; ++j) {
            if (i == j) {
                std::cout << "Computing u_" << i + 1 << i + 1 << std::endl;
            } else {
                std::cout << "Computing u_" << i + 1 << j + 1 << std::endl;
            }

            // Element-wise product of q_i and q_j
            CAROM::Matrix source(q[i].numRows(), q[i].numColumns(), true);
            for (int row = 0; row < q[i].numRows(); ++row) {
                for (int col = 0; col < q[i].numColumns(); ++col) {
                    source.item(row, col) = q[i].item(row, col) * q[j].item(row, col);
                }
            }

            // Solve Poisson equation: Delta u = source
            CAROM::Matrix u_ij(q[i].numRows(), q[i].numColumns(), true);
            solvePoisson(source, u_ij);

            // Immediately save the result to a file after computation
            std::string filename = "u_" + std::to_string(i + 1) + std::to_string(j + 1) + ".h5";
            u_ij.write(filename);  // Save the matrix to an HDF5 file
            std::cout << "Saved " << filename << std::endl;
        }
    }
}

int main(int argc, char* argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<std::string> sample_names;
    int dim = 0;
    int r = 0;  // Number of basis functions (or grid functions)
    std::string generator_filename = "total";
    bool subtract_mean = false;
    bool subtract_offset = false;

    if (argc >= 2) {
        int i = 1;
        while (i < argc) {
            if (!strcmp(argv[i], "basis") || !strcmp(argv[i], "-b")) {
                if (rank == 0) std::cout << "Argument " << i << " identified as basis or -b" << std::endl;
            }
            else if (!strcmp(argv[i], "mean") || !strcmp(argv[i], "-m")) {
                if (rank == 0) std::cout << "Will subtract mean" << std::endl;
                subtract_mean = true;
            }
            else if (!strcmp(argv[i], "offset") || !strcmp(argv[i], "-o")) {
                if (rank == 0) std::cout << "Will subtract offset" << std::endl;
                subtract_offset = true;
            }
            else {
                sample_names.push_back(argv[i]);
            }
            i += 1;
        }
    } else {
        if (rank == 0) std::cout << "No arguments passed." << std::endl;
        return 1;
    }

    // Read dimension and number of basis functions
    std::vector<CAROM::Matrix> q;  // Vector of grid functions q_i
    r = sample_names.size();
    if (rank == 0) std::cout << "Reading " << r << " grid functions" << std::endl;

    // Read the basis functions (q_i)
    for (const auto& sample_name : sample_names) {
        CAROM::BasisReader reader(sample_name);
        int func_dim = reader.getDim("basis");  // Assuming basis functions
        if (dim == 0) dim = func_dim;
        CAROM::Matrix q_i(func_dim, func_dim, true);
        reader.readBasis("basis", q_i);
        q.push_back(q_i);
    }

    // Compute and immediately save u_{ij} = (Delta)^{-1} (q_i * q_j)
    computeUij(r, q, rank);

    MPI_Finalize();
    return 0;
}
