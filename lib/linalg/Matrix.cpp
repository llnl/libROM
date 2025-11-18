/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

// Description: A simple, parallel Matrix class with the utility needed to
//              support the basis generation methods of this library.  A
//              distributed Matrix has its rows distributed across processors.

#include "Matrix.h"
#include "utils/HDFDatabase.h"
#include "utils/mpi_utils.h"

#include "mpi.h"
#include <string.h>
#include <vector>
#include <random>

#ifdef CAROM_HAS_ELEMENTAL
#include <El.hpp>
#endif

#include "scalapack_wrapper.h"

/* Use automatically detected Fortran name-mangling scheme */
#define dsyev CAROM_FC_GLOBAL(dsyev, DSYEV)
#define dgeev CAROM_FC_GLOBAL(dgeev, DGEEV)
#define dgetrf CAROM_FC_GLOBAL(dgetrf, DGETRF)
#define dgetri CAROM_FC_GLOBAL(dgetri, DGETRI)
#define dgeqp3 CAROM_FC_GLOBAL(dgeqp3, DGEQP3)
#define dgesdd CAROM_FC_GLOBAL(dgesdd, DGESDD)

extern "C" {
// Compute eigenvalue and eigenvectors of real symmetric matrix.
    void dsyev(char*, char*, int*, double*, int*, double*, double*, int*, int*);

// Compute eigenvalue and eigenvectors of real non-symmetric matrix.
    void dgeev(char*, char*, int*, double*, int*, double*, double*, double*, int*,
               double*, int*, double*, int*, int*);

// LU decomposition of a general matrix.
    void dgetrf(int*, int*, double*, int*, int*, int*);

// Generate inverse of a matrix given its LU decomposition.
    void dgetri(int*, double*, int*, int*, double*, int*, int*);

// BLAS-3 version of QR decomposition with column pivoting
    void dgeqp3(int*, int*, double*, int*, int*, double*, double*, int*, int*);

// Serial SVD of a matrix.
    void dgesdd(char*, int*, int*, double*, int*,
                double*, double*, int*, double*, int*,
                double*, int*, int*, int*);
}

namespace CAROM {

Matrix::Matrix() :
    d_mat(NULL),
    d_alloc_size(0),
    d_distributed(false),
    d_owns_data(true)
{}

Matrix::Matrix(
    int num_rows,
    int num_cols,
    bool distributed,
    bool randomized) :
    d_mat(0),
    d_alloc_size(0),
    d_distributed(distributed),
    d_owns_data(true)
{
    CAROM_VERIFY(num_rows > 0);
    CAROM_VERIFY(num_cols > 0);
    int mpi_init;
    MPI_Initialized(&mpi_init);
    if (mpi_init) {
        MPI_Comm_size(MPI_COMM_WORLD, &d_num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &d_rank);
    }
    else {
        d_num_procs = 1;
        d_rank = 0;
    }
    setSize(num_rows, num_cols);
    if (randomized) {
        std::default_random_engine generator;
        std::normal_distribution<double> normal_distribution(0.0, 1.0);
        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < num_cols; j++) {
                item(i,j) = normal_distribution(generator);
            }
        }
    }
}

Matrix::Matrix(
    double* mat,
    int num_rows,
    int num_cols,
    bool distributed,
    bool copy_data) :
    d_mat(0),
    d_alloc_size(0),
    d_distributed(distributed),
    d_owns_data(copy_data)
{
    CAROM_VERIFY(mat != 0);
    CAROM_VERIFY(num_rows > 0);
    CAROM_VERIFY(num_cols > 0);
    int mpi_init;
    MPI_Initialized(&mpi_init);
    if (mpi_init) {
        MPI_Comm_size(MPI_COMM_WORLD, &d_num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &d_rank);
    }
    else {
        d_num_procs = 1;
        d_rank = 0;
    }
    if (copy_data) {
        setSize(num_rows, num_cols);
        memcpy(d_mat, mat, d_alloc_size*sizeof(double));
    }
    else {
        // Check integer multiplication overflow
        if (num_rows > INT_MAX / num_cols)
            CAROM_ERROR("Matrix::Matrix- new size exceeds maximum integer value!\n");

        d_mat = mat;
        d_alloc_size = num_rows*num_cols;
        d_num_cols = num_cols;
        d_num_rows = num_rows;
        if (d_distributed) {
            calculateNumDistributedRows();
        }
    }
}

Matrix::Matrix(
    const Matrix& other) :
    d_mat(0),
    d_alloc_size(0),
    d_distributed(other.d_distributed),
    d_owns_data(true)
{
    int mpi_init;
    MPI_Initialized(&mpi_init);
    if (mpi_init) {
        MPI_Comm_size(MPI_COMM_WORLD, &d_num_procs);
        MPI_Comm_rank(MPI_COMM_WORLD, &d_rank);
    }
    else {
        d_num_procs = 1;
        d_rank = 0;
    }
    setSize(other.d_num_rows, other.d_num_cols);
    memcpy(d_mat, other.d_mat, d_alloc_size*sizeof(double));
}

Matrix::~Matrix()
{
    if (d_owns_data && d_mat) {
        delete [] d_mat;
    }
}

Matrix&
Matrix::operator = (
    const Matrix& rhs)
{
    d_distributed = rhs.d_distributed;
    d_num_procs = rhs.d_num_procs;
    setSize(rhs.d_num_rows, rhs.d_num_cols);
    memcpy(d_mat, rhs.d_mat, d_num_rows*d_num_cols*sizeof(double));
    return *this;
}

Matrix&
Matrix::operator += (
    const Matrix& rhs)
{
    CAROM_VERIFY(rhs.d_num_rows == d_num_rows);
    CAROM_VERIFY(rhs.d_num_cols == d_num_cols);
    for(int i=0; i<d_num_rows*d_num_cols; ++i) d_mat[i] += rhs.d_mat[i];
    return *this;
}

Matrix&
Matrix::operator -= (
    const Matrix& rhs)
{
    CAROM_VERIFY(rhs.d_num_rows == d_num_rows);
    CAROM_VERIFY(rhs.d_num_cols == d_num_cols);
    for(int i=0; i<d_num_rows*d_num_cols; ++i) d_mat[i] -= rhs.d_mat[i];
    return *this;
}

bool
Matrix::balanced() const
{
    // A Matrix is "balanced" (load-balanced for distributed dense matrix
    // computations) if:
    //
    // (1) the number of rows owned by each process in any pair of
    // processes on the communicator differs by at most one
    //
    // (2) process j has no fewer rows than k if j is less than k (j and
    // k are both integers corresponding to process ranks)

    // Serial matrices are balanced by definition; one process owns all
    // rows
    if (!distributed()) return true;

    const MPI_Comm comm = MPI_COMM_WORLD;

    // Otherwise, get the total number of rows of the matrix.
    int num_total_rows = numDistributedRows();

    const int first_rank_with_fewer = num_total_rows % d_num_procs;
    int my_rank;
    CAROM_VERIFY(MPI_Comm_rank(comm, &my_rank) == MPI_SUCCESS);

    const int  min_rows_per_rank = num_total_rows / d_num_procs;
    const bool has_extra_row     = my_rank < first_rank_with_fewer;
    const int  max_rows_on_rank  = min_rows_per_rank + has_extra_row;
    const bool has_enough_rows   = (d_num_rows >= min_rows_per_rank);
    const bool has_too_many_rows = (d_num_rows > max_rows_on_rank);

    int result = (has_enough_rows && !has_too_many_rows);
    const int reduce_count = 1;
    CAROM_VERIFY(MPI_Allreduce(MPI_IN_PLACE,
                               &result,
                               reduce_count,
                               MPI_INT,
                               MPI_LAND,
                               comm) == MPI_SUCCESS);

    return result;
}

Matrix&
Matrix::operator = (
    const double a)
{
    for(int i=0; i<d_num_rows*d_num_cols; ++i) {
        d_mat[i] = a;
    }
    return *this;
}

std::unique_ptr<Matrix>
Matrix::getFirstNColumns(int n) const
{
    CAROM_VERIFY(n > 0 && n <= d_num_cols);

    Matrix* first_n_columns = new Matrix(d_num_rows, n, d_distributed);
    getFirstNColumns(n, *first_n_columns);
    return std::unique_ptr<Matrix>(first_n_columns);
}

void
Matrix::getFirstNColumns(
    int n,
    Matrix& result) const
{
    CAROM_VERIFY(result.distributed() == distributed());
    CAROM_VERIFY(n > 0 && n <= d_num_cols);

    // Size result correctly.
    result.setSize(d_num_rows, n);

    for (int i = 0; i < d_num_rows; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result.item(i, j) = item(i, j);
        }
    }
}

void
Matrix::mult(
    const Matrix& other,
    Matrix& result) const
{
    CAROM_VERIFY(result.distributed() == distributed());
    CAROM_VERIFY(!other.distributed());
    CAROM_VERIFY(numColumns() == other.numRows());

    // Size result correctly.
    result.setSize(d_num_rows, other.d_num_cols);

    // Do the multiplication.
    for (int this_row = 0; this_row < d_num_rows; ++this_row) {
        for (int other_col = 0; other_col < other.d_num_cols; ++other_col) {
            double result_val = 0.0;
            for (int entry = 0; entry < d_num_cols; ++entry) {
                result_val += item(this_row, entry)*other.item(entry, other_col);
            }
            result.item(this_row, other_col) = result_val;
        }
    }
}

void
Matrix::mult(
    const Vector& other,
    Vector& result) const
{
    CAROM_VERIFY(result.distributed() == distributed());
    CAROM_VERIFY(!other.distributed());
    CAROM_VERIFY(numColumns() == other.dim());

    // Size result correctly.
    result.setSize(d_num_rows);

    // Do the multiplication.
    for (int this_row = 0; this_row < d_num_rows; ++this_row) {
        double result_val = 0.0;
        for (int entry = 0; entry < d_num_cols; ++entry) {
            result_val += item(this_row, entry)*other.item(entry);
        }
        result.item(this_row) = result_val;
    }
}

void
Matrix::pointwise_mult(
    int this_row,
    const Vector& other,
    Vector& result) const
{
    // TODO: change the CAROM_ASSERTs to CAROM_VERIFYs or generalize and eliminate the checks.
    CAROM_ASSERT(!result.distributed());
    CAROM_ASSERT(!distributed());
    CAROM_VERIFY(!other.distributed());
    CAROM_VERIFY(numColumns() == other.dim());

    // Do the multiplication.
    for (int entry = 0; entry < d_num_cols; ++entry) {
        result.item(entry) = item(this_row, entry)*other.item(entry);
    }
}

void
Matrix::pointwise_mult(
    int this_row,
    Vector& other) const
{
    CAROM_ASSERT(!distributed());
    CAROM_VERIFY(!other.distributed());
    CAROM_VERIFY(numColumns() == other.dim());

    // Do the multiplication.
    for (int entry = 0; entry < d_num_cols; ++entry) {
        other.item(entry) *= item(this_row, entry);
    }
}

void
Matrix::elementwise_mult(
    const Matrix& other,
    Matrix& result) const
{
    CAROM_VERIFY(result.distributed() == distributed());
    CAROM_VERIFY(distributed() == other.distributed());
    CAROM_VERIFY(numRows() == other.numRows());
    CAROM_VERIFY(numColumns() == other.numColumns());

    // Size result correctly.
    result.setSize(d_num_rows, other.d_num_cols);

    // Do the multiplication.
    for (int this_row = 0; this_row < d_num_rows; ++this_row) {
        for (int other_col = 0; other_col < d_num_cols; ++other_col) {
            result.item(this_row, other_col) = item(this_row,
                                                    other_col) * other.item(this_row, other_col);
        }
    }
}

void
Matrix::elementwise_square(
    Matrix& result) const
{
    // Size result correctly.
    result.setSize(d_num_rows, d_num_cols);

    // Do the pointwise square.
    for (int this_row = 0; this_row < d_num_rows; ++this_row) {
        for (int this_col = 0; this_col < d_num_cols; ++this_col) {
            const double a = item(this_row, this_col);
            result.item(this_row, this_col) = a * a;
        }
    }
}

void
Matrix::multPlus(
    Vector& a,
    const Vector& b,
    double c) const
{
    CAROM_VERIFY(a.distributed() == distributed());
    CAROM_VERIFY(!b.distributed());
    CAROM_VERIFY(numColumns() == b.dim());
    CAROM_VERIFY(numRows() == a.dim());

    for (int this_row = 0; this_row < d_num_rows; ++this_row) {
        double tmp = 0.0;
        for (int this_col = 0; this_col < d_num_cols; ++this_col) {
            tmp += item(this_row, this_col)*b.item(this_col);
        }
        a.item(this_row) += tmp*c;
    }
}

void
Matrix::transposeMult(
    const Matrix& other,
    Matrix& result) const
{
    CAROM_VERIFY(!result.distributed());
    CAROM_VERIFY(distributed() == other.distributed());
    CAROM_VERIFY(numRows() == other.numRows());

    // Size result correctly.
    result.setSize(d_num_cols, other.d_num_cols);

    // Do the multiplication.
    for (int this_col = 0; this_col < d_num_cols; ++this_col) {
        for (int other_col = 0; other_col < other.d_num_cols; ++other_col) {
            double result_val = 0.0;
            for (int entry = 0; entry < d_num_rows; ++entry) {
                result_val += item(entry, this_col)*other.item(entry, other_col);
            }
            result.item(this_col, other_col) = result_val;
        }
    }
    if (d_distributed && d_num_procs > 1) {
        int new_mat_size = d_num_cols*other.d_num_cols;
        MPI_Allreduce(MPI_IN_PLACE,
                      &result.item(0, 0),
                      new_mat_size,
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD);
    }
}

void
Matrix::transposeMult(
    const Vector& other,
    Vector& result) const
{
    CAROM_VERIFY(!result.distributed());
    CAROM_VERIFY(distributed() == other.distributed());
    CAROM_VERIFY(numRows() == other.dim());

    // If the result has not been allocated then do so.  Otherwise size it
    // correctly.
    result.setSize(d_num_cols);

    // Do the multiplication.
    for (int this_col = 0; this_col < d_num_cols; ++this_col) {
        double result_val = 0.0;
        for (int entry = 0; entry < d_num_rows; ++entry) {
            result_val += item(entry, this_col)*other.item(entry);
        }
        result.item(this_col) = result_val;
    }
    if (d_distributed && d_num_procs > 1) {
        MPI_Allreduce(MPI_IN_PLACE,
                      &result.item(0),
                      d_num_cols,
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD);
    }
}

void
Matrix::getColumn(int column,
                  Vector& result) const
{
    result.setSize(d_num_rows);
    for (int i = 0; i < d_num_rows; i++) {
        result.item(i) = item(i, column);
    }
}

void
Matrix::inverse(
    Matrix& result) const
{
    CAROM_VERIFY(!result.distributed() && result.numRows() == numRows() &&
                 result.numColumns() == numColumns());
    CAROM_VERIFY(!distributed());
    CAROM_VERIFY(numRows() == numColumns());

    // Size result correctly.
    result.setSize(d_num_rows, d_num_cols);

    // Call lapack routines to do the inversion.
    // Set up some stuff the lapack routines need.
    int info;
    int mtx_size = d_num_rows;
    int lwork = mtx_size*mtx_size;
    int* ipiv = new int [mtx_size];
    double* work = new double [lwork];
    // To use lapack we need a column major representation of this which is
    // essentially the transform of this.  Use result for this representation.
    for (int row = 0; row < mtx_size; ++row) {
        for (int col = 0; col < mtx_size; ++col) {
            result.item(col, row) = item(row, col);
        }
    }
    // Now call lapack to do the inversion.
    dgetrf(&mtx_size, &mtx_size, result.d_mat, &mtx_size, ipiv, &info);
    CAROM_VERIFY(info == 0);

    dgetri(&mtx_size, result.d_mat, &mtx_size, ipiv, work, &lwork, &info);
    CAROM_VERIFY(info == 0);

    // Result now has the inverse in a column major representation.  Put it
    // into row major order.
    for (int row = 0; row < mtx_size; ++row) {
        for (int col = row+1; col < mtx_size; ++col) {
            double tmp = result.item(row, col);
            result.item(row, col) = result.item(col, row);
            result.item(col, row) = tmp;
        }
    }

    delete [] ipiv;
    delete [] work;
}

void Matrix::transpose()
{
    CAROM_VERIFY(!distributed() && numRows() == numColumns());  // Avoid resizing
    const int n = numRows();
    for (int i=0; i<n-1; ++i)
    {
        for (int j=i+1; j<n; ++j)
        {
            const double t = d_mat[i*n+j];
            d_mat[i*n+j] = d_mat[j*n+i];
            d_mat[j*n+i] = t;
        }
    }
}

void
Matrix::inverse()
{
    CAROM_VERIFY(!distributed());
    CAROM_VERIFY(numRows() == numColumns());

    // Call lapack routines to do the inversion.
    // Set up some stuff the lapack routines need.
    int info;
    int mtx_size = d_num_rows;
    int lwork = mtx_size*mtx_size;
    int* ipiv = new int [mtx_size];
    double* work = new double [lwork];
    // To use lapack we need a column major representation of this which is
    // essentially the transform of this.
    for (int row = 0; row < mtx_size; ++row) {
        for (int col = row+1; col < mtx_size; ++col) {
            double tmp = item(row, col);
            item(row, col) = item(col, row);
            item(col, row) = tmp;
        }
    }
    // Now call lapack to do the inversion.
    dgetrf(&mtx_size, &mtx_size, d_mat, &mtx_size, ipiv, &info);
    CAROM_VERIFY(info == 0);

    dgetri(&mtx_size, d_mat, &mtx_size, ipiv, work, &lwork, &info);
    CAROM_VERIFY(info == 0);

    // This now has its inverse in a column major representation.  Put it into
    // row major representation.
    for (int row = 0; row < mtx_size; ++row) {
        for (int col = row+1; col < mtx_size; ++col) {
            double tmp = item(row, col);
            item(row, col) = item(col, row);
            item(col, row) = tmp;
        }
    }

    delete [] ipiv;
    delete [] work;
}

void Matrix::transposePseudoinverse()
{
    CAROM_VERIFY(!distributed());
    CAROM_VERIFY(numRows() >= numColumns());

    if (numRows() == numColumns())
    {
        inverse();
    }
    else
    {
        std::unique_ptr<Matrix> AtA = this->transposeMult(*this);

        // Directly invert AtA, which is a bad idea if AtA is not small.
        AtA->inverse();

        // Pseudoinverse is (AtA)^{-1}*this^T, but we store the transpose of the result in this, namely this*(AtA)^{-T}.
        Vector row(numColumns(), false);
        Vector res(numColumns(), false);
        for (int i=0; i<numRows(); ++i)
        {   // Compute i-th row of this multiplied by (AtA)^{-T}, whose transpose is (AtA)^{-1} times i-th row transposed.
            for (int j=0; j<numColumns(); ++j)
                row.item(j) = this->item(i,j);

            AtA->mult(row, res);

            // Overwrite i-th row with transpose of result.
            for (int j=0; j<numColumns(); ++j)
                this->item(i,j) = res.item(j);
        }
    }
}

void
Matrix::print(const char * prefix) const
{
    int my_rank;
    const bool success = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    CAROM_ASSERT(success);

    std::string filename_str = prefix + std::to_string(my_rank);
    const char * filename = filename_str.c_str();
    FILE * pFile = fopen(filename,"w");
    for (int row = 0; row < d_num_rows; ++row) {
        for (int col = 0; col < d_num_cols; ++col) {
            fprintf(pFile, " %25.20e\t", item(row,col));
        }
        fprintf(pFile, "\n");
    }
    fclose(pFile);
}

void
Matrix::write(const std::string& base_file_name) const
{
    CAROM_VERIFY(!base_file_name.empty());

    int mpi_init;
    MPI_Initialized(&mpi_init);
    int rank;
    if (mpi_init) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    else {
        rank = 0;
    }

    char tmp[100];
    sprintf(tmp, ".%06d", rank);
    std::string full_file_name = base_file_name + tmp;
    HDFDatabase database;
    database.create(full_file_name);

    sprintf(tmp, "distributed");
    database.putInteger(tmp, d_distributed);
    sprintf(tmp, "num_rows");
    database.putInteger(tmp, d_num_rows);
    sprintf(tmp, "num_cols");
    database.putInteger(tmp, d_num_cols);
    sprintf(tmp, "data");
    database.putDoubleArray(tmp, d_mat, d_num_rows*d_num_cols);
    database.close();
}

void
Matrix::read(const std::string& base_file_name)
{
    CAROM_VERIFY(!base_file_name.empty());

    int mpi_init;
    MPI_Initialized(&mpi_init);
    int rank;
    if (mpi_init) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &d_num_procs);
    }
    else {
        rank = 0;
        d_num_procs = 1;
    }

    char tmp[100];
    sprintf(tmp, ".%06d", rank);
    std::string full_file_name = base_file_name + tmp;
    HDFDatabase database;
    database.open(full_file_name, "r");

    sprintf(tmp, "distributed");
    int distributed;
    database.getInteger(tmp, distributed);
    d_distributed = bool(distributed);
    int num_rows;
    sprintf(tmp, "num_rows");
    database.getInteger(tmp, num_rows);
    int num_cols;
    sprintf(tmp, "num_cols");
    database.getInteger(tmp, num_cols);
    setSize(num_rows,num_cols);
    sprintf(tmp, "data");
    database.getDoubleArray(tmp, d_mat, d_alloc_size);
    d_owns_data = true;
    database.close();
}

void
Matrix::local_read(const std::string& base_file_name, int rank)
{
    CAROM_VERIFY(!base_file_name.empty());

    int mpi_init;
    MPI_Initialized(&mpi_init);
    if (mpi_init) {
        MPI_Comm_size(MPI_COMM_WORLD, &d_num_procs);
    }
    else {
        d_num_procs = 1;
    }

    char tmp[100];
    sprintf(tmp, ".%06d", rank);
    std::string full_file_name = base_file_name + tmp;
    HDFDatabase database;
    database.open(full_file_name, "r");

    sprintf(tmp, "distributed");
    int distributed;
    database.getInteger(tmp, distributed);
    d_distributed = bool(distributed);
    int num_rows;
    sprintf(tmp, "num_rows");
    database.getInteger(tmp, num_rows);
    int num_cols;
    sprintf(tmp, "num_cols");
    database.getInteger(tmp, num_cols);
    setSize(num_rows,num_cols);
    sprintf(tmp, "data");
    database.getDoubleArray(tmp, d_mat, d_alloc_size);
    d_owns_data = true;
    database.close();
}

void
Matrix::distribute(const int &local_num_rows)
{
    CAROM_VERIFY(!distributed());
    CAROM_VERIFY(d_owns_data);

    std::vector<int> row_offsets;
    int num_total_rows = get_global_offsets(local_num_rows, row_offsets,
                                            MPI_COMM_WORLD);
    CAROM_VERIFY(num_total_rows == d_num_rows);
    int local_offset = row_offsets[d_rank] * d_num_cols;
    const int new_size = local_num_rows * d_num_cols;

    double *d_new_mat = new double [new_size];
    if (new_size > 0)
        memcpy(d_new_mat, &d_mat[local_offset], 8 * new_size);

    delete [] d_mat;
    d_mat = d_new_mat;
    d_alloc_size = new_size;

    d_num_distributed_rows = d_num_rows;
    d_num_rows = local_num_rows;

    d_distributed = true;
}

void
Matrix::gather()
{
    CAROM_VERIFY(distributed());
    CAROM_VERIFY(d_owns_data);

    std::vector<int> row_offsets;
    const int num_total_rows = get_global_offsets(d_num_rows, row_offsets,
                               MPI_COMM_WORLD);
    CAROM_VERIFY(num_total_rows == d_num_distributed_rows);
    const int new_size = d_num_distributed_rows * d_num_cols;

    int *data_offsets = new int[row_offsets.size() - 1];
    int *data_cnts = new int[row_offsets.size() - 1];
    for (int k = 0; k < row_offsets.size() - 1; k++)
    {
        data_offsets[k] = row_offsets[k] * d_num_cols;
        data_cnts[k] = (row_offsets[k+1] - row_offsets[k]) * d_num_cols;
    }

    double *d_new_mat = new double [new_size] {0.0};
    CAROM_VERIFY(MPI_Allgatherv(d_mat, d_num_rows * d_num_cols, MPI_DOUBLE,
                                d_new_mat, data_cnts, data_offsets, MPI_DOUBLE,
                                MPI_COMM_WORLD) == MPI_SUCCESS);

    delete [] d_mat;
    delete [] data_offsets;
    delete [] data_cnts;
    d_mat = d_new_mat;
    d_alloc_size = new_size;

    d_num_rows = d_num_distributed_rows;

    d_distributed = false;
}

void
Matrix::calculateNumDistributedRows() {
    if (d_distributed && d_num_procs > 1) {
        int num_total_rows = d_num_rows;
        CAROM_VERIFY(MPI_Allreduce(MPI_IN_PLACE,
                                   &num_total_rows,
                                   1,
                                   MPI_INT,
                                   MPI_SUM,
                                   MPI_COMM_WORLD) == MPI_SUCCESS);
        d_num_distributed_rows = num_total_rows;
    }
    else {
        d_num_distributed_rows = d_num_rows;
    }
}

std::unique_ptr<Matrix> Matrix::qr_factorize() const
{
    std::vector<Matrix*> QR;
    qr_factorize(false, QR);
    return std::unique_ptr<Matrix>(QR[0]);
}

void Matrix::qr_factorize(std::vector<std::unique_ptr<Matrix>> & QR) const
{
    std::vector<Matrix*> QRraw;  // Raw pointers
    qr_factorize(true, QRraw);
    QR.clear();
    for (int i=0; i<2; ++i)
        QR.push_back(std::unique_ptr<Matrix>(QRraw[i]));
}

void Matrix::qr_factorize(bool computeR,
                          std::vector<Matrix*> & QR) const
{
    const int myid = d_rank;
    const int ncols = numColumns();

    std::vector<int> row_offset(d_num_procs + 1);
    row_offset[d_num_procs] = numDistributedRows();
    row_offset[myid] = numRows();

    CAROM_VERIFY(MPI_Allgather(MPI_IN_PLACE,
                               1,
                               MPI_INT,
                               row_offset.data(),
                               1,
                               MPI_INT,
                               MPI_COMM_WORLD) == MPI_SUCCESS);
    for (int i = d_num_procs - 1; i >= 0; i--) {
        row_offset[i] = row_offset[i + 1] - row_offset[i];
    }

    CAROM_VERIFY(row_offset[0] == 0);

    SLPK_Matrix slpk;

    int nrow_blocks = d_num_procs;
    int ncol_blocks = 1;

    int blocksize = row_offset[d_num_procs] / d_num_procs;
    if (row_offset[d_num_procs] % d_num_procs != 0) blocksize += 1;
    initialize_matrix(&slpk, ncols, numDistributedRows(),
                      ncol_blocks, nrow_blocks, ncols, blocksize);
    for (int rank = 0; rank < d_num_procs; ++rank) {
        scatter_block(&slpk, 1, row_offset[rank] + 1,
                      getData(), ncols,
                      row_offset[rank + 1] - row_offset[rank], rank);
    }

    QRManager QRmgr;
    qr_init(&QRmgr, &slpk);
    lqfactorize(&QRmgr);

    Matrix *R_matrix = nullptr;
    if (computeR)
    {
        // Obtain L from the distributed overwritten matrix QRmgr.A
        Matrix* L_dist_matrix = new Matrix(row_offset[myid + 1] - row_offset[myid],
                                           ncols, distributed());

        for (int rank = 0; rank < d_num_procs; ++rank) {
            gather_block(&L_dist_matrix->item(0, 0), QRmgr.A, 1,
                         row_offset[rank] + 1, ncols,
                         row_offset[rank + 1] - row_offset[rank], rank);
        }

        R_matrix = new Matrix(ncols, ncols, false);
        if (myid == 0)
        {
            const int numLocalRowsR = std::min(ncols, row_offset[myid + 1]);
            for (int i=0; i<numLocalRowsR; ++i)
            {
                for (int j=i; j<ncols; ++j)
                    (*R_matrix)(i,j) = (*L_dist_matrix)(i,j);

                for (int j=0; j<i; ++j)
                    (*R_matrix)(i,j) = 0.0;
            }
        }

        // Gather any rows of R to root from other ranks.
        const int maxLocalRowsR = std::min(ncols, row_offset[myid + 1]);
        const int numLocalRowsR = std::max(0, maxLocalRowsR - row_offset[myid]);

        std::vector<int> allNumRowsR(d_num_procs);
        MPI_Gather(&numLocalRowsR, 1, MPI_INT, allNumRowsR.data(), 1, MPI_INT,
                   0, MPI_COMM_WORLD);

        std::vector<double> localRowsR;

        if (myid > 0)
        {
            localRowsR.resize(numLocalRowsR * ncols);
            for (int i=0; i<numLocalRowsR; ++i)
            {
                const int row = row_offset[myid] + i;
                for (int j=row; j<ncols; ++j)
                    localRowsR[(i * ncols) + j] = (*L_dist_matrix)(i,j);

                for (int j=0; j<row; ++j)
                    localRowsR[(i * ncols) + j] = 0.0;
            }
        }
        else
            localRowsR.resize(1);  // Just to have a valid pointer from data() call.

        delete L_dist_matrix;

        std::vector<double> recvRowsR;
        if (myid == 0)
            recvRowsR.resize((ncols - numLocalRowsR) * ncols);

        if (recvRowsR.size() == 0)
            recvRowsR.resize(1);  // Just to have a valid pointer from data() call.

        std::vector<int> disp(d_num_procs);
        disp[0] = 0;
        allNumRowsR[0] = 0;
        for (int i=1; i<d_num_procs; ++i)
        {
            allNumRowsR[i] *= ncols;
            disp[i] = disp[i - 1] + allNumRowsR[i-1];
        }

        MPI_Gatherv(localRowsR.data(), myid == 0 ? 0 : numLocalRowsR * ncols,
                    MPI_DOUBLE, recvRowsR.data(),
                    allNumRowsR.data(), disp.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (myid == 0)
        {
            for (int i=numLocalRowsR; i<ncols; ++i)
            {
                for (int j=0; j<ncols; ++j)
                    (*R_matrix)(i,j) = recvRowsR[(i - numLocalRowsR) * ncols + j];
            }
        }

        MPI_Bcast(R_matrix->getData(), ncols * ncols, MPI_DOUBLE, 0,
                  MPI_COMM_WORLD);
    }

    // Manipulate QRmgr.A to get elementary household reflectors.
    for (int i = row_offset[myid]; i < ncols; i++) {
        for (int j = 0; j < i - row_offset[myid] && j < row_offset[myid + 1]; j++) {
            QRmgr.A->mdata[j * QRmgr.A->mm + i] = 0;
        }
        if (i < row_offset[myid + 1]) {
            QRmgr.A->mdata[(i - row_offset[myid]) * QRmgr.A->mm + i] = 1;
        }
    }

    // Obtain Q
    lqcompute(&QRmgr);
    Matrix *qr_factorized_matrix = new Matrix(row_offset[myid + 1] -
            row_offset[myid],
            ncols, distributed());

    for (int rank = 0; rank < d_num_procs; ++rank) {
        gather_block(&qr_factorized_matrix->item(0, 0), QRmgr.A, 1,
                     row_offset[rank] + 1, ncols,
                     row_offset[rank + 1] - row_offset[rank], rank);
    }

    free_matrix_data(&slpk);
    release_context(&slpk);

    free(QRmgr.tau);
    free(QRmgr.ipiv);

    QR.resize(2);
    QR[0] = qr_factorized_matrix;
    QR[1] = R_matrix;
}

void
Matrix::qrcp_pivots_transpose(int* row_pivot,
                              int* row_pivot_owner,
                              int  pivots_requested) const
{
    if(!distributed()) {
        return qrcp_pivots_transpose_serial(row_pivot,
                                            row_pivot_owner,
                                            pivots_requested);
    }
    else {
        return qrcp_pivots_transpose_distributed(row_pivot,
                row_pivot_owner,
                pivots_requested);
    }
}

void
Matrix::qrcp_pivots_transpose_serial(int* row_pivot,
                                     int* row_pivot_owner,
                                     int  pivots_requested) const
{
    // This method assumes this matrix is serial
    CAROM_VERIFY(!distributed());

    // Number of pivots requested can't exceed the number of rows of the
    // matrix
    CAROM_VERIFY(pivots_requested <= numRows());
    CAROM_VERIFY(pivots_requested > 0);

    // Make sure arrays are allocated before entry; this method does not
    // own the input pointers
    CAROM_VERIFY(row_pivot != NULL);
    CAROM_VERIFY(row_pivot_owner != NULL);

    // Get dimensions of transpose of matrix
    int num_rows_of_transpose = numColumns();
    int num_cols_of_transpose = numRows();

    // LAPACK routines tend to overwrite their inputs, but we'd like to
    // keep the basis matrix and use it in later computations, so copy
    // the basis matrix here.
    Matrix scratch(*this);

    // Allocate work arrays; work array for QR must be at least 1 plus 3
    // times the number of columns of its matrix input. This algorithm
    // applies QR to transposed basis matrix, so the applicable
    // dimension is the number of rows of the basis matrix. It's
    // possible to get better performance by computing the optimal block
    // size and then using that value to size the work array; see the
    // LAPACK source code and documentation for details.
    int lwork = 20 * num_cols_of_transpose + 1;
    double* work = new double[lwork];
    double* tau  = new double[std::min(num_rows_of_transpose,
                                       num_cols_of_transpose)];
    int* pivot = new int[num_cols_of_transpose] ();
    int info;

    // Compute the QR decomposition with column pivots of the transpose
    // of this matrix by abusing the fact that the C++ memory model is
    // row-major format, which is the transpose of the Fortran memory
    // model (which is column-major). Passing the row-major data
    // looks like an in-place transposition to Fortran.
    dgeqp3(&num_rows_of_transpose,
           &num_cols_of_transpose,
           scratch.d_mat,
           &num_rows_of_transpose,
           pivot,
           tau,
           work,
           &lwork,
           &info);

    // Fail if error in LAPACK routine.
    CAROM_VERIFY(info == 0);

    // Assume communicator is MPI_COMM_WORLD and get rank of this
    // process
    int is_mpi_initialized, is_mpi_finalized;
    CAROM_VERIFY(MPI_Initialized(&is_mpi_initialized) == MPI_SUCCESS);
    CAROM_VERIFY(MPI_Finalized(&is_mpi_finalized) == MPI_SUCCESS);
    int my_rank = 0;
    if(is_mpi_initialized && !is_mpi_finalized) {
        const MPI_Comm my_comm = MPI_COMM_WORLD;
        CAROM_VERIFY(MPI_Comm_rank(my_comm, &my_rank) == MPI_SUCCESS);
    }

    // Copy over pivots and subtract one to convert them from a
    // Fortran-based indexing convention (first element of 1-D array by
    // default corresponds to index of 1, though this convention can be
    // overridden) to a C-based indexing convention (first element of
    // 1-D array corresponds to index of 0).
    for (int i = 0; i < pivots_requested; i++) {
        row_pivot[i]       = pivot[i] - 1;
        row_pivot_owner[i] = my_rank;
    }

    // Free arrays
    delete [] work;
    delete [] tau;
    delete [] pivot;
}

void
Matrix::qrcp_pivots_transpose_distributed(int* row_pivot,
        int* row_pivot_owner,
        int  pivots_requested)
const
{
    // Check if distributed; otherwise, use serial implementation
    CAROM_VERIFY(distributed());

#ifdef CAROM_HAS_ELEMENTAL
    // Shim to design interface; not implemented yet

    // Elemental implementation
    return qrcp_pivots_transpose_distributed_elemental
           (row_pivot, row_pivot_owner, pivots_requested);
#else
    qrcp_pivots_transpose_distributed_scalapack
    (row_pivot, row_pivot_owner, pivots_requested);
#endif
}

void
Matrix::qrcp_pivots_transpose_distributed_scalapack
(int* row_pivot, int* row_pivot_owner, int pivots_requested) const
{
    // Check if distributed; otherwise, use serial implementation
    CAROM_VERIFY(distributed());

    int num_total_rows = d_num_rows;
    CAROM_VERIFY(MPI_Allreduce(MPI_IN_PLACE,
                               &num_total_rows,
                               1,
                               MPI_INT,
                               MPI_SUM,
                               MPI_COMM_WORLD) == MPI_SUCCESS);

    int *row_offset = new int[d_num_procs + 1];
    row_offset[d_num_procs] = num_total_rows;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    row_offset[my_rank] = d_num_rows;

    CAROM_VERIFY(MPI_Allgather(MPI_IN_PLACE,
                               1,
                               MPI_INT,
                               row_offset,
                               1,
                               MPI_INT,
                               MPI_COMM_WORLD) == MPI_SUCCESS);

    for (int i = d_num_procs - 1; i >= 0; i--) {
        row_offset[i] = row_offset[i + 1] - row_offset[i];
    }
    CAROM_VERIFY(row_offset[0] == 0);

    SLPK_Matrix slpk;
    int blocksize = row_offset[d_num_procs] / d_num_procs;
    if (row_offset[d_num_procs] % d_num_procs != 0) blocksize += 1;

    initialize_matrix(&slpk, d_num_cols, row_offset[d_num_procs], 1, d_num_procs, 1,
                      blocksize);  // transposed

    CAROM_VERIFY(d_num_cols <=
                 pivots_requested); // Otherwise, take a submatrix for the QR (not implemented).

    for (int rank = 0; rank < d_num_procs; ++rank) {
        // Take the row-major data in d_mat and put it in a transposed column-major array in slpk
        scatter_block(&slpk, 1, row_offset[rank]+1,
                      d_mat,
                      d_num_cols, row_offset[rank+1] - row_offset[rank],
                      rank);
    }

    QRManager QRmgr;
    qr_init(&QRmgr, &slpk);
    qrfactorize(&QRmgr);

    // Just gather the pivots to root and discard the factorization
    CAROM_VERIFY(0 < pivots_requested && pivots_requested <= QRmgr.ipivSize);
    CAROM_VERIFY(pivots_requested <= std::max(d_num_rows, d_num_cols));

    const int scount = std::max(0, std::min(pivots_requested,
                                            row_offset[my_rank+1]) - row_offset[my_rank]);
    int *mypivots = (scount > 0) ? new int[scount] : NULL;

    for (int i=0; i<scount; ++i)
        mypivots[i] = QRmgr.ipiv[i]-1;  // Make it 0-based

    int *rcount = (my_rank == 0) ? new int[d_num_procs] : NULL;
    int *rdisp = (my_rank == 0) ? new int[d_num_procs] : NULL;

    MPI_Gather(&scount, 1, MPI_INT, rcount, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
    {
        rdisp[0] = 0;
        for (int i = 1; i<d_num_procs; ++i)
            rdisp[i] = rdisp[i-1] + rcount[i-1];

        CAROM_VERIFY(rdisp[d_num_procs-1] + rcount[d_num_procs-1] == pivots_requested);
    }

    MPI_Gatherv(mypivots, scount, MPI_INT, row_pivot, rcount, rdisp, MPI_INT, 0,
                MPI_COMM_WORLD);

    delete [] mypivots;

    if (my_rank == 0)
    {
        for (int i=0; i<pivots_requested; ++i)
        {
            row_pivot_owner[i] = -1;
            for (int j=d_num_procs-1; j>=0; --j)
            {
                if (row_offset[j] <= row_pivot[i])
                {
                    row_pivot_owner[i] = j;
                    break;
                }
            }

            // Note that row_pivot[i] is a global index.
            CAROM_VERIFY(row_pivot_owner[i] >= 0);
            CAROM_VERIFY(row_offset[row_pivot_owner[i]] <= row_pivot[i]
                         && row_pivot[i] < row_offset[row_pivot_owner[i]+1]);
        }
    }
    else
    {
        for (int i=0; i<scount; ++i)
            row_pivot[i] = QRmgr.ipiv[i]-1;  // Make it 0-based
    }

    free_matrix_data(&slpk);
    release_context(&slpk);

    free(QRmgr.tau);
    free(QRmgr.ipiv);

    delete [] rcount;
    delete [] rdisp;
    delete [] row_offset;
}

void
Matrix::qrcp_pivots_transpose_distributed_elemental
(int* row_pivot, int* row_pivot_owner, int pivots_requested)
const
{
#ifdef CAROM_HAS_ELEMENTAL
    // Check if distributed; otherwise, use serial implementation
    CAROM_VERIFY(distributed());

    // Check if balanced
    if (balanced()) {
        qrcp_pivots_transpose_distributed_elemental_balanced
        (row_pivot, row_pivot_owner, pivots_requested);
    }
    else {
        qrcp_pivots_transpose_distributed_elemental_unbalanced
        (row_pivot, row_pivot_owner, pivots_requested);
    }
#else
    CAROM_VERIFY(false);
#endif
}

void
Matrix::qrcp_pivots_transpose_distributed_elemental_balanced
(int* row_pivot, int* row_pivot_owner, int pivots_requested)
const
{
#ifdef CAROM_HAS_ELEMENTAL
    // Compute pivots redundantly across all processes using the QRCP
    // from the distributed dense linear algebra library Elemental.

    // The following assumptions are made in this implementation just to
    // get a version up and running:
    //
    // (1) this->balanced() == true; // The matrix is balanced
    //
    // (2) Process 0 is the master rank of the object
    //
    // (3) Matrix rows are distributed block-cyclically over all processes
    //     starting on process zero with row 0, in increasing order of row
    //     index (all elements of a given row are still stored on the same
    //     process)
    //
    // (4) This Matrix is distributed over the MPI_COMM_WORLD
    //     communicator
    //
    // Some of these assumptions can be relaxed if the Matrix object
    // stores more information.

    // Check if distributed and balanced
    CAROM_VERIFY(distributed() && balanced());

    // Make sure arrays are allocated before entry; this method does not
    // own the input pointers
    CAROM_VERIFY(row_pivot != NULL);
    CAROM_VERIFY(row_pivot_owner != NULL);

    // Compute total number of rows to set global sizes of matrix
    const MPI_Comm comm    = MPI_COMM_WORLD;
    const int master_rank  = 0;

    int num_total_rows     = d_num_rows;
    const int reduce_count = 1;
    CAROM_VERIFY(MPI_Allreduce(MPI_IN_PLACE,
                               &num_total_rows,
                               reduce_count,
                               MPI_INT,
                               MPI_SUM,
                               comm) == MPI_SUCCESS);

    // Number of pivots requested can't exceed the total number of rows
    // of the matrix
    CAROM_VERIFY(pivots_requested <= num_total_rows);
    CAROM_VERIFY(pivots_requested > 0);

    // To compute a column-pivoted QRCP using Elemental, we need to
    // first get the data into datatypes Elemental can operate on.

    // Construct a process grid on the communicator that is 1
    // by (# processes), in column-major order; each process in the grid
    // will own its own rows of the matrix
    const El::Grid grid(comm, 1);

    // The row communicator in the grid should have the same number
    // of processes as the comm owned by the Matrix
    CAROM_VERIFY(El::mpi::Size(grid.RowComm()) == d_num_procs);

    // Instantiate the transposed matrix; Elemental calls the number of
    // rows the "height" of the matrix, and the number of columns the
    // "width" of the matrix
    El::Int height = static_cast<El::Int>(numColumns());
    El::Int width  = static_cast<El::Int>(numRows());
    El::Int root   = static_cast<El::Int>(master_rank);
    El::DistMatrix<double> scratch(height, width, grid, root);

    // Columns of the matrix should be distributed round-robin; each
    // element of a column should be on the same process. The
    // distribution should satisfy two invariants:
    //
    // (1) Each "global row" should have the same rank in its column
    // communicator
    //
    // (2) For the scratch matrix global column index j, the process (on
    // the row communicator) that owns j should be process rank (j %
    // d_num_procs).
    //
    CAROM_VERIFY(scratch.RowOwner(0) == scratch.RowOwner(1));
    int my_rank;
    CAROM_VERIFY(MPI_Comm_rank(comm, &my_rank) == MPI_SUCCESS);
    El::Int rank_as_row = static_cast<El::Int>(my_rank);
    CAROM_VERIFY(scratch.ColOwner(rank_as_row) == rank_as_row);

    // Set up work matrices
    El::DistMatrix<double> householder_scalars(grid);
    El::DistMatrix<double> diagonal(grid);
    El::DistPermutation perm(grid);
    El::QRCtrl<double> ctrl;

    // Build the copy of the matrix element by element, mapping
    // local indices to global indices. The Elemental::DistMatrix indices
    // are the transpose of the Matrix indices

    //  Elemental assigns elements of a matrix in a block-cyclic fashion
    // based on the size (specifically, the height and width) of the
    // El::Grid object; see
    // http://libelemental.org/documentation/dev/core/dist_matrix/Element/MC_MR.html
    // for details on the default element distribution, which is what
    // this code uses.
    //
    // If the matrix is balanced, then the matrix rows (i.e., columns of
    // the scratch, transpose matrix) should conform to the row
    // distribution of the Elemental process grid, and additional
    // communication (beyond possibly in local to global index mapping
    // queries) is not needed to assign elements in the proper places
    // because the matrix is already load-balanced. Furthermore, all of
    // the elements of a row of this matrix (column of scratch,
    // transpose matrix) are on a single process. Since the matrix rows
    // (columns of scratch) don't have to be redistributed, we can use
    // Elemental's native (global <--> local) indexing to construct a
    // map between local indices on a given process and global indices
    // of the distributed matrix, and Elemental will also give us the
    // correct process rank owning any given column of the scratch
    // matrix.
    for (int row = 0; row < d_num_rows; row++) {
        El::Int el_loc_col = static_cast<El::Int>(row);
        El::Int el_global_col = scratch.GlobalCol(el_loc_col);
        for (int col = 0; col < d_num_cols; col++) {
            El::Int el_loc_row = static_cast<El::Int>(col);
            El::Int el_global_row = scratch.GlobalRow(el_loc_row);
            scratch.Set(el_global_row, el_global_col, this->item(row, col));
        }
    }

    // After transferring the data over to Elemental's native data
    // types, we compute the QRCP.
    El::QR(scratch, householder_scalars, diagonal, perm); // add ctrl if needed

    // Then, we transfer the pivots into the pivot array
    // stored redundantly on each process.
    for (size_t i = 0; i < pivots_requested; i++) {
        El::Int el_i = static_cast<El::Int>(i);
        El::Int el_perm_i = perm.Image(el_i);

        // The permutation is computed in terms of global indices, so
        // we need to compute the local column pivot of the transpose;
        // this will be a row pivot
        El::Int el_loc_i = scratch.LocalCol(el_perm_i);
        int loc_i = static_cast<int>(el_loc_i);
        row_pivot[i] = loc_i;

        // The global index of the permutation can also be used to figure
        // out which process owns that pivot row, because this process is
        // also the process that owns that global column of the scratch
        // matrix
        El::Int el_owner = scratch.ColOwner(el_perm_i);
        int owner = static_cast<int>(el_owner);
        row_pivot_owner[i] = owner;
    }
#else
    CAROM_VERIFY(false);
#endif
}

void
Matrix::qrcp_pivots_transpose_distributed_elemental_unbalanced
(int* row_pivot, int* row_pivot_owner, int pivots_requested)
const
{
#ifdef CAROM_HAS_ELEMENTAL
    // Compute pivots redundantly across all processes using the QRCP
    // from the distributed dense linear algebra library Elemental.

    // The following assumptions are made in this implementation just to
    // get a version up and running:
    //
    // (1) Process 0 is the master rank of the object
    //
    // (2) This Matrix is distributed over the MPI_COMM_WORLD
    //     communicator
    //
    // Some of these assumptions can be relaxed if the Matrix object
    // stores more information.

    // Check if distributed and unbalanced
    CAROM_VERIFY(distributed() && !balanced());

    // Make sure arrays are allocated before entry; this method does not
    // own the input pointers
    CAROM_VERIFY(row_pivot != NULL);
    CAROM_VERIFY(row_pivot_owner != NULL);

    // Compute total number of rows to set global sizes of matrix
    const MPI_Comm comm    = MPI_COMM_WORLD;
    const int master_rank  = 0;

    int num_total_rows     = d_num_rows;
    const int reduce_count = 1;
    CAROM_VERIFY(MPI_Allreduce(MPI_IN_PLACE,
                               &num_total_rows,
                               reduce_count,
                               MPI_INT,
                               MPI_SUM,
                               comm) == MPI_SUCCESS);

    // Number of pivots requested can't exceed the total number of rows
    // of the matrix
    CAROM_VERIFY(pivots_requested <= num_total_rows);
    CAROM_VERIFY(pivots_requested > 0);

    // To compute a column-pivoted QRCP using Elemental, we need to
    // first get the data into datatypes Elemental can operate on.

    // Construct a process grid on the communicator that is 1
    // by (# processes), in column-major order; each process in the grid
    // will own its own rows of the matrix
    const El::Grid grid(comm, 1);

    // The row communicator in the grid should have the same number
    // of processes as the comm owned by the Matrix
    CAROM_VERIFY(El::mpi::Size(grid.RowComm()) == d_num_procs);

    // Instantiate the transposed matrix; Elemental calls the number of
    // rows the "height" of the matrix, and the number of columns the
    // "width" of the matrix
    El::Int height = static_cast<El::Int>(numColumns());
    El::Int width  = static_cast<El::Int>(numRows());
    El::Int root   = static_cast<El::Int>(master_rank);
    El::DistMatrix<double> scratch(height, width, grid, root);

    // Columns of the matrix should be distributed round-robin; each
    // element of a column should be on the same process. The
    // distribution should satisfy two invariants:
    //
    // (1) Each "global row" should have the same rank in its column
    // communicator
    //
    // (2) For the scratch matrix global column index j, the process (on
    // the row communicator) that owns j should be process rank (j %
    // d_num_procs).
    //
    CAROM_VERIFY(scratch.RowOwner(0) == scratch.RowOwner(1));
    int my_rank;
    CAROM_VERIFY(MPI_Comm_rank(comm, &my_rank) == MPI_SUCCESS);
    El::Int rank_as_row = static_cast<El::Int>(my_rank);
    CAROM_VERIFY(scratch.ColOwner(rank_as_row) == rank_as_row);

    // Set up work matrices
    El::DistMatrix<double> householder_scalars(grid);
    El::DistMatrix<double> diagonal(grid);
    El::DistPermutation perm(grid);
    El::QRCtrl<double> ctrl;

    // Build the copy of the matrix element by element, mapping local
    // indices to global indices. The Elemental::DistMatrix indices are
    // the transpose of the Matrix indices. The El::Grid object should
    // be constructed so that each column of the scratch matrix is owned
    // by a single process.
    //
    // If the matrix is unbalanced, matrix elements need to be
    // redistributed among processes; then, for the purposes of
    // computing the QR decomposition only, we redistribute matrix
    // elements in the scratch matrix. First, we compute global to
    // (process rank, local) index map. The mapping chosen is not
    // terribly performant -- it may do more data movement than
    // necessary when assigning matrix elements -- but is easy to
    // implement.

    int *row_offset = new int[d_num_procs + 1];
    row_offset[d_num_procs + 1] = num_total_rows;

    row_offset[my_rank] = d_num_rows;
    const int send_recv_count = 1;
    CAROM_VERIFY(MPI_Allgather(MPI_IN_PLACE,
                               send_recv_count,
                               MPI_INT,
                               row_offset,
                               send_recv_count,
                               MPI_INT,
                               comm) == MPI_SUCCESS);

    for (int i = d_num_procs - 1; i >= 0; i--) {
        row_offset[i] = row_offset[i + 1] - row_offset[i];
    }
    CAROM_VERIFY(row_offset[0] == 0);

    // Use the computed (local <--> global) index map to assign elements
    // to the scratch matrix
    for (int row = row_offset[my_rank]; row < row_offset[my_rank + 1]; row++) {
        El::Int el_loc_col = static_cast<El::Int>(row - row_offset[my_rank]);
        El::Int el_global_col = scratch.GlobalCol(el_loc_col);
        for (int col = 0; col < d_num_cols; col++) {
            El::Int el_loc_row = static_cast<El::Int>(col);
            El::Int el_global_row = scratch.GlobalRow(el_loc_row);
            scratch.Set(el_global_row, el_global_col, this->item(row, col));
        }
    }

    // After transferring the data over to Elemental's native data
    // types, we compute the QRCP.
    El::QR(scratch, householder_scalars, diagonal, perm); // add ctrl if needed

    // Then, we transfer the pivots into the pivot array
    // stored redundantly on each process.
    for (size_t i = 0; i < pivots_requested; i++) {
        El::Int el_i = static_cast<El::Int>(i);
        El::Int el_perm_i = perm.Image(el_i);

        // The permutation is computed in terms of global row indices of
        // the Matrix, so we need to compute the rank that owns the column
        // of scratch (remember: scratch is the transpose of this Matrix)
        // corresponding to this global index
        int global_row_index = static_cast<int>(el_perm_i);
        int rank;
        for (rank = 0; rank < d_num_procs; rank++) {
            bool is_at_or_above_lower_bound = (global_row_index >= row_offset[rank]);
            bool is_below_upper_bound = (global_row_index < row_offset[rank + 1]);
            if (is_at_or_above_lower_bound && is_below_upper_bound) {
                row_pivot_owner[i] = rank;
                break;
            }
        }

        // The local row index is then the global row index minus the
        // row_offset computed above in the simple (local <--> global) index
        // map computed above
        row_pivot[i] = global_row_index - row_offset[rank];
    }

    // Free arrays
    delete [] row_offset;
#else
    CAROM_VERIFY(false);
#endif
}

void
Matrix::orthogonalize(bool double_pass, double zero_tol)
{
    int const num_passes = double_pass ? 2 : 1;

    for (int work = 0; work < d_num_cols; ++work)
    {
        // Orthogonalize the column (twice if double_pass == true).
        for (int k = 0; k < num_passes; k++)
        {
            for (int col = 0; col < work; ++col)
            {
                double factor = 0.0;

                for (int i = 0; i < d_num_rows; ++i)
                    factor += item(i, col) * item(i, work);

                if (d_distributed && d_num_procs > 1)
                {
                    CAROM_VERIFY( MPI_Allreduce(MPI_IN_PLACE, &factor, 1,
                                                MPI_DOUBLE, MPI_SUM,
                                                MPI_COMM_WORLD)
                                  == MPI_SUCCESS );
                }
                for (int i = 0; i < d_num_rows; ++i)
                    item(i, work) -= factor * item(i, col);
            }
        }

        // Normalize the column.
        double norm = 0.0;

        for (int i = 0; i < d_num_rows; ++i)
            norm += item(i, work) * item(i, work);

        if (d_distributed && d_num_procs > 1)
        {
            CAROM_VERIFY( MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE,
                                        MPI_SUM, MPI_COMM_WORLD)
                          == MPI_SUCCESS );
        }
        if (norm > zero_tol)
        {
            norm = 1.0 / sqrt(norm);
            for (int i = 0; i < d_num_rows; ++i)
                item(i, work) *= norm;
        }
    }
}

void
Matrix::orthogonalize_last(int ncols, bool double_pass, double zero_tol)
{
    if (ncols == -1) ncols = d_num_cols;
    CAROM_VERIFY((ncols > 0) && (ncols <= d_num_cols));

    const int last_col = ncols - 1; // index of column to be orthonormalized

    int const num_passes = double_pass ? 2 : 1;

    // Orthogonalize the column (twice if double_pass == true).
    for (int k = 0; k < num_passes; k++)
    {
        for (int col = 0; col < last_col; ++col)
        {
            double factor = 0.0;

            for (int i = 0; i < d_num_rows; ++i)
                factor += item(i, col) * item(i, last_col);

            if (d_distributed && d_num_procs > 1)
            {
                CAROM_VERIFY( MPI_Allreduce(MPI_IN_PLACE, &factor, 1, MPI_DOUBLE,
                                            MPI_SUM, MPI_COMM_WORLD)
                              == MPI_SUCCESS );
            }
            for (int i = 0; i < d_num_rows; ++i)
                item(i, last_col) -= factor * item(i, col);
        }
    }

    // Normalize the column.
    double norm = 0.0;

    for (int i = 0; i < d_num_rows; ++i)
        norm += item(i, last_col) * item(i, last_col);

    if (d_distributed && d_num_procs > 1)
    {
        CAROM_VERIFY( MPI_Allreduce(MPI_IN_PLACE, &norm, 1, MPI_DOUBLE,
                                    MPI_SUM, MPI_COMM_WORLD) == MPI_SUCCESS );
    }
    if (norm > zero_tol)
    {
        norm = 1.0 / sqrt(norm);
        for (int i = 0; i < d_num_rows; ++i)
            item(i, last_col) *= norm;
    }
}

void
Matrix::rescale_rows_max()
{
    // Rescale every matrix row by its maximum absolute value.
    // In the Matrix class, columns are distributed row wise, but rows are
    // not distributed; namely, each process acts on a number of full rows.
    // Therefore, no MPI communication is needed.

    for (int i = 0; i < d_num_rows; i++)
    {
        // Find the row's max absolute value.
        double row_max = fabs(item(i, 0));
        for (int j = 1; j < d_num_cols; j++)
        {
            if (fabs(item(i, j)) > row_max)
                row_max = fabs(item(i, j));
        }

        // Rescale every row entry, if max nonzero.
        if (row_max > 1.0e-14)
        {
            for (int j = 0; j < d_num_cols; j++)
                item(i, j) /= row_max;
        }
    }
}

void
Matrix::rescale_cols_max()
{
    // Rescale every matrix column by its maximum absolute value.
    // Matrix columns are distributed row wise, so MPI communication is needed
    // to get the maximum of each column across all processes.

    // Find each column's max absolute value in the current process.
    double local_max[d_num_cols];
    for (int j = 0; j < d_num_cols; j++)
    {
        local_max[j] = fabs(item(0, j));
        for (int i = 1; i < d_num_rows; i++)
        {
            if (fabs(item(i, j)) > local_max[j])
                local_max[j] = fabs(item(i, j));
        }
    }

    // Get the max across all processes, if applicable.
    double global_max[d_num_cols];
    if (d_distributed && d_num_procs > 1)
    {
        MPI_Allreduce(&local_max, &global_max, d_num_cols, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
    }
    else
    {
        for (int i = 0; i < d_num_cols; i++)
            global_max[i] = local_max[i];
    }

    // Rescale each column's entries, if max nonzero.
    for (int j = 0; j < d_num_cols; j++)
    {
        if (global_max[j] > 1.0e-14)
        {
            double tmp = 1.0 / global_max[j];
            for (int i = 0; i < d_num_rows; i++)
                item(i, j) *= tmp;
        }
    }
}

Matrix outerProduct(const Vector &v, const Vector &w)
{
    /*
     * There are two cases of concern:
     *
     * 1) w is not distributed
     *
     * 2) w is distributed
     *
     */
    int result_num_rows = v.dim();
    int result_num_cols;
    bool is_distributed = v.distributed();
    Vector gathered_w;

    /*
     * Gather all of the entries in w on each process into a Vector stored
     * redundantly (i.e., not distributed) on each process. If the Vector w
     * is not distributed, this step is trivial.
     */
    if (!w.distributed())
    {
        result_num_cols = w.dim();
        gathered_w = w;
    }
    else // w.distributed() is true
    {
        // Get the number of columns on each processor and gather these
        // counts into an array. Use std::vector as an array substitute
        // because variable-length arrays aren't in the C++ standard,
        // but the storage for std::vector containers must be contiguous
        // as defined by the C++ standard.
        int process_local_num_cols = w.dim();
        MPI_Comm comm = MPI_COMM_WORLD;
        MPI_Datatype num_cols_datatype = MPI_INT;
        int num_procs;
        MPI_Comm_size(comm, &num_procs);
        std::vector<int> num_cols_on_proc(num_procs, 0);
        int num_procs_send_count = 1;
        int num_procs_recv_count = 1;
        MPI_Allgather(&process_local_num_cols,
                      num_procs_send_count,
                      num_cols_datatype,
                      num_cols_on_proc.data(),
                      num_procs_recv_count,
                      num_cols_datatype,
                      comm);

        // Compute the displacements for the entries in the gathered
        // Vector, along with the total number of columns in the result.
        std::vector<int> gathered_w_displacements(num_procs, 0);
        result_num_cols = 0;
        for (int i = 0; i < num_cols_on_proc.size(); i++)
        {
            gathered_w_displacements.at(i) = result_num_cols;
            result_num_cols += num_cols_on_proc.at(i);
        }

        // Gather the data from each process onto each process -- this
        // step is an Allgatherv (because difference processes may
        // have different numbers of entries.
        std::vector<double> w_data(process_local_num_cols);
        for (int i = 0; i < w_data.size(); i++)
        {
            w_data.at(i) = w(i);
        }

        std::vector<double> gathered_w_data(result_num_cols);
        MPI_Datatype entry_datatype = MPI_DOUBLE;
        MPI_Allgatherv(w_data.data(),
                       process_local_num_cols,
                       entry_datatype,
                       gathered_w_data.data(),
                       num_cols_on_proc.data(),
                       gathered_w_displacements.data(),
                       entry_datatype,
                       comm);

        gathered_w.setSize(result_num_cols);
        for (int i = 0; i < gathered_w_data.size(); i++)
        {
            gathered_w(i) = gathered_w_data.at(i);
        }
    }

    /* Create the matrix */
    Matrix result(result_num_rows, result_num_cols, is_distributed);

    /* Compute the outer product using the gathered copy of w. */
    for (int i = 0; i < result_num_rows; i++)
    {
        for (int j = 0; j < result_num_cols; j++)
        {
            result(i, j) = v(i) * gathered_w(j);
        }
    }

    return result;
}

Matrix DiagonalMatrixFactory(const Vector &v)
{
    const int resultNumRows = v.dim();
    int resultNumColumns, processRowStartIndex, processNumColumns;
    const bool isDistributed = v.distributed();

    /* If v isn't distributed, sizing the output matrix is trivial. */
    if (false == isDistributed)
    {
        resultNumColumns = processNumColumns = resultNumRows;
        processRowStartIndex = 0;
    }
    else /* true == isDistributed */
    {
        using sizeType = std::vector<int>::size_type;

        /**
         * Get the number of rows on each process; these process row
         * counts must be summed to get the number of columns on each
         * process.
         */
        const MPI_Comm comm = MPI_COMM_WORLD;
        int numProcesses;
        MPI_Comm_size(comm, &numProcesses);
        std::vector<int>
        numRowsOnProcess(static_cast<sizeType>(numProcesses));
        const int one = 1;
        const MPI_Datatype indexType = MPI_INT;
        MPI_Allgather(&resultNumRows, one, indexType,
                      numRowsOnProcess.data(), one, indexType, comm);

        /**
         * Compute the row starting index and total number of rows
         * over all processes -- which is also the total number of
         * columns. Also compute the starting row index on each
         * process.
         *
         * Assume that Matrix rows are contiguous sets of row indices,
         * e.g., if a four row matrix is partititioned over two
         * processes, then process zero on the MPI_Comm owns rows 0
         * and 1, and process one on the MPI_Comm owns rows 2 and 3.
         */
        std::vector<int> rowStartIndexOnProcess(numProcesses);
        resultNumColumns = 0;
        for (sizeType i = 0; i < rowStartIndexOnProcess.size(); i++)
        {
            rowStartIndexOnProcess.at(i) = resultNumColumns;
            resultNumColumns += numRowsOnProcess.at(i);
        }
        int processNumber;
        MPI_Comm_rank(comm, &processNumber);
        processRowStartIndex =
            rowStartIndexOnProcess.at(static_cast<sizeType>(processNumber));
    } /* end (true == isDistributed) */

    /**
     * Create the diagonal matrix and assign process local entries in v
     * to process local entries in the diagonal matrix.
     */
    Matrix diagonalMatrix(resultNumRows, resultNumColumns, isDistributed);
    for (int i = 0; i < resultNumRows; i++)
    {
        for (int j = 0; j < resultNumColumns; j++)
        {
            /**
             * Off-diagonal matrix entries are zero; diagonal matrix entries
             * come from the input Vector.
             */
            const double entry = (j == (i + processRowStartIndex)) ? v(i) : 0.0;
            diagonalMatrix(i, j) = entry;
        }
    }

    return diagonalMatrix;
}

Matrix IdentityMatrixFactory(const Vector &v)
{
    Vector temporary(v);
    temporary = 1.0;
    return DiagonalMatrixFactory(temporary);
}

EigenPair SymmetricRightEigenSolve(const Matrix & A)
{
    char jobz = 'V', uplo = 'U';

    int info;
    int k = A.numColumns();
    int lwork = std::max(1, 10*k-1);
    double* work = new double [lwork];
    double* eigs = new double [k];
    Matrix* ev = new Matrix(A);

    // ev now in a row major representation.  Put it
    // into column major order.
    for (int row = 0; row < k; ++row) {
        for (int col = row+1; col < k; ++col) {
            double tmp = ev->item(row, col);
            ev->item(row, col) = ev->item(col, row);
            ev->item(col, row) = tmp;
        }
    }

    // Now call lapack to do the eigensolve.
    dsyev(&jobz, &uplo, &k, ev->getData(), &k, eigs, work, &lwork, &info);

    // Eigenvectors now in a column major representation.  Put it
    // into row major order.
    for (int row = 0; row < k; ++row) {
        for (int col = row+1; col < k; ++col) {
            double tmp = ev->item(row, col);
            ev->item(row, col) = ev->item(col, row);
            ev->item(col, row) = tmp;
        }
    }

    EigenPair eigenpair;
    eigenpair.ev = ev;
    for (int i = 0; i < k; i++)
    {
        eigenpair.eigs.push_back(eigs[i]);
    }

    delete [] work;
    delete [] eigs;

    return eigenpair;
}

ComplexEigenPair NonSymmetricRightEigenSolve(const Matrix & A)
{
    char jobvl = 'N', jobrl = 'V';

    int info;
    int k = A.numColumns();
    int lwork = std::max(k*k, 10*k);
    double* work = new double [lwork];
    double* e_real = new double [k];
    double* e_imaginary = new double [k];
    double* ev_l = NULL;
    Matrix* ev_r = new Matrix(k, k, false);
    Matrix* A_copy = new Matrix(A);

    // A now in a row major representation.  Put it
    // into column major order.
    for (int row = 0; row < k; ++row) {
        for (int col = row+1; col < k; ++col) {
            double tmp = A_copy->item(row, col);
            A_copy->item(row, col) = A_copy->item(col, row);
            A_copy->item(col, row) = tmp;
        }
    }

    // Now call lapack to do the eigensolve.
    dgeev(&jobvl, &jobrl, &k, A_copy->getData(), &k, e_real, e_imaginary, ev_l,
          &k, ev_r->getData(), &k, work, &lwork, &info);

    // Eigenvectors now in a column major representation.  Put it
    // into row major order.
    for (int row = 0; row < k; ++row) {
        for (int col = row+1; col < k; ++col) {
            double tmp = ev_r->item(row, col);
            ev_r->item(row, col) = ev_r->item(col, row);
            ev_r->item(col, row) = tmp;
        }
    }

    ComplexEigenPair eigenpair;
    eigenpair.ev_real = new Matrix(k, k, false);
    eigenpair.ev_imaginary = new Matrix(k, k, false);

    // Separate lapack eigenvector into real and imaginary parts
    for (int i = 0; i < k; ++i)
    {
        for (int row = 0; row < k; ++row) {
            eigenpair.ev_real->item(row, i) = ev_r->item(row, i);
        }
        if (e_imaginary[i] != 0)
        {
            for (int row = 0; row < k; ++row) {
                eigenpair.ev_real->item(row, i + 1) = ev_r->item(row, i);
                eigenpair.ev_imaginary->item(row, i) = ev_r->item(row, i + 1);
                eigenpair.ev_imaginary->item(row, i + 1) = -ev_r->item(row, i + 1);
            }

            // Skip the next eigenvalue since it'll be part of the complex
            // conjugate pair.
            ++i;
        }
    }

    for (int i = 0; i < k; i++)
    {
        eigenpair.eigs.push_back(std::complex<double>(e_real[i], e_imaginary[i]));
    }

    delete [] work;
    delete [] e_real;
    delete [] e_imaginary;
    delete ev_r;
    delete A_copy;

    return eigenpair;
}

void SerialSVD(Matrix* A,
               Matrix* U,
               Vector* S,
               Matrix* V)
{
    CAROM_VERIFY(!A->distributed());
    int m = A->numRows();
    int n = A->numColumns();

    Matrix* A_copy = new Matrix(*A);
    if (U == NULL)
    {
        U = new Matrix(m, std::min(m, n), false);
    }
    else
    {
        CAROM_VERIFY(!U->distributed());
        U->setSize(m, std::min(m, n));
    }
    if (V == NULL)
    {
        CAROM_VERIFY(!V->distributed());
        V = new Matrix(std::min(m, n), n, false);
    }
    else
    {
        V->setSize(std::min(m, n), n);
    }
    if (S == NULL)
    {
        CAROM_VERIFY(!S->distributed());
        S = new Vector(n, false);
    }
    else
    {
        S->setSize(n);
    }

    char jobz = 'S';
    int lda = m;
    int ldu = m;
    int ldv = n;
    int mn = std::min(m, n);
    int lwork = 4 * mn * mn + 7 * mn;
    double* work = new double [lwork];
    int iwork[8*std::min(m, n)];
    int info;

    dgesdd(&jobz, &m, &n, A_copy->getData(), &lda, S->getData(), U->getData(), &ldu,
           V->getData(),
           &ldv, work, &lwork, iwork, &info);

    CAROM_VERIFY(info == 0);

    delete [] work;
    delete A_copy;
}

SerialSVDDecomposition SerialSVD(Matrix* A)
{
    CAROM_VERIFY(!A->distributed());
    Matrix* U = NULL;
    Vector* S = NULL;
    Matrix* V = NULL;

    SerialSVD(A, U, S, V);

    SerialSVDDecomposition decomp;
    decomp.U = U;
    decomp.S = S;
    decomp.V = V;

    return decomp;
}

// Compute the product A^T * B, where A is represented by the space-time
// product of As and At, and likewise for B.
std::unique_ptr<Matrix> SpaceTimeProduct(const CAROM::Matrix & As,
        const CAROM::Matrix & At,
        const CAROM::Matrix & Bs, const CAROM::Matrix & Bt,
        const std::vector<double> *tscale,
        const bool At0at0, const bool Bt0at0, const bool lagB,
        const bool skip0)
{
    // TODO: implement reduction in parallel for the spatial matrices
    CAROM_VERIFY(As.distributed() && Bs.distributed());

    const int AtOS = At0at0 ? 1 : 0;
    const int BtOS0 = Bt0at0 ? 1 : 0;
    const int BtOS = BtOS0 + (lagB ? 1 : 0);

    const int nrows = As.numColumns();
    const int ncols = Bs.numColumns();

    const int nspace = As.numRows();
    const int ntime = At.numRows() + AtOS;

    CAROM_VERIFY(nspace == Bs.numRows() && ntime == Bt.numRows() + BtOS0);

    // For now, we assume one time vector for each space vector
    CAROM_VERIFY(nrows == At.numColumns() && ncols == Bt.numColumns());

    //const int k0 = (At0at0 || Bt0at0 || lagB) std::max(AtOS, BtOS) : 0;
    const int k0AB = std::max(AtOS, BtOS);
    const int k00 = skip0 ? 1 : 0;
    const int k0 = std::max(k0AB, k00);

    CAROM_VERIFY(tscale == NULL || (tscale->size() == ntime-1 && k0 > 0));

    Matrix* p = new CAROM::Matrix(nrows, ncols, false);

    for (int i=0; i<nrows; ++i)
    {
        for (int j=0; j<ncols; ++j)
        {
            double pij = 0.0;

            for (int k=k0; k<ntime; ++k)
            {
                //const double At_k = (At0at0 && k == 0) ? 0.0 : At->item(k - AtOS,i);
                //const double Bt_k = (Bt0at0 && k == 0) ? 0.0 : Bt->item(k - BtOS,j);
                const double At_k = At.item(k - AtOS,i);
                const double Bt_k = Bt.item(k - BtOS,j);
                const double ts = (tscale == NULL) ? 1.0 : (*tscale)[k - 1];

                double spij = 0.0;  // sum over spatial entries
                for (int m=0; m<nspace; ++m)
                    spij += As.item(m,i) * Bs.item(m,j);

                pij += spij * ts * At_k * Bt_k;
            }

            p->item(i, j) = pij;
        }
    }

    return std::unique_ptr<Matrix>(p);
}

} // end namespace CAROM
