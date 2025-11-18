/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

// Description: A simple, parallel Vector class with the utility needed to
//              support the basis generation methods of this library.  A
//              distributed Vector has its rows distributed across processors.

#include "Vector.h"
#include "utils/HDFDatabase.h"
#include "utils/mpi_utils.h"

#include "mpi.h"

#include <cmath>
#include <string.h>

namespace CAROM {

Vector::Vector() :
    d_vec(NULL),
    d_alloc_size(0),
    d_distributed(false),
    d_owns_data(true)
{}

Vector::Vector(
    int dim,
    bool distributed) :
    d_vec(NULL),
    d_alloc_size(0),
    d_distributed(distributed),
    d_owns_data(true)
{
    CAROM_VERIFY(dim > 0);
    setSize(dim);
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
}

Vector::Vector(
    double* vec,
    int dim,
    bool distributed,
    bool copy_data) :
    d_vec(NULL),
    d_alloc_size(0),
    d_distributed(distributed),
    d_owns_data(copy_data)
{
    CAROM_VERIFY(vec != 0);
    CAROM_VERIFY(dim > 0);
    if (copy_data) {
        setSize(dim);
        memcpy(d_vec, vec, d_alloc_size*sizeof(double));
    }
    else {
        d_vec = vec;
        d_alloc_size = dim;
        d_dim = dim;
    }
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
}

Vector::Vector(
    const Vector& other) :
    d_vec(NULL),
    d_alloc_size(0),
    d_distributed(other.d_distributed),
    d_owns_data(true)
{
    setSize(other.d_dim);
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
    memcpy(d_vec, other.d_vec, d_alloc_size*sizeof(double));
}

Vector::~Vector()
{
    if (d_owns_data && d_vec) {
        delete [] d_vec;
    }
}

Vector&
Vector::operator = (
    const Vector& rhs)
{
    d_distributed = rhs.d_distributed;
    d_num_procs = rhs.d_num_procs;
    setSize(rhs.d_dim);
    memcpy(d_vec, rhs.d_vec, d_dim*sizeof(double));
    return *this;
}

Vector&
Vector::operator += (
    const Vector& rhs)
{
    CAROM_ASSERT(d_dim == rhs.d_dim);
    for(int i=0; i<d_dim; ++i) d_vec[i] += rhs.d_vec[i];
    return *this;
}

Vector&
Vector::operator -= (
    const Vector& rhs)
{
    CAROM_ASSERT(d_dim == rhs.d_dim);
    for(int i=0; i<d_dim; ++i) d_vec[i] -= rhs.d_vec[i];
    return *this;
}

Vector&
Vector::operator = (const double& a)
{
    for(int i=0; i<d_dim; ++i) d_vec[i] = a;
    return *this;
}

Vector&
Vector::operator *= (const double& a)
{
    for(int i=0; i<d_dim; ++i) d_vec[i] *= a;
    return *this;
}

Vector&
Vector::transform(std::function<void(const int size, double* vector)>
                  transformer) {
    transformer(d_dim, d_vec);
    return *this;
}

void
Vector::transform(Vector& result,
                  std::function<void(const int size, double* vector)> transformer) const {
    result.setSize(d_dim);
    result.d_distributed = d_distributed;
    transformer(d_dim, result.d_vec);
}

Vector&
Vector::transform(
    std::function<void(const int size, double* origVector, double* resultVector)>
    transformer) {
    Vector* origVector = new Vector(*this);
    transformer(d_dim, origVector->d_vec, d_vec);
    delete origVector;
    return *this;
}

void
Vector::transform(Vector& result,
                  std::function<void(const int size, double* origVector, double* resultVector)>
                  transformer) const {
    Vector* origVector = new Vector(*this);
    result.setSize(d_dim);
    result.d_distributed = d_distributed;
    transformer(d_dim, origVector->d_vec, result.d_vec);
    delete origVector;
}

double
Vector::inner_product(
    const Vector& other) const
{
    CAROM_ASSERT(dim() == other.dim());
    CAROM_ASSERT(distributed() == other.distributed());
    double ip;
    double local_ip = 0.0;
    for (int i = 0; i < d_dim; ++i) {
        local_ip += d_vec[i]*other.d_vec[i];
    }
    if (d_num_procs > 1 && d_distributed) {
        MPI_Allreduce(&local_ip, &ip, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    else {
        ip = local_ip;
    }
    return ip;
}

double
Vector::norm() const
{
    double norm = sqrt(norm2());
    return norm;
}

double
Vector::norm2() const
{
    double norm2 = inner_product(*this);
    return norm2;
}

double
Vector::normalize()
{
    double Norm = norm();
    for (int i = 0; i < d_dim; ++i) {
        d_vec[i] /= Norm;
    }
    return Norm;
}

void
Vector::plus(
    const Vector& other,
    Vector& result) const
{
    CAROM_ASSERT(result.distributed() == distributed());
    CAROM_ASSERT(distributed() == other.distributed());
    CAROM_ASSERT(dim() == other.dim());

    // Size result correctly.
    result.setSize(d_dim);

    // Do the addition.
    for (int i = 0; i < d_dim; ++i) {
        result.d_vec[i] = d_vec[i] + other.d_vec[i];
    }
}

void
Vector::plusAx(
    double factor,
    const Vector& other,
    Vector& result) const
{
    CAROM_ASSERT(result.distributed() == distributed());
    CAROM_ASSERT(distributed() == other.distributed());
    CAROM_ASSERT(dim() == other.dim());

    // Size result correctly.
    result.setSize(d_dim);

    // Do the addition.
    for (int i = 0; i < d_dim; ++i) {
        result.d_vec[i] = d_vec[i] + factor*other.d_vec[i];
    }
}

void
Vector::plusEqAx(
    double factor,
    const Vector& other)
{
    CAROM_ASSERT(distributed() == other.distributed());
    CAROM_ASSERT(dim() == other.dim());

    // Do the addition.
    for (int i = 0; i < d_dim; ++i) {
        d_vec[i] += factor*other.d_vec[i];
    }
}

void
Vector::minus(
    const Vector& other,
    Vector& result) const
{
    CAROM_ASSERT(result.distributed() == distributed());
    CAROM_ASSERT(distributed() == other.distributed());
    CAROM_ASSERT(dim() == other.dim());

    // Size result correctly.
    result.setSize(d_dim);

    // Do the subtraction.
    for (int i = 0; i < d_dim; ++i) {
        result.d_vec[i] = d_vec[i] - other.d_vec[i];
    }
}

void
Vector::mult(
    double factor,
    Vector& result) const
{
    CAROM_ASSERT(result.distributed() == distributed());

    // Size result correctly.
    result.setSize(d_dim);

    // Do the multiplication.
    for (int i = 0; i < d_dim; ++i) {
        result.d_vec[i] = factor*d_vec[i];
    }
}

void
Vector::write(const std::string& base_file_name)
{
    CAROM_ASSERT(!base_file_name.empty());

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
    sprintf(tmp, "dim");
    database.putInteger(tmp, d_dim);
    sprintf(tmp, "data");
    database.putDoubleArray(tmp, d_vec, d_dim);
    database.close();
}

void
Vector::print(const char * prefix) const
{
    int my_rank;
    const bool success = MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    CAROM_ASSERT(success);

    std::string filename_str = prefix + std::to_string(my_rank);
    const char * filename = filename_str.c_str();
    FILE * pFile = fopen(filename,"w");
    for (int k = 0; k < d_dim; ++k) {
        fprintf(pFile, " %25.20e\n", d_vec[k]);
    }
    fclose(pFile);
}

void
Vector::read(const std::string& base_file_name)
{
    CAROM_ASSERT(!base_file_name.empty());

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
    database.open(full_file_name, "r");

    sprintf(tmp, "distributed");
    int distributed;
    database.getInteger(tmp, distributed);
    d_distributed = bool(distributed);
    int dim;
    sprintf(tmp, "dim");
    database.getInteger(tmp, dim);
    setSize(dim);
    sprintf(tmp, "data");
    database.getDoubleArray(tmp, d_vec, d_alloc_size);
    d_owns_data = true;
    if (mpi_init) {
        MPI_Comm_size(MPI_COMM_WORLD, &d_num_procs);
    }
    else {
        d_num_procs = 1;
    }
    database.close();
}

void
Vector::local_read(const std::string& base_file_name, int rank)
{
    CAROM_ASSERT(!base_file_name.empty());

    int mpi_init;
    MPI_Initialized(&mpi_init);

    char tmp[100];
    sprintf(tmp, ".%06d", rank);
    std::string full_file_name = base_file_name + tmp;
    HDFDatabase database;
    database.open(full_file_name, "r");

    sprintf(tmp, "distributed");
    int distributed;
    database.getInteger(tmp, distributed);
    d_distributed = bool(distributed);
    int dim;
    sprintf(tmp, "dim");
    database.getInteger(tmp, dim);
    setSize(dim);
    sprintf(tmp, "data");
    database.getDoubleArray(tmp, d_vec, d_alloc_size);
    d_owns_data = true;
    if (mpi_init) {
        MPI_Comm_size(MPI_COMM_WORLD, &d_num_procs);
    }
    else {
        d_num_procs = 1;
    }
    database.close();
}

double Vector::localMin(int nmax)
{
    const int n = nmax > 0 ? nmax : d_dim;
    double v = d_vec[0];

    for (int i=1; i<n; ++i)
    {
        if (d_vec[i] < v)
            v = d_vec[i];
    }

    return v;
}

void
Vector::distribute(const int local_dim)
{
    CAROM_VERIFY(!distributed());
    CAROM_VERIFY(d_owns_data);
    CAROM_VERIFY(local_dim >= 0);

    std::vector<int> row_offsets;
    int global_dim = get_global_offsets(local_dim, row_offsets,
                                        MPI_COMM_WORLD);
    CAROM_VERIFY(global_dim == d_dim);
    int local_offset = row_offsets[d_rank];
    const int new_size = local_dim;

    double *d_new_vec = new double [new_size];
    if (new_size > 0)
        memcpy(d_new_vec, &d_vec[local_offset], 8 * new_size);

    delete [] d_vec;
    d_vec = d_new_vec;
    d_alloc_size = new_size;
    d_dim = new_size;

    d_distributed = true;
}

void
Vector::gather()
{
    CAROM_VERIFY(distributed());
    CAROM_VERIFY(d_owns_data);

    std::vector<int> row_offsets;
    const int global_dim = get_global_offsets(d_dim, row_offsets,
                           MPI_COMM_WORLD);
    const int new_size = global_dim;

    int *data_offsets = new int[row_offsets.size() - 1];
    int *data_cnts = new int[row_offsets.size() - 1];
    for (int k = 0; k < row_offsets.size() - 1; k++)
    {
        data_offsets[k] = row_offsets[k];
        data_cnts[k] = (row_offsets[k+1] - row_offsets[k]);
    }

    double *d_new_vec = new double [new_size] {0.0};
    CAROM_VERIFY(MPI_Allgatherv(d_vec, d_dim, MPI_DOUBLE,
                                d_new_vec, data_cnts, data_offsets, MPI_DOUBLE,
                                MPI_COMM_WORLD) == MPI_SUCCESS);

    delete [] d_vec;
    delete [] data_offsets;
    delete [] data_cnts;
    d_vec = d_new_vec;
    d_alloc_size = new_size;
    d_dim = new_size;

    d_distributed = false;
}

int getCenterPoint(const std::vector<const Vector*>& points,
                   bool use_centroid)
{
    int center_point;

    // get center-most point
    // obtain the centroid and find the point closest to centroid
    if (use_centroid)
    {
        Vector centroid(*points[0]);
        for (int i = 1; i < points.size(); i++) {
            centroid += *points[i];
        }
        centroid.mult(1.0 / points.size(), centroid);

        double min_dist_to_centroid;
        for (int i = 0; i < points.size(); i++) {
            Vector diff;
            centroid.minus(*points[i], diff);
            double dist_to_centroid = diff.norm();
            if (i == 0 || dist_to_centroid < min_dist_to_centroid)
            {
                min_dist_to_centroid = dist_to_centroid;
                center_point = i;
            }
        }
    }

    // otherwise, do a brute force search to obtain
    // the point closest to all other points
    else
    {
        std::vector<double> dist_to_points(points.size(), 0);
        for (int i = 0; i < points.size(); i++) {
            for (int j = i + 1; j < points.size(); j++) {
                Vector diff;
                points[i]->minus(*points[j], diff);
                double dist = diff.norm();
                dist_to_points[i] += dist;
                dist_to_points[j] += dist;
            }
        }

        double closest_dist_to_points = INT_MAX;
        for (int i = 0; i < dist_to_points.size(); i++) {
            if (dist_to_points[i] < closest_dist_to_points)
            {
                closest_dist_to_points = dist_to_points[i];
                center_point = i;
            }
        }
    }

    return center_point;
}

int getCenterPoint(const std::vector<Vector>& points,
                   bool use_centroid)
{
    std::vector<const Vector*> temp_points;
    for (int i = 0; i < points.size(); i++)
    {
        temp_points.push_back(&points[i]);
    }
    return getCenterPoint(temp_points, use_centroid);
}

int getClosestPoint(const std::vector<const Vector*>& points,
                    const Vector & test_point)
{
    int closest_point = 0;
    double closest_dist_to_test_point = INT_MAX;
    for (int i = 0; i < points.size(); i++) {
        Vector diff;
        test_point.minus(*points[i], diff);
        double dist = diff.norm();
        if (dist < closest_dist_to_test_point)
        {
            closest_dist_to_test_point = dist;
            closest_point = i;
        }
    }

    return closest_point;
}

int getClosestPoint(const std::vector<Vector>& points,
                    const Vector & test_point)
{
    std::vector<const Vector*> temp_points;
    for (int i = 0; i < points.size(); i++)
    {
        temp_points.push_back(&points[i]);
    }
    return getClosestPoint(temp_points, test_point);
}

}
