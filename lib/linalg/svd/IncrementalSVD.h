/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

// Description: The abstract incremental SVD algorithm defines algorithm
//              interface.

#ifndef included_IncrementalSVD_h
#define included_IncrementalSVD_h

#include "SVD.h"
#include "linalg/Options.h"
#include "utils/Database.h"

namespace CAROM {

/**
 * Abstract class IncrementalSVD defines the internal API of the
 * incremental SVD algorithm.
 */
class IncrementalSVD : public SVD
{
public:
    /**
     * @brief Constructor.
     *
     * @param[in] options The struct containing the options for this abstract
     *                    SVD class.
     * @param[in] basis_file_name The base part of the name of the file
     *                            containing the basis vectors.  Each process
     *                            will append its process ID to this base
     *                            name.
     * @see Options
     */
    IncrementalSVD(
        Options options,
        const std::string& basis_file_name);

    /**
     * @brief Destructor.
     */
    virtual
    ~IncrementalSVD();

    /**
     * @brief Sample new state, u_in, at the given time.
     *
     * @pre u_in != 0
     *
     * @param[in] u_in The state at the specified time.
     * @param[in] add_without_increase If true, addLinearlyDependent is invoked.
     *
     * @return True if the sampling was successful.
     */
    virtual
    bool
    takeSample(
        double* u_in,
        bool add_without_increase = false) override;

    /**
     * @brief Returns the basis vectors for the current time interval as a
     *        Matrix.
     *
     * @return The basis vectors for the current time interval.
     */
    std::shared_ptr<const Matrix>
    getSpatialBasis() override;

    /**
     * @brief Returns the temporal basis vectors for the current time interval
     *        as a Matrix.
     *
     * @return The temporal basis vectors for the current time interval.
     */
    std::shared_ptr<const Matrix>
    getTemporalBasis() override;

    /**
     * @brief Returns the singular values for the current time interval.
     *
     * @return The singular values for the current time interval.
     */
    std::shared_ptr<const Vector>
    getSingularValues() override;

    /**
     * @brief Returns the snapshot matrix for the current time interval.
     *
     * @return The snapshot matrix for the current time interval.
     */
    std::shared_ptr<const Matrix>
    getSnapshotMatrix() override;

protected:
    /**
     * @brief Constructs the first SVD.
     *
     * @pre u != 0
     *
     * @param[in] u The first state.
     */
    virtual
    void
    buildInitialSVD(
        double* u) = 0;

    /**
     * @brief Adds the new sampled the state vector, u, to the system.
     *
     * @pre u != 0
     *
     * @param[in] u The new state.
     * @param[in] add_without_increase If true, addLinearlyDependent is invoked.
     *
     * @return True if building the incremental SVD was successful.
     */
    virtual
    bool
    buildIncrementalSVD(
        double* u, bool add_without_increase = false);

    /**
     * @brief Computes the current basis vectors.
     */
    virtual
    void
    computeBasis() = 0;

    /**
     * @brief Construct the matrix Q whose SVD is needed.
     *
     * @pre l.dim() == numSamples()
     *
     * @param[out] Q The matrix to be constructed. [d_S,l; 0,k]
     * @param[in] l The last column of Q.
     * @param[in] k The lower right element of Q.
     */
    void
    constructQ(
        double*& Q,
        const Vector & l,
        double k);

    /**
     * @brief Given a matrix, A, returns 2 of the 3 components of its
     *        singular value decomposition. The right singular vectors are not
     *        needed and therefore not returned.
     *
     * @pre A != 0
     *
     * @param[in] A The matrix whose SVD is needed.
     * @param[out] U The left singular vectors of A.
     * @param[out] S The singular values of A.
     * @param[out] V The right singular vectors of A.
     *
     * @return True if the SVD succeeded.
     */
    bool
    svd(
        double* A,
        Matrix*& U,
        Matrix*& S,
        Matrix*& V);

    /**
     * @brief Add a linearly dependent sample to the SVD.
     *
     * @param[in] A The left singular vectors.
     * @param[in] W The right singular vectors.
     * @param[in] sigma The singular values.
     */
    virtual
    void
    addLinearlyDependentSample(
        const Matrix & A,
        const Matrix & W,
        const Matrix & sigma) = 0;

    /**
     * @brief Add a new, unique sample to the SVD.
     *
     * @param[in] j The new column of d_U.
     * @param[in] A The left singular vectors.
     * @param[in] W The right singular vectors.
     * @param[in] sigma The singular values.
     */
    virtual
    void
    addNewSample(
        const Vector & j,
        const Matrix & A,
        const Matrix & W,
        const Matrix & sigma) = 0;

    /**
     * @brief The number of samples stored.
     *
     * @return The number of samples stored.
     */
    int
    numSamples()
    {
        return d_num_samples;
    }

    /**
     * @brief Computes and returns the orthogonality of m.
     *
     * @param[in] m The matrix to check.
     *
     * @return The orthogonality of m.
     */
    double
    checkOrthogonality(
        const Matrix & m);

    /**
     * @brief Tolerance to determine whether or not a sample is linearly
     * dependent.
     */
    double d_linearity_tol;

    /**
     * @brief Whether to skip linearly dependent samples.
     */
    bool d_skip_linearly_dependent;

    /**
     * @brief The maximum basis dimension
     */
    int d_max_basis_dimension;

    /**
     * @brief The total number of processors.
     */
    int d_size;

    /**
     * @brief The rank of the processor owning this object.
     */
    int d_rank;

    /**
     * @brief The dimension of the system on each processor.
     */
    std::vector<int> d_proc_dims;

    /**
     * @brief The total dimension of the system.
     */
    long int d_total_dim;

    /**
     * @brief If true the state of the SVD will be written to disk when the
     *        object is deleted. If there are multiple time intervals then
     *        the state will not be saved as restoring such a state makes no
     *        sense.
     */
    bool d_save_state;

    /**
     * @brief Whether to update the right singular vectors.
     */
    bool d_update_right_SV;

    /**
     * @brief Pointer to the database that will hold saved state data if the
     *        state is to be saved.
     */
    Database* d_state_database;

    /**
     * @brief The name of file to which state is saved or restored from.
     */
    std::string d_state_file_name;

    /**
     * @brief MPI message tag.
     */
    static const int COMMUNICATE_U;

private:
    /**
     * @brief Unimplemented default constructor.
     */
    IncrementalSVD();

    /**
     * @brief Unimplemented copy constructor.
     */
    IncrementalSVD(
        const IncrementalSVD& other);

    /**
     * @brief Unimplemented assignment operator.
     */
    IncrementalSVD&
    operator = (
        const IncrementalSVD& rhs);
};

}

#endif
