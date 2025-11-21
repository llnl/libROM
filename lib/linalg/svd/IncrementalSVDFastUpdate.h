/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

// Description: The concrete implementation of the incremental SVD algorithm
//              using Matthew Brand's "fast update" method.

#ifndef included_IncrementalSVDFastUpdate_h
#define included_IncrementalSVDFastUpdate_h

#include "IncrementalSVD.h"
#include "linalg/Options.h"

namespace CAROM {

/**
 * Class IncrementalSVDFastUpdate implements Brand's fast update incremental SVD
 * algorithm by implementing the pure virtual methods of the IncrementalSVD
 * base class.
 */
class IncrementalSVDFastUpdate : public IncrementalSVD
{
public:
    /**
     * @brief Destructor.
     */
    ~IncrementalSVDFastUpdate();

private:
    friend class BasisGenerator;

    /**
     * @brief Constructor.
     *
     * @param[in] options The struct containing the options for this SVD
     *                    implementation.
     * @param[in] basis_file_name The base part of the name of the file
     *                            containing the basis vectors.  Each process
     *                            will append its process ID to this base
     *                            name.
     * @see Options
     */
    IncrementalSVDFastUpdate(
        Options options,
        const std::string& basis_file_name);

    /**
     * @brief Unimplemented default constructor.
     */
    IncrementalSVDFastUpdate();

    /**
     * @brief Unimplemented copy constructor.
     */
    IncrementalSVDFastUpdate(
        const IncrementalSVDFastUpdate& other);

    /**
     * @brief Unimplemented assignment operator.
     */
    IncrementalSVDFastUpdate&
    operator = (
        const IncrementalSVDFastUpdate& rhs);

    /**
     * @brief Constructs the first SVD.
     *
     * @pre u != 0
     * @pre time >= 0.0
     *
     * @param[in] u The first state.
     * @param[in] time The simulation time for the first state.
     */
    void
    buildInitialSVD(
        double* u) override;

    /**
     * @brief Computes the current basis vectors.
     */
    void
    computeBasis() override;

    /**
     * @brief Add a linearly dependent sample to the SVD.
     *
     * @param[in] A The left singular vectors.
     * @param[in] W The right singular vectors.
     * @param[in] sigma The singular values.
     */
    void
    addLinearlyDependentSample(
        const Matrix & A,
        const Matrix & W,
        const Matrix & sigma) override;

    /**
     * @brief Add a new, unique sample to the SVD.
     *
     * @param[in] j The new column of d_U.
     * @param[in] A The left singular vectors.
     * @param[in] W The right singular vectors.
     * @param[in] sigma The singular values.
     */
    void
    addNewSample(
        const Vector & j,
        const Matrix & A,
        const Matrix & W,
        const Matrix & sigma) override;

    /**
     * @brief The matrix U'. U' is not distributed and the entire matrix
     *        exists on each processor.
     */
    std::unique_ptr<Matrix> d_Up;

    /**
     * @brief The tolerance value used to remove small singular values.
     */
    double d_singular_value_tol;

};

}

#endif
