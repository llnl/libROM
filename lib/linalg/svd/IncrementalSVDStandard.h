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
//              that is equivalent to but computationally more expensive than
//              the "fast update" method.

#ifndef included_IncrementalSVDStandard_h
#define included_IncrementalSVDStandard_h

#include "IncrementalSVD.h"
#include "linalg/Options.h"

namespace CAROM {

/**
 * Class IncrementalSVDStandard embodies the standard incremental SVD algorithm.
 */
class IncrementalSVDStandard : public IncrementalSVD
{
public:
    /**
     * @brief Destructor.
     */
    ~IncrementalSVDStandard();

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
    IncrementalSVDStandard(
        Options options,
        const std::string& basis_file_name);

    /**
     * @brief Unimplemented default constructor.
     */
    IncrementalSVDStandard();

    /**
     * @brief Unimplemented copy constructor.
     */
    IncrementalSVDStandard(
        const IncrementalSVDStandard& other);

    /**
     * @brief Unimplemented assignment operator.
     */
    IncrementalSVDStandard&
    operator = (
        const IncrementalSVDStandard& rhs);

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
};

}

#endif
