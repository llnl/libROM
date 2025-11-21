/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

#ifndef MFEMUTILITIES_H
#define MFEMUTILITIES_H

#include "mfem.hpp"
#include "linalg/Matrix.h"

using namespace mfem;

namespace CAROM {

/**
 * @brief This function computes a reduced, non-distributed matrix C^t AB stored
 *        identically (redundantly) on every process.
 *
 * @param[in] A The distributed HypreParMatrix (an MFEM class) A in C^t AB.
 *
 * @param[in] B The distributed Matrix B in C^t AB.
 *
 * @param[in] C The distributed Matrix C in C^t AB.
 *
 * @param[out] CtAB The non-distributed Matrix C^t AB.
 *
 */
void ComputeCtAB(const HypreParMatrix& A,
                 const CAROM::Matrix& B,
                 const CAROM::Matrix& C,
                 CAROM::Matrix& CtAB);

/**
 * @brief This function computes a reduced, non-distributed vector C^t AB stored
 *        identically (redundantly) on every process.
 *
 * @param[in] A The distributed HypreParMatrix (an MFEM class) A in C^t AB.
 *
 * @param[in] B The distributed HypreParVector B in C^t AB.
 *
 * @param[in] C The distributed Matrix C in C^t AB.
 *
 * @param[out] CtAB The non-distributed Vector C^t AB.
 *
 */
void ComputeCtAB_vec(const HypreParMatrix& A,
                     const HypreParVector& B,
                     const CAROM::Matrix& C,
                     CAROM::Vector& CtAB_vec);

/**
 * @brief Helper function to ensure that @p t is within a given percentage of
 * the domain relative to the center of the mesh. Performs the check for each
 * dimension of the mesh (works if mesh is 2D or 3D).
 *
 * @param bb_min Minimum corner of mesh bounding box.
 * See mfem::Mesh::GetBoundingBox().
 *
 * @param bb_max Maximum corner of mesh bounding box.
 * See mfem::Mesh::GetBoundingBox().
 *
 * @param t Point to check if within given @p limit percentage of mesh domain
 * relative to mesh center.
 *
 * @param limit Fractional percentage (from [0, 1]) of mesh domain to check
 * bounds of @p t.
 *
 * @note This will throw an error and exit if the check fails.
 */
void verify_within_portion(const mfem::Vector &bb_min,
                           const mfem::Vector &bb_max,
                           const mfem::Vector &t, const double limit);

/**
 * @brief Maps a value from [-1, 1] to the corresponding mesh domain [bb_min, bb_max]
 *
 * @param bb_min Minimum value of domain
 *
 * @param bb_max Maximum value of domain
 *
 * @param fraction Value between [-1, 1] to map
 *
 * @returns @p fraction mapped to [bb_min, bb_max]
 * @see map_from_ref_mesh
 */
double map_to_ref_mesh(const double &bb_min, const double &bb_max,
                       const double &fraction);

/**
 * @brief Maps a value within mesh domain [bb_min, bb_max] to the corresponding
 * value between [-1, 1]
 *
 * @param bb_min Minimum value of domain
 *
 * @param bb_max Maximum value of domain
 *
 * @param value Value between [bb_min, bb_max] to map
 *
 * @returns @p value mapped to [-1, 1]
 * @see map_to_ref_mesh
 */
double map_from_ref_mesh(const double &bb_min, const double &bb_max,
                         const double &value);

}  // namespace CAROM

#endif // MFEMUTILITIES_H
