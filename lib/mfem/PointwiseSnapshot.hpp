/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

#ifndef POINTWISE_SNAPSHOT
#define POINTWISE_SNAPSHOT

#include "mfem.hpp"

using namespace mfem;
#ifdef MFEM_USE_GSLIB

namespace CAROM {

class PointwiseSnapshot
{
public:
    PointwiseSnapshot(const int sdim, const int *dims_);

    void SetMesh(ParMesh *pmesh);
    void GetSnapshot(ParGridFunction const& f, mfem::Vector & s);

    ~PointwiseSnapshot();

private:
    FindPointsGSLIB *finder;

    int npoints;
    int dims[3];
    const int spaceDim;

    mfem::Vector domainMin, domainMax;
    mfem::Vector xyz;
};

}
#endif // MFEM_USE_GSLIB

#endif // POINTWISE_SNAPSHOT
