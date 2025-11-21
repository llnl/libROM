#include "PointwiseSnapshot.hpp"

#ifdef MFEM_USE_GSLIB
using std::cout;
using std::endl;
namespace CAROM {

PointwiseSnapshot::PointwiseSnapshot(const int sdim, const int *dims_)
    : finder(nullptr), spaceDim(sdim)
{
    MFEM_VERIFY(1 < sdim && sdim < 4, "");

    npoints = 1;
    for (int i=0; i<spaceDim; ++i)
    {
        dims[i] = dims_[i];
        npoints *= dims[i];
    }

    for (int i=spaceDim; i<3; ++i)
        dims[i] = 1;

    xyz.SetSize(npoints * spaceDim);
    xyz = 0.0;
}

void PointwiseSnapshot::SetMesh(ParMesh *pmesh)
{
    if (finder) finder->FreeData();  // Free the internal gslib data.
    delete finder;

    MFEM_VERIFY(pmesh->Dimension() == spaceDim, "");
    MFEM_VERIFY(pmesh->SpaceDimension() == spaceDim, "");

    pmesh->GetBoundingBox(domainMin, domainMax, 0);

    double h[3] = {0.0, 0.0, 0.0};
    for (int i=0; i<spaceDim; ++i)
        h[i] = (domainMax[i] - domainMin[i]) / ((double) (dims[i] - 1));

    const int rank = pmesh->GetMyRank();
    if (rank == 0)
    {
        cout << "PointwiseSnapshot on bounding box from (";
        for (int i=0; i<spaceDim; ++i)
        {
            cout << domainMin[i];
            if (i < spaceDim-1) cout << ", ";
        }
        cout << ") to (";
        for (int i=0; i<spaceDim; ++i)
        {
            cout << domainMax[i];
            if (i < spaceDim-1) cout << ", ";
        }
        cout << ")" << endl;
    }

    for (int k=0; k<dims[2]; ++k)
    {
        const double pz = spaceDim > 2 ? domainMin[2] + k*h[2] : 0.0;
        const int osk = k*dims[0]*dims[1];

        for (int j=0; j<dims[1]; ++j)
        {
            const double py = domainMin[1] + j*h[1];
            const int osj = (j*dims[0]) + osk;

            for (int i=0; i<dims[0]; ++i)
            {
                const double px = domainMin[0] + i*h[0];
                xyz[i + osj] = px;
                if (spaceDim > 1) xyz[npoints + i + osj] = py;
                if (spaceDim > 2) xyz[2*npoints + i + osj] = pz;
            }
        }
    }

    finder = new FindPointsGSLIB(MPI_COMM_WORLD);
    finder->Setup(*pmesh);
    finder->SetL2AvgType(FindPointsGSLIB::AvgType::NONE);
}

void PointwiseSnapshot::GetSnapshot(ParGridFunction const& f, mfem::Vector & s)
{
    const int vdim = f.FESpace()->GetVDim();
    s.SetSize(npoints * vdim);

    finder->Interpolate(xyz, f, s);

    Array<unsigned int> code_out = finder->GetCode();

    MFEM_VERIFY(code_out.Size() == npoints, "");

    // Note that Min() and Max() are not defined for Array<unsigned int>
    //MFEM_VERIFY(code_out.Min() >= 0 && code_out.Max() < 2, "");
    int cmin = code_out[0];
    int cmax = code_out[0];
    for (auto c : code_out)
    {
        if (c < cmin)
            cmin = c;

        if (c > cmax)
            cmax = c;
    }

    MFEM_VERIFY(cmin >= 0 && cmax < 2, "");
}

PointwiseSnapshot::~PointwiseSnapshot()
{
    finder->FreeData();  // Free the internal gslib data.
    delete finder;
}

} // namespace CAROM
#endif
