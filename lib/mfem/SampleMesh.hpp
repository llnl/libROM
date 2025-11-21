/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

#ifndef SAMPLEMESH_H
#define SAMPLEMESH_H

#include <set>
#include <string>

#include "mfem.hpp"
#include "mpi.h"

#include "linalg/Matrix.h"

using namespace mfem;
using std::vector;
using std::set;
using std::map;
using std::string;

namespace CAROM {

void SampleVisualization(ParMesh *pmesh, set<int> const& elems,
                         set<int> const& intElems, set<int> const& faces,
                         set<int> const& edges, set<int> const& vertices,
                         string const& filename,
                         mfem::Vector *elemCount=nullptr,
                         double elementScaling=1.0);

/**
 * Class SampleMeshManager constructs and manages a sample mesh and finite element spaces on the sample mesh.
 * The spaces defined on the sample mesh are referred to as sample mesh spaces, and they generally contain
 * the sampled DOFs along with other unsampled DOFs. This class can be used to work with vectors and basis
 * matrices in the sample mesh spaces.
 */

class SampleMeshManager
{
public:
    /**
     * @brief Constructor creating a SampleMeshManager with arbitrarily many ParFiniteElementSpaces on a full-order mesh.
     *
     * @pre fespace_.size() > 0
     *
     * @param[in] fespace_ Arbitrarily many ParFiniteElementSpaces defined on the same full-order mesh. Each sampled variable
     *                     must be defined on one of these spaces. All spaces must be specified in the constructor.
     *
     * @param[in] visFileName If non-empty, this filename is used for VisIt and ParaView output of the sample mesh
     *                        and the sampled DOFs on the full-order mesh.
     *
     * @param[in] visScale    Constant value for indicating the sample mesh elements in visualization.
     */
    SampleMeshManager(vector<ParFiniteElementSpace*> & fespace_,
                      string visFileName="", double visScale=1.0);

    /**
     * @brief Register a variable and set its sampled DOFs.
     *
     * @param[in] variable String containing the name of the variable, used for look-up.
     *
     * @param[in] space Index of the space in fespace (see constructor) where the variable is defined.
     *
     * @param[in] sample_dofs_v Local sampled DOF indices on all MPI processes (cf. DEIM function).
     *
     * @param[in] num_sample_dofs_per_proc Number of sampled DOFs on each MPI process (cf. DEIM function).
     */
    void RegisterSampledVariable(const string variable, const int space,
                                 vector<int> const& sample_dofs_v, vector<int> const& num_sample_dofs_per_proc);

    /**
     * @brief Construct the sample mesh, after registering all sampled variables.
     */
    void ConstructSampleMesh();

    /**
     * @brief Gather the rows of a distributed basis CAROM::Matrix corresponding to a sample mesh space.
     *
     * @param[in] variable String containing the name of the variable, used for look-up.
     *
     * @param[in] B Distributed matrix with columns containing full-order vectors.
     *
     * @param[in] rdim Number of columns of B to extract (starting with the first).
     *
     * @param[out] Bsp Matrix with rows corresponding to sample mesh space DOFs, gathered only to the root process (MPI rank 0).
     */
    void GatherDistributedMatrixRows(const string variable, CAROM::Matrix const& B,
                                     const int rdim, CAROM::Matrix& Bsp) const;

    /**
     * @brief Returns a sample mesh space.
     *
     * @param[in] space Index of the finite element space (corresponds to fespace_ in constructor).
     *
     * @return The sample mesh space with the given index.
     */
    ParFiniteElementSpace* GetSampleFESpace(const int space) const {
        return spfespace[space];
    }

    /**
     * @brief Returns the sample mesh.
     *
     * @return The sample mesh, defined only on the root process (MPI rank 0).
     */
    ParMesh* GetSampleMesh() const {
        MFEM_VERIFY(finalized, "Sample mesh is not constructed");
        return sample_pmesh;
    }

    /**
     * @brief Returns the number of sampled DOFs on all processes for a variable.
     *
     * @param[in] variable String containing the name of the variable, used for look-up.
     *
     * @return Number of sampled DOFs on all processes for a variable.
     */
    int GetNumVarSamples(const string variable) const;

    /**
     * @brief Sets a vector of sampled DOFs from a vector on the sample mesh space.
     *
     * @param[in] variable String containing the name of the variable, used for look-up.
     *
     * @param[in] v Vector on the sample mesh space.
     *
     * @param[out] s Vector of sampled DOFs on all processes.
     */
    void GetSampledValues(const string variable, mfem::Vector const& v,
                          CAROM::Vector & s) const;


    /**
     * @brief Returns a set of indices of local FOM mesh elements corresponding
     *        to sample elements.
     *
     * @return Pointer to a set of local FOM mesh element indices.
     */
    set<int>* GetSampleElements() {
        return &elems;
    }

    /**
     * @brief Writes a variable sample DOF map to file, which can be read by SampleDOFSelector::ReadMapFromFile
     *        in order to use SampleDOFSelector::GetSampledValues when this SampleMeshManager object is not available.
     *
     * @param[in] variable String containing the name of the variable, used for look-up.
     *
     * @param[in] file_name Name of the output file.
     */
    void WriteVariableSampleMap(const string variable, string file_name) const;

    /**
     * @brief Destructor.
    */
    ~SampleMeshManager()
    {
        for (int i=0; i<nspaces; ++i)
            delete spfespace[i];

        delete sample_pmesh;
    }

private:
    void CreateSampleMesh();

    int GetVariableIndex(const string variable) const;

    void SetSampleMaps();
    void FinishSampleMaps();

    bool finalized;

    const int nspaces;
    int nvar = 0;

    map<string, int> vmap;  // Variable name to index map

    int myid, nprocs;
    MPI_Comm root_comm;

    vector<ParFiniteElementSpace*> fespace;
    vector<ParFiniteElementSpace*> spfespace;

    ParMesh *pmesh = nullptr;
    ParMesh *sample_pmesh = nullptr;

    vector<vector<set<int>>> sample_dofs_proc;

    vector<int> sample_dofs;  // block true DOFs for all spaces, for all processes.

    vector<int> num_sample_dofs_per_proc_merged;

    vector<int> varSpace;
    vector<vector<int>> sample_dofs_var;
    vector<vector<int>> num_sample_dofs_per_proc_var;
    vector<vector<int>> s2sp_var;
    vector<int> s2sp, st2sp;
    vector<int>
    sprows;  // Local true DOFs on the original full mesh, restricted to the sample mesh stencil.
    vector<int> all_sprows;  // sprows gathered over all processes

    vector<int> spaceTOS, spaceOS, spaceOSSP;
    vector<vector<int>> spaceOSall;

    set<int> elems;

    string filename;  // For visualization output

    double elemVisScale;  // Scaling for sample element visualization
};

/**
 * Class SampleDOFSelector performs the same function GetSampledValues as in SampleMeshManager, when a
 * SampleMeshManager object is not available. The intended usage is that SampleMeshManager is constructed
 * in a preprocessing phase when the full-order mesh and basis matrices are loaded in parallel, and then
 * in the serial online phase SampleDOFSelector reads sample DOF maps for each sampled variable in order
 * to select sampled DOF values from the sample mesh spaces.
 */
class SampleDOFSelector {
public:
    SampleDOFSelector() { }

    /**
     * @brief Reads a variable sample DOF map from file, written by SampleMeshManager::WriteVariableSampleMap,
     *        in order to use GetSampledValues when a SampleMeshManager object is not available.
     *
     * @param[in] variable String containing the name of the variable, used for look-up.
     *
     * @param[in] file_name Name of the output file.
     */
    void ReadMapFromFile(const string variable, string file_name);

    /**
     * @brief Sets a vector of sampled DOFs from a vector on the sample mesh space. Note that this function
     *        is the same as SampleMeshManager::GetSampledValues and is used when SampleMeshManager is not available.
     *
     * @param[in] variable String containing the name of the variable, used for look-up.
     *
     * @param[in] v Vector on the sample mesh space.
     *
     * @param[out] s Vector of sampled DOFs on all processes.
     */
    void GetSampledValues(const string variable, mfem::Vector const& v,
                          CAROM::Vector & s) const;

    /**
       * @brief Destructor.
      */
    ~SampleDOFSelector()
    { }

private:
    int nvar = 0;
    map<string, int> vmap;  // Variable name to index map
    vector<vector<int>> s2sp_var;

    int GetVariableIndex(const string variable) const;
};

}  // namespace CAROM

#endif // SAMPLEMESH_H
