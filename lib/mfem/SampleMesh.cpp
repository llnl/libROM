/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

#include "SampleMesh.hpp"

using std::endl;
using std::make_pair;
using std::ofstream;
using std::ifstream;

namespace CAROM {

#define FULL_DOF_STENCIL

void FindStencilElements(const vector<int>& sample_dofs_gid,
                         set<int>& elements,
                         ParFiniteElementSpace& fespace)
{
    for (int k = 0; k < fespace.GetParMesh()->GetNE(); k++)
    {
        Array<int> dofs;
        fespace.GetElementVDofs(k, dofs);

        for (vector<int>::const_iterator it = sample_dofs_gid.begin();
                it != sample_dofs_gid.end(); ++it)
        {
            for (int i = 0; i < dofs.Size(); i++)
            {
                const int dof_i = dofs[i] >= 0 ? dofs[i] : -1 - dofs[i];
                int global_dof = fespace.GetGlobalTDofNumber(dof_i);
                if (global_dof == *it)
                {
                    elements.insert(k);
                    goto next;
                }
            }
        }
next:
        continue;
    }
}

void FindSampledMeshEntities(const int type, const vector<int>& sample_dofs_gid,
                             set<int>& entities, ParFiniteElementSpace& fespace)
{
    int N = 0;
    switch (type)
    {
    case 0:
        N = fespace.GetParMesh()->GetNE();
        break;
    case 1:
        N = fespace.GetParMesh()->GetNFaces();
        break;
    case 2:
        N = fespace.GetParMesh()->GetNEdges();
        break;
    case 3:
        N = fespace.GetParMesh()->GetNV();
        break;
    }

    for (int k = 0; k < N; k++)
    {
        Array<int> dofs;

        switch (type)
        {
        case 0:
            fespace.GetElementInteriorVDofs(k, dofs);
            break;
        case 1:
            fespace.GetFaceVDofs(k, dofs);
            break;
        case 2:
            fespace.GetEdgeVDofs(k, dofs);
            break;
        case 3:
            fespace.GetVertexVDofs(k, dofs);
            break;
        }

        for (vector<int>::const_iterator it = sample_dofs_gid.begin();
                it != sample_dofs_gid.end(); ++it)
        {
            for (int i = 0; i < dofs.Size(); i++)
            {
                const int dof_i = dofs[i] >= 0 ? dofs[i] : -1 - dofs[i];
                int global_dof = fespace.GetGlobalTDofNumber(dof_i);
                if (global_dof == *it)
                {
                    entities.insert(k);
                    goto next;
                }
            }
        }
next:
        continue;
    }

}

void GetLocalSampleMeshElements(ParMesh& pmesh, ParFiniteElementSpace& fespace,
                                const vector<int>& sample_dofs,
                                const vector<int>& local_num_sample_dofs, set<int>& elems,
                                bool getSampledEntities, set<int>& intElems, set<int>& faces, set<int>& edges,
                                set<int>& vertices)
{
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    const int num_procs = local_num_sample_dofs.size();

    // Construct the map of local true dofs to global true dofs.
    map<int, int> ltdof2gtdof;
    const int ndofs = fespace.GetVSize();
    for (int i = 0; i < fespace.GetVSize(); ++i) {
        int ltdof = fespace.GetLocalTDofNumber(i);

        if (ltdof != -1) {
            int gtdof = fespace.GetGlobalTDofNumber(i);
            ltdof2gtdof[ltdof] = gtdof;
        }
    }

    // Convert local sample dof indices into global sample dof indices.
    int* offsets = new int [num_procs];
    offsets[0] = 0;
    for (int i = 1; i < num_procs; ++i) {
        offsets[i] = offsets[i-1] + local_num_sample_dofs[i-1];
    }

    vector<int> local_sample_dofs_gid;
    int my_offset = offsets[myid];
    for (int i = 0; i < local_num_sample_dofs[myid]; ++i) {
        map<int, int>::const_iterator it = ltdof2gtdof.find(sample_dofs[my_offset+i]);
        MFEM_VERIFY(it != ltdof2gtdof.end(), "");
        local_sample_dofs_gid.push_back(ltdof2gtdof[sample_dofs[my_offset+i]]);
    }

    int total_num_sample_dofs = static_cast<int>(sample_dofs.size());
    vector<int> sample_dofs_gid(total_num_sample_dofs);
    MPI_Allgatherv(&local_sample_dofs_gid[0], local_num_sample_dofs[myid],
                   MPI_INT, &sample_dofs_gid[0], &local_num_sample_dofs[0],
                   offsets, MPI_INT, MPI_COMM_WORLD);

    // Get the processor local element ids of all elements connected to any
    // sample_dofs_gid and the total number of elements in the stencil mesh.
    elems.clear();
    FindStencilElements(sample_dofs_gid, elems, fespace);

    if (getSampledEntities)
    {
        FindSampledMeshEntities(0, sample_dofs_gid, intElems, fespace);
        if (fespace.GetParMesh()->Dimension() == 3) FindSampledMeshEntities(1,
                    sample_dofs_gid, faces, fespace);
        FindSampledMeshEntities(2, sample_dofs_gid, edges, fespace);
        FindSampledMeshEntities(3, sample_dofs_gid, vertices, fespace);
    }

    delete [] offsets;
}

void SplitDofsIntoBlocks(const vector<int>& Ntrue, const vector<int>& dofs,
                         const vector<int>& local_num_dofs,
                         vector<vector<int>>& dofs_block, vector<vector<int>>& dofs_block_todofs,
                         vector<vector<int>>& local_num_dofs_block)
{
    const int num_procs = local_num_dofs.size();
    const int nspaces = Ntrue.size();

    MFEM_VERIFY(nspaces > 0 && nspaces == dofs_block.size()
                && nspaces == dofs_block_todofs.size()
                && nspaces == local_num_dofs_block.size(), "");

    vector<vector<vector<int>>> procDofs_block(nspaces);
    vector<vector<int>> allNtrue(nspaces);

    for (int i=0; i<nspaces; ++i)
    {
        local_num_dofs_block[i].resize(num_procs);
        local_num_dofs_block[i].assign(local_num_dofs_block[i].size(), 0);
        procDofs_block[i].resize(num_procs);

        allNtrue[i].resize(num_procs);
        MPI_Allgather(&Ntrue[i], 1, MPI_INT, allNtrue[i].data(), 1, MPI_INT,
                      MPI_COMM_WORLD);
    }

    vector<int> spaceOS(nspaces);
    spaceOS[0] = 0;

    int os = 0;
    vector<int>::const_iterator it = dofs.begin();
    for (int p=0; p<num_procs; ++p)
    {
        for (int i=1; i<nspaces; ++i)
        {
            spaceOS[i] = spaceOS[i-1] + allNtrue[i-1][p];
        }

        for (int i=0; i<local_num_dofs[p]; ++i, ++it)
        {
            bool found = false;
            for (int s=nspaces-1; s>=0; --s)
            {
                if (*it >= spaceOS[s])
                {
                    dofs_block[s].push_back(*it - spaceOS[s]);
                    dofs_block_todofs[s].push_back(os + i);
                    local_num_dofs_block[s][p]++;
                    found = true;
                    break;
                }
            }
            MFEM_VERIFY(found, "Space not found");
        }

        os += local_num_dofs[p];
    }

    MFEM_VERIFY(it == dofs.end(), "");
}

void InsertElementDofs(ParFiniteElementSpace& fespace, const int elId,
                       const int offset, set<int>& element_dofs)
{
    Array<int> dofs;
    fespace.GetElementVDofs(elId, dofs);
    for (int i = 0; i < dofs.Size(); ++i) {
        const int dof_i = dofs[i] >= 0 ? dofs[i] : -1 - dofs[i];
#ifdef FULL_DOF_STENCIL
        element_dofs.insert(offset + dof_i);
#else
        int ltdof = fespace.GetLocalTDofNumber(dof_i);
        if (ltdof != -1) {
            element_dofs.insert(offset + ltdof);
        }
#endif
    }
}

// dofs are full dofs
void AugmentDofListWithOwnedDofs(vector<int>& mixedDofs,
                                 vector<ParFiniteElementSpace*> & fespace)
{
    const int nspaces = fespace.size();
    vector<int> spaceOS(nspaces);

    spaceOS[0] = 0;
    for (int i=1; i<nspaces; ++i)
        spaceOS[i] = spaceOS[i-1] + fespace[i-1]->GetVSize();

    vector<vector<int> > dofs(nspaces);
    set<int> mixedDofSet;
    for (auto i : mixedDofs)
    {
        mixedDofSet.insert(i);

        int space = 0;
        for (int j=nspaces-1; j > 0; --j)
        {
            if (i >= spaceOS[j])
            {
                space = j;
                break;
            }
        }

        dofs[space].push_back(i - spaceOS[space]);
    }

    int nprocs = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    vector<int> cts(nprocs);
    vector<int> offsets(nprocs);

    for (int s=0; s<nspaces; ++s)  // loop over fespaces
    {
        const int ndofs = dofs[s].size();
        MPI_Allgather(&ndofs, 1, MPI_INT, cts.data(), 1, MPI_INT, MPI_COMM_WORLD);

        offsets[0] = 0;
        for (int i = 1; i < nprocs; ++i)
            offsets[i] = offsets[i-1] + cts[i-1];

        vector<int> gdofs(ndofs);
        for (int i = 0; i<ndofs; ++i)
        {
            gdofs[i] = fespace[s]->GetGlobalTDofNumber(dofs[s][i]);
        }

        vector<int> gdofsGathered(offsets[nprocs-1] + cts[nprocs-1]);

        MPI_Allgatherv(gdofs.data(), ndofs, MPI_INT,
                       gdofsGathered.data(), cts.data(), offsets.data(), MPI_INT, MPI_COMM_WORLD);

        set<int> allgdofs;

        for (auto i : gdofsGathered)
            allgdofs.insert(i);

        for (int i=0; i<fespace[s]->GetVSize(); ++i)
        {
            int ltdof = fespace[s]->GetLocalTDofNumber(i);
            if (ltdof != -1)
            {
                const int g = fespace[s]->GetGlobalTDofNumber(i);
                set<int>::iterator it = allgdofs.find(g);
                if (it != allgdofs.end())
                {
                    // Dof i should be included in mixedDofs. First check whether it is already included.
                    const int j = spaceOS[s] + i;
                    set<int>::iterator it = mixedDofSet.find(j);
                    if (it == mixedDofSet.end())
                    {
                        mixedDofs.push_back(j);
                        mixedDofSet.insert(j);
                    }
                }
            }
        }
    }
}

void BuildSampleMesh(ParMesh& pmesh, vector<ParFiniteElementSpace*> & fespace,
                     const set<int>& elems, Mesh*& sample_mesh, vector<int>& stencil_dofs,
                     vector<int>& elemLocalIndices, vector<map<int, int> >& elemLocalIndicesInverse)
{
    // Get the number of stencil elements coming from each processor and the
    // total number of stencil elements.
    int num_procs = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const int d = pmesh.Dimension();

    // Must be first order, to get a bijection between vertices and DOF's.
    H1_FECollection h1_coll(1, d);

    // This constructor effectively sets vertex (DOF) global indices.
    ParFiniteElementSpace H1DummySpace(&pmesh, &h1_coll);

    vector<int> procNumElems(num_procs);

    const int local_num_elems = elems.size();
    int* cts = new int [num_procs];
    int* offsets = new int [num_procs];
    MPI_Allgather(&local_num_elems, 1, MPI_INT, cts, 1, MPI_INT, MPI_COMM_WORLD);
    int sample_mesh_num_elems = 0;
    offsets[0] = 0;
    for (int i = 0; i < num_procs; ++i) {
        sample_mesh_num_elems += cts[i];
        procNumElems[i] = cts[i];
        cts[i] *= 2;
        if (i > 0)
            offsets[i] = offsets[i-1] + cts[i-1];
    }

    const int nspaces = fespace.size();

    vector<int> spaceN(nspaces);
    for (int i=0; i<nspaces; ++i)
    {
#ifdef FULL_DOF_STENCIL
        spaceN[i] = fespace[i]->GetVSize();
#else
        spaceN[i] = fespace[i]->TrueVSize();
#endif
    }

    // Each process generates for each of its stencil elements a list of global
    // TDofs (which are equivalent to vertices), a list of element attributes,
    // and a list of vertices (coords). These get communicated to P0. For each
    // stencil mesh element, P0 goes through the list of vertices and constructs
    // a map<int, int> mapping these global TDofs to local vertex ids. For each
    // unique TDof it finds, it adds a vertex to the sample mesh. Then it adds
    // each element, attribute, and connectivity.
    vector<int> my_element_attr(
        2*local_num_elems);  // Stores element attribute and local index
    set<int> element_dofs;
    int attr_idx = 0;
    int conn_idx = 0;
    int coords_idx = 0;
    Array<int> elVert;
    pmesh.GetElementVertices(*elems.begin(), elVert);
    int numElVert = elVert.Size();  // number of vertices per element
    MFEM_VERIFY(numElVert > 0, "");

    // vertex global indices, for each element
    vector<int> my_element_vgid(numElVert*local_num_elems);
    vector<double> my_element_coords(d*numElVert*local_num_elems);

    for (set<int>::iterator it = elems.begin(); it != elems.end(); ++it) {
        const int elId = *it;
        int os = 0;
        for (int i=0; i<nspaces; ++i)
        {
            InsertElementDofs(*fespace[i], elId, os, element_dofs);
            os += spaceN[i];
        }

        pmesh.GetElementVertices(elId, elVert);
        MFEM_VERIFY(numElVert == elVert.Size(),
                    "Assuming a uniform element type in the mesh.");
        // NOTE: to be very careful, it should be verified that this is the same across all processes.

        Array<int> dofs;
        H1DummySpace.GetElementDofs(elId, dofs);
        MFEM_VERIFY(numElVert == dofs.Size(),
                    "Assuming a bijection between vertices and H1 dummy space DOF's");

        for (int i = 0; i < numElVert; ++i) {
            my_element_vgid[conn_idx++] = H1DummySpace.GetGlobalTDofNumber(dofs[i]);
            double* coords = pmesh.GetVertex(elVert[i]);
            for (int j=0; j<d; ++j)
                my_element_coords[coords_idx++] = coords[j];
        }

        my_element_attr[attr_idx++] = pmesh.GetAttribute(elId);
        my_element_attr[attr_idx++] = elId;
    }

    stencil_dofs.assign(element_dofs.begin(), element_dofs.end());

#ifdef FULL_DOF_STENCIL
    AugmentDofListWithOwnedDofs(stencil_dofs, fespace);
#endif

    MFEM_VERIFY(coords_idx == d*numElVert*local_num_elems, "");
    MFEM_VERIFY(conn_idx == numElVert*local_num_elems, "");
    MFEM_VERIFY(attr_idx == 2*local_num_elems, "");

    // Gather all the element attributes from all processors.

    vector<int> element_attr(2*sample_mesh_num_elems);
    MPI_Allgatherv(&my_element_attr[0], 2*local_num_elems, MPI_INT,
                   &element_attr[0], cts, offsets, MPI_INT, MPI_COMM_WORLD);

    // Gather all the element connectivities from all processors.
    offsets[0] = 0;
    cts[0] = numElVert*cts[0]/2;
    for (int i = 1; i < num_procs; ++i) {
        cts[i] = numElVert*cts[i]/2;
        offsets[i] = offsets[i-1] + cts[i-1];
    }
    vector<int> element_vgid(numElVert*sample_mesh_num_elems);
    MPI_Allgatherv(&my_element_vgid[0], numElVert*local_num_elems, MPI_INT,
                   &element_vgid[0], cts, offsets, MPI_INT, MPI_COMM_WORLD);

    // Gather all the element coordinates from all processors.
    offsets[0] = 0;
    cts[0] = d*cts[0];
    for (int i = 1; i < num_procs; ++i) {
        cts[i] = d*cts[i];
        offsets[i] = offsets[i-1] + cts[i-1];
    }
    vector<double> element_coords(d*numElVert*sample_mesh_num_elems);
    MPI_Allgatherv(&my_element_coords[0], d*numElVert*local_num_elems, MPI_DOUBLE,
                   &element_coords[0], cts, offsets, MPI_DOUBLE, MPI_COMM_WORLD);
    delete [] cts;
    delete [] offsets;

    // element_vgid holds vertices as global ids. Vertices may be shared
    // between elements so we don't know the number of unique vertices in the
    // sample mesh. Find all the unique vertices and construct the map of
    // global dof ids to local dof ids (vertices). Keep track of the number of
    // unique vertices.
    set<int> unique_gdofs;
    map<int, int> unique_gdofs_first_appearance;
    for (int i = 0; i < numElVert*sample_mesh_num_elems; ++i) {
        int gdof = element_vgid[i];
        if (unique_gdofs.insert(gdof).second) {
            unique_gdofs_first_appearance.insert(make_pair(gdof, i));
        }
    }
    int sample_mesh_num_verts = unique_gdofs.size();
    map<int, int> unique_gdofs_2_vertex;
    map<int, int> vertex_2_unique_gdofs;
    int idx = 0;
    for (set<int>::iterator it = unique_gdofs.begin();
            it != unique_gdofs.end(); ++it) {
        unique_gdofs_2_vertex.insert(make_pair(*it, idx));
        vertex_2_unique_gdofs.insert(make_pair(idx, *it));
        ++idx;
    }

    // Now we have enough info to build the sample mesh.
    printf("sample mesh has %d elements, %d vertices\n",
           sample_mesh_num_elems, sample_mesh_num_verts);

    sample_mesh = new Mesh(d, sample_mesh_num_verts, sample_mesh_num_elems);

    // For each vertex that we found, add its coordinates to the stencil mesh.
    for (int vert = 0; vert < sample_mesh_num_verts; ++vert) {
        int unique_gdof = vertex_2_unique_gdofs[vert];
        int first_conn_ref = unique_gdofs_first_appearance[unique_gdof];
        int coord_loc = d*first_conn_ref;
        sample_mesh->AddVertex(&element_coords[coord_loc]);
    }

    // Now add each element and give it its attributes and connectivity.
    const int elGeom = pmesh.GetElementBaseGeometry(0);
    idx = 0;
    elemLocalIndices.resize(sample_mesh_num_elems);
    elemLocalIndicesInverse.resize(num_procs);
    int ielem = 0;
    for (int p=0; p<num_procs; ++p)
    {
        for (int i=0; i<procNumElems[p]; ++i, ++ielem)
        {
            Element* sel = sample_mesh->NewElement(elGeom);
            sel->SetAttribute(element_attr[2*ielem]);

            elemLocalIndices[ielem] = element_attr[(2*ielem)+1];
            elemLocalIndicesInverse[p][elemLocalIndices[ielem]] = ielem;

            Array<int> sv(numElVert);
            for (int vert = 0; vert < numElVert; ++vert) {
                sv[vert] = unique_gdofs_2_vertex[element_vgid[idx++]];
            }
            sel->SetVertices(sv);

            sample_mesh->AddElement(sel);
        }
    }

    MFEM_VERIFY(ielem == sample_mesh_num_elems, "");

    sample_mesh->FinalizeTopology();
}

void GetLocalDofsToLocalElementMap(ParFiniteElementSpace& fespace,
                                   const vector<int>& dofs,
                                   const vector<int>& localNumDofs,
                                   const set<int>& elems,
                                   vector<int>& dofToElem,
                                   vector<int>& dofToElemDof,
                                   const bool useTDof)
{
    int myid;
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    int myoffset = 0;
    for (int p=0; p<myid; ++p)
        myoffset += localNumDofs[p];

    dofToElem.resize(localNumDofs[myid]);
    dofToElemDof.resize(localNumDofs[myid]);

    Array<int> eldofs;

    for (int i=0; i<localNumDofs[myid]; ++i)
    {
        dofToElem[i] = -1;
        for (set<int>::const_iterator it = elems.begin(); it != elems.end(); ++it)
        {
            const int elId = *it;
            fespace.GetElementVDofs(elId, eldofs);

            for (int j=0; j<eldofs.Size(); ++j)
            {
                const int eldof_j = eldofs[j] >= 0 ? eldofs[j] : -1 - eldofs[j];
                int ltdof = fespace.GetLocalTDofNumber(eldof_j);
#ifdef FULL_DOF_STENCIL
                const int dof_j = useTDof ? ltdof : eldof_j;
                if (dof_j == dofs[myoffset + i])  // dofs contains true DOF's.
#else
                if (ltdof == dofs[myoffset + i])  // dofs contains true DOF's.
#endif
                {
                    // Possibly overwrite another element index, which is fine
                    // since we just want any element index.
                    dofToElem[i] = elId;
                    dofToElemDof[i] = j;
                }
            }
        }

#ifndef FULL_DOF_STENCIL
        const int dte = dofToElem[i];
        MFEM_VERIFY(dofToElem[i] >= 0, "");
#endif
    }
}

void Set_s2sp(const int myid, const int num_procs, vector<int> const& spNtrue,
              const int global_num_sample_dofs, const vector<int>& local_num_sample_dofs,
              const vector<vector<int> >& local_num_sample_dofs_sub,
              const vector<vector<int> >& localSampleDofsToElem_sub,
              const vector<vector<int> >& localSampleDofsToElemDof_sub,
              const vector<vector<int> >& sample_dofs_sub_to_sample_dofs,
              const vector<map<int, int> >& elemLocalIndicesInverse,
              vector<ParFiniteElementSpace*> & spfespace,
              vector<int>& s2sp)
{
    // Set s2sp, which is a map from the selected indices in sample_dofs for all processes to DOF's in sample_mesh.
    // This is done by the following steps:
    //   1. For each local sample DOF, determine its fespace and find an element to which it belongs and its local index with respect to the element DOF ordering.
    //   2. Communicate this data (one process-local element index and an element DOF index) from each process to P0.
    //   3. On P0, loop over each process's sample DOF's and get their process-local element index, determine the sample mesh element,
    //      and then get the sample mesh DOF from the sample mesh element DOF's. This goes into s2sp.
    // Note that two sample DOF's could be mapped to the same DOF in sample_mesh, which is fine. This may happen if a DOF is shared by two processes.
    // The DEIM implementation only sees vectors on each process, without knowledge that the application may identify two DOF's.
    // Fortunately, DEIM will not select two or more indices on different processes that are equivalent as DOF's, because the corresponding rows
    // of the basis matrix would then be identical, and then the corresponding entries of the residual used in DEIM would be identical.
    // Since DEIM does not select the same index twice, it also will not select two indices that have the same row data.
    // Thus s2sp will not map two indices selected by DEIM to the same DOF in sample_mesh, but it may map two original mesh DOF's that are in
    // the sample_mesh stencil to the same sample_mesh DOF, which should not cause any problems.

    const int nspaces = spfespace.size();
    MFEM_VERIFY(nspaces == spNtrue.size(), "");

    int mySampleDofOffset = 0;
    vector<int> os;
    os.assign(nspaces, 0);
    for (int p=0; p<myid; ++p)
    {
        mySampleDofOffset += local_num_sample_dofs[p];
        for (int i=0; i<nspaces; ++i)
            os[i] += local_num_sample_dofs_sub[i][p];
    }

    // Gather all the sample DOF to element and element DOF indices.
    vector<int> mySampleToElement(2*local_num_sample_dofs[myid]);

    // Initialize with invalid values, to verify later that everything was set.
    mySampleToElement.assign(mySampleToElement.size(), -1);

    for (int s=0; s<nspaces; ++s)  // Loop over subspaces
    {
        for (int i=0; i<local_num_sample_dofs_sub[s][myid]; ++i)
        {
            const int sdi = sample_dofs_sub_to_sample_dofs[s][os[s] + i] -
                            mySampleDofOffset;
            mySampleToElement[2*sdi] = localSampleDofsToElem_sub[s][i];
            mySampleToElement[(2*sdi)+1] = localSampleDofsToElemDof_sub[s][i];
        }
    }

#ifndef FULL_DOF_STENCIL
    for (int i=0; i<mySampleToElement.size(); ++i)
    {
        MFEM_VERIFY(mySampleToElement[i] >= 0, "");
    }
#endif

    int* cts = new int [num_procs];
    int* offsets = new int [num_procs];

    offsets[0] = 0;
    cts[0] = 2*local_num_sample_dofs[0];
    for (int i = 1; i < num_procs; ++i) {
        cts[i] = 2*local_num_sample_dofs[i];
        offsets[i] = offsets[i-1] + cts[i-1];
    }

    // TODO: replace Allgatherv with just a gather to root?

    vector<int> sampleToElement(2*global_num_sample_dofs);
    MPI_Allgatherv(&mySampleToElement[0], 2*local_num_sample_dofs[myid], MPI_INT,
                   &sampleToElement[0], cts, offsets, MPI_INT, MPI_COMM_WORLD);

    delete [] cts;
    delete [] offsets;

    // The remaining code in this function only sets s2sp, only on the root process.
    if (myid != 0)
        return;

    vector<int> spaceOS(nspaces);

    spaceOS[0] = 0;
    for (int i=1; i<nspaces; ++i)
        spaceOS[i] = spaceOS[i-1] + spfespace[i-1]->GetVSize();

    s2sp.resize(global_num_sample_dofs);
    // Initialize with invalid values, to verify later that everything was set.
    s2sp.assign(s2sp.size(), -1);

    for (int s=0; s<nspaces; ++s)
        os[s] = 0;

    int soffset = 0;

    for (int p=0; p<num_procs; ++p)
    {
        for (int s=0; s<nspaces; ++s)  // Loop over subspaces
        {
            for (int i=0; i<local_num_sample_dofs_sub[s][p]; ++i)
            {
                const int sdi = sample_dofs_sub_to_sample_dofs[s][os[s] + i];
                const int procElementIndex = sampleToElement[2*sdi];
#ifdef FULL_DOF_STENCIL
                if (procElementIndex == -1)
                    continue;
#endif
                const int procElementDofIndex = sampleToElement[(2*sdi)+1];
                map<int, int>::const_iterator it = elemLocalIndicesInverse[p].find(
                                                       procElementIndex);

                MFEM_VERIFY(it != elemLocalIndicesInverse[p].end(), "");
                MFEM_VERIFY(it->first == procElementIndex, "");

                const int sampleMeshElement = it->second;
                Array<int> eldofs;
                spfespace[s]->GetElementVDofs(sampleMeshElement, eldofs);
                const int eldof = eldofs[procElementDofIndex] >= 0 ?
                                  eldofs[procElementDofIndex] : -1 - eldofs[procElementDofIndex];
                s2sp[sdi] = eldof + spaceOS[s];
            }

            os[s] += local_num_sample_dofs_sub[s][p];
        }

        soffset += local_num_sample_dofs[p];
    }

#ifdef FULL_DOF_STENCIL

#else
    for (int i=0; i<s2sp.size(); ++i)
    {
        MFEM_VERIFY(s2sp[i] >= 0, "");
    }
#endif
}

#ifdef FULL_DOF_STENCIL
void Finish_s2sp_augmented(const int rank, const int nprocs,
                           vector<ParFiniteElementSpace*> & fespace,
                           vector<vector<int>>& dofs_block, vector<vector<int> >& dofs_sub_to_sdofs,
                           vector<vector<int> >& local_num_dofs_sub, const bool dofsTrue,
                           vector<int> & s2sp_)
{
    const int nspaces = fespace.size();

    vector<int> s2sp;
    {
        int n = rank == 0 ? s2sp_.size() : 0;
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
        s2sp.resize(n);
        if (rank == 0)
            s2sp = s2sp_;

        MPI_Bcast(s2sp.data(), n, MPI_INT, 0, MPI_COMM_WORLD);
    }

    for (int s=0; s<nspaces; ++s)  // loop over spaces
    {
        int os = 0;
        for (int p=0; p<rank; ++p)
            os += local_num_dofs_sub[s][p];

        {
            int sum = 0;
            for (int p=0; p<nprocs; ++p)
                sum += local_num_dofs_sub[s][p];
            MFEM_VERIFY(dofs_sub_to_sdofs[s].size() == sum
                        && dofs_block[s].size() == sum, "");
        }

        const int ndof = local_num_dofs_sub[s][rank];
        MFEM_VERIFY(os + ndof <= dofs_block[s].size(), "");

        vector<int> gid(2*ndof);

        if (dofsTrue)
        {
            vector<int> fdofs;

            map<int, int> tdofIndex;
            for (int i=0; i<ndof; ++i)
            {
                tdofIndex[dofs_block[s][os + i]] = i;
            }

            fdofs.assign(ndof, -1);
            for (int i=0; i<fespace[s]->GetVSize(); ++i)
            {
                const int ltdof = fespace[s]->GetLocalTDofNumber(i);
                map<int, int>::const_iterator it = tdofIndex.find(ltdof);
                if (it != tdofIndex.end())
                {
                    MFEM_VERIFY(it->first == ltdof && fdofs[it->second] == -1, "");
                    fdofs[it->second] = i;
                }
            }

            bool fdofsSet = true;
            for (int i=0; i<ndof; ++i)
            {
                if (fdofs[i] < 0)
                    fdofsSet = false;
            }

            MFEM_VERIFY(fdofsSet, "");
            for (int i=0; i<ndof; ++i)
            {
                gid[2*i] = fespace[s]->GetGlobalTDofNumber(fdofs[i]);
            }
        }
        else
        {
            for (int i=0; i<ndof; ++i)
            {
                gid[2*i] = fespace[s]->GetGlobalTDofNumber(dofs_block[s][os + i]);
            }
        }

        for (int i=0; i<ndof; ++i)
        {
            gid[(2*i) + 1] = s2sp[dofs_sub_to_sdofs[s][os + i]];
        }

        vector<int> counts(nprocs);
        vector<int> offsets(nprocs);

        offsets[0] = 0;
        for (int i=0; i<nprocs; ++i)
        {
            counts[i] = 2 * local_num_dofs_sub[s][i];
            if (i > 0)
                offsets[i] = offsets[i-1] + counts[i-1];
        }

        vector<int> allgid(offsets[nprocs-1] + counts[nprocs-1]);

        MPI_Gatherv(gid.data(), 2*ndof, MPI_INT, allgid.data(), counts.data(),
                    offsets.data(), MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            map<int, int> gi2sp;
            // Set gi2sp
            for (int p=0; p<nprocs; ++p)
            {
                for (int i=0; i<local_num_dofs_sub[s][p]; ++i)
                {
                    const int g = allgid[offsets[p] + (2*i)];
                    const int sp = allgid[offsets[p] + (2*i) + 1];

                    map<int, int>::const_iterator it = gi2sp.find(g);
                    if (it == gi2sp.end())
                        gi2sp[g] = sp;
                    else
                    {
                        MFEM_VERIFY(it->first == g, "");
                        if (it->second == -1)
                            gi2sp[g] = sp;
                        else
                        {
                            MFEM_VERIFY(it->second == sp, "");
                        }
                    }
                }
            }

            os = 0;
            for (int p=0; p<nprocs; ++p)
            {
                for (int i=0; i<local_num_dofs_sub[s][p]; ++i)
                {
                    const int g = allgid[offsets[p] + (2*i)];

                    map<int, int>::const_iterator it = gi2sp.find(g);
                    MFEM_VERIFY(it != gi2sp.end() && it->first == g, "");

                    if (s2sp[dofs_sub_to_sdofs[s][os + i]] != -1)
                    {
                        MFEM_VERIFY(s2sp[dofs_sub_to_sdofs[s][os + i]] == it->second, "");
                    }

                    s2sp[dofs_sub_to_sdofs[s][os + i]] = it->second;
                }

                os += local_num_dofs_sub[s][p];
            }

            for (int i=0; i<s2sp_.size(); ++i)
            {
                if (s2sp_[i] == -1)
                    s2sp_[i] = s2sp[i];
                else
                    MFEM_VERIFY(s2sp_[i] == s2sp[i], "Consistency check");
            }
        }
    }

    for (int i=0; i<s2sp_.size(); ++i)
        MFEM_VERIFY(s2sp_[i] >= 0 && s2sp_[i] == s2sp[i], "");
}
#endif

void ParaViewPrintAttributes(const string &fname,
                             Mesh &mesh,
                             int entity_dim,
                             const Array<int> *el_number=nullptr,
                             const Array<int> *vert_number=nullptr)
{
    ofstream out(fname + ".vtu");

    out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\"";
    out << " byte_order=\"" << VTKByteOrder() << "\">\n";
    out << "<UnstructuredGrid>\n";

    const string fmt_str = "ascii";

    int dim = mesh.Dimension();
    int ne = 0;
    if (entity_dim == 1)
    {
        if (dim == 1) {
            ne = mesh.GetNE();
        }
        else {
            ne = mesh.GetNEdges();
        }
    }
    else if (entity_dim == 2)
    {
        if (dim == 2) {
            ne = mesh.GetNE();
        }
        else {
            ne = mesh.GetNFaces();
        }
    }
    else if (entity_dim == 3)
    {
        ne = mesh.GetNE();
    }
    int np = mesh.GetNV();

    auto get_geom = [mesh,entity_dim,dim](int i)
    {
        if (entity_dim == 1) {
            return Geometry::SEGMENT;
        }
        else if (entity_dim == 2 && dim > 2) {
            return mesh.GetFaceGeometry(i);
        }
        else {
            return mesh.GetElementGeometry(i);
        }
    };

    auto get_verts = [mesh,entity_dim,dim](int i, Array<int> &v)
    {
        if (entity_dim == dim) {
            mesh.GetElementVertices(i, v);
        }
        else if (entity_dim == 1) {
            mesh.GetEdgeVertices(i, v);
        }
        else if (entity_dim == 2) {
            mesh.GetFaceVertices(i, v);
        }
    };

    out << "<Piece NumberOfPoints=\"" << np << "\" NumberOfCells=\""
        << ne << "\">\n";

    // print out the points
    out << "<Points>\n";
    out << "<DataArray type=\"" << "Float64"
        << "\" NumberOfComponents=\"3\" format=\"" << fmt_str << "\">\n";
    for (int i = 0; i < np; i++)
    {
        const double *v = mesh.GetVertex(i);
        for (int d = 0; d < 3; ++ d)
        {
            if (d < mesh.SpaceDimension()) {
                out << v[d] << " ";
            }
            else {
                out << "0.0 ";
            }
        }
        out << '\n';
    }
    out << "</DataArray>" << endl;
    out << "</Points>" << endl;

    out << "<Cells>" << endl;
    out << "<DataArray type=\"Int32\" Name=\"connectivity\" format=\""
        << fmt_str << "\">" << endl;
    for (int i = 0; i < ne; i++)
    {
        Array<int> v;
        Geometry::Type geom = get_geom(i);
        get_verts(i, v);
        const int *p = VTKGeometry::VertexPermutation[geom];
        for (int j = 0; j < v.Size(); ++j)
        {
            out << v[p ? p[j] : j] << " ";
        }
        out << '\n';
    }
    out << "</DataArray>" << endl;

    out << "<DataArray type=\"Int32\" Name=\"offsets\" format=\""
        << fmt_str << "\">" << endl;
    // offsets
    int coff = 0;
    for (int i = 0; i < ne; ++i)
    {
        Array<int> v;
        get_verts(i, v);
        coff += v.Size();
        out << coff << '\n';
    }
    out << "</DataArray>" << endl;
    out << "<DataArray type=\"UInt8\" Name=\"types\" format=\""
        << fmt_str << "\">" << endl;
    // cell types
    for (int i = 0; i < ne; i++)
    {
        Geometry::Type geom = get_geom(i);
        out << VTKGeometry::Map[geom] << '\n';
    }
    out << "</DataArray>" << endl;
    out << "</Cells>" << endl;

    out << "<CellData Scalars=\"attribute\">" << endl;

    if (el_number)
    {
        string array_name;
        if (entity_dim == dim) {
            array_name = "element number";
        }
        else if (entity_dim == 2) {
            array_name = "face number";
        }
        else if (entity_dim == 1) {
            array_name = "edge number";
        }
        out << "<DataArray type=\"Int32\" Name=\""
            << array_name << "\" format=\""
            << fmt_str << "\">" << endl;
        for (int i = 0; i < ne; i++)
        {
            out << (*el_number)[i] << '\n';
        }
        out << "</DataArray>" << endl;
    }
    out << "</CellData>" << endl;

    if (vert_number)
    {
        out << "<PointData>" << endl;
        out << "<DataArray type=\"Int32\" Name=\"vertex number\" format=\""
            << fmt_str << "\">" << endl;
        for (int i = 0; i < np; i++)
        {
            out << (*vert_number)[i] << '\n';
        }
        out << "</DataArray>" << endl;
        out << "</PointData>" << endl;
    }

    out << "</Piece>\n"; // need to close the piece open in the PrintVTU method
    out << "</UnstructuredGrid>\n";
    out << "</VTKFile>" << endl;
}

SampleMeshManager::SampleMeshManager(vector<ParFiniteElementSpace*> & fespace_,
                                     string visFileName, double visScale) :
    nspaces(fespace_.size()), fespace(fespace_), filename(visFileName),
    elemVisScale(visScale)
{
    MFEM_VERIFY(nspaces > 0, "Must provide at least one finite element space");
    spfespace.assign(nspaces, nullptr);

    pmesh = fespace[0]->GetParMesh();

    for (int i=1; i<nspaces; ++i)
    {
        MFEM_VERIFY(pmesh == fespace[i]->GetParMesh(),
                    "All spaces must use the same full-order mesh");
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int color = (myid != 0);
    const int status = MPI_Comm_split(MPI_COMM_WORLD, color, myid, &root_comm);
    MFEM_VERIFY(status == MPI_SUCCESS,
                "Construction of hyperreduction comm failed");

    sample_dofs_proc.assign(nspaces, vector<set<int>> (nprocs));

    finalized = false;
}

void SampleMeshManager::RegisterSampledVariable(const string variable,
        const int space, vector<int> const& sample_dofs_v,
        vector<int> const& num_sample_dofs_per_proc)
{
    MFEM_VERIFY(!finalized,
                "Cannot register a variable in a finalized SampleMeshManager");
    MFEM_VERIFY(0 <= space && space < nspaces, "Invalid space index");
    MFEM_VERIFY(num_sample_dofs_per_proc.size() == nprocs, "");

    {
        auto search = vmap.find(variable);
        MFEM_VERIFY(search == vmap.end(),
                    "Variable " + variable + " is already registered!");
    }
    vmap[variable] = nvar;

    for (int p=0; p<nprocs; ++p)
    {
        int os = 0;
        for (int q=0; q<p; ++q)
        {
            os += num_sample_dofs_per_proc[q];
        }

        if (p == nprocs-1)
        {
            MFEM_VERIFY(os + num_sample_dofs_per_proc[p] == sample_dofs_v.size(), "");
        }

        for (int j=0; j<num_sample_dofs_per_proc[p]; ++j)
            sample_dofs_proc[space][p].insert(sample_dofs_v[os + j]);
    }

    num_sample_dofs_per_proc_var.push_back(num_sample_dofs_per_proc);
    sample_dofs_var.push_back(sample_dofs_v);
    varSpace.push_back(space);
    nvar = varSpace.size();
}

void SampleMeshManager::SetSampleMaps()
{
    num_sample_dofs_per_proc_merged.assign(nprocs, 0);

    vector<vector<int>> allspaceTOS(nspaces);

    for (int i=0; i<nspaces; ++i)
    {
        allspaceTOS[i].resize(nprocs);
        MPI_Allgather(&spaceTOS[i], 1, MPI_INT, allspaceTOS[i].data(), 1, MPI_INT,
                      MPI_COMM_WORLD);
    }

    for (int p=0; p<nprocs; ++p)
    {
        vector<int> sample_dofs_os(nspaces);

        for (int i=0; i<nspaces; ++i)
        {
            num_sample_dofs_per_proc_merged[p] += sample_dofs_proc[i][p].size();
            const int os = allspaceTOS[i][p];
            sample_dofs_os[i] = sample_dofs.size();

            for (set<int>::const_iterator it = sample_dofs_proc[i][p].begin();
                    it != sample_dofs_proc[i][p].end(); ++it)
            {
                sample_dofs.push_back(os + (*it));
            }
        }

        // For each variable v and each of the num_sample_dofs_per_proc[v][p]
        // samples, set s2sp_var[v][] to be its index in sample_dofs.
        s2sp_var.resize(nvar);
        for (int v=0; v<nvar; ++v)
        {
            const int space = varSpace[v];

            int os = 0;
            for (int q=0; q<p; ++q)
                os += num_sample_dofs_per_proc_var[v][q];

            const int nvs = sample_dofs_var[v].size();
            s2sp_var[v].resize(nvs);

            for (int j=0; j<num_sample_dofs_per_proc_var[v][p]; ++j)
            {
                const int sample = sample_dofs_var[v][os + j];

                // Note: this has quadratic complexity and could be improved
                // with a std::map<int, int>, but it should not be a bottleneck.
                int k = -1;
                int cnt = 0;
                for (set<int>::const_iterator it = sample_dofs_proc[space][p].begin();
                        it != sample_dofs_proc[space][p].end(); ++it, ++cnt)
                {
                    if (*it == sample)
                    {
                        MFEM_VERIFY(k == -1, "");
                        k = cnt;
                    }
                }

                MFEM_VERIFY(k >= 0, "");
                s2sp_var[v][os + j] = sample_dofs_os[space] + k;
            }
        }
    }  // loop over p
}

void SampleMeshManager::ConstructSampleMesh()
{
    MFEM_VERIFY(!finalized, "Sample mesh is already constructed");

    nvar = varSpace.size();
    MFEM_VERIFY(nvar == sample_dofs_var.size()
                && nvar == num_sample_dofs_per_proc_var.size(), "");

    spaceOS.assign(nspaces + 1, 0);
    spaceTOS.assign(nspaces + 1, 0);
    spaceOSSP.assign(nspaces, 0);

    for (int i=0; i<nspaces; ++i)
    {
        spaceOS[i+1] = spaceOS[i] + fespace[i]->GetVSize();
        spaceTOS[i+1] = spaceTOS[i] + fespace[i]->GetTrueVSize();
    }

    SetSampleMaps();
    CreateSampleMesh();

    if (myid == 0)
    {
        for (int i = 0; i < nspaces - 1; ++i)
        {
            spaceOSSP[i+1] = spaceOSSP[i] + spfespace[i]->GetVSize();
        }

        sample_pmesh->EnsureNodes();
    }

    FinishSampleMaps();

    finalized = true;
}

void SampleMeshManager::FinishSampleMaps()
{
    {
        // TODO: if this is used in more than one place, then store it in the class.
        // Set spaceOSall by gathering spaceTOS from all processes, for all spaces.
        vector<int> data(nprocs * (nspaces+1));
        MPI_Allgather(spaceTOS.data(), nspaces+1, MPI_INT, data.data(), nspaces+1,
                      MPI_INT, MPI_COMM_WORLD);

        spaceOSall.assign(nspaces+1, vector<int> (nprocs, 0));
        for (int p=0; p<nprocs; ++p)
            for (int s=0; s<nspaces+1; ++s)
                spaceOSall[s][p] = data[(p*(nspaces+1)) + s];
    }

    if (myid != 0)
        return;

    int offset = 0;
    for (int p=0; p<nprocs; ++p)
    {
        for (int i=0; i<num_sample_dofs_per_proc_merged[p]; ++i)
        {
            // Determine space index s based on offsets
            int s = nspaces - 1;
            while (sample_dofs[offset + i] < spaceOSall[s][p])
                s--;
        }

        offset += num_sample_dofs_per_proc_merged[p];
    }

    MFEM_VERIFY(s2sp.size() == offset, "");

    // Define the map s2sp_var from variable samples to sample mesh dofs.
    for (int v=0; v<nvar; ++v)
    {
        const int s = varSpace[v];
        int os_p = 0;
        for (int p=0; p<nprocs; ++p)
        {
            for (int j=0; j<num_sample_dofs_per_proc_var[v][p]; ++j)
            {
                MFEM_VERIFY(spaceOSall[s][p] <= sample_dofs[s2sp_var[v][os_p + j]]
                            && sample_dofs[s2sp_var[v][os_p + j]] < spaceOSall[s+1][p], "");
                const int spId = s2sp[s2sp_var[v][os_p + j]];
                s2sp_var[v][os_p + j] = spId - spaceOSSP[s];
            }

            os_p += num_sample_dofs_per_proc_var[v][p];
        }

        MFEM_VERIFY(os_p == sample_dofs_var[v].size(), "");
    }
}

int SampleMeshManager::GetNumVarSamples(const string variable) const
{
    const int var = GetVariableIndex(variable);
    return sample_dofs_var[var].size();
}

void SampleMeshManager::GetSampledValues(const string variable,
        mfem::Vector const& v, CAROM::Vector & s) const
{
    const int var = GetVariableIndex(variable);
    const int n = s2sp_var[var].size();
    const int space = varSpace[var];
    MFEM_VERIFY(s.dim() == n, "");
    MFEM_VERIFY(v.Size() == spfespace[space]->GetTrueVSize(), "");
    for (int i=0; i<n; ++i)
        s(i) = v[s2sp_var[var][i]];
}

void SampleMeshManager::WriteVariableSampleMap(const string variable,
        string file_name) const
{
    const int var = GetVariableIndex(variable);
    ofstream file;
    file.open(file_name);
    for (int i=0; i<s2sp_var[var].size(); ++i)
    {
        file << s2sp_var[var][i] << endl;
    }
    file.close();
}

void GatherDistributedMatrixRows_aux(const CAROM::Matrix& B, const int rdim,
#ifdef FULL_DOF_STENCIL
                                     const int os0, const int os1, const int ossp,
                                     ParFiniteElementSpace& fespace,
#endif
                                     const vector<int>& st2sp, const vector<int>& sprows,
                                     const vector<int>& all_sprows, CAROM::Matrix& Bsp)
{
    // Create B sample+ matrix Bsp (B^s+)
    // On P0 get number of rows each processor contributes to Bsp.

    int num_procs, myid;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MFEM_VERIFY(rdim <= B.numColumns(), "");

    vector<int> allos0(num_procs);
    vector<int> allos1(num_procs);

    MPI_Allgather(&os0, 1, MPI_INT, allos0.data(), 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&os1, 1, MPI_INT, allos1.data(), 1, MPI_INT, MPI_COMM_WORLD);

    const int num_sprows = static_cast<int>(sprows.size());

#ifdef FULL_DOF_STENCIL
    int num_sprows_true = 0;
    for (auto i : sprows)
    {
        if (i >= os0 && i < os1 && fespace.GetLocalTDofNumber(i - os0) >= 0)
        {
            num_sprows_true++;
        }
    }

    vector<int> sprows_true(num_sprows_true);
    {
        int tcnt = 0;
        for (int j=0; j<sprows.size(); ++j)
        {
            const int i = sprows[j];
            if (i >= os0 && i < os1 && fespace.GetLocalTDofNumber(i - os0) >= 0)
            {
                sprows_true[tcnt] = j;
                tcnt++;
            }
        }

        MFEM_VERIFY(tcnt == num_sprows_true, "");
    }
#endif

    int* cts = new int [num_procs];
#ifdef FULL_DOF_STENCIL
    MPI_Allgather(&num_sprows_true, 1, MPI_INT, cts, 1, MPI_INT, MPI_COMM_WORLD);
    int* allNSP = new int [num_procs];
    MPI_Allgather(&num_sprows, 1, MPI_INT, allNSP, 1, MPI_INT, MPI_COMM_WORLD);
    int* offsetSP = new int [num_procs];
    offsetSP[0] = 0;
    for (int i = 1; i < num_procs; ++i) {
        offsetSP[i] = offsetSP[i-1] + allNSP[i-1];
    }
#else
    MPI_Allgather(&num_sprows, 1, MPI_INT, cts, 1, MPI_INT, MPI_COMM_WORLD);
#endif
    int* offsets = new int [num_procs];
    offsets[0] = 0;
    for (int i = 1; i < num_procs; ++i) {
        offsets[i] = offsets[i-1] + cts[i-1];
    }

#ifdef FULL_DOF_STENCIL
    const int total_num_sprows_true = offsets[num_procs-1] + cts[num_procs-1];
    vector<int> all_sprows_true(total_num_sprows_true);

    MPI_Allgatherv(sprows_true.data(), num_sprows_true,
                   MPI_INT, all_sprows_true.data(), cts,
                   offsets, MPI_INT, MPI_COMM_WORLD);
#else
    MFEM_VERIFY(offsets[num_procs-1] + cts[num_procs-1] == all_sprows.size(), "");
#endif

    if (myid == 0) {
        const int Nsp = Bsp.numRows();

        for (int i = 0; i < num_sprows; i++) {
            int row = sprows[i];
            if (os0 <= row && row < os1)
            {
#ifdef FULL_DOF_STENCIL
                const int ltdof = fespace.GetLocalTDofNumber(row - os0);
                if (ltdof >= 0 && st2sp[i] >= 0)
                {
                    MFEM_VERIFY(0 <= st2sp[i] - ossp && st2sp[i] - ossp < Nsp, "");

                    for (int j = 0; j < rdim; ++j)
                        Bsp(st2sp[i] - ossp, j) = B(ltdof, j);
                }
#else
                for (int j = 0; j < rdim; ++j)
                    Bsp(st2sp[i], j) = B(row, j);
#endif
            }
        }

#ifdef FULL_DOF_STENCIL
        int Bsp_row = num_sprows_true;
#else
        int Bsp_row = num_sprows;
#endif
        MPI_Status status;
        for (int i = 1; i < num_procs; ++i) {
            for (int j = 0; j < cts[i]; ++j) {
#ifdef FULL_DOF_STENCIL
                const int sti = offsetSP[i] + all_sprows_true[Bsp_row];
                if (allos0[i] <= all_sprows[sti] && all_sprows[sti] < allos1[i])
                {
                    MFEM_VERIFY(0 <= st2sp[sti] - ossp && st2sp[sti] - ossp < Bsp.numRows(), "");
                    // Note that this may redundantly overwrite some rows corresponding to shared DOF's.
                    MPI_Recv(&Bsp(st2sp[sti] - ossp, 0), rdim, MPI_DOUBLE,
                             i, offsets[i]+j, MPI_COMM_WORLD, &status);
                }
#else
                if (allos0[i] <= all_sprows[Bsp_row] && all_sprows[Bsp_row] < allos1[i])
                {
                    // Note that this may redundantly overwrite some rows corresponding to shared DOF's.
                    MPI_Recv(&Bsp(st2sp[Bsp_row], 0), rdim, MPI_DOUBLE,
                             i, offsets[i]+j, MPI_COMM_WORLD, &status);
                }
#endif
                ++Bsp_row;
            }
        }

#ifdef FULL_DOF_STENCIL
        MFEM_VERIFY(Bsp_row == all_sprows_true.size(), "");
#else
        MFEM_VERIFY(Bsp_row == all_sprows.size(), "");
#endif
    }
    else {
        double* v = new double [rdim];
#ifdef FULL_DOF_STENCIL
        for (int i = 0; i < num_sprows_true; i++) {
            int row = -1;
            if (os0 <= sprows[sprows_true[i]] && sprows[sprows_true[i]] < os1)
                row = fespace.GetLocalTDofNumber(sprows[sprows_true[i]] - os0);

            MFEM_VERIFY(row >= 0, "");

            if (os0 <= sprows[sprows_true[i]] && sprows[sprows_true[i]] < os1)
#else
        for (int i = 0; i < num_sprows; i++) {
            int row = sprows[i];
            if (os0 <= row && row < os1)
#endif
            {
                for (int j = 0; j < rdim; ++j)
                    v[j] = B(row, j);
                MPI_Send(v, rdim, MPI_DOUBLE, 0, offsets[myid]+i, MPI_COMM_WORLD);
            }
        }
        delete [] v;
    }
    delete [] cts;
    delete [] offsets;
#ifdef FULL_DOF_STENCIL
    delete [] allNSP;
    delete [] offsetSP;
#endif
}

int SampleMeshManager::GetVariableIndex(const string variable) const
{
    auto search = vmap.find(variable);
    MFEM_VERIFY(search != vmap.end(),
                "Variable " + variable + " is not registered!");
    const int var = search->second;
    MFEM_VERIFY(0 <= var && var < nvar, "Invalid variable index");
    return var;
}

void SampleMeshManager::GatherDistributedMatrixRows(const string variable,
        CAROM::Matrix const& B, const int rdim, CAROM::Matrix& Bsp) const
{
    const int var = GetVariableIndex(variable);
    const int s = varSpace[var];

    GatherDistributedMatrixRows_aux(B, rdim, spaceOS[s], spaceOS[s+1], spaceOSSP[s],
                                    *fespace[s], st2sp, sprows, all_sprows, Bsp);
}

void SampleMeshManager::CreateSampleMesh()
{
    MFEM_VERIFY(nspaces > 0, "");

    vector<vector<int> > sample_dofs_block(nspaces);  // True DOF's
    vector<vector<int> > sample_dofs_sub_to_sample_dofs(nspaces);
    vector<vector<int> > local_num_sample_dofs_sub(nspaces);
    vector<vector<int> > localSampleDofsToElem_sub(nspaces);
    vector<vector<int> > localSampleDofsToElemDof_sub(nspaces);

    vector<int> Ntrue(nspaces);
    for (int i=0; i<nspaces; ++i)
        Ntrue[i] = fespace[i]->TrueVSize();

    vector<int>& local_num_sample_dofs = num_sample_dofs_per_proc_merged;

    SplitDofsIntoBlocks(Ntrue, sample_dofs, local_num_sample_dofs,
                        sample_dofs_block, sample_dofs_sub_to_sample_dofs, local_num_sample_dofs_sub);

    const bool getSampledEntities = !filename.empty();
    // Local mesh entities containing sampled DOFs, used only for ParaView visualization.
    set<int> intElems, faces, edges, vertices;

    // Find all local elements that should be included, using all spaces.
    GetLocalSampleMeshElements(*pmesh, *fespace[0], sample_dofs_block[0],
                               local_num_sample_dofs_sub[0], elems,
                               getSampledEntities, intElems, faces, edges, vertices);
    GetLocalDofsToLocalElementMap(*fespace[0], sample_dofs_block[0],
                                  local_num_sample_dofs_sub[0], elems, localSampleDofsToElem_sub[0],
                                  localSampleDofsToElemDof_sub[0], true);
    for (int i=1; i<nspaces; ++i)
    {
        set<int> elems_i;
        set<int> intElems_i, faces_i, edges_i,
            vertices_i;  // Local mesh entities containing sampled DOFs, used only for ParaView visualization.

        GetLocalSampleMeshElements(*pmesh, *fespace[i], sample_dofs_block[i],
                                   local_num_sample_dofs_sub[i], elems_i,
                                   getSampledEntities, intElems_i, faces_i, edges_i, vertices_i);
        GetLocalDofsToLocalElementMap(*fespace[i], sample_dofs_block[i],
                                      local_num_sample_dofs_sub[i], elems_i, localSampleDofsToElem_sub[i],
                                      localSampleDofsToElemDof_sub[i], true);

        // Merge the elements found for all spaces.
        elems.insert(elems_i.begin(), elems_i.end());

        if (getSampledEntities)
        {
            intElems.insert(intElems_i.begin(), intElems_i.end());
            faces.insert(faces_i.begin(), faces_i.end());
            edges.insert(edges_i.begin(), edges_i.end());
            vertices.insert(vertices_i.begin(), vertices_i.end());
        }
    }

    vector<int> elemLocalIndices;
    vector<map<int, int> > elemLocalIndicesInverse;
    Mesh *sample_mesh = 0;
    BuildSampleMesh(*pmesh, fespace, elems, sample_mesh, sprows, elemLocalIndices,
                    elemLocalIndicesInverse);

    MFEM_VERIFY(sample_mesh->GetNE() == elemLocalIndices.size(), "");

    if (myid == 0)
    {
        sample_pmesh = new ParMesh(root_comm, *sample_mesh);
        delete sample_mesh;

        // Create fespaces on sample mesh
        for (int i=0; i<nspaces; ++i)
            spfespace[i] = new ParFiniteElementSpace(sample_pmesh, fespace[i]->FEColl(),
                    fespace[i]->GetVDim());
    }

    vector<int> spNtrue(nspaces);
    for (int i=0; i<nspaces; ++i)
        spNtrue[i] = myid == 0 ? spfespace[i]->TrueVSize() : 0;

    Set_s2sp(myid, nprocs, spNtrue, sample_dofs.size(), local_num_sample_dofs,
             local_num_sample_dofs_sub, localSampleDofsToElem_sub,
             localSampleDofsToElemDof_sub, sample_dofs_sub_to_sample_dofs,
             elemLocalIndicesInverse, spfespace, s2sp);

#ifdef FULL_DOF_STENCIL
    Finish_s2sp_augmented(myid, nprocs, fespace, sample_dofs_block,
                          sample_dofs_sub_to_sample_dofs, local_num_sample_dofs_sub, true, s2sp);
#endif

    // Prepare for setting st2sp

    const int numStencil = sprows.size();
    vector<int> local_num_stencil_dofs(nprocs);

    MPI_Allgather(&numStencil, 1, MPI_INT, &local_num_stencil_dofs[0], 1, MPI_INT,
                  MPI_COMM_WORLD);

    int* offsets = new int [nprocs];
    offsets[0] = 0;
    for (int i = 1; i < nprocs; ++i)
        offsets[i] = offsets[i-1] + local_num_stencil_dofs[i-1];

    const int total_num_stencil_dofs = offsets[nprocs-1] +
                                       local_num_stencil_dofs[nprocs-1];

    all_sprows.resize(total_num_stencil_dofs);
    MPI_Allgatherv(&sprows[0], local_num_stencil_dofs[myid],
                   MPI_INT, &all_sprows[0], &local_num_stencil_dofs[0],
                   offsets, MPI_INT, MPI_COMM_WORLD);

    // Note that all_stencil_dofs may contain DOF's on different processes that
    // are identical (shared DOF's), which is fine (see comments in Set_s2sp).

    delete [] offsets;

    vector<vector<int>> stencil_dofs_block(nspaces);
    vector<vector<int> > stencil_dofs_sub_to_stencil_dofs(nspaces);
    vector<vector<int> > local_num_stencil_dofs_sub(nspaces);
    vector<vector<int> > localStencilDofsToElem_sub(nspaces);
    vector<vector<int> > localStencilDofsToElemDof_sub(nspaces);

#ifdef FULL_DOF_STENCIL
    vector<int> Nfull(nspaces);
    for (int i=0; i<nspaces; ++i)
        Nfull[i] = fespace[i]->GetVSize();

    SplitDofsIntoBlocks(Nfull, all_sprows, local_num_stencil_dofs,
                        stencil_dofs_block, stencil_dofs_sub_to_stencil_dofs,
                        local_num_stencil_dofs_sub);
#else
    SplitDofsIntoBlocks(Ntrue, all_sprows, local_num_stencil_dofs,
                        stencil_dofs_block, stencil_dofs_sub_to_stencil_dofs,
                        local_num_stencil_dofs_sub);
#endif

    for (int i=0; i<nspaces; ++i)
    {
        GetLocalDofsToLocalElementMap(*fespace[i], stencil_dofs_block[i],
                                      local_num_stencil_dofs_sub[i], elems, localStencilDofsToElem_sub[i],
                                      localStencilDofsToElemDof_sub[i], false);
    }

    Set_s2sp(myid, nprocs, spNtrue, all_sprows.size(), local_num_stencil_dofs,
             local_num_stencil_dofs_sub, localStencilDofsToElem_sub,
             localStencilDofsToElemDof_sub, stencil_dofs_sub_to_stencil_dofs,
             elemLocalIndicesInverse, spfespace, st2sp);

#ifdef FULL_DOF_STENCIL
    Finish_s2sp_augmented(myid, nprocs, fespace, stencil_dofs_block,
                          stencil_dofs_sub_to_stencil_dofs, local_num_stencil_dofs_sub, false, st2sp);
#endif

    if (!filename.empty())
    {
        SampleVisualization(pmesh, elems, intElems, faces, edges, vertices,
                            filename, nullptr, elemVisScale);
    }
}

void SampleVisualization(ParMesh *pmesh, set<int> const& elems,
                         set<int> const& intElems, set<int> const& faces,
                         set<int> const& edges, set<int> const& vertices,
                         string const& filename, mfem::Vector *elemCount,
                         double elementScaling)
{
    // Write files for global pmesh and a piecewise constant function with 1 on
    // sample mesh elements and 0 elsewhere. Note that this is a visualization of
    // the sample mesh elements, not the sampled DOFs. That is, it shows all
    // elements that contain a sampled DOF, which could be at a vertex, edge,
    // face, or volume.

    // Construct a piecewise constant space on the global mesh.
    L2_FECollection l2_coll(1, pmesh->Dimension());  // order 1
    ParFiniteElementSpace pwc_space(pmesh, &l2_coll);

    ParGridFunction marker(&pwc_space);
    marker = 0.0;
    if (elemCount)
    {
        MFEM_VERIFY(elemCount->Size() == pmesh->GetNE(), "");
        for (int e=0; e<pmesh->GetNE(); ++e)
        {
            Array<int> dofs;
            pwc_space.GetElementVDofs(e, dofs);

            for (int i = 0; i < dofs.Size(); i++)
            {
                const int dof_i = dofs[i] >= 0 ? dofs[i] : -1 - dofs[i];
                marker[dof_i] = (*elemCount)[e];
            }
        }
    }
    else
    {
        for (set<int>::const_iterator it = elems.begin(); it != elems.end(); ++it)
        {
            Array<int> dofs;
            pwc_space.GetElementVDofs(*it, dofs);

            for (int i = 0; i < dofs.Size(); i++)
            {
                const int dof_i = dofs[i] >= 0 ? dofs[i] : -1 - dofs[i];
                marker[dof_i] = elementScaling;
            }
        }
    }

    VisItDataCollection visit_dc(filename, pmesh);
    visit_dc.RegisterField("Sample Elements", &marker);
    visit_dc.SetFormat(DataCollection::SERIAL_FORMAT);
    visit_dc.Save();

    // Write ParaView files to visualize sampled elements, edges, and vertices
    int ne = pmesh->GetNE();
    int nv = pmesh->GetNV();
    int nf = pmesh->GetNFaces();
    int nedge = pmesh->GetNEdges();

    Array<int> e(ne), v(nv), f(nf), edge(nedge);
    e = 0;
    v = 0;
    f = 0;
    edge = 0;

    for (set<int>::const_iterator it = intElems.begin(); it != intElems.end(); ++it)
    {
        e[*it] = 1;
    }

    for (set<int>::const_iterator it = faces.begin(); it != faces.end(); ++it)
    {
        f[*it] = 1;
    }

    for (set<int>::const_iterator it = edges.begin(); it != edges.end(); ++it)
    {
        edge[*it] = 1;
    }

    for (set<int>::const_iterator it = vertices.begin(); it != vertices.end(); ++it)
    {
        v[*it] = 1;
    }

    /*
    for (int i=0; i<ne/2; ++i) { e[i] = 0; }
    for (int i=0; i<nv/2; ++i) { v[i] = 0; }
    for (int i=0; i<nf/2; ++i) { f[i] = 0; }
    for (int i=0; i<nedge/2; ++i) { edge[i] = 0; }
    */

    ParaViewPrintAttributes(filename + "_pv", *pmesh, pmesh->Dimension(), &e, &v);
    if (pmesh->Dimension() == 3)
    {
        ParaViewPrintAttributes(filename + "_pv_face", *pmesh, 2, &f, &v);
    }
    ParaViewPrintAttributes(filename + "_pv_edge", *pmesh, 1, &edge, &v);
}

void SampleDOFSelector::ReadMapFromFile(const string variable, string file_name)
{
    {
        auto search = vmap.find(variable);
        MFEM_VERIFY(search == vmap.end(),
                    "Map for variable " + variable + " is already read!");
    }
    vmap[variable] = nvar;

    vector<int> v;
    ifstream file;
    file.open(file_name);
    string line;
    while(getline(file, line)) {
        v.push_back(stoi(line));
    }
    file.close();

    s2sp_var.push_back(v);
    nvar = s2sp_var.size();
}

int SampleDOFSelector::GetVariableIndex(const string variable) const
{
    auto search = vmap.find(variable);
    MFEM_VERIFY(search != vmap.end(),
                "Variable " + variable + " is not registered!");
    const int var = search->second;
    MFEM_VERIFY(0 <= var && var < nvar, "Invalid variable index");
    return var;
}

void SampleDOFSelector::GetSampledValues(const string variable,
        mfem::Vector const& v, CAROM::Vector & s) const
{
    const int var = GetVariableIndex(variable);

    const int n = s2sp_var[var].size();
    MFEM_VERIFY(s.dim() == n, "");
    for (int i=0; i<n; ++i)
        s(i) = v[s2sp_var[var][i]];
}

} // namespace CAROM
