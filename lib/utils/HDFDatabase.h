/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

// Description: The concrete database implementation using HDF5.

#ifndef included_HDFDatabase_h
#define included_HDFDatabase_h

#include "Database.h"
#include "Utilities.h"
#include "hdf5.h"
#include <string>

namespace CAROM {

/**
 * HDFDatabase implements the interface of Database for HDF5 database files.
 */
class HDFDatabase : public Database
{
public:
    /**
     * @brief Default constructor.
     */
    HDFDatabase();

    /**
     * @brief Destructor.
     */
    virtual
    ~HDFDatabase();

    /**
     * @brief Creates a new HDF5 database file with the supplied name.
     *
     * @param[in] file_name Name of HDF5 database file to create.
     * @param[in] comm MPI communicator for distributed data I/O.
     *                 If not MPI_COMM_NULL, each process creates a file
     *                 whose name is file_name extended with process rank in 6 digits.
     *
     * @return True if file create was successful.
     */
    virtual
    bool
    create(
        const std::string& file_name,
        const MPI_Comm comm=MPI_COMM_NULL) override;

    /**
     * @brief Opens an existing HDF5 database file with the supplied name.
     *
     * @param[in] file_name Name of existing HDF5 database file to open.
     * @param[in] type Read/write type ("r"/"wr")
     *                 If not MPI_COMM_NULL, each process opens a file
     *                 whose name is file_name extended with process rank in 6 digits.
     *
     * @return True if file open was successful.
     */
    virtual
    bool
    open(
        const std::string& file_name,
        const std::string& type,
        const MPI_Comm comm=MPI_COMM_NULL) override;

    /**
     * @brief Closes the currently open HDF5 database file.
     *
     * @return True if the file close was successful.
     */
    virtual
    bool
    close() override;

    /**
     * @brief Writes an array of integers associated with the supplied key to
     * the currently open HDF5 database file.
     *
     * @pre !key.empty()
     * @pre data != nullptr
     * @pre nelements > 0
     *
     * @param[in] key The key associated with the array of values to be
     *                written.
     * @param[in] data The array of integer values to be written.
     * @param[in] nelements The number of integers in the array.
     * @param[in] distributed True if data is a distributed integer array.
     *                        HDFDatabase writes the array in file-per-process,
     *                        where each file is written serially by one process.
     */
    virtual
    void
    putIntegerArray(
        const std::string& key,
        const int* const data,
        int nelements,
        const bool distributed=false) override;

    /**
     * @brief Writes an array of doubles associated with the supplied key to
     * the currently open HDF5 database file.
     *
     * @pre !key.empty()
     * @pre data != nullptr
     * @pre nelements > 0
     *
     * @param[in] key The key associated with the array of values to be
     *                written.
     * @param[in] data The array of double values to be written.
     * @param[in] nelements The number of doubles in the array.
     * @param[in] distributed True if data is a distributed double array.
     *                        HDFDatabase writes the array in file-per-process,
     *                        where each file is written serially by one process.
     */
    virtual
    void
    putDoubleArray(
        const std::string& key,
        const double* const data,
        int nelements,
        const bool distributed=false) override;

    /**
     * @brief Writes a vector of doubles associated with the supplied key to
     * the currently open HDF5 database file.
     *
     * @pre !key.empty()
     * @pre data != nullptr
     * @pre nelements > 0
     *
     * @param[in] key The key associated with the vector of values to be
     *                written.
     * @param[in] data The vector of double values to be written.
     * @param[in] nelements The number of doubles in the vector.
     * @param[in] distributed True if data is a distributed double array.
     *                        HDFDatabase writes the array in file-per-process,
     *                        where each file is written serially by one process.
     */
    virtual
    void
    putDoubleVector(
        const std::string& key,
        const std::vector<double>& data,
        int nelements,
        const bool distributed=false) override;

    /**
     * @brief Reads an array of integers associated with the supplied key
     * from the currently open HDF5 database file.
     *
     * @pre !key.empty()
     * @pre data != nullptr || nelements == 0
     *
     * @param[in] key The key associated with the array of values to be
     *                read.
     * @param[out] data The allocated array of integer values to be read.
     * @param[in] nelements The number of integers in the array.
     * @param[in] distributed True if data is a distributed integer array.
     *                        HDFDatabase reads the array in file-per-process,
     *                        where each file is read serially by one process.
     */
    virtual
    void
    getIntegerArray(
        const std::string& key,
        int* data,
        int nelements,
        const bool distributed=false) override;

    /**
     * @brief Count the number of elements in an array of doubles associated
     * with the supplied key from the currently open HDF5 database file.
     *
     * @pre !key.empty()
     *
     * @param[in] key The key associated with the array of values to be read.
     */
    virtual
    int
    getDoubleArraySize(const std::string& key) override;

    /**
     * @brief Reads an array of doubles associated with the supplied key
     * from the currently open HDF5 database file.
     *
     * @pre !key.empty()
     * @pre data != nullptr || nelements == 0
     *
     * @param[in] key The key associated with the array of values to be
     *                read.
     * @param[out] data The allocated array of double values to be read.
     * @param[in] nelements The number of doubles in the array.
     * @param[in] distributed True if data is a distributed double array.
     *                        HDFDatabase reads the array in file-per-process,
     *                        where each file is read serially by one process.
     */
    virtual
    void
    getDoubleArray(
        const std::string& key,
        double* data,
        int nelements,
        const bool distributed=false) override;

    /**
     * @brief Reads a sub-array of doubles associated with the supplied key
     * from the currently open HDF5 database file.
     *
     * @pre !key.empty()
     * @pre data != nullptr || nelements == 0
     *
     * @param[in] key The key associated with the array of values to be
     *                read.
     * @param[out] data The allocated sub-array of double values to be read.
     * @param[in] nelements The number of doubles in the full array.
     * @param[in] idx The set of indices in the sub-array.
     * @param[in] distributed True if data is a distributed double array.
     *                        HDFDatabase reads the array in file-per-process,
     *                        where each file is read serially by one process.
     */
    virtual
    void
    getDoubleArray(
        const std::string& key,
        double* data,
        int nelements,
        const std::vector<int>& idx,
        const bool distributed=false) override;

    /**
     * @brief Reads an array of doubles associated with the supplied key
     * from the currently open HDF5 database file.
     *
     * @pre !key.empty()
     * @pre data != nullptr || nelements == 0
     *
     * @param[in] key The key associated with the array of values to be
     *                read.
     * @param[out] data The allocated array of double values to be read.
     * @param[in] nelements The number of doubles in the array.
     * @param[in] offset The initial offset in the array.
     *                   Typically, this is a column index of the matrix data.
     * @param[in] block_size The block size to read from the HDF5 dataset.
     *                       Typically, this is a number of columns of the matrix data.
     * @param[in] stride The stride to read from the HDF5 dataset.
     *                   Typically, this is the total number of columns of the matrix data.
     * @param[in] distributed True if data is a distributed double array.
     *                        HDFDatabase reads the array in file-per-process,
     *                        where each file is read serially by one process.
     */
    virtual
    void
    getDoubleArray(
        const std::string& key,
        double* data,
        int nelements,
        int offset,
        int block_size,
        int stride,
        const bool distributed=false) override;

protected:

    /**
     * @brief Returns true if the specified key represents an integer entry.
     *        If the key does not exist or if the string is empty then false is
     *        returned.
     *
     * @param[in] key The key associated with the data we are interested in.
     *
     * @return True if the data associated with key is an integer array.
     */
    bool
    isInteger(
        const std::string& key);

    /**
     * @brief Returns true if the specified key represents a double entry.
     *        If the key does not exist or if the string is empty then false is
     *        returned.
     *
     * @param[in] key The key associated with the data we are interested in.
     *
     * @return True if the data associated with key is a double array.
     */
    bool
    isDouble(
        const std::string& key);

    /**
     * @brief Write an attribute to the specified dataset.
     *
     * @param[in] type_key The attribute to be written.
     * @param[in] dataset_id ID of the dataset key will be written to.
     */
    virtual
    void
    writeAttribute(
        int type_key,
        hid_t dataset_id);

    /**
     * @brief Read an attribute from the specified dataset.
     *
     * @param[in] dataset_id ID of the dataset key will be written to.
     *
     * @return The attribute.
     */
    int
    readAttribute(
        hid_t dataset_id);

    /**
     * @brief True if the HDF5 database is mounted to a file.
     */
    bool d_is_file;

    /**
     * @brief ID of file attached to database or -1 if not mounted to a file.
     */
    hid_t d_file_id;

    /**
     * @brief ID of the group attached to the database.
     * */
    hid_t d_group_id;

    /**
     * @brief The key representing a double array.
     * */
    static const int KEY_DOUBLE_ARRAY;

    /**
     * @brief The key representing an integer array.
     * */
    static const int KEY_INT_ARRAY;

    /**
     * @brief MPI process rank.
     */
    int d_rank;

private:
    /**
     * @brief Unimplemented copy constructor.
     */
    HDFDatabase(
        const HDFDatabase& other);

    /**
     * @brief Unimplemented assignment operator.
     */
    HDFDatabase&
    operator = (
        const HDFDatabase& rhs);
};

}

#endif
