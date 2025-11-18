/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

// Description: This class greedily selects parameter points
//              for the construction of a ROM database. The implemented
//              greedy algorithm is derived from algorithm 2 of Choi et. al's
//              paper "Gradient-based constrained optimization using a database
//              of linear reduced-order models": https://arxiv.org/abs/1506.07849
//
//              The greedy algorithm workflow is as follows:
//              1. Construct the GreedySampler by giving it a
//                 domain of parameter points.
//              2. Request a parameter point to create a local ROM at.
//              3. Request a parameter point to compute an error error indicator
//                 for using the nearest local ROM.
//              4. Give the computed error indicator to the GreedySampler.
//              5. Repeat steps 3 and 4 until the GreedySampler
//                 no longer requires more error indicators to be computed.
//              6. Repeat steps 2 to 5 until the GreedySampler
//                 no longer requires more local ROMs to be created.
//              7. The ROM database is now complete, meeting the error tolerance
//                 for all parameter points within the domain.

#ifndef included_GreedySampler_h
#define included_GreedySampler_h

#include "linalg/Vector.h"
#include <string>
#include <vector>
#include <set>
#include <random>

/* Use C++11 built-in shared pointers if available; else fallback to Boost. */
#if __cplusplus >= 201103L
#include <memory>
#else
#include <boost/shared_ptr.hpp>
#endif

namespace CAROM {

class Vector;

/**
 * struct GreedyErrorIndicatorPoint is a struct containing the information
 * required to calculate the error indicator at a specified parameter point.
 */
struct GreedyErrorIndicatorPoint {

    /**
     * @brief The parameter point to calculate the error indicator.
     */
    std::shared_ptr<Vector> point;

    /**
     * @brief The parameter point of the closest local ROM.
     */
    std::shared_ptr<Vector> localROM;
};

/**
 * Class GreedySampler is a class defining the interface of a
 *       greedy algorithm that given a domain of parameter points, iteratively
 *       returns the next best parameter point to sample in order to create
 *       a ROM database efficiently.
 */
class GreedySampler
{
public:
    /**
     * @brief Constructor.
     *
     * @param[in] parameter_points A vector of CAROM::Vectors containing many
     *                             user-defined parameter points from which a
     *                             subset will be chosen for building ROMs.
     * @param[in] check_local_rom Compute local ROM error indicator each iteration.
     * @param[in] relative_error_tolerance The relative error tolerance value
     *                                     for which to end the algorithm.
     * @param[in] alpha A constant factor to increase the error indicator
     *                  tolerance each iteration.
     * @param[in] max_clamp A scalar factor representing the maximum amount
     *                      the error indicator tolerance can change per iteration.
     * @param[in] subset_size The size of the random subset.
     * @param[in] convergence_subset_size The size of the convergence subset.
     * @param[in] output_log_path The path to the output log file. If not used,
     *                            outputs to stdout.
     * @param[in] warm_start_file_name The path to the HDF5 file of a previous
     *                                 database to use as a warm start.
     * @param[in] use_centroid Whether to use the centroid heuristic when
     *                         determining the first parameter point to sample.
     * @param[in] random_seed A random seed.
     * @param[in] debug_algorithm Whether to turn off all randomness for
     *                            debugging purposes.
     */
    GreedySampler(
        const std::vector<Vector> & parameter_points,
        bool check_local_rom,
        double relative_error_tolerance,
        double alpha,
        double max_clamp,
        int subset_size,
        int convergence_subset_size,
        std::string output_log_path = "",
        std::string warm_start_file_name = "",
        bool use_centroid = true,
        int random_seed = 1,
        bool debug_algorithm = false);

    /**
     * @brief Constructor.
     *
     * @param[in] parameter_points A vector of doubles containing
     *                             the different parameter points.
     * @param[in] check_local_rom Compute local ROM error indicator each iteration.
     * @param[in] relative_error_tolerance The relative error tolerance value
     *                                     for which to end the algorithm.
     * @param[in] alpha A alpha constant to increase greedy algorithm by each
     *                  iteration.
     * @param[in] max_clamp A scalar factor representing the maximum amount
     *                      the error indicator tolerance can change per iteration.
     * @param[in] subset_size The size of the random subset.
     * @param[in] convergence_subset_size The size of the convergence subset.
     * @param[in] output_log_path The path to the output log file. If not used,
     *                            outputs to stdout.
     * @param[in] warm_start_file_name The path to the HDF5 file of a previous
     *                                 database to use as a warm start.
     * @param[in] use_centroid Whether to use the centroid heuristic when
     *                         determining the first parameter point to sample.
     * @param[in] random_seed A random seed.
     * @param[in] debug_algorithm Whether to turn off all randomness for
     *                            debugging purposes.
     */
    GreedySampler(
        const std::vector<double> & parameter_points,
        bool check_local_rom,
        double relative_error_tolerance,
        double alpha,
        double max_clamp,
        int subset_size,
        int convergence_subset_size,
        std::string output_log_path = "",
        std::string warm_start_file_name = "",
        bool use_centroid = true,
        int random_seed = 1,
        bool debug_algorithm = false);

    /**
     * @brief Constructor.
     *
     * @param[in] param_space_min A CAROM::Vector representing the minimum
     *                            of the parameter space domain.
     * @param[in] param_space_max A CAROM::Vector representing the maximum
     *                            of the parameter space domain.
     * @param[in] num_parameter_points The maximum number of parameter points
     *                                 to sample.
     * @param[in] check_local_rom Compute local ROM error indicator each iteration.
     * @param[in] relative_error_tolerance The relative error tolerance value
     *                                     for which to end the algorithm.
     * @param[in] alpha A alpha constant to increase greedy algorithm by each
     *                  iteration.
     * @param[in] max_clamp A scalar factor representing the maximum amount
     *                      the error indicator tolerance can change per iteration.
     * @param[in] subset_size The size of the random subset.
     * @param[in] convergence_subset_size The size of the convergence subset.
     * @param[in] output_log_path The path to the output log file. If not used,
     *                            outputs to stdout.
     * @param[in] warm_start_file_name The path to the HDF5 file of a previous
     *                                 database to use as a warm start.
     * @param[in] use_centroid Whether to use the centroid heuristic when
     *                         determining the first parameter point to sample.
     * @param[in] random_seed A random seed.
     * @param[in] debug_algorithm Whether to turn off all randomness for
     *                            debugging purposes.
     */
    GreedySampler(
        const Vector & param_space_min,
        const Vector & param_space_max,
        int num_parameter_points,
        bool check_local_rom,
        double relative_error_tolerance,
        double alpha,
        double max_clamp,
        int subset_size,
        int convergence_subset_size,
        std::string output_log_path = "",
        std::string warm_start_file_name = "",
        bool use_centroid = true,
        int random_seed = 1,
        bool debug_algorithm = false);

    /**
     * @brief Constructor.
     *
     * @param[in] param_space_min A double representing the minimum
     *                            of the parameter space domain.
     * @param[in] param_space_max A double representing the maximum
     *                            of the parameter space domain.
     * @param[in] num_parameter_points The maximum number of parameter points
     *                                 to sample.
     * @param[in] check_local_rom Compute local ROM error indicator each iteration.
     * @param[in] relative_error_tolerance The relative error tolerance value
     *                                     for which to end the algorithm.
     * @param[in] alpha A alpha constant to increase greedy algorithm by each
     *                  iteration.
     * @param[in] max_clamp A scalar factor representing the maximum amount
     *                     the error indicator tolerance can change per iteration.
     * @param[in] subset_size The size of the random subset.
     * @param[in] convergence_subset_size The size of the convergence subset.
     * @param[in] output_log_path The path to the output log file. If not used,
     *                            outputs to stdout.
     * @param[in] warm_start_file_name The path to the HDF5 file of a previous
     *                                 database to use as a warm start.
     * @param[in] use_centroid Whether to use the centroid heuristic when
     *                         determining the first parameter point to sample.
     * @param[in] random_seed A random seed.
     * @param[in] debug_algorithm Whether to turn off all randomness for
     *                            debugging purposes.
     */
    GreedySampler(
        double param_space_min,
        double param_space_max,
        int num_parameter_points,
        bool check_local_rom,
        double relative_error_tolerance,
        double alpha,
        double max_clamp,
        int subset_size,
        int convergence_subset_size,
        std::string output_log_path = "",
        std::string warm_start_file_name = "",
        bool use_centroid = true,
        int random_seed = 1,
        bool debug_algorithm = false);

    /**
     * @brief Constructor.
     *
     * @param[in] base_file_name The base part of the file of the
     *                           database to load when restarting from a save.
     * @param[in] output_log_path The path to the output log file. If not used,
     *                            outputs to stdout.
     */
    GreedySampler(
        std::string base_file_name,
        std::string output_log_path = "");

    /**
     * @brief Destructor.
     */
    virtual ~GreedySampler();

    /**
     * @brief Returns the next parameter point for which sampling is required.
     *
     * @return A shared pointer holding the point or NULL if a point is not
     *         ready to be sampled.
     */
    std::shared_ptr<Vector>
    getNextParameterPoint();

    /**
     * @brief Returns the next parameter point for which a relative error
     *        is required.
     *
     * @return A struct holding the point and the location of the nearest local
     *         ROM or NULL if no point is required.
     */
    GreedyErrorIndicatorPoint
    getNextPointRequiringRelativeError();

    /**
     * @brief Returns the next parameter point for which an error indicator
     *        is required.
     *
     * @return A struct holding the point and the location of the nearest local
     *         ROM or NULL if no point is required.
     */
    GreedyErrorIndicatorPoint
    getNextPointRequiringErrorIndicator();

    /**
     * @brief Set the relative error of the specified point.
     *
     * @param[in] error The relative error.
     */
    void
    setPointRelativeError(double error);

    /**
     * @brief Set the error indicator error of the specified parameter point.
     *
     * @param[in] error The error indicator value.
     * @param[in] vec_size The size of the vector the error indicator was
     *                     obtained from. This is used for normalization.
     */
    void
    setPointErrorIndicator(double error, int vec_size);

    /**
     * @brief Returns the index of the nearest non-sampled parameter point to
     *         the given point.
     *
     * @param[in] point The CAROM::Vector holding the point.
     *
     * @return An integer holding the nearest non-sampled point to
     *         the given point or -1 if none exist.
     */
    int
    getNearestNonSampledPoint(const Vector & point);

    /**
     * @brief Returns the nearest local ROM to the specified parameter point.
     *
     * @param[in] point The CAROM::Vector holding the point.
     *
     * @return A shared pointer holding the point or NULL if no local ROM exists.
     */
    std::shared_ptr<Vector>
    getNearestROM(const Vector & point);

    /**
     * @brief Get the domain of the parameter points.
     *
     * @return A vector holding all parameter point candidates.
     */
    std::vector<Vector>
    getParameterPointDomain();

    /**
     * @brief Get the sampled parameter points.
     *
     * @return A vector holding all sampled parameter points.
     */
    std::vector<Vector>
    getSampledParameterPoints();

    /**
     * @brief Save the object state to a file.
     *
     * @param[in] base_file_name The base part of the file to save the
     *                           database to.
     */
    virtual void
    save(std::string base_file_name);

    /**
     * @brief Check if the greedy algorithm procedure is complete.
     *
     * @return A boolean representing if the greedy algorithm procedure is
     *           complete.
     */
    bool
    isComplete();

protected:

    /**
     * @brief Do a warm start by adding the sampled parameters from the database
     *         in a file to the current database.
     *
     * @param[in] warm_start_file_name The path to the HDF5 file of a previous
     *                                 database to use as a warm start.
     */
    void addDatabaseFromFile(
        std::string const& warm_start_file_name);

    /**
     * @brief Load the object state from a file.
     *
     * @param[in] base_file_name The base part of the file to load the
     *                           database from.
     */
    virtual void load(
        std::string base_file_name);

    /**
     * @brief Construct the list of parameter point candidates to sample.
     */
    virtual void constructParameterPoints() = 0;

    /**
     * @brief Construct the list of parameter point candidates to sample.
     */
    void checkParameterPointInput();

    /**
     * @brief Construct the GreedySampler object.
     */
    void constructObject(
        bool check_local_rom,
        double relative_error_tolerance,
        double alpha,
        double max_clamp,
        int subset_size,
        int convergence_subset_size,
        std::string output_log_path,
        bool use_centroid,
        int random_seed,
        bool debug_algorithm);

    /**
     * @brief Initialize the list of parameter point candidates to sample.
     */
    void initializeParameterPoints();

    /**
     * @brief Get the next subset point requiring an error indicator.
     *
     * @return A struct holding the point and the location of the nearest local
     *         ROM or NULL if no point is required.
     */
    GreedyErrorIndicatorPoint getNextSubsetPointRequiringErrorIndicator();

    /**
     * @brief Get the next convergence point requiring an error indicator.
     *
     * @return A struct holding the point and the location of the nearest local
     *         ROM or NULL if no point is required.
     */
    GreedyErrorIndicatorPoint
    getNextConvergencePointRequiringErrorIndicator();

    /**
     * @brief Generate a vector of random points.
     *
     * @param[in] num_points The number of points to generate
     *
     * @return A vector of random points.
     */
    std::vector<Vector> generateRandomPoints(int num_points);

    /**
     * @brief Print to output_log_file or cout.
     */
    void agnosticPrint(std::string str);

    /**
     * @brief Print the error indicator.
     *
     * @param[in] errorIndicatorPoint The vector where the error indicator
                                      was obtained.
     * @param[in] proc_errors The error indicator value.
     *
     */
    void printErrorIndicator(const Vector & errorIndicatorPoint,
                             double proc_errors);

    /**
     * @brief Print that the error indicator was not met.
     */
    void printErrorIndicatorToleranceNotMet();

    /**
     * @brief Print the sampling type.
     *
     * @param[in] sampling_type The sampling type.
     *
     */
    void printSamplingType(std::string sampling_type);

    /**
     * @brief Print that convergence was achieved.
     */
    void printConvergenceAchieved();

    /**
     * @brief Set the error indicator for a subset point.
     *
     * @param[in] proc_errors The error indicator value.
     *
     */
    void setSubsetErrorIndicator(double proc_errors);

    /**
     * @brief Set the error indicator for a convergence point.
     *
     * @param[in] proc_errors The error indicator value.
     *
     */
    void setConvergenceErrorIndicator(double proc_errors);

    /**
     * @brief Generate a new set of convergence points.
     */
    void generateConvergenceSubset();

    /**
     * @brief Switch to convergence mode.
     */
    void startConvergence();

    /**
     * @brief Get the next parameter point to sample after a convergence failure.
     */
    virtual void getNextParameterPointAfterConvergenceFailure() = 0;

    /**
     * @brief Returns the index to the nearest local ROM to the specified
     *        parameter point index in the parameter point list.
     *
     * @param[in] index The index of the specified parameter point.
     * @param[in] ignore_self Whether to ignore the specified parameter point
     *                        if a local ROM was built there.
     *
     * @return The index of the point in the list of parameters.
     */
    int
    getNearestROMIndexToParameterPoint(int index, bool ignore_self);

    /**
     * @brief The parameter points to explore.
     */
    std::vector<Vector> d_parameter_points;

    /**
     * @brief The convergence parameter points to explore.
     */
    std::vector<Vector> d_convergence_points;

    /**
     * @brief The minimum value of the parameter space.
     */
    Vector d_min_param_point;

    /**
     * @brief The maximum value of the parameter space.
     */
    Vector d_max_param_point;

    /**
     * @brief The parameter points that were already sampled.
     */
    std::set<int> d_parameter_sampled_indices;

    /**
     * @brief The parameter point indices (used to generate the random subsets).
     */
    std::vector<int> d_parameter_point_random_indices;

    /**
     * @brief The current errors of the parameter points.
     */
    std::vector<double> d_parameter_point_errors;

    /**
     * @brief The local ROMs used to obtain the current errors of the parameter points.
     */
    std::vector<int> d_parameter_point_local_rom;

    /**
     * @brief Output log path.
     */
    std::string d_output_log_path;

    /**
     * @brief The current max error of the parameter points of the current iteration.
     */
    double d_max_error;

    /**
     * @brief The current relative error of the current iteration.
     */
    double d_curr_relative_error;

    /**
     * @brief The alpha constant.
     */
    double d_alpha;

    /**
     * @brief The max_clamp constant.
     */
    double d_max_clamp;

    /**
     * @brief The error indicator tolerance used to terminate the greedy procedure.
     */
    double d_error_indicator_tol;

    /**
     * @brief The relative error tolerance used to terminate the greedy procedure.
     */
    double d_relative_error_tol;

    /**
     * @brief The maximum number of parameter points to sample.
     */
    int d_num_parameter_points;

    /**
     * @brief The size of the subset of parameter points used per iteration.
     */
    int d_subset_size;

    /**
     * @brief The size of the subset of parameter points used to check convergence.
     */
    int d_convergence_subset_size;

    /**
     * @brief The next parameter point to sample.
     */
    int d_next_point_to_sample;

    /**
     * @brief The next parameter point requiring a error indicator.
     */
    int d_next_point_requiring_error_indicator;

    /**
     * @brief Whether the use the centroid heuristic for obtaining the first
     *        parameter point.
     */
    bool d_use_centroid;

    /**
     * @brief Whether the check the last sampled local ROM's error indicator
     *        after each iteration.
     */
    bool d_check_local_rom;

    /**
     * @brief Whether the database has already computed a new parameter point
     *        for the current iteration.
     */
    bool d_iteration_started;

    /**
     * @brief Whether the database is in the convergence verifying phase.
     */
    bool d_convergence_started;

    /**
     * @brief Whether the database has already computed a new parameter point
     *        to sample.
     */
    bool d_next_parameter_point_computed;

    /**
     * @brief Whether the database has already computed a new paramter point
     *        requiring a error indicator.
     */
    bool d_point_requiring_error_indicator_computed;

    /**
     * @brief Whether the database has already created a random subset
     *        for this iteration.
     */
    bool d_subset_created;

    /**
     * @brief Turn off randomness for debugging purposes.
     *
     */
    bool d_debug_algorithm;

    /**
     * @brief An internal counter.
     *
     */
    int d_counter;

    /**
     * @brief An internal subset counter.
     *
     */
    int d_subset_counter;

    /**
     * @brief Whether the greedy procedure has completed.
     */
    bool d_procedure_completed;

    /**
     * @brief The rank of the given processor.
     */
    int d_rank;

    /**
     * @brief Then random seed used to generate subsets.
     */
    int d_random_seed;

    /**
     * @brief Random engine used to generate subsets.
     */
    std::default_random_engine rng;
};

/**
 * @brief Create a greedy error indicator point.
 *
 * @param[in] point The error indicator point.
 * @param[in] localROM The point of the nearest ROM.
 *
 * @return A struct holding @a point and @a localROM.
 */
GreedyErrorIndicatorPoint createGreedyErrorIndicatorPoint(
    const std::shared_ptr<Vector> & point,
    const std::shared_ptr<Vector> & localROM);

/**
 * @brief Given a vector, find the nearest point in a domain.
 *
 * @param[in] paramPoints The domain to search.
 * @param[in] point The specified point.
 *
 * @return A vector representing the nearest point in a domain.
 */
Vector getNearestPoint(const std::vector<Vector> & paramPoints,
                       const Vector & point);

/**
 * @brief Given a double, find the nearest point in a domain.
 *
 * @param[in] paramPoints The domain to search.
 * @param[in] point The specified point.
 *
 * @return A double representing the nearest point in a domain.
 */
double getNearestPoint(const std::vector<double> & paramPoints, double point);

/**
 * @brief Given a vector, find the nearest point in a domain.
 *
 * @param[in] paramPoints The domain to search.
 * @param[in] point The specified point.
 *
 * @return The index of the nearest point in a domain.
 */
int getNearestPointIndex(const std::vector<Vector> & paramPoints,
                         Vector & point);

/**
 * @brief Given a double, find the nearest point in a domain.
 *
 * @param[in] paramPoints The domain to search.
 * @param[in] point The specified point.
 *
 * @return The index of the nearest point in a domain.
 */
int getNearestPointIndex(const std::vector<double> & paramPoints, double point);
}

#endif
