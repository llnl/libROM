/******************************************************************************
 *
 * Copyright (c) 2013-2024, Lawrence Livermore National Security, LLC
 * and other libROM project developers. See the top-level COPYRIGHT
 * file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 *
 *****************************************************************************/

// Description: This source file is a test runner that uses the Google Test
// Framework to run unit tests on the CAROM::GNAT class.

#include <iostream>

#ifdef CAROM_HAS_GTEST
#include<gtest/gtest.h>
#include <mpi.h>
#include "hyperreduction/GNAT.h"
#include "linalg/Matrix.h"
#define _USE_MATH_DEFINES
#include <cmath>

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
    SUCCEED();
}

TEST(GNATSerialTest, Test_GNAT)
{

    // Orthonormal input matrix to GNAT
    double* orthonormal_mat = new double[50] {
        -0.1067,   -0.4723,   -0.4552,    0.1104,   -0.2337,
            0.1462,    0.6922,   -0.2716,    0.1663,    0.3569,
            0.4087,   -0.3437,    0.4952,   -0.3356,    0.3246,
            0.2817,   -0.0067,   -0.0582,   -0.0034,    0.0674,
            0.5147,    0.1552,   -0.1635,   -0.3440,   -0.3045,
            -0.4628,    0.0141,   -0.1988,   -0.5766,    0.0150,
            -0.2203,    0.3283,    0.2876,   -0.4597,   -0.1284,
            -0.0275,    0.1202,   -0.0924,   -0.2290,   -0.3808,
            0.4387,   -0.0199,   -0.3338,   -0.1711,   -0.2220,
            0.0101,    0.1807,    0.4488,    0.3219,   -0.6359
        };

    // Result of GNAT (f_basis_sampled_inv)
    double* GNAT_true_ans = new double[25] {
        -0.295811, -0.264874,  1.02179,  -1.05194,  -0.554046,
            -0.270643,  1.05349,    0.119162,  0.541832,  0.646459,
            -1.33334,  -0.874864,  -0.276067, -0.27327,   0.124747,
            0.672776,  0.538704,  -0.735484, -0.794417,  0.388543,
            -0.682073, -0.049598, -0.51706,  -0.457748, -1.11295
        };

    int num_cols = 5;
    int num_rows = 10;

    CAROM::Matrix u(orthonormal_mat, num_rows, num_cols, false);
    double* GNAT_res = NULL;
    std::vector<int> f_sampled_row(num_cols, 0);
    std::vector<int> f_sampled_row_true_ans{0, 1, 4, 5, 9};
    std::vector<int> f_sampled_rows_per_proc(1, 0);
    CAROM::Matrix f_basis_sampled_inv = CAROM::Matrix(num_cols, num_cols, false);
    CAROM::GNAT(u, num_cols, f_sampled_row, f_sampled_rows_per_proc,
                f_basis_sampled_inv, 0, 1);

    for (int i = 0; i < num_cols; i++) {
        EXPECT_EQ(f_sampled_row[i], f_sampled_row_true_ans[i]);
    }

    // Compare the norm between the GNAT result and the true GNAT answer
    double l2_norm_diff = 0.0;
    for (int i = 0; i < num_cols; i++) {
        for (int j = 0; j < num_cols; j++) {
            l2_norm_diff += pow(abs(GNAT_true_ans[i * num_cols + j] - f_basis_sampled_inv(i,
                                    j)), 2);
        }
    }
    l2_norm_diff = sqrt(l2_norm_diff);

    // Allow for some error due to float rounding
    EXPECT_TRUE(l2_norm_diff < 1e-5);
}

TEST(GNATSerialTest, Test_GNAT_oversampling)
{

    // Orthonormal input matrix to GNAT
    double* orthonormal_mat = new double[50] {
        -0.1067,   -0.4723,   -0.4552,    0.1104,   -0.2337,
            0.1462,    0.6922,   -0.2716,    0.1663,    0.3569,
            0.4087,   -0.3437,    0.4952,   -0.3356,    0.3246,
            0.2817,   -0.0067,   -0.0582,   -0.0034,    0.0674,
            0.5147,    0.1552,   -0.1635,   -0.3440,   -0.3045,
            -0.4628,    0.0141,   -0.1988,   -0.5766,    0.0150,
            -0.2203,    0.3283,    0.2876,   -0.4597,   -0.1284,
            -0.0275,    0.1202,   -0.0924,   -0.2290,   -0.3808,
            0.4387,   -0.0199,   -0.3338,   -0.1711,   -0.2220,
            0.0101,    0.1807,    0.4488,    0.3219,   -0.6359
        };

    // Result of GNAT (f_basis_sampled_inv)
    double* GNAT_true_ans = new double[45] {
        -0.111754, -0.472181, -0.454143,  0.110436,  -0.234925,
            0.169535,  0.691715, -0.276502,  0.166065,   0.362544,
            0.443111, -0.344589,  0.488125, -0.336035,   0.332789,
            0.556025,  0.154167, -0.172054, -0.344493,  -0.294606,
            -0.498551,  0.0149608,-0.191435, -0.576216,   0.00647047,
            -0.247487,  0.328931,  0.293216, -0.459384,  -0.134887,
            -0.036157,  0.120386, -0.0906109,-0.228902, -0.382862,
            0.478391, -0.0208761,-0.342018, -0.171577,  -0.2125,
            -0.0109879, 0.181169,  0.453212,  0.322187,   -0.640965
        };

    int num_cols = 5;
    int num_rows = 10;
    int num_samples = 9;

    CAROM::Matrix u(orthonormal_mat, num_rows, num_cols, false);
    double* GNAT_res = NULL;
    std::vector<int> f_sampled_row(num_samples, 0);
    std::vector<int> f_sampled_row_true_ans{0, 1, 2, 4, 5, 6, 7, 8, 9};
    std::vector<int> f_sampled_rows_per_proc(1, 0);
    CAROM::Matrix f_basis_sampled_inv = CAROM::Matrix(num_samples, num_cols, false);
    CAROM::GNAT(u, num_cols, f_sampled_row, f_sampled_rows_per_proc,
                f_basis_sampled_inv, 0, 1, num_samples);

    for (int i = 0; i < num_samples; i++) {
        EXPECT_EQ(f_sampled_row[i], f_sampled_row_true_ans[i]);
    }

    // Compare the norm between the GNAT result and the true GNAT answer
    double l2_norm_diff = 0.0;
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_cols; j++) {
            l2_norm_diff += pow(abs(GNAT_true_ans[i * num_cols + j] - f_basis_sampled_inv(i,
                                    j)), 2);
        }
    }
    l2_norm_diff = sqrt(l2_norm_diff);

    // Allow for some error due to float rounding
    EXPECT_TRUE(l2_norm_diff < 1e-5);
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
#else // #ifndef CAROM_HAS_GTEST
int main()
{
    std::cout << "libROM was compiled without Google Test support, so unit "
              << "tests have been disabled. To enable unit tests, compile "
              << "libROM with Google Test support." << std::endl;
}
#endif // #endif CAROM_HAS_GTEST
