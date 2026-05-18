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
// Framework to run unit tests on the CAROM::DEIM class.

#include <iostream>

#ifdef CAROM_HAS_GTEST
#include<gtest/gtest.h>
#include <mpi.h>
#include "hyperreduction/DEIM.h"
#include "linalg/Matrix.h"
#define _USE_MATH_DEFINES
#include <cmath>

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
    SUCCEED();
}

TEST(DEIMSerialTest, Test_DEIM)
{

    // Orthonormal input matrix to DEIM
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

    // Result of DEIM (f_basis_sampled_inv)
    double* DEIM_true_ans = new double[25] {
        -0.295811, -0.264874,  1.02179,  -1.05194,  -0.554046,
            -0.270643,  1.05349,    0.119162,  0.541832,  0.646459,
            -1.33334,  -0.874864,  -0.276067, -0.27327,   0.124747,
            0.672776,  0.538704,  -0.735484, -0.794417,  0.388543,
            -0.682073, -0.049598, -0.51706,  -0.457748, -1.11295
        };

    int num_cols = 5;
    int num_rows = 10;

    CAROM::Matrix u(orthonormal_mat, num_rows, num_cols, false);
    double* DEIM_res = NULL;
    std::vector<int> f_sampled_row(num_cols, 0);
    std::vector<int> f_sampled_row_true_ans{0, 1, 4, 5, 9};
    std::vector<int> f_sampled_rows_per_proc(1, 0);
    CAROM::Matrix f_basis_sampled_inv = CAROM::Matrix(num_cols, num_cols, false);
    CAROM::DEIM(u, num_cols, f_sampled_row, f_sampled_rows_per_proc,
                f_basis_sampled_inv, 0, 1);

    for (int i = 0; i < num_cols; i++) {
        EXPECT_EQ(f_sampled_row[i], f_sampled_row_true_ans[i]);
    }

    // Compare the norm between the DEIM result and the true DEIM answer
    double l2_norm_diff = 0.0;
    for (int i = 0; i < num_cols; i++) {
        for (int j = 0; j < num_cols; j++) {
            l2_norm_diff += pow(abs(DEIM_true_ans[i * num_cols + j] - f_basis_sampled_inv(i,
                                    j)), 2);
        }
    }
    l2_norm_diff = sqrt(l2_norm_diff);

    // Allow for some error due to float rounding
    EXPECT_TRUE(l2_norm_diff < 1e-5);
}

TEST(DEIMSerialTest, Test_DEIM_decreased_used_basis_vectors)
{

    // Orthonormal input matrix to DEIM
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

    // Result of DEIM (f_basis_sampled_inv)
    double* DEIM_true_ans = new double[9] {
        -0.331632, -0.690455,  2.07025,
            -0.541131,  1.17546,  -0.446068,
            -1.55764,  -1.05777,  -0.022448
        };

    int num_cols = 5;
    int num_rows = 10;
    int num_basis_vectors_used = 3;

    CAROM::Matrix u(orthonormal_mat, num_rows, num_cols, false);
    double* DEIM_res = NULL;
    std::vector<int> f_sampled_row(num_basis_vectors_used, 0);
    std::vector<int> f_sampled_row_true_ans{0, 1, 4};
    std::vector<int> f_sampled_rows_per_proc(1, 0);
    CAROM::Matrix f_basis_sampled_inv = CAROM::Matrix(num_basis_vectors_used,
                                        num_basis_vectors_used, false);
    CAROM::DEIM(u, num_basis_vectors_used, f_sampled_row, f_sampled_rows_per_proc,
                f_basis_sampled_inv, 0, 1);

    for (int i = 0; i < num_basis_vectors_used; i++) {
        EXPECT_EQ(f_sampled_row[i], f_sampled_row_true_ans[i]);
    }

    // Compare the norm between the DEIM result and the true DEIM answer
    double l2_norm_diff = 0.0;
    for (int i = 0; i < num_basis_vectors_used; i++) {
        for (int j = 0; j < num_basis_vectors_used; j++) {
            l2_norm_diff += pow(abs(DEIM_true_ans[i * num_basis_vectors_used + j] -
                                    f_basis_sampled_inv(i, j)), 2);
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
