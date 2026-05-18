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
// Framework to run unit tests on the CAROM::QDEIM class.

#include <iostream>

#ifdef CAROM_HAS_GTEST
#include<gtest/gtest.h>
#include <mpi.h>
#include "hyperreduction/QDEIM.h"
#include "linalg/Matrix.h"
#define _USE_MATH_DEFINES
#include <cmath>

/**
 * Simple smoke test to make sure Google Test is properly linked
 */
TEST(GoogleTestFramework, GoogleTestFrameworkFound) {
    SUCCEED();
}

TEST(QDEIMSerialTest, Test_QDEIM)
{

    // Orthonormal input matrix to QDEIM
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

    // Result of QDEIM (f_basis_sampled_inv)
    double* QDEIM_true_ans = new double[25] {
        0.439578,  0.704765,   0.90859,  -0.80199,   0.333326,
        0.989496, -0.108636,   0.0666524, 0.626509,  0.341482,
        0.195839,  0.938665,  -0.634648,  0.425609,  0.807025,
        0.437564, -0.00667263,-0.5862,   -1.0578,    0.736468,
        0.647878,  0.639428,  -0.711286, -0.0697963, -0.527859
    };

    int num_cols = 5;
    int num_rows = 10;

    CAROM::Matrix u(orthonormal_mat, num_rows, num_cols, true);
    double* QDEIM_res = NULL;
    std::vector<int> f_sampled_row(num_cols, 0);
    std::vector<int> f_sampled_row_true_ans{1, 2, 4, 6, 9};
    std::vector<int> f_sampled_rows_per_proc(1, 0);
    CAROM::Matrix f_basis_sampled_inv = CAROM::Matrix(num_cols, num_cols, false);
    CAROM::QDEIM(u, num_cols, f_sampled_row, f_sampled_rows_per_proc,
                 f_basis_sampled_inv, 0, 1, num_cols);

    for (int i = 0; i < num_cols; i++) {
        EXPECT_EQ(f_sampled_row[i], f_sampled_row_true_ans[i]);
    }

    // Compare the norm between the QDEIM result and the true QDEIM answer
    double l2_norm_diff = 0.0;
    for (int i = 0; i < num_cols; i++) {
        for (int j = 0; j < num_cols; j++) {
            l2_norm_diff += pow(abs(QDEIM_true_ans[i * num_cols + j] - f_basis_sampled_inv(
                                        i, j)), 2);
        }
    }
    l2_norm_diff = sqrt(l2_norm_diff);


    // Allow for some error due to float rounding
    EXPECT_TRUE(l2_norm_diff < 1e-5);
}

TEST(QDEIMSerialTest, Test_QDEIM_gpode_oversampling)
{

    // Orthonormal input matrix to QDEIM
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

    // Result of QDEIM (f_basis_sampled_inv)
    double* QDEIM_true_ans = new double[40] {
        -0.114191, -0.464064,   -0.460252,  0.0950032, -0.260738,
            0.172884,   0.680565,   -0.268109,  0.187266,   0.398005,
            0.450012,  -0.367566,    0.505419, -0.292346,   0.405863,
            0.546255,   0.186697,   -0.196538, -0.406346,  -0.39806,
            -0.506112,  0.0401362,  -0.210384, -0.624084,  -0.0735944,
            -0.255659,  0.356138,    0.272739, -0.511115,  -0.221413,
            0.472064,   0.00019259, -0.357875, -0.211637,  -0.279505,
            -0.0179827, 0.204459,    0.435683,  0.277904,  -0.715033
        };

    int num_cols = 5;
    int num_rows = 10;
    int num_samples = 8;

    CAROM::Matrix u(orthonormal_mat, num_rows, num_cols, true);
    double* QDEIM_res = NULL;
    std::vector<int> f_sampled_row(num_samples, 0);
    std::vector<int> f_sampled_row_true_ans{0, 1, 2, 4, 5, 6, 8, 9};
    std::vector<int> f_sampled_rows_per_proc(1, 0);
    CAROM::Matrix f_basis_sampled_inv = CAROM::Matrix(num_samples, num_cols, false);
    CAROM::QDEIM(u, num_cols, f_sampled_row, f_sampled_rows_per_proc,
                 f_basis_sampled_inv, 0, 1, num_samples);

    for (int i = 0; i < num_samples; i++) {
        EXPECT_EQ(f_sampled_row[i], f_sampled_row_true_ans[i]);
    }

    // Compare the norm between the QDEIM result and the true QDEIM answer
    double l2_norm_diff = 0.0;
    for (int i = 0; i < num_samples; i++) {
        for (int j = 0; j < num_cols; j++) {
            l2_norm_diff += pow(abs(QDEIM_true_ans[i * num_cols + j] - f_basis_sampled_inv(
                                        i, j)), 2);
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
