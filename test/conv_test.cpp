
#/*
   Comp Eng 3DY4 (Computer Systems Integration Project)

   Department of Electrical and Computer Engineering
   McMaster University
   Ontario, Canada
*/

#include <limits.h>
#include "dy4.h"
#include "iofunc.h"
#include "filter.h"
#include "gtest/gtest.h"

namespace {

    class Resampling_Fixture : public ::testing::Test {
    public:
        const int N = 1024;     // Input signal size
        const int M = 101;      // Filter kernel size
        const real EPSILON = 1e-4;
        const int decim = 10;   // Downsampling factor
        const int up = 3;       // Upsampling factor

        std::vector<real> x, h, y_ref, y_test;
        std::vector<real> state_ref, state_test;

        Resampling_Fixture() {
            x.resize(N);
            h.resize(M);
            state_ref.resize(M - 1, 0.0);
            state_test.resize(M - 1, 0.0);
        }

        void SetUp() override {
            generate_random_values(x, -1.0, 1.0);
            generate_random_values(h, -1.0, 1.0);
        }

        void downsample(std::vector<real> &y, const std::vector<real> &x, int decim) {
            int output_size = x.size() / decim;
            y.resize(output_size);
            for (int i = 0; i < output_size; i++) {
                y[i] = x[i * decim];
            }
        }
    };

    TEST_F(Resampling_Fixture, EfficientConvolveDownsample_MatchesReference) {
        // Reset states
        std::fill(state_ref.begin(), state_ref.end(), 0.0);
        std::fill(state_test.begin(), state_test.end(), 0.0);

        // Reference: Convolve then downsample
        convolveFIR(y_ref, x, h, state_ref);
        downsample(y_ref, y_ref, decim);

        // Test implementation
        efficient_convolve_downsample(y_test, x, h, state_test, decim);

        // Verify sizes match
        ASSERT_EQ(y_ref.size(), y_test.size()) << "Output sizes differ!";

        // Verify values match
        for (int i = 0; i < (int)y_ref.size(); i++) {
            EXPECT_NEAR(y_ref[i], y_test[i], EPSILON) << "Mismatch at index " << i;
        }

        // Verify states match
        for (int i = 0; i < (int)state_ref.size(); i++) {
            EXPECT_NEAR(state_ref[i], state_test[i], EPSILON) << "State mismatch at index " << i;
        }
    }

    TEST_F(Resampling_Fixture, Resampler_MatchesReference) {
        // Reset states
        std::fill(state_ref.begin(), state_ref.end(), 0.0);
        std::fill(state_test.begin(), state_test.end(), 0.0);

        // Reference implementation
        std::vector<real> upsampled_x;
        
        // 1. Upsample by inserting zeros
        upsampled_x.resize(x.size() * up, 0.0);
        for (int i = 0; i < (int)x.size(); i++) {
            upsampled_x[i * up] = x[i];
        }

        // 2. Convolve (this updates state_ref with upsampled_x samples)
        std::vector<real> filtered_x;
        convolveFIR(filtered_x, upsampled_x, h, state_ref);

        // 3. Downsample
        downsample(y_ref, filtered_x, decim);

        // Test implementation
        resampler(y_test, x, h, state_test, decim, up);

        // Verify outputs match
        ASSERT_EQ(y_ref.size(), y_test.size()) << "Output sizes differ!";
        for (int i = 0; i < (int)y_ref.size(); i++) {
            EXPECT_NEAR(y_ref[i], y_test[i], EPSILON) << "Output mismatch at index " << i;
        }

        // Verify state contains correct original samples (last M-1 samples of x)
        int expected_state_samples = std::min((int)x.size(), (int)state_test.size());
        for (int i = 0; i < expected_state_samples; i++) {
            EXPECT_NEAR(x[x.size() - expected_state_samples + i], 
                       state_test[i], EPSILON) 
                << "State should contain last " << expected_state_samples 
                << " original samples at index " << i;
        }
    }

} // namespace