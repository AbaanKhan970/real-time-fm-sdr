/*
   Comp Eng 3DY4 (Computer Systems Integration Project)
   Department of Electrical and Computer Engineering
   McMaster University
   Ontario, Canada
*/

#include <benchmark/benchmark.h>
#include "utils_bench.h"
#include "dy4.h"
#include "iofunc.h"
#include "filter.h"

#define RANGE_MULTIPLIER 2
#define MIN_INPUT_SIZE 32768
#define MAX_INPUT_SIZE (4 * MIN_INPUT_SIZE)
#define MIN_FILTER_SIZE 101
#define MAX_FILTER_SIZE (1 * MIN_FILTER_SIZE)

const int lower_bound = -1;
const int upper_bound = 1;
const int default_decim = 10;
const int default_up = 3;

static void Bench_convolveFIR(benchmark::State& state) {
    int N = state.range(0);
    int M = state.range(1);

    std::vector<real> x(N);
    std::vector<real> h(M);
    std::vector<real> y(N + M - 1);
    std::vector<real> state_buf(M - 1, 0.0);

    generate_random_values(x, lower_bound, upper_bound);
    generate_random_values(h, lower_bound, upper_bound);

    for (auto _ : state) {
        convolveFIR(y, x, h, state_buf);
    }
}

BENCHMARK(Bench_convolveFIR)
    ->ArgsProduct({benchmark::CreateRange(MIN_INPUT_SIZE, MAX_INPUT_SIZE, RANGE_MULTIPLIER),
                   benchmark::CreateRange(MIN_FILTER_SIZE, MAX_FILTER_SIZE, RANGE_MULTIPLIER)});

static void Bench_efficient_convolve_downsample(benchmark::State& state) {
    int N = state.range(0);
    int M = state.range(1);
    int decim = default_decim;

    std::vector<real> x(N);
    std::vector<real> h(M);
    std::vector<real> y(N / decim);
    std::vector<real> state_buf(M - 1, 0.0);

    generate_random_values(x, lower_bound, upper_bound);
    generate_random_values(h, lower_bound, upper_bound);

    for (auto _ : state) {
        efficient_convolve_downsample(y, x, h, state_buf, decim);
    }
}

BENCHMARK(Bench_efficient_convolve_downsample)
    ->ArgsProduct({benchmark::CreateRange(MIN_INPUT_SIZE, MAX_INPUT_SIZE, RANGE_MULTIPLIER),
                   benchmark::CreateRange(MIN_FILTER_SIZE, MAX_FILTER_SIZE, RANGE_MULTIPLIER)});

static void Bench_resampler(benchmark::State& state) {
    int N = state.range(0);
    int M = state.range(1);
    int decim = default_decim;
    int up = default_up;

    std::vector<real> x(N);
    std::vector<real> h(M);
    std::vector<real> y(N * up / decim);
    std::vector<real> state_buf(M - 1, 0.0);

    generate_random_values(x, lower_bound, upper_bound);
    generate_random_values(h, lower_bound, upper_bound);

    for (auto _ : state) {
        resampler(y, x, h, state_buf, decim, up);
    }
}

BENCHMARK(Bench_resampler)
    ->ArgsProduct({benchmark::CreateRange(MIN_INPUT_SIZE, MAX_INPUT_SIZE, RANGE_MULTIPLIER),
                   benchmark::CreateRange(MIN_FILTER_SIZE, MAX_FILTER_SIZE, RANGE_MULTIPLIER)});

/*
The resampler appears to take more CPU time than efficient_convolve_downsample despite being theoretically more efficient because of several key factors in its operation:
1.Upsampling Overhead
The resampler performs upsampling (by 3x) before convolution, which:
	Increases the data volume it processes by 3x
	Requires additional zero-insertion operations
	Makes the convolution work with 3x more samples

2.Output Size Difference
For the same input size N:
	efficient_convolve_downsample outputs N/10 samples
	resampler outputs (N*3)/10 samples (30% more output)

3.Performance Comparison Breakdown:
Operation						Input Size	Effective Work	 Output Size	  Relative Time
convolveFIR							N			N×M				N+M-1		  2.98ms
efficient_convolve_downsample		N		  (N×M)/10			N/10		  0.32ms
resampler							N		  (3N×M)/10			3N/10		  0.73ms

4.Why It's Still More Efficient:
	Despite the higher absolute time:
	It's doing more work (3x upsampling)
	It's producing more output (30% more samples)
	The time increase is proportional to the additional work:
		0.32ms × 2.3 ≈ 0.73ms (matches the observed ratio)

*/