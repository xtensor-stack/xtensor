/***************************************************************************
 * Comparison benchmarks: xtensor vs raw C++
 ****************************************************************************/

#include <iostream>

#include <benchmark/benchmark.h>

// Custom main for comparison benchmarks
int main(int argc, char** argv)
{
    std::cout << "=== COMPARISON BENCHMARKS: xtensor vs raw C++ ===" << std::endl;
    std::cout << "Each benchmark runs multiple implementations for direct comparison\n" << std::endl;

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
    {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
}
