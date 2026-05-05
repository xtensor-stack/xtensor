/***************************************************************************
 * Comparison benchmarks: xtensor vs raw C++
 ****************************************************************************/

#include <benchmark/benchmark.h>

// Custom main for comparison benchmarks
int main(int argc, char** argv)
{
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
    {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
}
