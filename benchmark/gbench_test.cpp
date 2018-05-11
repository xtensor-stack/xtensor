#include <benchmark/benchmark.h>
#include <string>

namespace xt {
	template <class V>
	inline void init_benchmark_data(V& lhs, V& rhs, std::size_t size0, std::size_t size1)
	{
	    using T = typename V::value_type;
	    for (std::size_t i = 0; i < size0; ++i)
	    {
	        for (std::size_t j = 0; j < size1; ++j)
	        {
	            lhs(i, j) = T(0.5) * T(j) / T(j + 1) + std::sqrt(T(i)) * T(9.) / T(size1);
	            rhs(i, j) = T(10.2) / T(i + 2) + T(0.25) * T(j);
	        }
	    }
	}

	template <class V>
	inline void init_xtensor_benchmark(V& lhs, V& rhs, V& res,
	                                   std::size_t size0, size_t size1)
	{
	    lhs.resize({ size0, size1 });
	    rhs.resize({ size0, size1 });
	    res.resize({ size0, size1 });
	    init_benchmark_data(lhs, rhs, size0, size1);
	}

    template <class E>
    inline auto benchmark_xtensor(const E& x, const E& y, E& res, std::size_t count)
    {
        duration_type t_res = duration_type::max();
        for (std::size_t i = 0; i < count; ++i)
        {
            auto start = std::chrono::steady_clock::now();
            xt::noalias(res) = 3 * x - 2 * y;
            auto end = std::chrono::steady_clock::now();
            auto tmp = end - start;
            t_res = tmp < t_res ? tmp : t_res;
        }
        return t_res;
    }
}

static void BM_StringCreation(benchmark::State& state) {
  for (auto _ : state)
    std::string empty_string;
}
// Register the function as a benchmark
BENCHMARK(BM_StringCreation);

// Define another benchmark
static void BM_StringCopy(benchmark::State& state) {
  std::string x = "hello";
  for (auto _ : state)
    std::string copy(x);
}
BENCHMARK(BM_StringCopy);

BENCHMARK_MAIN();