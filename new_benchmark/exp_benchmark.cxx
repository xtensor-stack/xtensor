#include "marray_benchmark.hxx"

#include <cmath> // std::pow


// xtensor 
#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>


// blitz (to compare with)
#include <blitz/array.h>



ARRAY_BENCH(expr, xt){

    // setup
    const auto s = int64_t(state.range(0));

    // todo there must be more xtensor-ish ways
    std::vector<int> shape(DIM, s);

    xt::xtensor<T, DIM> a = xt::ones<T>(shape);
    xt::xtensor<T, DIM> b = xt::ones<T>(shape);
    xt::xtensor<T, DIM> c = xt::ones<T>(shape);
    xt::xtensor<T, DIM> d = xt::ones<T>(shape);
    xt::xtensor<T, DIM> e = xt::ones<T>(shape);

    // run
    while (state.KeepRunning()){
        e = (a+b) + (c+d);
        benchmark::DoNotOptimize(e);
    }

    // how much work was done
    state.SetBytesProcessed(int64_t(state.iterations())*std::pow(s,DIM));
};

REG_ARRAY_BENCH(expr, xt, 1, int)->RangeMultiplier(2)->Range(2<<2,2<<20);
REG_ARRAY_BENCH(expr, xt, 2, int)->RangeMultiplier(2)->Range(2<<6,2<<12);




ARRAY_BENCH(expr, blitz){
    
    // setup
    const auto s = int64_t(state.range(0));
    auto a = blitz::Array<T, DIM>(s);
    auto b = blitz::Array<T, DIM>(s);
    auto c = blitz::Array<T, DIM>(s);
    auto d = blitz::Array<T, DIM>(s);
    auto e = blitz::Array<T, DIM>(s);

    a = T(1);
    b = T(1);
    c = T(1);
    d = T(1);

    // run
    while (state.KeepRunning()){
        e = (a+b) + (c+d);
        benchmark::DoNotOptimize(e);
    }

    // how much work was done
    state.SetBytesProcessed(int64_t(state.iterations())*std::pow(s,DIM));
};

REG_ARRAY_BENCH(expr, blitz, 1, int)->RangeMultiplier(2)->Range(2<<2,2<<20);
REG_ARRAY_BENCH(expr, blitz, 2, int)->RangeMultiplier(2)->Range(2<<6,2<<12);



ARRAY_BENCH(expr, explicit_code){
    // setup
    const auto s = int64_t(state.range(0));
    const auto size = int64_t(std::pow(s,DIM));

    auto a = new T[size];
    auto b = new T[size];
    auto c = new T[size];
    auto d = new T[size];
    auto e = new T[size];

    // run
    while (state.KeepRunning()){
        for(auto i=0; i<size; ++i){
            e[i] = (a[i] + b[i]) + (c[i] + d[i]);
        }

    }

    // how much work was done
    state.SetBytesProcessed(int64_t(state.iterations())*std::pow(s,DIM));
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] d;
    delete[] e;

};
REG_ARRAY_BENCH(expr, explicit_code, 1, int)->RangeMultiplier(2)->Range(2<<2,2<<20);
REG_ARRAY_BENCH(expr, explicit_code, 2, int)->RangeMultiplier(2)->Range(2<<6,2<<12);


BENCHMARK_MAIN();