#include "marray_benchmark.hxx"

#include <cmath>     // std::pow
#include <algorithm> // std::fill

// xtensor 
#include <xtensor/xtensor.hpp>
#include <xtensor/xarray.hpp>


// blitz (to compare with)
#include <blitz/array.h>


// to avoid dublicate code AND avoid erroneous benchmarks 
// we encode the actual expression in structs which are 
// used by all impls:
struct expr_2_a{
    template<class A,class B, class RES>
    static void op(const A & a, const B & b, RES & res){
        res = a + b;
    }
    static std::string name(){
        return "res = a + b";
    }
};
struct expr_2_b{
    template<class A,class B, class RES>
    static void op(const A & a, const B & b, RES & res){
        res = 2*a + 3*b + 5;
    }
    static std::string name(){
        return "res = 2*a + 3*b + 5";
    }
};

struct expr_3_a{
    template<class A,class B, class C, class RES>
    static void op(const A & a, const B & b, const C & c, RES & res){
        res = a + b + c;
    }
    static std::string name(){
        return "res = a + b + c";
    }
};
struct expr_3_b{
    template<class A,class B, class C, class RES>
    static void op(const A & a, const B & b, const C & c, RES & res){
        res = 2*a + 3*b + 5  + c;
    }
    static std::string name(){
        return "res = 2*a + 3*b + 5 + c";
    }
};





// benchmark expressions where 2 arrays 
// are involved to create a result array

template<std::size_t DIM, class T, class EXPRESSION>
void bm_xt_expr_2(benchmark::State& state) {

    const auto s = int64_t(state.range(0));
    std::vector<int> shape(DIM, s);
    xt::xtensor<T, DIM> a    = xt::ones<T>(shape);
    xt::xtensor<T, DIM> b    = xt::ones<T>(shape);
    xt::xtensor<T, DIM> res  = xt::ones<T>(shape);
    while (state.KeepRunning()) {
        EXPRESSION::op(a, b, res);  
    }
    state.SetBytesProcessed(int64_t(state.iterations())*std::pow(s,DIM));
}

template<std::size_t DIM, class T, class EXPRESSION>
void bm_blitz_expr_2(benchmark::State& state) {
    const auto s = int64_t(state.range(0));
    auto a   = blitz::Array<T, DIM>(s);
    auto b   = blitz::Array<T, DIM>(s);
    auto res = blitz::Array<T, DIM>(s);
    a = T(1);
    b = T(1);
    res =T(1);
    while (state.KeepRunning()) {
        EXPRESSION::op(a, b, res);  
    }
    state.SetBytesProcessed(int64_t(state.iterations())*std::pow(s,DIM));
}

template<std::size_t DIM, class T, class EXPRESSION>
void bm_opt_expr_2(benchmark::State& state) {
    const auto s = int64_t(state.range(0));
    const uint64_t size = std::pow(s,DIM);
    auto a =   new T[size];
    auto b =   new T[size];
    auto res = new T[size];
    // to really do the same as the array
    std::fill(a,a+size,1);
    std::fill(b,b+size,1);
    std::fill(res,res+size,1);
    while (state.KeepRunning()) {
        for(auto i=0; i<size; ++i){
            EXPRESSION::op(a[i], b[i], res[i]);
        }
    }
    delete[] a;
    delete[] b;
    delete[] res;
    state.SetBytesProcessed(int64_t(state.iterations())*s);
}


// benchmark expressions where 2 arrays  are involved to create a result array
EXPR_BENCHMARK(bm_xt_expr_2,    1, int, expr_2_a, "xtensor")->RangeMultiplier(2)->Range(2<<4, 2<<18);
EXPR_BENCHMARK(bm_xt_expr_2,    2, int, expr_2_a, "xtensor")->RangeMultiplier(2)->Range(2<<4, 2<<9);
EXPR_BENCHMARK(bm_xt_expr_2,    3, int, expr_2_a, "xtensor")->RangeMultiplier(2)->Range(2<<4, 2<<5);
EXPR_BENCHMARK(bm_xt_expr_2,    1, int, expr_2_b, "xtensor")->RangeMultiplier(2)->Range(2<<4, 2<<18);
EXPR_BENCHMARK(bm_xt_expr_2,    2, int, expr_2_b, "xtensor")->RangeMultiplier(2)->Range(2<<4, 2<<9);
EXPR_BENCHMARK(bm_xt_expr_2,    3, int, expr_2_b, "xtensor")->RangeMultiplier(2)->Range(2<<4, 2<<5);

EXPR_BENCHMARK(bm_blitz_expr_2, 1, int, expr_2_a,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<18);
EXPR_BENCHMARK(bm_blitz_expr_2, 2, int, expr_2_a,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<9);
EXPR_BENCHMARK(bm_blitz_expr_2, 3, int, expr_2_a,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<5);
EXPR_BENCHMARK(bm_blitz_expr_2, 1, int, expr_2_b,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<18);
EXPR_BENCHMARK(bm_blitz_expr_2, 2, int, expr_2_b,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<9);
EXPR_BENCHMARK(bm_blitz_expr_2, 3, int, expr_2_b,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<5);

EXPR_BENCHMARK(bm_opt_expr_2, 1, int, expr_2_a,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<18);
EXPR_BENCHMARK(bm_opt_expr_2, 2, int, expr_2_a,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<9);
EXPR_BENCHMARK(bm_opt_expr_2, 3, int, expr_2_a,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<5);
EXPR_BENCHMARK(bm_opt_expr_2, 1, int, expr_2_b,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<18);
EXPR_BENCHMARK(bm_opt_expr_2, 2, int, expr_2_b,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<9);
EXPR_BENCHMARK(bm_opt_expr_2, 3, int, expr_2_b,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<5);






// benchmark expressions where 3 arrays 
// are involved to create a result array

template<std::size_t DIM, class T, class EXPRESSION>
void bm_xt_expr_3(benchmark::State& state) {

    const auto s = int64_t(state.range(0));
    std::vector<int> shape(DIM, s);
    xt::xtensor<T, DIM> a    = xt::ones<T>(shape);
    xt::xtensor<T, DIM> b    = xt::ones<T>(shape);
    xt::xtensor<T, DIM> c    = xt::ones<T>(shape);
    xt::xtensor<T, DIM> res  = xt::ones<T>(shape);
    while (state.KeepRunning()) {
        EXPRESSION::op(a, b,  c, res);  
    }
    state.SetBytesProcessed(int64_t(state.iterations())*std::pow(s,DIM));
}

template<std::size_t DIM, class T, class EXPRESSION>
void bm_blitz_expr_3(benchmark::State& state) {
    const auto s = int64_t(state.range(0));
    auto a   = blitz::Array<T, DIM>(s);
    auto b   = blitz::Array<T, DIM>(s);
    auto c   = blitz::Array<T, DIM>(s);
    auto res = blitz::Array<T, DIM>(s);
    a = T(1);
    b = T(1);
    c = T(1);
    res = T(1);
    while (state.KeepRunning()) {
        EXPRESSION::op(a, b, c, res);  
    }
    state.SetBytesProcessed(int64_t(state.iterations())*std::pow(s,DIM));
}

template<std::size_t DIM, class T, class EXPRESSION>
void bm_opt_expr_3(benchmark::State& state) {
    const auto s = int64_t(state.range(0));
    const uint64_t size = std::pow(s,DIM);
    auto a =   new T[size];
    auto b =   new T[size];
    auto c =   new T[size];
    auto res = new T[size];
    // to really do the same as the array
    std::fill(a,a+size,1);
    std::fill(b,b+size,1);
    std::fill(c,c+size,1);
    std::fill(res,res+size,1);
    while (state.KeepRunning()) {
        for(auto i=0; i<size; ++i){
            EXPRESSION::op(a[i], b[i], c[i], res[i]);
        }
    }
    delete[] a;
    delete[] b;
    delete[] c;
    delete[] res;
    state.SetBytesProcessed(int64_t(state.iterations())*s);
}



// benchmark expressions where 2 arrays  are involved to create a result array
EXPR_BENCHMARK(bm_xt_expr_3,    1, int, expr_3_a, "xtensor")->RangeMultiplier(2)->Range(2<<4, 2<<18);
EXPR_BENCHMARK(bm_xt_expr_3,    2, int, expr_3_a, "xtensor")->RangeMultiplier(2)->Range(2<<4, 2<<9);
EXPR_BENCHMARK(bm_xt_expr_3,    3, int, expr_3_a, "xtensor")->RangeMultiplier(2)->Range(2<<4, 2<<5);
EXPR_BENCHMARK(bm_xt_expr_3,    1, int, expr_3_b, "xtensor")->RangeMultiplier(2)->Range(2<<4, 2<<18);
EXPR_BENCHMARK(bm_xt_expr_3,    2, int, expr_3_b, "xtensor")->RangeMultiplier(2)->Range(2<<4, 2<<9);
EXPR_BENCHMARK(bm_xt_expr_3,    3, int, expr_3_b, "xtensor")->RangeMultiplier(2)->Range(2<<4, 2<<5);

EXPR_BENCHMARK(bm_blitz_expr_3, 1, int, expr_3_a,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<18);
EXPR_BENCHMARK(bm_blitz_expr_3, 2, int, expr_3_a,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<9);
EXPR_BENCHMARK(bm_blitz_expr_3, 3, int, expr_3_a,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<5);
EXPR_BENCHMARK(bm_blitz_expr_3, 1, int, expr_3_b,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<18);
EXPR_BENCHMARK(bm_blitz_expr_3, 2, int, expr_3_b,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<9);
EXPR_BENCHMARK(bm_blitz_expr_3, 3, int, expr_3_b,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<5);

EXPR_BENCHMARK(bm_opt_expr_3, 1, int, expr_3_a,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<18);
EXPR_BENCHMARK(bm_opt_expr_3, 2, int, expr_3_a,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<9);
EXPR_BENCHMARK(bm_opt_expr_3, 3, int, expr_3_a,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<5);
EXPR_BENCHMARK(bm_opt_expr_3, 1, int, expr_3_b,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<18);
EXPR_BENCHMARK(bm_opt_expr_3, 2, int, expr_3_b,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<9);
EXPR_BENCHMARK(bm_opt_expr_3, 3, int, expr_3_b,   "blitz")->RangeMultiplier(2)->Range(2<<4, 2<<5);








BENCHMARK_MAIN();