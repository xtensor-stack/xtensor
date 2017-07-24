#include <benchmark/benchmark.h>
#include <string>
#include <sstream>





template<class T>
struct TypeToName;

#define TYPE_TO_NAME_IMPL(TYPE) \
template<> \
struct TypeToName<TYPE>{ \
    static std::string op(){ \
        return #TYPE; \
    } \
} 

TYPE_TO_NAME_IMPL(double);
TYPE_TO_NAME_IMPL(float);
TYPE_TO_NAME_IMPL(int);

namespace mabm{

    enum PlotType{
        Boxplot = 0,
        Plot    = 1,
        Heatmap = 2
    };





    template<std::size_t DIM, class T>
    std::string nameGen(
        const std::string & name,
        const std::string & impl
    ){
        std::stringstream ss;
        //json j;
        ss<<"name="<<name<<" "
          <<"impl="<<impl<<" "
          <<"DIM="<<DIM<<" "
          <<"T="<<TypeToName<T>::op()<<" "
        ;
        return ss.str();
    }

};






#define ARRAY_BENCH(BM_NAME, IMPL_NAME) \
template<std::size_t DIM, class T> \
struct BM_NAME##IMPL_NAME { \
    void operator()(benchmark::State& state)const; \
}; \
template<std::size_t DIM, class T> \
void BM_NAME##IMPL_NAME <DIM, T>::operator()(benchmark::State& state)const\




#define REG_ARRAY_BENCH(BM_NAME, IMPL_NAME, DIM, T) \
const auto BM_NAME##IMPL_NAME##DIM##T##___ = benchmark::RegisterBenchmark( \
    mabm::nameGen<DIM, T>( \
        #BM_NAME, \
        #IMPL_NAME \
    ).c_str(), \
    BM_NAME##IMPL_NAME<DIM, T>() \
)