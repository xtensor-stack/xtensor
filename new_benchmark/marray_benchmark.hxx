#include <benchmark/benchmark.h>
#include <string>
#include <sstream>




#define REGISTER_EXPRESSION_BENCHMARK(FUNCTION, DIM, TYPE, EXPRESSION, IMPL) \
    BENCHMARK_TEMPLATE(FUNCTION, DIM, TYPE, EXPRESSION) \
        ->AddMetaData("name",EXPRESSION::name())\
        ->ArgNames({"shape"}) \
        ->AddMetaData("group" ,"expressions") \
        ->AddMetaData("impl" ,IMPL) \
        ->AddMetaData("dimension" ,#DIM) \
        ->AddMetaData("value_type" ,#TYPE)

