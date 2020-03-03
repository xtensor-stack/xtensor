
#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "gtest/gtest.h"


namespace xt {

template<typename OutType, typename CreateType>
xtensor<OutType, 2> get(const size_t vShape) {
    xtensor<CreateType, 2> vTensor = ones<CreateType>({ vShape, vShape });
    return vTensor;
}

TEST(InvalidWriteOfSizeTest, FloatInt8ValgrindFail) {
    using CreateType = float;
    using OutType = uint8_t;

    xtensor<OutType, 2> vExpected = ones<CreateType>({ 1024, 1024 });
    xtensor<OutType, 2> vResult = get<OutType, CreateType>(vExpected.shape(0));
    EXPECT_EQ(vResult, vExpected);
}

TEST(InvalidWriteOfSizeTest, DoubleInt8ValgrindFail) {
    using CreateType = double;
    using OutType = uint8_t;

    xtensor<OutType, 2> vExpected = ones<CreateType>({ 1024, 1024 });
    xtensor<OutType, 2> vResult = get<OutType, CreateType>(vExpected.shape(0));
    EXPECT_EQ(vResult, vExpected);
}

TEST(InvalidWriteOfSizeTest, DoubleInt16ValgrindFail) {
    using CreateType = double;
    using OutType = uint16_t;

    xtensor<OutType, 2> vExpected = ones<CreateType>({ 1024, 1024 });
    xtensor<OutType, 2> vResult = get<OutType, CreateType>(vExpected.shape(0));
    EXPECT_EQ(vResult, vExpected);
}

// These do not produce any errors with valgrind
TEST(InvalidWriteOfSizeTest, FloatInt16ValgrindSuccess) {
    using CreateType = float;
    using OutType = uint16_t;

    xtensor<OutType, 2> vExpected = ones<CreateType>({ 1024, 1024 });
    xtensor<OutType, 2> vResult = get<OutType, CreateType>(vExpected.shape(0));
    EXPECT_EQ(vResult, vExpected);
}

TEST(InvalidWriteOfSizeTest, DoubleInt32ValgrindSuccess) {
    using CreateType = double;
    using OutType = uint32_t;

    xtensor<OutType, 2> vExpected = ones<CreateType>({ 1024, 1024 });
    xtensor<OutType, 2> vResult = get<OutType, CreateType>(vExpected.shape(0));
    EXPECT_EQ(vResult, vExpected);
}

TEST(InvalidWriteOfSizeTest, DoubleFloatValgrindSuccess) {
    using CreateType = double;
    using OutType = float;

    xtensor<OutType, 2> vExpected = ones<CreateType>({ 1024, 1024 });
    xtensor<OutType, 2> vResult = get<OutType, CreateType>(vExpected.shape(0));
    EXPECT_EQ(vResult, vExpected);
}

}