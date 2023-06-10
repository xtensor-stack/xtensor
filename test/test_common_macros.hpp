#ifndef TEST_COMMON_MACROS_HPP
#define TEST_COMMON_MACROS_HPP

#if defined(XTENSOR_DISABLE_EXCEPTIONS)
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#endif

#include "xtensor/xtensor_config.hpp"

#include "doctest/doctest.h"
#include "test_utils.hpp"

#if defined(XTENSOR_DISABLE_EXCEPTIONS)
#warning "XT_EXPECT_THROW, XT_ASSERT_THROW, XT_EXPECT_ANY_THROW and XT_ASSERT_ANY_THROW are disabled"
#define XT_EXPECT_THROW(x, y)
#define XT_ASSERT_THROW(x, y)
#define XT_EXPECT_ANY_THROW(x)
#define XT_ASSERT_ANY_THROW(x)
#define XT_EXPECT_NO_THROW(x) x;
#define XT_ASSERT_NO_THROW(x) x;

#else

#define XT_EXPECT_THROW(x, y) CHECK_THROWS_AS(x, y);
#define XT_ASSERT_THROW(x, y) REQUIRE_THROWS_AS(x, y);
#define XT_EXPECT_ANY_THROW(x) CHECK_THROWS_AS(x, std::exception);
#define XT_ASSERT_ANY_THROW(x) REQUIRE_THROWS_AS(x, std::exception);
#define XT_EXPECT_NO_THROW(x) x;
#define XT_ASSERT_NO_THROW(x) x;
#endif


#define EXPECT_NO_THROW(x) XT_EXPECT_NO_THROW(x)
#define EXPECT_THROW(x, y) XT_EXPECT_THROW(x, y)

#define TEST(A, B) TEST_CASE(#A "." #B)
#define EXPECT_EQ(A, B) CHECK_EQ(A, B)
#define EXPECT_NE(A, B) CHECK_NE(A, B)
#define EXPECT_LE(A, B) CHECK_LE(A, B)
#define EXPECT_GE(A, B) CHECK_GE(A, B)
#define EXPECT_LT(A, B) CHECK_LT(A, B)
#define EXPECT_GT(A, B) CHECK_GT(A, B)
#define EXPECT_TRUE(A) CHECK_EQ(A, true)
#define EXPECT_FALSE(A) CHECK_FALSE(A)

#define ASSERT_EQ(A, B) REQUIRE_EQ(A, B)
#define ASSERT_NE(A, B) REQUIRE_NE(A, B)
#define ASSERT_LE(A, B) REQUIRE_LE(A, B)
#define ASSERT_GE(A, B) REQUIRE_GE(A, B)
#define ASSERT_LT(A, B) REQUIRE_LT(A, B)
#define ASSERT_GT(A, B) REQUIRE_GT(A, B)
#define ASSERT_TRUE(A) REQUIRE_EQ(A, true)
#define ASSERT_FALSE(A) REQUIRE_FALSE(A)

#define EXPECT_DOUBLE_EQ(x, y) CHECK(xt::scalar_near(x, y));
#define EXPECT_TENSOR_EQ(x, y) CHECK(xt::tensor_near(x, y));

#define TEST_F(FIXTURE_CLASS, NAME) TEST_CASE_FIXTURE(FIXTURE_CLASS, #NAME)

#define HETEROGEN_PARAMETRIZED_DEFINE(ID) TEST_CASE_TEMPLATE_DEFINE(#ID, TypeParam, ID)

#define HETEROGEN_PARAMETRIZED_TEST_APPLY(ID, TEST_FUNC) \
    TEST_CASE_TEMPLATE_APPLY(ID, augment_t<std::decay_t<decltype(TEST_FUNC())>>)

#endif
