#ifndef TEST_COMMON_MACROS_HPP
#define TEST_COMMON_MACROS_HPP

#include "gtest/gtest.h"
#include "xtensor/xtensor_config.hpp"

#if defined(XTENSOR_DISABLE_EXCEPTIONS)

#define XT_EXPECT_THROW(x, y) EXPECT_DEATH_IF_SUPPORTED(x, "");
#define XT_ASSERT_THROW(x, y) ASSERT_DEATH_IF_SUPPORTED(x, "");
#define XT_EXPECT_ANY_THROW(x) EXPECT_DEATH_IF_SUPPORTED(x, "");
#define XT_ASSERT_ANY_THROW(x) ASSERT_DEATH_IF_SUPPORTED(x, "");
#define XT_EXPECT_NO_THROW(x) x;
#define XT_ASSERT_NO_THROW(x) x;

#else

#define XT_EXPECT_THROW(x, y) EXPECT_THROW(x, y);
#define XT_ASSERT_THROW(x, y) EXPECT_THROW(x, y);
#define XT_EXPECT_ANY_THROW(x) EXPECT_ANY_THROW(x);
#define XT_ASSERT_ANY_THROW(x) ASSERT_ANY_THROW(x);
#define XT_EXPECT_NO_THROW(x) EXPECT_NO_THROW(x);
#define XT_ASSERT_NO_THROW(x) ASSERT_NO_THROW(x);
#endif

#endif
