#ifdef __clang__
#if __has_feature(memory_sanitizer)
// Suppress MSan false positives in doctest's String union comparison.
// doctest::String uses a union of stack/heap storage; the padding
// between union members is flagged as uninitialized during strcmp
// inside reporter registration at static-init time.
// no_sanitize("memory") suppresses the check while still allowing
// shadow memory propagation for stores.
#pragma clang attribute push(__attribute__((no_sanitize("memory"))), apply_to = function)
#endif
#endif

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#if defined(XTENSOR_DISABLE_EXCEPTIONS)
#define DOCTEST_CONFIG_NO_EXCEPTIONS_BUT_WITH_ALL_ASSERTS
#endif
#include "doctest/doctest.h"

#ifdef __clang__
#if __has_feature(memory_sanitizer)
#pragma clang attribute pop
#endif
#endif
