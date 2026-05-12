#ifndef XTENSOR_DOCTEST_WRAPPER_HPP
#define XTENSOR_DOCTEST_WRAPPER_HPP

// Suppress MSan false positives in doctest's String union comparison.
// doctest::String uses a union of stack/heap storage; the padding
// between union members is flagged as uninitialized during strcmp
// inside reporter registration at static-init time, and during
// strlen when constructing exception messages.
// no_sanitize("memory") suppresses the checks while still allowing
// shadow memory propagation for stores.
#ifdef __clang__
#if __has_feature(memory_sanitizer)
#pragma clang attribute push(__attribute__((no_sanitize("memory"))), apply_to = function)
#endif
#endif

#include <doctest/doctest.h>

#ifdef __clang__
#if __has_feature(memory_sanitizer)
#pragma clang attribute pop
#endif
#endif

#endif // XTENSOR_DOCTEST_WRAPPER_HPP
