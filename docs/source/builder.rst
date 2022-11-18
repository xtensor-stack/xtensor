.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Expression builders
===================

*xtensor* provides functions to ease the build of common N-dimensional expressions. The expressions
returned by these functions implement the laziness of *xtensor*, that is, they don't hold any value.
Values are computed upon request.

Ones and zeros
--------------

- :cpp:func:`xt::zeros(shape) <xt::zeros>`: generates an expression containing zeros of the specified shape.
- :cpp:func:`xt::ones(shape) <xt::ones>`: generates an expression containing ones of the specified shape.
- :cpp:func:`xt::eye(shape, k=0) <xt::eye>`: generates an expression of the specified shape, with ones on the k-th diagonal.
- :cpp:func:`xt::eye(n, k = 0) <xt::eye>`: generates an expression of shape ``(n, n)`` with ones on the k-th diagonal.

Numerical ranges
----------------

- :cpp:func:`xt::arange(start=0, stop, step=1) <xt::arange>`: generates numbers evenly spaced within given half-open interval.
- :cpp:func:`xt::linspace(start, stop, num_samples) <xt::linspace>`: generates num_samples evenly spaced numbers over given interval.
- :cpp:func:`xt::logspace(start, stop, num_samples) <xt::logspace>`: generates num_samples evenly spaced on a log scale over given interval

Joining expressions
-------------------

- :cpp:func:`xt::concatenate(tuple, axis=0) <xt::concatenate>`: concatenates a list of expressions along the given axis.
- :cpp:func:`xt::stack(tuple, axis=0) <xt::stack>`: stacks a list of expressions along the given axis.
- :cpp:func:`xt::hstack(tuple) <xt::hstack>`: stacks expressions in sequence horizontally (i.e. column-wise).
- :cpp:func:`xt::vstack(tuple) <xt::vstack>`: stacks expressions in sequence vertically (i.e. row wise).

Random distributions
--------------------

.. warning:: xtensor uses a lazy generator for random numbers.
   You need to assign them or use :cpp:func:`xt::eval` to keep the generated values consistent.

- :cpp:func:`xt::random::rand(shape, lower, upper) <xt::random::rand>`: generates an expression of the specified
  shape, containing uniformly distributed random numbers in the half-open interval [lower, upper).
- :cpp:func:`xt::random::randint(shape, lower, upper) <xt::random::randint>`: generates an expression of the specified
  shape, containing uniformly distributed random integers in the half-open interval [lower, upper).
- :cpp:func:`xt::random::randn(shape, mean, std_dev) <xt::random::randn>`: generates an expression of the specified
  shape, containing numbers sampled from the Normal random number distribution.
- :cpp:func:`xt::random::binomial(shape, trials, prob) <xt::random::binomial>`: generates an expression of the specified
  shape, containing numbers sampled from the binomial random number distribution.
- :cpp:func:`xt::random::geometric(shape, prob) <xt::random::geometric>`: generates an expression of the specified shape,
  containing numbers sampled from the geometric random number distribution.
- :cpp:func:`xt::random::negative_binomial(shape, k, prob) <xt::random::negative_binomial>`: generates an expression
  of the specified shape, containing numbers sampled from the negative binomial random number distribution.
- :cpp:func:`xt::random::poisson(shape, rate) <xt::random::poisson>`: generates an expression of the specified shape,
  containing numbers sampled from the Poisson random number distribution.
- :cpp:func:`xt::random::exponential(shape, rate) <xt::random::exponential>`: generates an expression of the specified
  shape, containing numbers sampled from the exponential random number distribution.
- :cpp:func:`xt::random::gamma(shape, alpha, beta) <xt::random::gamma>`: generates an expression of the specified shape,
  containing numbers sampled from the gamma random number distribution.
- :cpp:func:`xt::random::weibull(shape, a, b) <xt::random::weibull>`: generates an expression of the specified shape,
  containing numbers sampled from the Weibull random number distribution.
- :cpp:func:`xt::random::extreme_value(shape, a, b) <xt::random::extreme_value>`: generates an expression of the
  specified shape, containing numbers sampled from the extreme value random number distribution.
- :cpp:func:`xt::random::lognormal(shape, a, b) <xt::random::lognormal>`: generates an expression of the specified
  shape, containing numbers sampled from the Log-Normal random number distribution.
- :cpp:func:`xt::random::chi_squared(shape, a, b) <xt::random::chi_squared>`: generates an expression of the specified
  shape, containing numbers sampled from the chi-squared random number distribution.
- :cpp:func:`xt::random::cauchy(shape, a, b) <xt::random::cauchy>`: generates an expression of the specified shape,
  containing numbers sampled from the Cauchy random number distribution.
- :cpp:func:`xt::random::fisher_f(shape, m, n) <xt::random::fisher_f>`: generates an expression of the specified shape,
  containing numbers sampled from the Fisher-f random number distribution.
- :cpp:func:`xt::random::student_t(shape, n) <xt::random::student_t>`: generates an expression of the specified shape,
  containing numbers sampled from the Student-t random number distribution.

Meshes
------

- :cpp:func:`xt::meshgrid(x1, x2,...) <xt::meshgrid>`: generates N-D coordinate expressions given
  one-dimensional coordinate arrays ``x1``, ``x2``...
  If specified vectors have lengths ``Ni = len(xi)``, meshgrid returns ``(N1, N2, N3,..., Nn)``-shaped arrays, with the elements
  of xi repeated to fill the matrix along the first dimension for x1, the second for x2 and so on.
