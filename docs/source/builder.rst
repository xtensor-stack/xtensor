.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Expression builders
===================

`xtensor` provides functions to ease the build of common N-dimensional expressions. The expressions
returned by these functions implement the laziness of `xtensor`, that is, they don't hold any value.
Values are computed upon request.

Ones and zeros
--------------

- ``zeros(shape)``: generates an expression containing zeros of the specified shape.
- ``ones(shape)``: generates an expression containing ones of the specified shape.
- ``eye(shape, k=0)``: generates an expression of the specified shape, with ones on the k-th diagonal.
- ``eye(n, k = 0)``: generates an expression of shape ``(n, n)`` with ones on the k-th diagonal.

Numerical ranges
----------------

- ``arange(start=0, stop, step=1)``: generates numbers evenly spaced within given half-open interval.
- ``linspace(start, stop, num_samples)``: generates num_samples evenly spaced numbers over given interval.
- ``logspace(start, stop, num_samples)``: generates num_samples evenly spaced on a log scale over given interval

Joining expressions
-------------------

- ``concatenate(tuple, axis=0)``: concatenates a list of expressions along the given axis.
- ``stack(tuple, axis=0)``: stacks a list of expressions along the given axis.
- ``hstack(tuple)``: stacks expressions in seqeunce horizontally (i.e. column wise).
- ``vstack(tuple)``: stacks expressions in seqeunce vertically (i.e. row wise).

Random distributions
--------------------

.. warning:: xtensor uses a lazy generator for random numbers. You need to assign them or use ``eval`` to keep the generated values consistent.

- ``rand(shape, lower, upper)``: generates an expression of the specified shape, containing uniformly
  distributed random numbers in the half-open interval [lower, upper).
- ``randint(shape, lower, upper)``: generates an expression of the specified shape, containing uniformly
  distributed random integers in the half-open interval [lower, upper).
- ``randn(shape, mean, std_dev)``: generates an expression of the specified shape, containing numbers
  sampled from the Normal random number distribution.
- ``binomial(shape, trials, prob)``: generates an expression of the specified shape, containing numbers
  sampled from the binomial random number distribution.
- ``geometric(shape, prob)``: generates an expression of the specified shape, containing numbers
  sampled from the geometric random number distribution.
- ``negative_binomial(shape, k, prob)``: generates an expression of the specified shape, containing numbers
  sampled from the negative binomial random number distribution.
- ``poisson(shape, rate)``: generates an expression of the specified shape, containing numbers
  sampled from the Poisson random number distribution.
- ``exponential(shape, rate)``: generates an expression of the specified shape, containing numbers
  sampled from the exponential random number distribution.
- ``gamma(shape, alpha, beta)``: generates an expression of the specified shape, containing numbers
  sampled from the gamma random number distribution.
- ``weibull(shape, a, b)``: generates an expression of the specified shape, containing numbers
  sampled from the Weibull random number distribution.
- ``extreme_value(shape, a, b)``: generates an expression of the specified shape, containing numbers
  sampled from the extreme value random number distribution.
- ``lognormal(shape, a, b)``: generates an expression of the specified shape, containing numbers
  sampled from the Log-Normal random number distribution.
- ``chi_squared(shape, a, b)``: generates an expression of the specified shape, containing numbers
  sampled from the chi-squared random number distribution.
- ``cauchy(shape, a, b)``: generates an expression of the specified shape, containing numbers
  sampled from the Cauchy random number distribution.
- ``fisher_f(shape, m, n)``: generates an expression of the specified shape, containing numbers
  sampled from the Fisher-f random number distribution.
- ``student_t(shape, n)``: generates an expression of the specified shape, containing numbers
  sampled from the Student-t random number distribution.

Meshes
------

- ``meshgrid(x1, x2,...)```: generates N-D coordinate expressions given one-dimensional coordinate arrays ``x1``, ``x2``...
  If specified vectors have lengths ``Ni = len(xi)``, meshgrid returns ``(N1, N2, N3,..., Nn)``-shaped arrays, with the elements
  of xi repeated to fill the matrix along the first dimension for x1, the second for x2 and so on.

