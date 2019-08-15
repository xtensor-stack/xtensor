.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

xrandom
=======

Defined in ``xtensor/xrandom.hpp``

.. warning:: xtensor uses a lazy generator for random numbers. You need to assign them or use ``eval`` to keep the generated values consistent.

.. doxygenfunction:: xt::random::get_default_random_engine
   :project: xtensor

.. doxygenfunction:: xt::random::seed
   :project: xtensor

.. doxygenfunction:: xt::random::rand(const S&, T, T, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::randint(const S&, T, T, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::randn(const S&, T, T, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::binomial(const S&, T, D, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::geometric(const S&, D, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::negative_binomial(const S&, T, D, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::poisson(const S&, D, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::exponential(const S&, T, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::gamma(const S&, T, T, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::weibull(const S&, T, T, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::extreme_value(const S&, T, T, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::lognormal(const S&, T, T, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::lognormal(const S&, T, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::cauchy(const S&, T, T, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::fisher_f(const S&, T, T, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::student_t(const S&, T, E&)
   :project: xtensor

.. doxygenfunction:: xt::random::choice
   :project: xtensor

.. doxygenfunction:: xt::random::shuffle
   :project: xtensor

.. doxygenfunction:: xt::random::permutation(T, E&)
   :project: xtensor
