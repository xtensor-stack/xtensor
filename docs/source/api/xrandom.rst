.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

xrandom
=======

Defined in ``xtensor/xrandom.hpp``

.. warning:: xtensor uses a lazy generator for random numbers. You need to assign them or use ``eval`` to keep the generated values consistent.

.. _random-get_default_random_engine-function-reference:
.. doxygenfunction:: xt::random::get_default_random_engine
   :project: xtensor

.. _random-seed-function-reference:
.. doxygenfunction:: xt::random::seed
   :project: xtensor

.. _random-rand-function-reference:
.. doxygenfunction:: xt::random::rand(const S&, T, T, E&)
   :project: xtensor

.. _random-randint-function-reference:
.. doxygenfunction:: xt::random::randint(const S&, T, T, E&)
   :project: xtensor

.. _random-randn-function-reference:
.. doxygenfunction:: xt::random::randn(const S&, T, T, E&)
   :project: xtensor

.. _random-binomial-function-reference:
.. doxygenfunction:: xt::random::binomial(const S&, T, D, E&)
   :project: xtensor

.. _random-geometric-function-reference:
.. doxygenfunction:: xt::random::geometric(const S&, D, E&)
   :project: xtensor

.. _random-negative_binomial-function-reference:
.. doxygenfunction:: xt::random::negative_binomial(const S&, T, D, E&)
   :project: xtensor

.. _random-poisson-function-reference:
.. doxygenfunction:: xt::random::poisson(const S&, D, E&)
   :project: xtensor

.. _random-exponential-function-reference:
.. doxygenfunction:: xt::random::exponential(const S&, T, E&)
   :project: xtensor

.. _random-gamma-function-reference:
.. doxygenfunction:: xt::random::gamma(const S&, T, T, E&)
   :project: xtensor

.. _random-weibull-function-reference:
.. doxygenfunction:: xt::random::weibull(const S&, T, T, E&)
   :project: xtensor

.. _random-extreme_value-function-reference:
.. doxygenfunction:: xt::random::extreme_value(const S&, T, T, E&)
   :project: xtensor

.. _random-lognormal-function-reference:
.. doxygenfunction:: xt::random::lognormal(const S&, T, T, E&)
   :project: xtensor

.. _random-cauchy-function-reference:
.. doxygenfunction:: xt::random::cauchy(const S&, T, T, E&)
   :project: xtensor

.. _random-fisher_f-function-reference:
.. doxygenfunction:: xt::random::fisher_f(const S&, T, T, E&)
   :project: xtensor

.. _random-student_t-function-reference:
.. doxygenfunction:: xt::random::student_t(const S&, T, E&)
   :project: xtensor

.. _random-choice-function-reference:
.. doxygenfunction:: xt::random::choice(const xexpression<T>&, std::size_t, bool, E&)
   :project: xtensor
.. doxygenfunction:: xt::random::choice(const xexpression<T>&, std::size_t, const xexpression<W>&, bool, E&)
   :project: xtensor

.. _random-shuffle-function-reference:
.. doxygenfunction:: xt::random::shuffle
   :project: xtensor

.. _random-permutation-function-reference:
.. doxygenfunction:: xt::random::permutation(T, E&)
   :project: xtensor
