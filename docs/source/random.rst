.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. _random:

******
Random
******

xt::random::seed
================

:ref:`xt::random::seed <random-seed-function-reference>`

Set seed for random number generator. A common practice to get a 'real' random number is to use:

.. code-block:: cpp

    #include <ctime>

    ...

    xt::random::seed(time(NULL));

xt::random::rand
================

:ref:`xt::random::rand <random-rand-function-reference>`

xt::random::randint
===================

:ref:`xt::random::randint <random-randint-function-reference>`

xt::random::randn
=================

:ref:`xt::random::randn <random-randn-function-reference>`

xt::random::binomial
====================

:ref:`xt::random::binomial <random-binomial-function-reference>`

xt::random::geometric
=====================

:ref:`xt::random::geometric <random-geometric-function-reference>`

xt::random::negative_binomial
=============================

:ref:`xt::random::negative_binomial <random-negative_binomial-function-reference>`

xt::random::poisson
===================

:ref:`xt::random::poisson <random-poisson-function-reference>`

xt::random::exponential
=======================

:ref:`xt::random::exponential <random-exponential-function-reference>`

xt::random::gamma
=================

:ref:`xt::random::gamma <random-gamma-function-reference>`

Produces (an array of) random positive floating-point values,
distributed according to the probability density:

.. math::

    P(x) = x^{\alpha-1} \frac{e^{-x / \beta}}{\beta^\alpha \; \Gamma(\alpha)}

where :math:`\alpha` is the shape (also known as :math:`k`) and :math:`\beta` the scale (also known as :math:`\theta`), and :math:`\Gamma` is the Gamma function.

.. note::

    Different from NumPy, the first argument is the shape of the output array.

.. seealso::

    *   :any:`numpy.random.gamma`
    *   `std::gamma_distribution <https://en.cppreference.com/w/cpp/numeric/random/gamma_distribution>`_
    *   `Weisstein, Eric W. "Gamma Distribution." From MathWorld – A Wolfram Web Resource. <http://mathworld.wolfram.com/GammaDistribution.html>`_
    *   `Wikipedia, "Gamma distribution". <https://en.wikipedia.org/wiki/Gamma_distribution>`_

xt::random::weibull
===================

:ref:`xt::random::weibull <random-weibull-function-reference>`

Produces (an array of) random positive floating-point values,
distributed according to the probability density:

.. math::

    P(x) = \frac{a}{b} \left( \frac{x}{b} \right)^{a - 1} e^{-(x / b)^a}

where :math:`a > 0` is the shape parameter and :math:`b > 0` the scale parameter.
In particular, a random variable is produced as

.. math::

    X = b (- \ln (U))^{1/a}

where :math:`U` is drawn from the uniform distribution (0, 1].

By default both the shape :math:`a = 1` and the scale :math:`b = 1`.
Note that you can specify only :math:`a` while choosing the default for :math:`b`.

.. note::

    Different from NumPy, the first argument is the shape of the output array.

.. seealso::

    *   :any:`numpy.random.weibull`
    *   `std::weibull_distribution <https://en.cppreference.com/w/cpp/numeric/random/weibull_distribution>`_
    *   `Wikipedia, "Weibull distribution". <https://en.wikipedia.org/wiki/Weibull_distribution>`_

xt::random::extreme_value
=========================

:ref:`xt::random::extreme_value <random-extreme_value-function-reference>`

xt::random::lognormal
=====================

:ref:`xt::random::lognormal <random-lognormal-function-reference>`

xt::random::cauchy
==================

:ref:`xt::random::cauchy <random-cauchy-function-reference>`

xt::random::fisher_f
====================

:ref:`xt::random::fisher_f <random-fisher_f-function-reference>`

xt::random::student_t
=====================

:ref:`xt::random::student_t <random-student_t-function-reference>`

xt::random::choice
==================

:ref:`xt::random::choice <random-choice-function-reference>`

xt::random::shuffle
===================

:ref:`xt::random::shuffle <random-shuffle-function-reference>`

xt::random::permutation
=======================

:ref:`xt::random::permutation <random-permutation-function-reference>`
