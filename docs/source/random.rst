.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. _random:

******
Random
******

:cpp:func:`xt::random::seed`
============================

Set seed for random number generator. A common practice to get a 'real' random number is to use:

.. code-block:: cpp

    #include <ctime>

    ...

    xt::random::seed(time(NULL));

:cpp:func:`xt::random::rand`
============================

:cpp:func:`xt::random::randint`
===============================

:cpp:func:`xt::random::randn`
=============================

:cpp:func:`xt::random::binomial`
================================

:cpp:func:`xt::random::geometric`
=================================

:cpp:func:`xt::random::negative_binomial`
=========================================

:cpp:func:`xt::random::poisson`
===============================

:cpp:func:`xt::random::exponential`
===================================

:cpp:func:`xt::random::gamma`
=============================

Produces (an array of) random positive floating-point values,
distributed according to the probability density:

.. math::

    P(x) = x^{\alpha-1} \frac{e^{-x / \beta}}{\beta^\alpha \; \Gamma(\alpha)}

where :math:`\alpha` is the shape (also known as :math:`k`) and :math:`\beta` the scale
(also known as :math:`\theta`), and :math:`\Gamma` is the Gamma function.

.. note::

    Different from NumPy, the first argument is the shape of the output array.

.. seealso::

    *   :any:`numpy.random.gamma`
    *   `std::gamma_distribution <https://en.cppreference.com/w/cpp/numeric/random/gamma_distribution>`_
    *   `Weisstein, Eric W. "Gamma Distribution." From MathWorld – A Wolfram Web Resource. <http://mathworld.wolfram.com/GammaDistribution.html>`_
    *   `Wikipedia, "Gamma distribution". <https://en.wikipedia.org/wiki/Gamma_distribution>`_

:cpp:func:`xt::random::weibull`
===============================

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

:cpp:func:`xt::random::extreme_value`
=====================================

:cpp:func:`xt::random::lognormal`
=================================

:cpp:func:`xt::random::cauchy`
==============================

:cpp:func:`xt::random::fisher_f`
================================

:cpp:func:`xt::random::student_t`
=================================

:cpp:func:`xt::random::choice`
==============================

:cpp:func:`xt::random::shuffle`
===============================

:cpp:func:`xt::random::permutation`
===================================
