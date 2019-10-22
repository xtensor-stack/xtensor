.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. _random:

******
Random
******

xt::random::gamma
=================

Produces (an array of) random positive floating-point values, distributed according to the probability density:

.. math::

    P(x) = x^{\alpha-1} \frac{e^{-x / \beta}}{\beta^\alpha \; \Gamma(\alpha)}

where :math:`\alpha` is the shape (also known as :math:`k`) and :math:`\beta` the scale (also known as :math:`\theta`), and :math:`\Gamma` is the Gamma function.

.. seealso::

    *   `numpy.random.gamma <https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.gamma.html#numpy.random.gamma>`_
    *   `std::gamma_distribution <https://en.cppreference.com/w/cpp/numeric/random/gamma_distribution>`_
    *   `Weisstein, Eric W. "Gamma Distribution." From MathWorld – A Wolfram Web Resource. <http://mathworld.wolfram.com/GammaDistribution.html>`_
    *   `Wikipedia, "Gamma distribution". <http://en.wikipedia.org/wiki/Gamma_distribution>`_
