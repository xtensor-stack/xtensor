.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. raw:: html

   <style>
   .rst-content table.docutils {
       width: 100%;
       table-layout: fixed;
   }

   table.docutils .line-block {
       margin-left: 0;
       margin-bottom: 0;
   }

   table.docutils code.literal {
       color: initial;
   }

   code.docutils {
       background: initial;
   }
   </style>

Mathematical functions
======================

.. toctree::

   operators

.. table::
   :widths: 30 70

   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator+`     | identity                                 |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator-`     | opposite                                 |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator+`     | addition                                 |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator-`     | substraction                             |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator*`     | multiplication                           |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator/`     | division                                 |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator||`    | logical or                               |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator&&`    | logical and                              |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator!`     | logical not                              |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::where`         | ternary selection                        |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::any`           | return true if any value is truthy       |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::all`           | return true if all the values are truthy |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator\<`    | element-wise lesser than                 |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator\<=`   | element-wise less or equal               |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator>`     | element-wise greater than                |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator>=`    | element-wise greater or equal            |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator==`    | expression equality                      |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator!=`    | expression inequality                    |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::equal`         | element-wise equality                    |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::not_equal`     | element-wise inequality                  |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::less`          | element-wise lesser than                 |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::less_equal`    | element-wise less or equal               |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::greater`       | element-wise greater than                |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::greater_equal` | element-wise greater or equal            |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::cast`          | element-wise ``static_cast``             |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator&`     | bitwise and                              |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator|`     | bitwise or                               |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator^`     | bitwise xor                              |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator~`     | bitwise not                              |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::left_shift`    | bitwise shift left                       |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::right_shift`   | bitwise shift right                      |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator\<\<`  | bitwise shift left                       |
   +-------------------------------+------------------------------------------+
   | :cpp:func:`xt::operator\>\>`  | bitwise shift right                      |
   +-------------------------------+------------------------------------------+

.. toctree::

   index_related

.. table::
   :widths: 30 70

   +------------------------------+----------------------+
   | :cpp:func:`xt::where`        | indices selection    |
   +------------------------------+----------------------+
   | :cpp:func:`xt::nonzero`      | indices selection    |
   +------------------------------+----------------------+
   | :cpp:func:`xt::argwhere`     | indices selection    |
   +------------------------------+----------------------+
   | :cpp:func:`xt::from_indices` | biulder from indices |
   +------------------------------+----------------------+

.. toctree::

   basic_functions

.. table::
   :widths: 30 70

   +---------------------------+----------------------------------------------------+
   | :cpp:func:`xt::abs`       | absolute value                                     |
   +---------------------------+----------------------------------------------------+
   | :cpp:func:`xt::fabs`      | absolute value                                     |
   +---------------------------+----------------------------------------------------+
   | :cpp:func:`xt::fmod`      | remainder of the floating point division operation |
   +---------------------------+----------------------------------------------------+
   | :cpp:func:`xt::remainder` | signed remainder of the division operation         |
   +---------------------------+----------------------------------------------------+
   | :cpp:func:`xt::fma`       | fused multiply-add operation                       |
   +---------------------------+----------------------------------------------------+
   | :cpp:func:`xt::minimum`   | element-wise minimum                               |
   +---------------------------+----------------------------------------------------+
   | :cpp:func:`xt::maximum`   | element-wise maximum                               |
   +---------------------------+----------------------------------------------------+
   | :cpp:func:`xt::fmin`      | element-wise minimum for floating point values     |
   +---------------------------+----------------------------------------------------+
   | :cpp:func:`xt::fmax`      | element-wise maximum for floating point values     |
   +---------------------------+----------------------------------------------------+
   | :cpp:func:`xt::fdim`      | element-wise positive difference                   |
   +---------------------------+----------------------------------------------------+
   | :cpp:func:`xt::clip`      | element-wise clipping operation                    |
   +---------------------------+----------------------------------------------------+
   | :cpp:func:`xt::sign`      | element-wise indication of the sign                |
   +---------------------------+----------------------------------------------------+

.. toctree::

   exponential_functions

.. table::
   :widths: 30 70

   +-----------------------+-----------------------------------------+
   | :cpp:func:`xt::exp`   | natural exponential function            |
   +-----------------------+-----------------------------------------+
   | :cpp:func:`xt::exp2`  | base 2 exponential function             |
   +-----------------------+-----------------------------------------+
   | :cpp:func:`xt::expm1` | natural exponential function, minus one |
   +-----------------------+-----------------------------------------+
   | :cpp:func:`xt::log`   | natural logarithm function              |
   +-----------------------+-----------------------------------------+
   | :cpp:func:`xt::log2`  | base 2 logarithm function               |
   +-----------------------+-----------------------------------------+
   | :cpp:func:`xt::log10` | base 10 logarithm function              |
   +-----------------------+-----------------------------------------+
   | :cpp:func:`xt::log1p` | natural logarithm of one plus function  |
   +-----------------------+-----------------------------------------+

.. toctree::

   power_functions

.. table::
   :widths: 30 70

   +-----------------------+----------------------+
   | :cpp:func:`xt::pow`   | power function       |
   +-----------------------+----------------------+
   | :cpp:func:`xt::sqrt`  | square root function |
   +-----------------------+----------------------+
   | :cpp:func:`xt::cbrt`  | cubic root function  |
   +-----------------------+----------------------+
   | :cpp:func:`xt::hypot` | hypotenuse function  |
   +-----------------------+----------------------+

.. toctree::

   trigonometric_functions

.. table::
   :widths: 30 70

   +-----------------------+---------------------------------------------+
   | :cpp:func:`xt::sin`   | sine function                               |
   +-----------------------+---------------------------------------------+
   | :cpp:func:`xt::cos`   | cosine function                             |
   +-----------------------+---------------------------------------------+
   | :cpp:func:`xt::tan`   | tangent function                            |
   +-----------------------+---------------------------------------------+
   | :cpp:func:`xt::asin`  | arc sine function                           |
   +-----------------------+---------------------------------------------+
   | :cpp:func:`xt::acos`  | arc cosine function                         |
   +-----------------------+---------------------------------------------+
   | :cpp:func:`xt::atan`  | arc tangent function                        |
   +-----------------------+---------------------------------------------+
   | :cpp:func:`xt::atan2` | arc tangent function, determining quadrants |
   +-----------------------+---------------------------------------------+

.. toctree::

   hyperbolic_functions

.. table::
   :widths: 30 70

   +-----------------------+-------------------------------------+
   | :cpp:func:`xt::sinh`  | hyperbolic sine function            |
   +-----------------------+-------------------------------------+
   | :cpp:func:`xt::cosh`  | hyperbolic cosine function          |
   +-----------------------+-------------------------------------+
   | :cpp:func:`xt::tanh`  | hyperbolic tangent function         |
   +-----------------------+-------------------------------------+
   | :cpp:func:`xt::asinh` | inverse hyperbolic sine function    |
   +-----------------------+-------------------------------------+
   | :cpp:func:`xt::acosh` | inverse hyperbolic cosine function  |
   +-----------------------+-------------------------------------+
   | :cpp:func:`xt::atanh` | inverse hyperbolic tangent function |
   +-----------------------+-------------------------------------+

.. toctree::

   error_functions

.. table::
   :widths: 30 70

   +------------------------+-----------------------------------------+
   | :cpp:func:`xt::erf`    | error function                          |
   +------------------------+-----------------------------------------+
   | :cpp:func:`xt::erfc`   | complementary error function            |
   +------------------------+-----------------------------------------+
   | :cpp:func:`xt::tgamma` | gamma function                          |
   +------------------------+-----------------------------------------+
   | :cpp:func:`xt::lgamma` | natural logarithm of the gamma function |
   +------------------------+-----------------------------------------+

.. toctree::

   nearint_operations

.. table::
   :widths: 30 70

   +---------------------------+----------------------------------------------+
   | :cpp:func:`xt::ceil`      | nearest integers not less                    |
   +---------------------------+----------------------------------------------+
   | :cpp:func:`xt::floor`     | nearest integers not greater                 |
   +---------------------------+----------------------------------------------+
   | :cpp:func:`xt::trunc`     | nearest integers not greater in magnitude    |
   +---------------------------+----------------------------------------------+
   | :cpp:func:`xt::round`     | nearest integers, rounding away from zero    |
   +---------------------------+----------------------------------------------+
   | :cpp:func:`xt::nearbyint` | nearest integers using current rounding mode |
   +---------------------------+----------------------------------------------+
   | :cpp:func:`xt::rint`      | nearest integers using current rounding mode |
   +---------------------------+----------------------------------------------+

.. toctree::

   classif_functions

.. table::
   :widths: 30 70

   +--------------------------+----------------------------------+
   | :cpp:func:`xt::isfinite` | checks for finite values         |
   +--------------------------+----------------------------------+
   | :cpp:func:`xt::isinf`    | checks for infinite values       |
   +--------------------------+----------------------------------+
   | :cpp:func:`xt::isnan`    | checks for NaN values            |
   +--------------------------+----------------------------------+
   | :cpp:func:`xt::isclose`  | element-wise closeness detection |
   +--------------------------+----------------------------------+
   | :cpp:func:`xt::allclose` | closeness reduction              |
   +--------------------------+----------------------------------+

.. toctree::

   reducing_functions

.. table::
   :widths: 30 70

   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::sum`               | sum of elements over given axes                                     |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::prod`              | product of elements over given axes                                 |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::mean`              | mean of elements over given axes                                    |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::average`           | weighted average along the specified axis                           |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::variance`          | variance of elements over given axes                                |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::stddev`            | standard deviation of elements over given axes                      |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::diff`              | Calculate the n-th discrete difference along the given axis         |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::amax`              | amax of elements over given axes                                    |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::amin`              | amin of elements over given axes                                    |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::trapz`             | Integrate along the given axis using the composite trapezoidal rule |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::norm_l0`           | L0 pseudo-norm over given axes                                      |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::norm_l1`           | L1 norm over given axes                                             |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::norm_sq`           | Squared L2 norm over given axes                                     |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::norm_l2`           | L2 norm over given axes                                             |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::norm_linf`         | Infinity norm over given axes                                       |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::norm_lp_to_p`      | p_th power of Lp norm over given axes                               |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::norm_lp`           | Lp norm over given axes                                             |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::norm_induced_l1`   | Induced L1 norm of a matrix                                         |
   +-----------------------------------+---------------------------------------------------------------------+
   | :cpp:func:`xt::norm_induced_linf` | Induced L-infinity norm of a matrix                                 |
   +-----------------------------------+---------------------------------------------------------------------+

.. toctree::

   accumulating_functions

.. table::
   :widths: 30 70

   +-------------------------+------------------------------------------------+
   | :cpp:func:`xt::cumsum`  | Cumulative sum of elements over a given axis   |
   +-------------------------+------------------------------------------------+
   | :cpp:func:`xt::cumprod` | Cumulative product of elements over given axes |
   +-------------------------+------------------------------------------------+

.. toctree::

   nan_functions

.. table::
   :widths: 30 70

   +----------------------------+----------------------------------------------------------------------+
   | :cpp:func:`xt::nan_to_num` | Convert NaN and +/- inf to finite numbers                            |
   +----------------------------+----------------------------------------------------------------------+
   | :cpp:func:`xt::nanmin`     | Min of elements over a given axis, ignoring NaNs                     |
   +----------------------------+----------------------------------------------------------------------+
   | :cpp:func:`xt::nanmax`     | Max of elements over a given axis, ignoring NaNs                     |
   +----------------------------+----------------------------------------------------------------------+
   | :cpp:func:`xt::nansum`     | Sum of elements over a given axis, replacing NaN with 0              |
   +----------------------------+----------------------------------------------------------------------+
   | :cpp:func:`xt::nanprod`    | Product of elements over given axes, replacing NaN with 1            |
   +----------------------------+----------------------------------------------------------------------+
   | :cpp:func:`xt::nancumsum`  | Cumulative sum of elements over a given axis, replacing NaN with 0   |
   +----------------------------+----------------------------------------------------------------------+
   | :cpp:func:`xt::nancumprod` | Cumulative product of elements over given axes, replacing NaN with 1 |
   +----------------------------+----------------------------------------------------------------------+
   | :cpp:func:`xt::nanmean`    | Mean of elements over given axes, ignoring NaNs                      |
   +----------------------------+----------------------------------------------------------------------+
   | :cpp:func:`xt::nanvar`     | Variance of elements over given axes, ignoring NaNs                  |
   +----------------------------+----------------------------------------------------------------------+
   | :cpp:func:`xt::nanstd`     | Standard deviation of elements over given axes, ignoring NaNs        |
   +----------------------------+----------------------------------------------------------------------+
