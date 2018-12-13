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

+-----------------------------------------+------------------------------------------+
| :ref:`operator+ <identity-op-ref>`      | identity                                 |
+-----------------------------------------+------------------------------------------+
| :ref:`operator- <neg-op-ref>`           | opposite                                 |
+-----------------------------------------+------------------------------------------+
| :ref:`operator+ <plus-op-ref>`          | addition                                 |
+-----------------------------------------+------------------------------------------+
| :ref:`operator- <minus-op-ref>`         | substraction                             |
+-----------------------------------------+------------------------------------------+
| :ref:`operator* <mul-op-ref>`           | multiplication                           |
+-----------------------------------------+------------------------------------------+
| :ref:`operator/ <div-op-ref>`           | division                                 |
+-----------------------------------------+------------------------------------------+
| :ref:`operator|| <or-op-ref>`           | logical or                               |
+-----------------------------------------+------------------------------------------+
| :ref:`operator&& <and-op-ref>`          | logical and                              |
+-----------------------------------------+------------------------------------------+
| :ref:`operator! <not-op-ref>`           | logical not                              |
+-----------------------------------------+------------------------------------------+
| :ref:`where <where-op-ref>`             | ternary selection                        |
+-----------------------------------------+------------------------------------------+
| :ref:`any <any-op-ref>`                 | return true if any value is truthy       |
+-----------------------------------------+------------------------------------------+
| :ref:`all <all-op-ref>`                 | return true if all the values are truthy |
+-----------------------------------------+------------------------------------------+
| :ref:`operator\< <less-op-ref>`         | element-wise lesser than                 |
+-----------------------------------------+------------------------------------------+
| :ref:`operator\<= <less-eq-op-ref>`     | element-wise less or equal               |
+-----------------------------------------+------------------------------------------+
| :ref:`operator> <greater-op-ref>`       | element-wise greater than                |
+-----------------------------------------+------------------------------------------+
| :ref:`operator>= <greater-eq-op-ref>`   | element-wise greater or equal            |
+-----------------------------------------+------------------------------------------+
| :ref:`operator== <equal-op-ref>`        | expression equality                      |
+-----------------------------------------+------------------------------------------+
| :ref:`operator!= <nequal-op-ref>`       | expression inequality                    |
+-----------------------------------------+------------------------------------------+
| :ref:`equal <equal-fn-ref>`             | element-wise equality                    |
+-----------------------------------------+------------------------------------------+
| :ref:`not_equal <nequal-fn-ref>`        | element-wise inequality                  |
+-----------------------------------------+------------------------------------------+
| :ref:`less <less-fn-ref>`               | element-wise lesser than                 |
+-----------------------------------------+------------------------------------------+
| :ref:`less_equal <less-eq-fn-ref>`      | element-wise less or equal               |
+-----------------------------------------+------------------------------------------+
| :ref:`greater <greater-fn-ref>`         | element-wise greater than                |
+-----------------------------------------+------------------------------------------+
| :ref:`greater_equal <greate-eq-fn-ref>` | element-wise greater or equal            |
+-----------------------------------------+------------------------------------------+
| :ref:`cast <cast-ref>`                  | element-wise `static_cast`               |
+-----------------------------------------+------------------------------------------+
| :ref:`operator& <bitwise-and-op-ref>`   | bitwise and                              |
+-----------------------------------------+------------------------------------------+
| :ref:`operator| <bitwise-or-op-ref>`    | bitwise or                               |
+-----------------------------------------+------------------------------------------+
| :ref:`operator^ <bitwise-xor-op-ref>`   | bitwise xor                              |
+-----------------------------------------+------------------------------------------+
| :ref:`operator~ <bitwise-not-op-ref>`   | bitwise not                              |
+-----------------------------------------+------------------------------------------+
| :ref:`left_shift <left-shift-fn-ref>`   | bitwise shift left                       |
+-----------------------------------------+------------------------------------------+
| :ref:`right_shift <right-shift-fn-ref>` | bitwise shift right                      |
+-----------------------------------------+------------------------------------------+
| :ref:`operator\<\< <left-sh-op-ref>`    | bitwise shift left                       |
+-----------------------------------------+------------------------------------------+
| :ref:`operator\>\> <right-sh-op-ref>`   | bitwise shift right                      |
+-----------------------------------------+------------------------------------------+

.. toctree::

   index_related

+-----------------------------------------+------------------------------------------+
| :ref:`where <wherec-op-ref>`            | indices selection                        |
+-----------------------------------------+------------------------------------------+
| :ref:`nonzero <nonzero-op-ref>`         | indices selection                        |
+-----------------------------------------+------------------------------------------+
| :ref:`argwhere <argwhere-op-ref>`       | indices selection                        |
+-----------------------------------------+------------------------------------------+
| :ref:`from_indices <frindices-op-ref>`  | biulder from indices                     |
+-----------------------------------------+------------------------------------------+

.. toctree::

   basic_functions

+---------------------------------------+----------------------------------------------------+
| :ref:`abs <abs-function-reference>`   | absolute value                                     |
+---------------------------------------+----------------------------------------------------+
| :ref:`fabs <fabs-function-reference>` | absolute value                                     |
+---------------------------------------+----------------------------------------------------+
| :ref:`fmod <fmod-function-reference>` | remainder of the floating point division operation |
+---------------------------------------+----------------------------------------------------+
| :ref:`remainder <remainder-func-ref>` | signed remainder of the division operation         |
+---------------------------------------+----------------------------------------------------+
| :ref:`fma <fma-function-reference>`   | fused multiply-add operation                       |
+---------------------------------------+----------------------------------------------------+
| :ref:`minimum <minimum-func-ref>`     | element-wise minimum                               |
+---------------------------------------+----------------------------------------------------+
| :ref:`maximum <maximum-func-ref>`     | element-wise maximum                               |
+---------------------------------------+----------------------------------------------------+
| :ref:`fmin <fmin-function-reference>` | element-wise minimum for floating point values     |
+---------------------------------------+----------------------------------------------------+
| :ref:`fmax <fmax-function-reference>` | element-wise maximum for floating point values     |
+---------------------------------------+----------------------------------------------------+
| :ref:`fdim <fdim-function-reference>` | element-wise positive difference                   |
+---------------------------------------+----------------------------------------------------+
| :ref:`clip <clip-function-reference>` | element-wise clipping operation                    |
+---------------------------------------+----------------------------------------------------+
| :ref:`sign <sign-function-reference>` | element-wise indication of the sign                |
+---------------------------------------+----------------------------------------------------+

.. toctree::

   exponential_functions

+---------------------------------------+----------------------------------------------------+
| :ref:`exp <exp-function-reference>`   | natural exponential function                       |
+---------------------------------------+----------------------------------------------------+
| :ref:`exp2 <exp2-function-reference>` | base 2 exponential function                        |
+---------------------------------------+----------------------------------------------------+
| :ref:`expm1 <expm1-func-ref>`         | natural exponential function, minus one            |
+---------------------------------------+----------------------------------------------------+
| :ref:`log <log-function-reference>`   | natural logarithm function                         |
+---------------------------------------+----------------------------------------------------+
| :ref:`log2 <log2-function-reference>` | base 2 logarithm function                          |
+---------------------------------------+----------------------------------------------------+
| :ref:`log10 <log10-func-ref>`         | base 10 logarithm function                         |
+---------------------------------------+----------------------------------------------------+
| :ref:`log1p <log1p-func-ref>`         | natural logarithm of one plus function             |
+---------------------------------------+----------------------------------------------------+

.. toctree::

   power_functions

+---------------------------------------+----------------------------------------------------+
| :ref:`pow <pow-function-reference>`   | power function                                     |
+---------------------------------------+----------------------------------------------------+
| :ref:`sqrt <sqrt-function-reference>` | square root function                               |
+---------------------------------------+----------------------------------------------------+
| :ref:`cbrt <cbrt-function-reference>` | cubic root function                                |
+---------------------------------------+----------------------------------------------------+
| :ref:`hypot <hypot-func-ref>`         | hypotenuse function                                |
+---------------------------------------+----------------------------------------------------+

.. toctree::

   trigonometric_functions

+---------------------------------------+----------------------------------------------------+
| :ref:`sin <sin-function-reference>`   | sine function                                      |
+---------------------------------------+----------------------------------------------------+
| :ref:`cos <cos-function-reference>`   | cosine function                                    |
+---------------------------------------+----------------------------------------------------+
| :ref:`tan <tan-function-reference>`   | tangent function                                   |
+---------------------------------------+----------------------------------------------------+
| :ref:`asin <asin-function-reference>` | arc sine function                                  |
+---------------------------------------+----------------------------------------------------+
| :ref:`acos <acos-function-reference>` | arc cosine function                                |
+---------------------------------------+----------------------------------------------------+
| :ref:`atan <atan-function-reference>` | arc tangent function                               |
+---------------------------------------+----------------------------------------------------+
| :ref:`atan2 <atan2-func-ref>`         | arc tangent function, determining quadrants        |
+---------------------------------------+----------------------------------------------------+

.. toctree::

   hyperbolic_functions

+---------------------------------------+----------------------------------------------------+
| :ref:`sinh <sinh-function-reference>` | hyperbolic sine function                           |
+---------------------------------------+----------------------------------------------------+
| :ref:`cosh <cosh-function-reference>` | hyperbolic cosine function                         |
+---------------------------------------+----------------------------------------------------+
| :ref:`tanh <tanh-function-reference>` | hyperbolic tangent function                        |
+---------------------------------------+----------------------------------------------------+
| :ref:`asinh <asinh-func-ref>`         | inverse hyperbolic sine function                   |
+---------------------------------------+----------------------------------------------------+
| :ref:`acosh <acosh-func-ref>`         | inverse hyperbolic cosine function                 |
+---------------------------------------+----------------------------------------------------+
| :ref:`atanh <atanh-func-ref>`         | inverse hyperbolic tangent function                |
+---------------------------------------+----------------------------------------------------+

.. toctree::

   error_functions

+---------------------------------------+----------------------------------------------------+
| :ref:`erf <erf-function-reference>`   | error function                                     |
+---------------------------------------+----------------------------------------------------+
| :ref:`erfc <erfc-function-reference>` | complementary error function                       |
+---------------------------------------+----------------------------------------------------+
| :ref:`tgamma <tgamma-func-ref>`       | gamma function                                     |
+---------------------------------------+----------------------------------------------------+
| :ref:`lgamma <lgamma-func-ref>`       | natural logarithm of the gamma function            |
+---------------------------------------+----------------------------------------------------+

.. toctree::

   nearint_operations

+---------------------------------------+----------------------------------------------------+
| :ref:`ceil <ceil-function-reference>` | nearest integers not less                          |
+---------------------------------------+----------------------------------------------------+
| :ref:`floor <floor-func-ref>`         | nearest integers not greater                       |
+---------------------------------------+----------------------------------------------------+
| :ref:`trunc <trunc-func-ref>`         | nearest integers not greater in magnitude          |
+---------------------------------------+----------------------------------------------------+
| :ref:`round <round-func-ref>`         | nearest integers, rounding away from zero          |
+---------------------------------------+----------------------------------------------------+
| :ref:`nearbyint <nearbyint-func-ref>` | nearest integers using current rounding mode       |
+---------------------------------------+----------------------------------------------------+
| :ref:`rint <rint-function-reference>` | nearest integers using current rounding mode       |
+---------------------------------------+----------------------------------------------------+

.. toctree::

   classif_functions

+---------------------------------------+----------------------------------------------------+
| :ref:`isfinite <isfinite-func-ref>`   | checks for finite values                           |
+---------------------------------------+----------------------------------------------------+
| :ref:`isinf <isinf-func-ref>`         | checks for infinite values                         |
+---------------------------------------+----------------------------------------------------+
| :ref:`isnan <isnan-func-ref>`         | checks for NaN values                              |
+---------------------------------------+----------------------------------------------------+
| :ref:`isclose <isclose-func-ref>`     | element-wise closeness detection                   |
+---------------------------------------+----------------------------------------------------+
| :ref:`allclose <allclose-func-ref>`   | closeness reduction                                |
+---------------------------------------+----------------------------------------------------+

.. toctree::

   reducing_functions

+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`sum <sum-function-reference>`           | sum of elements over given axes                                     |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`prod <prod-function-reference>`         | product of elements over given axes                                 |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`mean <mean-function-reference>`         | mean of elements over given axes                                    |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`variance <variance-function-reference>` | variance of elements over given axes                                |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`stddev <stddev-function-reference>`     | standard deviation of elements over given axes                      |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`diff <diff-function-reference>`         | Calculate the n-th discrete difference along the given axis         |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`amax <amax-function-reference>`         | amax of elements over given axes                                    |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`amin <amin-function-reference>`         | amin of elements over given axes                                    |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`trapz <trapz-function-reference>`       | Integrate along the given axis using the composite trapezoidal rule |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`norm_l0 <norm-l0-func-ref>`             | L0 pseudo-norm over given axes                                      |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`norm_l1 <norm-l1-func-ref>`             | L1 norm over given axes                                             |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`norm_sq <norm-sq-func-ref>`             | Squared L2 norm over given axes                                     |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`norm_l2 <norm-l2-func-ref>`             | L2 norm over given axes                                             |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`norm_linf <norm-linf-func-ref>`         | Infinity norm over given axes                                       |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`norm_lp_to_p <nlptop-func-ref>`         | p_th power of Lp norm over given axes                               |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`norm_lp <norm-lp-func-ref>`             | Lp norm over given axes                                             |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`norm_induced_l1 <nind-l1-ref>`          | Induced L1 norm of a matrix                                         |
+-----------------------------------------------+---------------------------------------------------------------------+
| :ref:`norm_induced_linf <nilinf-ref>`         | Induced L-infinity norm of a matrix                                 |
+-----------------------------------------------+---------------------------------------------------------------------+

.. toctree::

   accumulating_functions

+---------------------------------------------+-------------------------------------------------+
| :ref:`cumsum <cumsum-function-reference>`   | cumulative sum of elements over a given axis    |
+---------------------------------------------+-------------------------------------------------+
| :ref:`cumprod <cumprod-function-reference>` | cumulative product of elements over given axes  |
+---------------------------------------------+-------------------------------------------------+

.. toctree::

   nan_functions

+---------------------------------------------------+------------------------------------------------------------+
| :ref:`nan_to_num <nan-to-num-function-reference>` | convert NaN and +/- inf to finite numbers                  |
+---------------------------------------------------+------------------------------------------------------------+
| :ref:`nansum <nansum-function-reference>`         | sum of elements over a given axis, replacing NaN with 0    |
+---------------------------------------------------+------------------------------------------------------------+
| :ref:`nanprod <nanprod-function-reference>`       | product of elements over given axes, replacing NaN with 1  |
+---------------------------------------------------+------------------------------------------------------------+
| :ref:`nancumsum <nancumsum-function-reference>`   | cumsum of elements over a given axis, replacing NaN with 0 |
+---------------------------------------------------+------------------------------------------------------------+
| :ref:`nancumprod <nancumprod-function-reference>` | cumprod of elements over given axes, replacing NaN with 1  |
+---------------------------------------------------+------------------------------------------------------------+
