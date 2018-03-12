.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. _closure-semantics-label:

Closure semantics
=================

The ``xtensor`` library is a tensor expression library implementing numpy-style broadcasting and universal functions but in a lazy fashion.

If ``x`` and ``y`` are two tensor expressions with compatible shapes, the result of ``x + y`` is not a tensor but an expression that does
not hold any value. Values of ``x + y`` are computed upon access or when the result is assigned to a container such as ``xt::xtensor`` or
``xt::xarray``. The same holds for most functions in xtensor, views, broadcasting views, etc.

In order to be able to perform the differed computation of ``x + y``, the returned expression must hold references, const references or
copies of the members ``x`` and ``y``, depending on how arguments were passed to ``operator+``. The actual types held by the expressions
are the **closure types**.

The concept of closure type is key in the implementation of ``xtensor`` and appears in all the expressions defined in xtensor, and the utility functions and metafunctions complement the tools of the standard library for the move semantics. 

Basic rules for determining closure types
-----------------------------------------

The two main requirements are the following:

- when an argument passed to the function returning an expression (here, ``operator+``) is an *rvalue*, the closure type is always a value and the ``rvalue`` is *moved*.
- when an argument passed to the function returning an expression is an *lvalue reference*, the closure type is a reference of the same type.

It is important for the closure type not to be a reference when the passed argument is an rvalue, which can result in dangling references.

Following the conventions of the C++ standard library for naming type traits, we provide two type traits classes providing an implementation of these rules
in the ``xutils.hpp`` header, ``closure``, and ``const_closure``. The latter adds the const qualifier even when the provided argument is not const.

.. code:: cpp

    template <class S>
    struct closure
    {
        using underlying_type = std::conditional_t<
            std::is_const<std::remove_reference_t<S>>::value,
            const std::decay_t<S>,
            std::decay_t<S>>;
        using type = typename std::conditional<
            std::is_lvalue_reference<S>::value,
            underlying_type&,
            underlying_type>::type;
    };

    template <class S>
    using closure_t = typename closure<S>::type;

The implementation for ``const_closure`` is slightly shorter.

.. code:: cpp

    template <class S>
    struct const_closure
    {
        using underlying_type = const std::decay_t<S>;
        using type = typename std::conditional<
            std::is_lvalue_reference<S>::value,
            underlying_type&,
            underlying_type>::type;
    };
        
    template <class S>
    using const_closure_t = typename const_closure<S>::type;

Using this mechanism, we were able to

- avoid dangling references in nested expressions,
- hold references whenever possible,
- take advantage of the move semantics when holding references is not possible.

Closure types and scalar wrappers
---------------------------------

A requirement for ``xtensor`` is the ability to mix scalars and tensors in tensor expressions. In order to do so,
scalar values are wrapped into the ``xscalar`` wrapper, which is a cheap 0-D tensor expression holding a single
scalar value.

For the xscalar to be a proper proxy on the scalar value, if actually holds a closure type on the scalar value.

The logic for this is encoded into xtensor's ``xclosure`` type trait.

.. code:: cpp

    template <class E, class EN = void>
    struct xclosure
    {
        using type = closure_t<E>;
    };

    template <class E>
    struct xclosure<E, disable_xexpression<std::decay_t<E>>>
    {
        using type = xscalar<closure_t<E>>;
    };

    template <class E>
    using xclosure_t = typename xclosure<E>::type;

In doing so, we ensure const-correctness, we avoid dangling reference, and ensure that lvalues remain lvalues.
The `const_xclosure` follows the same scheme:

.. code:: cpp

    template <class E, class EN = void>
    struct const_xclosure
    {
        using type = const_closure_t<E>;
    };

    template <class E>
    struct const_xclosure<E, disable_xexpression<std::decay_t<E>>>
    {
        using type = xscalar<std::decay_t<E>>;
    };

    template <class E>
    using const_xclosure_t = typename const_xclosure<E>::type;

Writing functions that return expressions
-----------------------------------------

*xtensor closure semantics are not meant to prevent users from doing mistakes, since it would also prevent them from doing something clever*.

This section covers cases where understanding C++ move semantics and xtensor closure semantics helps writing better code with xtensor.

Returning evaluated or unevaluated expressions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A key feature of xtensor is that a function returning e.g. ``x + y / z`` where ``x``, ``y`` and ``z`` are xtensor expressions does not actually perform any
computation. It is only evaluated upon access or assignment. The returned expression holds values or references for ``x``, ``y`` and ``z`` depending on the
lvalue-ness of the variables passed to the expression, using the *closure semantics* described earlier. This may result in dangling references when using
local variables of a function in an unevaluated expression unless one properly forwards / move the variables.

.. note::

   The following rule of thumbs prevents dangling references in the xtensor closure semantics:

   - If the laziness is not important for your use case, returning ``xt::eval(x + y / z)`` will return an evaluated container and avoid these complications.
   - Otherwise, the key is to *move* lvalues that become invalid when leaving the current scope.

**Example: moving local variables and forwarding universal references**

Let us first consider the following implementation of the ``mean`` function in xtensor:

.. raw:: html

    <style>
    .rst-content .admonition-title {
        display: none;
    }
    </style>

.. code:: cpp

    template <class E>
    inline auto mean(E&& e) noexcept
    {
        using value_type = typename std::decay_t<E>::value_type;
        auto size = e.size();
        auto s = sum(std::forward<E>(e));
        return std::move(s) / value_type(size); 
    }

The first thing to take into account is that the result of the final division is an expression, which performs the actual computation
upon access or assignment.

- In order to perform the division, the expression must hold the values or references on the numerator and denominator.
- Since ``s`` is a local variable, it will be destroyed upon leaving the scope of the function, and more importantly, it is an *lvalue*.
- A consequence of ``s`` being an lvalue and a local variable, is that the ``s / value_type(size)`` would end up holding a dangling const reference on ``s``.
- Hence we must call return ``std::move(s) / value_type(size)``.

The other place in this example where the C++ move semantics is used is the line ``s = sum(std::forward<E>(e))``. The goal is to have the unevaluated ``s`` expression
hold a const reference or a value for ``e`` depending on the lvalue-ness of the parameter passed to the function.


