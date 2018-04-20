.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

Internals of xtensor
====================

This section provides information about `xtensor`'s internals and its architecture. It is intended for developers
who want to contribute to `xtensor` or simply understand how it works under the hood. `xtensor` makes heavy use
of the CRTP pattern, template meta-programming, universal references and perfect forwarding. One should be familiar
with these notions before going any further.

.. toctree::

   concepts
   implementation_classes
   expression_tree.rst
   iterating_expression.rst
   assignment.rst
