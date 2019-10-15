.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

File input and output
=====================

``xtensor`` has some built-in mechanisms to make loading and saving data easy.
The base xtensor package allows to save and load data in the ``.csv``, ``.json`` and ``.npy``
format.
Please note that many more input and output formats are available in the `xtensor-io
<https://github.com/xtensor-stack/xtensor-io>`_ package.
``xtensor-io`` offers functions to load and store from image files (``jpg``, ``gif``, ``png``...),
sound files (``wav``, ``ogg``...), HDF5 files (``h5``, ``hdf5``, ...), and compressed numpy format (``npz``).


Loading CSV data into xtensor
-----------------------------

The following example code demonstrates how to use ``load_csv`` and ``dump_csv`` to load and
save data in the Comma-separated value format. The reference documentation is :doc:`api/xcsv`.

.. code::

    #include <istream>
    #include <fstream>
    #include <iostream>

    #include "xtensor/xarray.hpp"
    #include "xtensor/xcsv.hpp"

    int main()
    {
        ifstream in_file;
        in_file.open("in.csv");
        auto data = xt::load_csv<double>(in_file);

        ofstream out_file;
        out_file("out.csv");

        xt::xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        xt::dump_csv(out_file, a);

        return 0;
    }

Loading NPY data into xtensor
-----------------------------

The following example demonstrates how to load and store xtensor data in the ``npy`` "NumPy" format,
using the ``load_npy`` and ``dump_npy`` functions.
Reference documentation for the functions used is found here :doc:`api/xnpy`.

.. code::

    #include <istream>
    #include <iostream>
    #include <fstream>

    #include "xtensor/xarray.hpp"
    #include "xtensor/xnpy.hpp"

    int main()
    {
        // Note: you need to supply the data type you are loading
        //       in this case "double".
        auto data = xt::load_npy<double>("in.npy");

        xt::xarray<double> a = {{1,2,3,4}, {5,6,7,8}};
        xt::dump_npy("out.npy", a);

        return 0;
    }

Loading JSON data into xtensor
------------------------------

It's possible to load and dump data to json, using the json library written by
``nlohmann`` (https://nlohmann.github.io/json/) which offers a convenient way
to handle json data in C++. Note that the library needs to be separately installed.
The reference documentation is found :doc:`api/xjson`.


.. code::

    #include "xtensor/xjson.hpp"
    #include "xtensor/xarray.hpp"

    int main()
    {

        xt::xarray<double> t =  {{{1, 2},
                                  {3, 4}},
                                 {{1, 2},
                                  {3, 4}}};

        nlohmann::json jl = t;
        // To obtain the json serialized string
        std::string s = jl.dump();

        xt::xarray<double> res;
        auto j = "[[10.0,10.0],[10.0,10.0]]"_json;
        from_json(j, res);
    }
