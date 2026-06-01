/***************************************************************************
 * Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht    *
 *                                                                          *
 * Distributed under the terms of the BSD 3-Clause License.                 *
 *                                                                          *
 * The full license is in the file LICENSE, distributed with this software. *
 ****************************************************************************/

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <benchmark/benchmark.h>
#include <Python.h>

#include "xtensor/containers/xtensor.hpp"
#include "xtensor/core/xmath.hpp"
#include "xtensor/core/xnoalias.hpp"

namespace
{
    constexpr auto numpy_math_range_min = 64;
    constexpr auto numpy_math_range_max = 64 << 3;

    void set_single_thread_environment()
    {
        const char* vars[] = {
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS"
        };

        for (const char* var : vars)
        {
            setenv(var, "1", 1);
        }
    }

    [[noreturn]] void throw_python_error(const std::string& context)
    {
        PyErr_Print();
        throw std::runtime_error(context);
    }

    class py_object
    {
    public:

        py_object() = default;

        explicit py_object(PyObject* ptr)
            : m_ptr(ptr)
        {
        }

        py_object(const py_object&) = delete;
        py_object& operator=(const py_object&) = delete;

        py_object(py_object&& rhs) noexcept
            : m_ptr(rhs.m_ptr)
        {
            rhs.m_ptr = nullptr;
        }

        py_object& operator=(py_object&& rhs) noexcept
        {
            if (this != &rhs)
            {
                Py_XDECREF(m_ptr);
                m_ptr = rhs.m_ptr;
                rhs.m_ptr = nullptr;
            }
            return *this;
        }

        ~py_object()
        {
            Py_XDECREF(m_ptr);
        }

        PyObject* get() const
        {
            return m_ptr;
        }

    private:

        PyObject* m_ptr = nullptr;
    };

    py_object steal_reference(PyObject* ptr, const std::string& context)
    {
        if (ptr == nullptr)
        {
            throw_python_error(context);
        }
        return py_object(ptr);
    }

    std::string py_unicode_to_string(PyObject* obj, const std::string& context)
    {
        const char* value = PyUnicode_AsUTF8(obj);
        if (value == nullptr)
        {
            throw_python_error(context);
        }
        return std::string(value);
    }

    std::string py_object_to_string(PyObject* obj, const std::string& context)
    {
        auto string_object = steal_reference(PyObject_Str(obj), context);
        return py_unicode_to_string(string_object.get(), context);
    }

    PyObject* get_dict_item(PyObject* dict, const char* key)
    {
        if (dict == nullptr || !PyDict_Check(dict))
        {
            return nullptr;
        }
        return PyDict_GetItemString(dict, key);
    }

    std::string
    get_nested_config_string(PyObject* dict, std::initializer_list<const char*> keys, const std::string& fallback)
    {
        PyObject* current = dict;
        for (const char* key : keys)
        {
            current = get_dict_item(current, key);
            if (current == nullptr)
            {
                return fallback;
            }
        }
        return py_object_to_string(current, std::string("reading NumPy config value for ") + keys.begin()[0]);
    }

    template <class V>
    void init_benchmark_data(V& lhs, V& rhs, std::size_t rows, std::size_t cols)
    {
        using value_type = typename V::value_type;
        for (std::size_t i = 0; i < rows; ++i)
        {
            for (std::size_t j = 0; j < cols; ++j)
            {
                lhs(i, j) = value_type(0.5) * value_type(j) / value_type(j + 1)
                            + std::sqrt(value_type(i)) * value_type(9.) / value_type(cols);
                rhs(i, j) = value_type(10.2) / value_type(i + 2) + value_type(0.25) * value_type(j);
            }
        }
    }

    template <class V>
    void shift_benchmark_data(V& values, typename V::value_type offset)
    {
        for (auto& value : values)
        {
            value += offset;
        }
    }

    template <class V>
    void clamp_benchmark_data(V& values, typename V::value_type minimum)
    {
        for (auto& value : values)
        {
            if (value < minimum)
            {
                value = minimum;
            }
        }
    }

    template <class V>
    void clamp_benchmark_data(V& values, typename V::value_type minimum, typename V::value_type maximum)
    {
        for (auto& value : values)
        {
            if (value < minimum)
            {
                value = minimum;
            }
            else if (value > maximum)
            {
                value = maximum;
            }
        }
    }

    template <class V>
    void prepare_prod_benchmark_data(V& values)
    {
        using value_type = typename V::value_type;
        for (auto& value : values)
        {
            value = value_type(1.0) + value / value_type(100000.0);
        }
    }

    class numpy_context
    {
    public:

        static numpy_context& instance()
        {
            static numpy_context ctx;
            return ctx;
        }

        void print_stats() const
        {
            std::cout << "USING NUMPY\nNUMPY VERSION: " << m_version << "\nNUMPY THREADS: 1"
                      << "\nNUMPY BLAS: " << m_blas_name << "\nNUMPY LAPACK: " << m_lapack_name
                      << "\nNUMPY C COMPILER: " << m_c_compiler << "\nNUMPY CFLAGS: " << m_cflags
                      << "\nNUMPY C++ COMPILER: " << m_cxx_compiler << "\nNUMPY CXXFLAGS: " << m_cxxflags
                      << "\n\n";
        }

        PyObject* add() const
        {
            return m_add.get();
        }

        PyObject* multiply() const
        {
            return m_multiply.get();
        }

        PyObject* subtract() const
        {
            return m_subtract.get();
        }

        PyObject* divide() const
        {
            return m_divide.get();
        }

        PyObject* power() const
        {
            return m_power.get();
        }

        PyObject* hypot() const
        {
            return m_hypot.get();
        }

        PyObject* sin() const
        {
            return m_sin.get();
        }

        PyObject* cos() const
        {
            return m_cos.get();
        }

        PyObject* tan() const
        {
            return m_tan.get();
        }

        PyObject* asin() const
        {
            return m_asin.get();
        }

        PyObject* acos() const
        {
            return m_acos.get();
        }

        PyObject* atan() const
        {
            return m_atan.get();
        }

        PyObject* sinh() const
        {
            return m_sinh.get();
        }

        PyObject* cosh() const
        {
            return m_cosh.get();
        }

        PyObject* tanh() const
        {
            return m_tanh.get();
        }

        PyObject* asinh() const
        {
            return m_asinh.get();
        }

        PyObject* acosh() const
        {
            return m_acosh.get();
        }

        PyObject* atanh() const
        {
            return m_atanh.get();
        }

        PyObject* exp() const
        {
            return m_exp.get();
        }

        PyObject* exp2() const
        {
            return m_exp2.get();
        }

        PyObject* expm1() const
        {
            return m_expm1.get();
        }

        PyObject* log() const
        {
            return m_log.get();
        }

        PyObject* log10() const
        {
            return m_log10.get();
        }

        PyObject* log2() const
        {
            return m_log2.get();
        }

        PyObject* log1p() const
        {
            return m_log1p.get();
        }

        PyObject* sqrt() const
        {
            return m_sqrt.get();
        }

        PyObject* cbrt() const
        {
            return m_cbrt.get();
        }

        PyObject* ceil() const
        {
            return m_ceil.get();
        }

        PyObject* floor() const
        {
            return m_floor.get();
        }

        PyObject* trunc() const
        {
            return m_trunc.get();
        }

        PyObject* round() const
        {
            return m_round.get();
        }

        PyObject* mean() const
        {
            return m_mean.get();
        }

        PyObject* amin() const
        {
            return m_amin.get();
        }

        PyObject* amax() const
        {
            return m_amax.get();
        }

        PyObject* prod() const
        {
            return m_prod.get();
        }

        PyObject* sum() const
        {
            return m_sum.get();
        }

        py_object make_array_view(double* data, std::size_t rows, std::size_t cols) const
        {
            auto buffer = steal_reference(
                PyMemoryView_FromMemory(
                    reinterpret_cast<char*>(data),
                    static_cast<Py_ssize_t>(rows * cols * sizeof(double)),
                    PyBUF_WRITE
                ),
                "creating NumPy buffer view"
            );

            auto one_dimensional = steal_reference(
                PyObject_CallFunctionObjArgs(m_frombuffer.get(), buffer.get(), m_float64.get(), nullptr),
                "creating NumPy array from buffer"
            );

            return steal_reference(
                PyObject_CallMethod(one_dimensional.get(), "reshape", "(nn)", rows, cols),
                "reshaping NumPy array"
            );
        }

    private:

        numpy_context()
        {
            set_single_thread_environment();
            if (!Py_IsInitialized())
            {
                Py_Initialize();
            }

            m_numpy = steal_reference(PyImport_ImportModule("numpy"), "importing numpy");
            m_float64 = steal_reference(PyObject_GetAttrString(m_numpy.get(), "float64"), "loading numpy.float64");
            m_frombuffer = steal_reference(
                PyObject_GetAttrString(m_numpy.get(), "frombuffer"),
                "loading numpy.frombuffer"
            );
            m_add = steal_reference(PyObject_GetAttrString(m_numpy.get(), "add"), "loading numpy.add");
            m_multiply = steal_reference(
                PyObject_GetAttrString(m_numpy.get(), "multiply"),
                "loading numpy.multiply"
            );
            m_subtract = steal_reference(
                PyObject_GetAttrString(m_numpy.get(), "subtract"),
                "loading numpy.subtract"
            );
            m_divide = steal_reference(PyObject_GetAttrString(m_numpy.get(), "divide"), "loading numpy.divide");
            m_power = steal_reference(PyObject_GetAttrString(m_numpy.get(), "power"), "loading numpy.power");
            m_hypot = steal_reference(PyObject_GetAttrString(m_numpy.get(), "hypot"), "loading numpy.hypot");
            m_sin = steal_reference(PyObject_GetAttrString(m_numpy.get(), "sin"), "loading numpy.sin");
            m_cos = steal_reference(PyObject_GetAttrString(m_numpy.get(), "cos"), "loading numpy.cos");
            m_tan = steal_reference(PyObject_GetAttrString(m_numpy.get(), "tan"), "loading numpy.tan");
            m_asin = steal_reference(PyObject_GetAttrString(m_numpy.get(), "asin"), "loading numpy.asin");
            m_acos = steal_reference(PyObject_GetAttrString(m_numpy.get(), "acos"), "loading numpy.acos");
            m_atan = steal_reference(PyObject_GetAttrString(m_numpy.get(), "atan"), "loading numpy.atan");
            m_sinh = steal_reference(PyObject_GetAttrString(m_numpy.get(), "sinh"), "loading numpy.sinh");
            m_cosh = steal_reference(PyObject_GetAttrString(m_numpy.get(), "cosh"), "loading numpy.cosh");
            m_tanh = steal_reference(PyObject_GetAttrString(m_numpy.get(), "tanh"), "loading numpy.tanh");
            m_asinh = steal_reference(PyObject_GetAttrString(m_numpy.get(), "asinh"), "loading numpy.asinh");
            m_acosh = steal_reference(PyObject_GetAttrString(m_numpy.get(), "acosh"), "loading numpy.acosh");
            m_atanh = steal_reference(PyObject_GetAttrString(m_numpy.get(), "atanh"), "loading numpy.atanh");
            m_exp = steal_reference(PyObject_GetAttrString(m_numpy.get(), "exp"), "loading numpy.exp");
            m_exp2 = steal_reference(PyObject_GetAttrString(m_numpy.get(), "exp2"), "loading numpy.exp2");
            m_expm1 = steal_reference(PyObject_GetAttrString(m_numpy.get(), "expm1"), "loading numpy.expm1");
            m_log = steal_reference(PyObject_GetAttrString(m_numpy.get(), "log"), "loading numpy.log");
            m_log10 = steal_reference(PyObject_GetAttrString(m_numpy.get(), "log10"), "loading numpy.log10");
            m_log2 = steal_reference(PyObject_GetAttrString(m_numpy.get(), "log2"), "loading numpy.log2");
            m_log1p = steal_reference(PyObject_GetAttrString(m_numpy.get(), "log1p"), "loading numpy.log1p");
            m_sqrt = steal_reference(PyObject_GetAttrString(m_numpy.get(), "sqrt"), "loading numpy.sqrt");
            m_cbrt = steal_reference(PyObject_GetAttrString(m_numpy.get(), "cbrt"), "loading numpy.cbrt");
            m_ceil = steal_reference(PyObject_GetAttrString(m_numpy.get(), "ceil"), "loading numpy.ceil");
            m_floor = steal_reference(PyObject_GetAttrString(m_numpy.get(), "floor"), "loading numpy.floor");
            m_trunc = steal_reference(PyObject_GetAttrString(m_numpy.get(), "trunc"), "loading numpy.trunc");
            m_round = steal_reference(PyObject_GetAttrString(m_numpy.get(), "round"), "loading numpy.round");
            m_mean = steal_reference(PyObject_GetAttrString(m_numpy.get(), "mean"), "loading numpy.mean");
            m_amin = steal_reference(PyObject_GetAttrString(m_numpy.get(), "amin"), "loading numpy.amin");
            m_amax = steal_reference(PyObject_GetAttrString(m_numpy.get(), "amax"), "loading numpy.amax");
            m_prod = steal_reference(PyObject_GetAttrString(m_numpy.get(), "prod"), "loading numpy.prod");
            m_sum = steal_reference(PyObject_GetAttrString(m_numpy.get(), "sum"), "loading numpy.sum");

            auto version = steal_reference(
                PyObject_GetAttrString(m_numpy.get(), "__version__"),
                "loading numpy version"
            );
            m_version = py_unicode_to_string(version.get(), "reading numpy version");

            auto config_module = steal_reference(
                PyObject_GetAttrString(m_numpy.get(), "__config__"),
                "loading numpy.__config__"
            );
            auto config = steal_reference(
                PyObject_GetAttrString(config_module.get(), "CONFIG"),
                "loading numpy build config"
            );
            m_blas_name = get_nested_config_string(config.get(), {"Build Dependencies", "blas", "name"}, "unknown");
            m_lapack_name = get_nested_config_string(
                config.get(),
                {"Build Dependencies", "lapack", "name"},
                "unknown"
            );
            m_c_compiler = get_nested_config_string(config.get(), {"Compilers", "c", "commands"}, "unknown");
            m_cflags = get_nested_config_string(config.get(), {"Compilers", "c", "args"}, "unknown");
            m_cxx_compiler = get_nested_config_string(config.get(), {"Compilers", "c++", "commands"}, "unknown");
            m_cxxflags = get_nested_config_string(config.get(), {"Compilers", "c++", "args"}, "unknown");
        }

        std::string m_version;
        std::string m_blas_name;
        std::string m_lapack_name;
        std::string m_c_compiler;
        std::string m_cflags;
        std::string m_cxx_compiler;
        std::string m_cxxflags;
        py_object m_numpy;
        py_object m_float64;
        py_object m_frombuffer;
        py_object m_add;
        py_object m_multiply;
        py_object m_subtract;
        py_object m_divide;
        py_object m_power;
        py_object m_hypot;
        py_object m_sin;
        py_object m_cos;
        py_object m_tan;
        py_object m_asin;
        py_object m_acos;
        py_object m_atan;
        py_object m_sinh;
        py_object m_cosh;
        py_object m_tanh;
        py_object m_asinh;
        py_object m_acosh;
        py_object m_atanh;
        py_object m_exp;
        py_object m_exp2;
        py_object m_expm1;
        py_object m_log;
        py_object m_log10;
        py_object m_log2;
        py_object m_log1p;
        py_object m_sqrt;
        py_object m_cbrt;
        py_object m_ceil;
        py_object m_floor;
        py_object m_trunc;
        py_object m_round;
        py_object m_mean;
        py_object m_amin;
        py_object m_amax;
        py_object m_prod;
        py_object m_sum;
    };

    void call_numpy(PyObject* function, PyObject* args, PyObject* kwargs)
    {
        auto result = steal_reference(PyObject_Call(function, args, kwargs), "calling NumPy benchmark function");
        benchmark::DoNotOptimize(result.get());
        benchmark::ClobberMemory();
    }

    py_object make_out_kwargs(PyObject* out)
    {
        auto kwargs = steal_reference(PyDict_New(), "creating NumPy keyword arguments");
        if (PyDict_SetItemString(kwargs.get(), "out", out) != 0)
        {
            throw_python_error("setting NumPy out keyword argument");
        }
        return kwargs;
    }

    py_object make_axis_kwargs(long axis)
    {
        auto kwargs = steal_reference(PyDict_New(), "creating NumPy keyword arguments");
        auto axis_object = steal_reference(PyLong_FromLong(axis), "creating axis value");
        if (PyDict_SetItemString(kwargs.get(), "axis", axis_object.get()) != 0)
        {
            throw_python_error("setting NumPy axis keyword argument");
        }
        return kwargs;
    }

    struct numpy_binary_fixture
    {
        explicit numpy_binary_fixture(std::size_t size)
        {
            init(size, size);
        }

        void init(std::size_t rows, std::size_t cols)
        {
            lhs.resize({rows, cols});
            rhs.resize({rows, cols});
            res.resize({rows, cols});
            init_benchmark_data(lhs, rhs, rows, cols);
            np_lhs = numpy_context::instance().make_array_view(lhs.data(), rows, cols);
            np_rhs = numpy_context::instance().make_array_view(rhs.data(), rows, cols);
            np_res = numpy_context::instance().make_array_view(res.data(), rows, cols);
        }

        xt::xtensor<double, 2> lhs;
        xt::xtensor<double, 2> rhs;
        xt::xtensor<double, 2> res;
        py_object np_lhs;
        py_object np_rhs;
        py_object np_res;
    };

    struct numpy_unary_fixture
    {
        explicit numpy_unary_fixture(std::size_t size)
        {
            lhs.resize({size, size});
            res.resize({size, size});
            rhs.resize({size, size});
            init_benchmark_data(lhs, rhs, size, size);
            np_lhs = numpy_context::instance().make_array_view(lhs.data(), size, size);
            np_res = numpy_context::instance().make_array_view(res.data(), size, size);
        }

        xt::xtensor<double, 2> lhs;
        xt::xtensor<double, 2> rhs;
        xt::xtensor<double, 2> res;
        py_object np_lhs;
        py_object np_res;
    };

    struct numpy_reduce_fixture
    {
        explicit numpy_reduce_fixture(std::size_t size)
        {
            lhs.resize({size, size});
            rhs.resize({size, size});
            init_benchmark_data(lhs, rhs, size, size);
            np_lhs = numpy_context::instance().make_array_view(lhs.data(), size, size);
        }

        xt::xtensor<double, 2> lhs;
        xt::xtensor<double, 2> rhs;
        py_object np_lhs;
    };

    template <class XtensorOp>
    void run_binary_xtensor(benchmark::State& state, XtensorOp&& op)
    {
        numpy_binary_fixture fixture(static_cast<std::size_t>(state.range(0)));
        for (auto _ : state)
        {
            xt::noalias(fixture.res) = op(fixture.lhs, fixture.rhs);
            benchmark::DoNotOptimize(fixture.res.data());
        }
    }

    template <class Prepare>
    void run_binary_numpy(benchmark::State& state, PyObject* function, const char* context, Prepare&& prepare)
    {
        numpy_binary_fixture fixture(static_cast<std::size_t>(state.range(0)));
        prepare(fixture);
        auto args = steal_reference(PyTuple_Pack(2, fixture.np_lhs.get(), fixture.np_rhs.get()), context);
        auto kwargs = make_out_kwargs(fixture.np_res.get());
        for (auto _ : state)
        {
            call_numpy(function, args.get(), kwargs.get());
        }
    }

    template <class XtensorOp, class Prepare>
    void run_unary_xtensor(benchmark::State& state, XtensorOp&& op, Prepare&& prepare)
    {
        numpy_unary_fixture fixture(static_cast<std::size_t>(state.range(0)));
        prepare(fixture);
        for (auto _ : state)
        {
            xt::noalias(fixture.res) = op(fixture.lhs);
            benchmark::DoNotOptimize(fixture.res.data());
        }
    }

    template <class Prepare>
    void run_unary_numpy(benchmark::State& state, PyObject* function, const char* context, Prepare&& prepare)
    {
        numpy_unary_fixture fixture(static_cast<std::size_t>(state.range(0)));
        prepare(fixture);
        auto args = steal_reference(PyTuple_Pack(1, fixture.np_lhs.get()), context);
        auto kwargs = make_out_kwargs(fixture.np_res.get());
        for (auto _ : state)
        {
            call_numpy(function, args.get(), kwargs.get());
        }
    }

    template <class XtensorOp, class Prepare>
    void run_axis_reduce_xtensor(benchmark::State& state, XtensorOp&& op, std::size_t axis, Prepare&& prepare)
    {
        numpy_reduce_fixture fixture(static_cast<std::size_t>(state.range(0)));
        prepare(fixture);
        xt::xtensor<double, 1> res;
        res.resize({static_cast<std::size_t>(state.range(0))});
        for (auto _ : state)
        {
            res = op(fixture.lhs, axis);
            benchmark::DoNotOptimize(res.data());
        }
    }

    template <class Prepare>
    void
    run_axis_reduce_numpy(benchmark::State& state, PyObject* function, const char* context, long axis, Prepare&& prepare)
    {
        numpy_reduce_fixture fixture(static_cast<std::size_t>(state.range(0)));
        prepare(fixture);
        auto args = steal_reference(PyTuple_Pack(1, fixture.np_lhs.get()), context);
        auto kwargs = make_axis_kwargs(axis);
        for (auto _ : state)
        {
            call_numpy(function, args.get(), kwargs.get());
        }
    }

    template <class XtensorOp, class Prepare>
    void run_all_reduce_xtensor(benchmark::State& state, XtensorOp&& op, Prepare&& prepare)
    {
        numpy_reduce_fixture fixture(static_cast<std::size_t>(state.range(0)));
        prepare(fixture);
        for (auto _ : state)
        {
            auto res = op(fixture.lhs);
            benchmark::DoNotOptimize(res());
        }
    }

    template <class Prepare>
    void
    run_all_reduce_numpy(benchmark::State& state, PyObject* function, const char* context, Prepare&& prepare)
    {
        numpy_reduce_fixture fixture(static_cast<std::size_t>(state.range(0)));
        prepare(fixture);
        auto args = steal_reference(PyTuple_Pack(1, fixture.np_lhs.get()), context);
        for (auto _ : state)
        {
            call_numpy(function, args.get(), nullptr);
        }
    }

    template <class Prepare>
    void prepare_default(Prepare&)
    {
    }
}

namespace xt::numpy
{
    void print_stats()
    {
        numpy_context::instance().print_stats();
    }

    void add_xtensor(benchmark::State& state)
    {
        run_binary_xtensor(
            state,
            [](const auto& lhs, const auto& rhs)
            {
                return lhs + rhs;
            }
        );
    }

    void add_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_binary_numpy(state, context.add(), "packing NumPy add arguments", prepare_default<numpy_binary_fixture>);
    }

    void multiply_xtensor(benchmark::State& state)
    {
        run_binary_xtensor(
            state,
            [](const auto& lhs, const auto& rhs)
            {
                return lhs * rhs;
            }
        );
    }

    void multiply_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_binary_numpy(state, context.multiply(), "packing NumPy multiply arguments", prepare_default<numpy_binary_fixture>);
    }

    void subtract_xtensor(benchmark::State& state)
    {
        run_binary_xtensor(
            state,
            [](const auto& lhs, const auto& rhs)
            {
                return lhs - rhs;
            }
        );
    }

    void subtract_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_binary_numpy(state, context.subtract(), "packing NumPy subtract arguments", prepare_default<numpy_binary_fixture>);
    }

    void divide_xtensor(benchmark::State& state)
    {
        run_binary_xtensor(
            state,
            [](const auto& lhs, const auto& rhs)
            {
                return lhs / rhs;
            }
        );
    }

    void divide_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_binary_numpy(state, context.divide(), "packing NumPy divide arguments", prepare_default<numpy_binary_fixture>);
    }

    void power_xtensor(benchmark::State& state)
    {
        run_binary_xtensor(
            state,
            [](const auto& lhs, const auto& rhs)
            {
                return xt::pow(lhs, rhs);
            }
        );
    }

    void power_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_binary_numpy(state, context.power(), "packing NumPy power arguments", prepare_default<numpy_binary_fixture>);
    }

    void hypot_xtensor(benchmark::State& state)
    {
        run_binary_xtensor(
            state,
            [](const auto& lhs, const auto& rhs)
            {
                return xt::hypot(lhs, rhs);
            }
        );
    }

    void hypot_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_binary_numpy(state, context.hypot(), "packing NumPy hypot arguments", prepare_default<numpy_binary_fixture>);
    }

    void sin_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::sin(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void sin_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.sin(), "packing NumPy sin arguments", prepare_default<numpy_unary_fixture>);
    }

    void cos_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::cos(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void cos_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.cos(), "packing NumPy cos arguments", prepare_default<numpy_unary_fixture>);
    }

    void tan_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::tan(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void tan_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.tan(), "packing NumPy tan arguments", prepare_default<numpy_unary_fixture>);
    }

    void asin_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(
            state,
            [](const auto& lhs)
            {
                return xt::asin(lhs);
            },
            [](auto& fixture)
            {
                clamp_benchmark_data(fixture.lhs, -1.0, 1.0);
            }
        );
    }

    void asin_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(
            state,
            context.asin(),
            "packing NumPy asin arguments",
            [](auto& fixture)
            {
                clamp_benchmark_data(fixture.lhs, -1.0, 1.0);
            }
        );
    }

    void acos_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(
            state,
            [](const auto& lhs)
            {
                return xt::acos(lhs);
            },
            [](auto& fixture)
            {
                clamp_benchmark_data(fixture.lhs, -1.0, 1.0);
            }
        );
    }

    void acos_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(
            state,
            context.acos(),
            "packing NumPy acos arguments",
            [](auto& fixture)
            {
                clamp_benchmark_data(fixture.lhs, -1.0, 1.0);
            }
        );
    }

    void atan_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::atan(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void atan_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.atan(), "packing NumPy atan arguments", prepare_default<numpy_unary_fixture>);
    }

    void sinh_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::sinh(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void sinh_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.sinh(), "packing NumPy sinh arguments", prepare_default<numpy_unary_fixture>);
    }

    void cosh_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::cosh(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void cosh_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.cosh(), "packing NumPy cosh arguments", prepare_default<numpy_unary_fixture>);
    }

    void tanh_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::tanh(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void tanh_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.tanh(), "packing NumPy tanh arguments", prepare_default<numpy_unary_fixture>);
    }

    void asinh_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::asinh(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void asinh_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.asinh(), "packing NumPy asinh arguments", prepare_default<numpy_unary_fixture>);
    }

    void acosh_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(
            state,
            [](const auto& lhs)
            {
                return xt::acosh(lhs);
            },
            [](auto& fixture)
            {
                shift_benchmark_data(fixture.lhs, 1.0);
                clamp_benchmark_data(fixture.lhs, 1.0);
            }
        );
    }

    void acosh_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(
            state,
            context.acosh(),
            "packing NumPy acosh arguments",
            [](auto& fixture)
            {
                shift_benchmark_data(fixture.lhs, 1.0);
                clamp_benchmark_data(fixture.lhs, 1.0);
            }
        );
    }

    void atanh_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(
            state,
            [](const auto& lhs)
            {
                return xt::atanh(lhs);
            },
            [](auto& fixture)
            {
                clamp_benchmark_data(fixture.lhs, -0.999, 0.999);
            }
        );
    }

    void atanh_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(
            state,
            context.atanh(),
            "packing NumPy atanh arguments",
            [](auto& fixture)
            {
                clamp_benchmark_data(fixture.lhs, -0.999, 0.999);
            }
        );
    }

    void exp_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::exp(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void exp_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.exp(), "packing NumPy exp arguments", prepare_default<numpy_unary_fixture>);
    }

    void exp2_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::exp2(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void exp2_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.exp2(), "packing NumPy exp2 arguments", prepare_default<numpy_unary_fixture>);
    }

    void expm1_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::expm1(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void expm1_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.expm1(), "packing NumPy expm1 arguments", prepare_default<numpy_unary_fixture>);
    }

    void log_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(
            state,
            [](const auto& lhs)
            {
                return xt::log(lhs);
            },
            [](auto& fixture)
            {
                shift_benchmark_data(fixture.lhs, 1.0);
            }
        );
    }

    void log_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(
            state,
            context.log(),
            "packing NumPy log arguments",
            [](auto& fixture)
            {
                shift_benchmark_data(fixture.lhs, 1.0);
            }
        );
    }

    void log10_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(
            state,
            [](const auto& lhs)
            {
                return xt::log10(lhs);
            },
            [](auto& fixture)
            {
                shift_benchmark_data(fixture.lhs, 1.0);
            }
        );
    }

    void log10_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(
            state,
            context.log10(),
            "packing NumPy log10 arguments",
            [](auto& fixture)
            {
                shift_benchmark_data(fixture.lhs, 1.0);
            }
        );
    }

    void log2_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(
            state,
            [](const auto& lhs)
            {
                return xt::log2(lhs);
            },
            [](auto& fixture)
            {
                shift_benchmark_data(fixture.lhs, 1.0);
            }
        );
    }

    void log2_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(
            state,
            context.log2(),
            "packing NumPy log2 arguments",
            [](auto& fixture)
            {
                shift_benchmark_data(fixture.lhs, 1.0);
            }
        );
    }

    void log1p_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(
            state,
            [](const auto& lhs)
            {
                return xt::log1p(lhs);
            },
            [](auto& fixture)
            {
                clamp_benchmark_data(fixture.lhs, -0.999);
            }
        );
    }

    void log1p_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(
            state,
            context.log1p(),
            "packing NumPy log1p arguments",
            [](auto& fixture)
            {
                clamp_benchmark_data(fixture.lhs, -0.999);
            }
        );
    }

    void sqrt_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::sqrt(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void sqrt_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.sqrt(), "packing NumPy sqrt arguments", prepare_default<numpy_unary_fixture>);
    }

    void cbrt_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::cbrt(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void cbrt_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.cbrt(), "packing NumPy cbrt arguments", prepare_default<numpy_unary_fixture>);
    }

    void ceil_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::ceil(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void ceil_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.ceil(), "packing NumPy ceil arguments", prepare_default<numpy_unary_fixture>);
    }

    void floor_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::floor(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void floor_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.floor(), "packing NumPy floor arguments", prepare_default<numpy_unary_fixture>);
    }

    void trunc_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::trunc(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void trunc_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.trunc(), "packing NumPy trunc arguments", prepare_default<numpy_unary_fixture>);
    }

    void round_xtensor(benchmark::State& state)
    {
        run_unary_xtensor(state, [](const auto& lhs) { return xt::round(lhs); }, prepare_default<numpy_unary_fixture>);
    }

    void round_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_unary_numpy(state, context.round(), "packing NumPy round arguments", prepare_default<numpy_unary_fixture>);
    }

    void sum_axis0_xtensor(benchmark::State& state)
    {
        run_axis_reduce_xtensor(state, [](const auto& lhs, std::size_t axis) { return xt::sum(lhs, {axis}); }, 0, prepare_default<numpy_reduce_fixture>);
    }

    void sum_axis0_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_axis_reduce_numpy(state, context.sum(), "packing NumPy sum arguments", 0, prepare_default<numpy_reduce_fixture>);
    }

    void sum_axis1_xtensor(benchmark::State& state)
    {
        run_axis_reduce_xtensor(state, [](const auto& lhs, std::size_t axis) { return xt::sum(lhs, {axis}); }, 1, prepare_default<numpy_reduce_fixture>);
    }

    void sum_axis1_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_axis_reduce_numpy(state, context.sum(), "packing NumPy sum arguments", 1, prepare_default<numpy_reduce_fixture>);
    }

    void sum_all_xtensor(benchmark::State& state)
    {
        run_all_reduce_xtensor(state, [](const auto& lhs) { return xt::sum(lhs); }, prepare_default<numpy_reduce_fixture>);
    }

    void sum_all_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_all_reduce_numpy(state, context.sum(), "packing NumPy sum arguments", prepare_default<numpy_reduce_fixture>);
    }

    void mean_axis0_xtensor(benchmark::State& state)
    {
        run_axis_reduce_xtensor(state, [](const auto& lhs, std::size_t axis) { return xt::mean(lhs, {axis}); }, 0, prepare_default<numpy_reduce_fixture>);
    }

    void mean_axis0_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_axis_reduce_numpy(state, context.mean(), "packing NumPy mean arguments", 0, prepare_default<numpy_reduce_fixture>);
    }

    void mean_axis1_xtensor(benchmark::State& state)
    {
        run_axis_reduce_xtensor(state, [](const auto& lhs, std::size_t axis) { return xt::mean(lhs, {axis}); }, 1, prepare_default<numpy_reduce_fixture>);
    }

    void mean_axis1_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_axis_reduce_numpy(state, context.mean(), "packing NumPy mean arguments", 1, prepare_default<numpy_reduce_fixture>);
    }

    void mean_all_xtensor(benchmark::State& state)
    {
        run_all_reduce_xtensor(state, [](const auto& lhs) { return xt::mean(lhs); }, prepare_default<numpy_reduce_fixture>);
    }

    void mean_all_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_all_reduce_numpy(state, context.mean(), "packing NumPy mean arguments", prepare_default<numpy_reduce_fixture>);
    }

    void amin_axis0_xtensor(benchmark::State& state)
    {
        run_axis_reduce_xtensor(state, [](const auto& lhs, std::size_t axis) { return xt::amin(lhs, {axis}); }, 0, prepare_default<numpy_reduce_fixture>);
    }

    void amin_axis0_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_axis_reduce_numpy(state, context.amin(), "packing NumPy amin arguments", 0, prepare_default<numpy_reduce_fixture>);
    }

    void amin_axis1_xtensor(benchmark::State& state)
    {
        run_axis_reduce_xtensor(state, [](const auto& lhs, std::size_t axis) { return xt::amin(lhs, {axis}); }, 1, prepare_default<numpy_reduce_fixture>);
    }

    void amin_axis1_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_axis_reduce_numpy(state, context.amin(), "packing NumPy amin arguments", 1, prepare_default<numpy_reduce_fixture>);
    }

    void amin_all_xtensor(benchmark::State& state)
    {
        run_all_reduce_xtensor(state, [](const auto& lhs) { return xt::amin(lhs); }, prepare_default<numpy_reduce_fixture>);
    }

    void amin_all_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_all_reduce_numpy(state, context.amin(), "packing NumPy amin arguments", prepare_default<numpy_reduce_fixture>);
    }

    void amax_axis0_xtensor(benchmark::State& state)
    {
        run_axis_reduce_xtensor(state, [](const auto& lhs, std::size_t axis) { return xt::amax(lhs, {axis}); }, 0, prepare_default<numpy_reduce_fixture>);
    }

    void amax_axis0_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_axis_reduce_numpy(state, context.amax(), "packing NumPy amax arguments", 0, prepare_default<numpy_reduce_fixture>);
    }

    void amax_axis1_xtensor(benchmark::State& state)
    {
        run_axis_reduce_xtensor(state, [](const auto& lhs, std::size_t axis) { return xt::amax(lhs, {axis}); }, 1, prepare_default<numpy_reduce_fixture>);
    }

    void amax_axis1_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_axis_reduce_numpy(state, context.amax(), "packing NumPy amax arguments", 1, prepare_default<numpy_reduce_fixture>);
    }

    void amax_all_xtensor(benchmark::State& state)
    {
        run_all_reduce_xtensor(state, [](const auto& lhs) { return xt::amax(lhs); }, prepare_default<numpy_reduce_fixture>);
    }

    void amax_all_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_all_reduce_numpy(state, context.amax(), "packing NumPy amax arguments", prepare_default<numpy_reduce_fixture>);
    }

    void prod_axis0_xtensor(benchmark::State& state)
    {
        run_axis_reduce_xtensor(
            state,
            [](const auto& lhs, std::size_t axis)
            {
                return xt::prod(lhs, {axis});
            },
            0,
            [](auto& fixture)
            {
                prepare_prod_benchmark_data(fixture.lhs);
            }
        );
    }

    void prod_axis0_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_axis_reduce_numpy(
            state,
            context.prod(),
            "packing NumPy prod arguments",
            0,
            [](auto& fixture)
            {
                prepare_prod_benchmark_data(fixture.lhs);
            }
        );
    }

    void prod_axis1_xtensor(benchmark::State& state)
    {
        run_axis_reduce_xtensor(
            state,
            [](const auto& lhs, std::size_t axis)
            {
                return xt::prod(lhs, {axis});
            },
            1,
            [](auto& fixture)
            {
                prepare_prod_benchmark_data(fixture.lhs);
            }
        );
    }

    void prod_axis1_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_axis_reduce_numpy(
            state,
            context.prod(),
            "packing NumPy prod arguments",
            1,
            [](auto& fixture)
            {
                prepare_prod_benchmark_data(fixture.lhs);
            }
        );
    }

    void prod_all_xtensor(benchmark::State& state)
    {
        run_all_reduce_xtensor(
            state,
            [](const auto& lhs)
            {
                return xt::prod(lhs);
            },
            [](auto& fixture)
            {
                prepare_prod_benchmark_data(fixture.lhs);
            }
        );
    }

    void prod_all_numpy(benchmark::State& state)
    {
        auto& context = numpy_context::instance();
        run_all_reduce_numpy(
            state,
            context.prod(),
            "packing NumPy prod arguments",
            [](auto& fixture)
            {
                prepare_prod_benchmark_data(fixture.lhs);
            }
        );
    }

    BENCHMARK(add_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(add_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(multiply_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(multiply_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(subtract_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(subtract_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(divide_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(divide_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(power_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(power_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(hypot_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(hypot_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(sin_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(sin_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(cos_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(cos_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(tan_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(tan_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(asin_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(asin_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(acos_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(acos_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(atan_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(atan_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(sinh_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(sinh_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(cosh_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(cosh_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(tanh_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(tanh_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(asinh_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(asinh_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(acosh_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(acosh_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(atanh_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(atanh_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(exp_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(exp_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(exp2_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(exp2_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(expm1_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(expm1_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(log_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(log_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(log10_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(log10_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(log2_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(log2_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(log1p_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(log1p_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(sqrt_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(sqrt_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(cbrt_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(cbrt_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(ceil_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(ceil_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(floor_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(floor_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(trunc_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(trunc_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(round_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(round_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(sum_axis0_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(sum_axis0_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(sum_axis1_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(sum_axis1_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(sum_all_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(sum_all_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(mean_axis0_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(mean_axis0_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(mean_axis1_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(mean_axis1_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(mean_all_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(mean_all_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(amin_axis0_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(amin_axis0_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(amin_axis1_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(amin_axis1_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(amin_all_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(amin_all_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(amax_axis0_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(amax_axis0_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(amax_axis1_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(amax_axis1_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(amax_all_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(amax_all_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(prod_axis0_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(prod_axis0_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(prod_axis1_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(prod_axis1_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(prod_all_xtensor)->Range(numpy_math_range_min, numpy_math_range_max);
    BENCHMARK(prod_all_numpy)->Range(numpy_math_range_min, numpy_math_range_max);
}
