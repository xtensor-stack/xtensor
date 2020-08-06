#ifndef XTENSOR_FILE_ARRAY_HPP
#define XTENSOR_FILE_ARRAY_HPP

#include <istream>
#include <fstream>
#include <iostream>

#include "xarray.hpp"
#include "xnoalias.hpp"

namespace xt
{
    namespace detail
    {
        // Workaround for VS2015
        template <class E>
        using try_path = decltype(std::declval<E>().path());

        template <class E, template <class> class OP, class = void>
        struct file_helper_impl
        {
            static const char* path(const xexpression<E>& e)
            {
                return "";
            }
            using is_stored = std::false_type;
        };

        template <class E, template <class> class OP>
        struct file_helper_impl<E, OP, void_t<OP<E>>>
        {
            static const char* path(const xexpression<E>& e)
            {
                return e.derived_cast().path();
            }
            using is_stored = std::true_type;
        };

        template <class E>
        using file_helper = file_helper_impl<E, try_path>;
    }

    template<class E>
    constexpr bool is_stored(const xexpression<E>& e)
    {
        using return_type = typename detail::file_helper<E>::is_stored;
        return return_type::value;
    }

    template <class EC>
    class xfile_reference
    {
    public:

        xfile_reference(EC& value, bool& array_dirty)
        {
            m_pvalue = &value;
            m_parray_dirty = &array_dirty;
        }

        operator const EC() const
        {
            return *m_pvalue;
        }

        operator EC()
        {
            return *m_pvalue;
        }

        EC operator=(const EC value)
        {
            if (value != *m_pvalue)
            {
                *m_parray_dirty = true;
                *m_pvalue = value;
            }
            return *m_pvalue;
        }

    private:

        EC* m_pvalue;
        bool* m_parray_dirty;

    };

    template <class EC, class io_handler>
    class xfile_array;

    template <class EC, class io_handler>
    struct xcontainer_inner_types<xfile_array<EC, io_handler>>
    {
        using const_reference = const EC&;
        using reference = xfile_reference<EC>;
        using size_type = std::size_t;
        using storage_type = EC;
        using temporary_type = xfile_array<EC, io_handler>;
    };

    template <class EC, class io_handler>
    struct xiterable_inner_types<xfile_array<EC, io_handler>>
    {
        using inner_shape_type = typename xarray<EC>::shape_type;
        using const_stepper = xindexed_stepper<xfile_array<EC, io_handler>, true>;
        using stepper = xindexed_stepper<xfile_array<EC, io_handler>, false>;
    };

    template <class EC, class io_handler>
    class xfile_array: public xaccessible<xfile_array<EC, io_handler>>,
                       public xiterable<xfile_array<EC, io_handler>>,
                       public xcontainer_semantic<xfile_array<EC, io_handler>>
    {
    public:

        using const_reference = const EC&;
        using reference = xfile_reference<EC>;
        using self_type = xfile_array<EC, io_handler>;
        using semantic_base = xcontainer_semantic<self_type>;
        using iterable_base = xconst_iterable<self_type>;
        using const_stepper = typename iterable_base::const_stepper;
        using stepper = typename iterable_base::stepper;
        using inner_types = xcontainer_inner_types<self_type>;
        using size_type = typename inner_types::size_type;
        using storage_type = typename inner_types::storage_type;
        using value_type = storage_type;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using difference_type = std::ptrdiff_t;
        using shape_type = typename xarray<EC>::shape_type;
        using temporary_type = typename inner_types::temporary_type;
        using bool_load_type = xt::bool_load_type<value_type>;
        static constexpr layout_type static_layout = layout_type::dynamic;

        template <class O>
        const_stepper stepper_begin(const O& shape) const noexcept;
        template <class O>
        const_stepper stepper_end(const O& shape, layout_type) const noexcept;

        template <class O>
        stepper stepper_begin(const O& shape) noexcept;
        template <class O>
        stepper stepper_end(const O& shape, layout_type) noexcept;

        const auto& shape() const
        {
            return m_array.shape();
        }

        inline layout_type layout() const noexcept
        {
            return static_layout;
        }

        inline bool is_contiguous() const noexcept
        {
            return false;
        }

        xfile_array() {}

        ~xfile_array()
        {
            flush();
        }

        void flush()
        {
            if (m_array_dirty)
            {
                m_io_handler.write(m_array, m_path);
                m_array_dirty = false;
            }
        }

        xarray<EC>& array()
        {
            return m_array;
        }

        template <class C>
        void configure_format(C& config)
        {
            m_io_handler.configure(config);
        }

        void set_path(const char* path)
        {
            std::string p(path);
            set_path(p);
        }

        void set_path(std::string& path)
        {
            if (path != m_path)
            {
                // maybe write to old file
                if (m_array_dirty)
                {
                    m_io_handler.write(m_array, m_path);
                    m_array_dirty = false;
                }
                m_path = path;
                // read new file
                m_io_handler.read(m_array, path);
            }
        }

        template <class S>
        void resize(S& shape)
        {
            m_array.resize(shape);
            m_array = broadcast(0, shape);
        }

        template <class... Idxs>
        inline reference operator()(Idxs... idxs)
        {
            auto index = get_indexes(idxs...);
            return reference(m_array.element(index.cbegin(), index.cend()), m_array_dirty);
        }

        template <class... Idxs>
        inline const_reference operator()(Idxs... idxs) const
        {
            auto index = get_indexes(idxs...);
            return m_array.element(index.cbegin(), index.cend());
        }

        xfile_array(const xfile_array&) = default;
        xfile_array& operator=(const xfile_array&) = default;

        xfile_array(xfile_array&&) = default;
        xfile_array& operator=(xfile_array&&) = default;

        template <class E>
        xfile_array(const xexpression<E>& e, const char* path)
        {
            set_path(path);
            m_array_dirty = true;
            const auto& shape = e.derived_cast().shape();
            m_array.resize(shape);
            xstrided_slice_vector sv;
            for (auto i = 0; i < dimension(); i++)
                sv.push_back(all());
            noalias(m_array) = strided_view(e.derived_cast(), sv);
        }

        template <class E>
        xfile_array(const xexpression<E>& e)
        {
            const char* path = detail::file_helper<E>::path(e);
            *this = xfile_array<EC, io_handler>(e, path);
        }

        template <class E>
        self_type& operator=(const xexpression<E>& e)
        {
            return semantic_base::operator=(e);
        }

        reference operator[](const xindex& index)
        {
            return reference(m_array.element(index.cbegin(), index.cend()), m_array_dirty);
        }

        const_reference operator[](const xindex& index) const
        {
            return m_array.element(index.cbegin(), index.cend());
        }

        template <class It>
        inline reference element(It first, It last)
        {
            return reference(m_array.element(first, last), m_array_dirty);
        }

        template <class It>
        inline const_reference element(It first, It last) const
        {
            return m_array.element(first, last);
        }

        size_type dimension() const
        {
            return shape().size();
        }

        const char* path() const
        {
            return m_path.c_str();
        }

        template <class S>
        bool broadcast_shape(S& s, bool reuse_cache = false) const
        {
            return xt::broadcast_shape(shape(), s);
        }

    private:

        xarray<EC> m_array;
        bool m_array_dirty;
        io_handler m_io_handler;
        std::string m_path;

        template <class... Idxs>
        inline std::array<size_t, sizeof...(Idxs)> get_indexes(Idxs... idxs) const
        {
            std::array<size_t, sizeof...(Idxs)> indexes = {{idxs...}};
            return indexes;
        }

        template <class S>
        bool is_trivial_broadcast(const S& str) const noexcept
        {
            return false;
        }
    };

    template <class EC, class io_handler>
    template <class O>
    inline auto xfile_array<EC, io_handler>::stepper_begin(const O& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset);
    }

    template <class EC, class io_handler>
    template <class O>
    inline auto xfile_array<EC, io_handler>::stepper_end(const O& shape, layout_type) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset, true);
    }

    template <class EC, class io_handler>
    template <class O>
    inline auto xfile_array<EC, io_handler>::stepper_begin(const O& shape) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(this, offset);
    }

    template <class EC, class io_handler>
    template <class O>
    inline auto xfile_array<EC, io_handler>::stepper_end(const O& shape, layout_type) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(this, offset, true);
    }
}

#endif
