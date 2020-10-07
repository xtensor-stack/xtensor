#ifndef XTENSOR_CHUNKED_ARRAY_HPP
#define XTENSOR_CHUNKED_ARRAY_HPP

#include <vector>
#include <array>

#include "xnoalias.hpp"
#include "xstrided_view.hpp"
#include "xchunk_store_manager.hpp"

namespace xt
{

    template <class T>
    struct is_chunk_store_manager : std::false_type
    {
    };
    
    template <class T>
    struct is_chunk_store_manager<xchunk_store_manager<T>>
        : std::true_type
    {
    };

    /******************************
     * xchunked_array declaration *
     ******************************/

    class empty_extension {};

    template <class chunk_storage, class extension = empty_extension>
    class xchunked_array;

    template <class chunk_storage, class extension>
    struct xcontainer_inner_types<xchunked_array<chunk_storage, extension>>
    {
        using chunk_type = typename chunk_storage::value_type;
        using const_reference = typename chunk_type::const_reference;
        using reference = typename chunk_type::reference;
        using size_type = std::size_t;
        using storage_type = chunk_type;
        using temporary_type = xchunked_array<chunk_storage, extension>;
    };

    template <class chunk_storage, class extension>
    struct xiterable_inner_types<xchunked_array<chunk_storage, extension>>
    {
        using chunk_type = typename chunk_storage::value_type;
        using inner_shape_type = typename chunk_type::shape_type;
        using const_stepper = xindexed_stepper<xchunked_array<chunk_storage, extension>, true>;
        using stepper = xindexed_stepper<xchunked_array<chunk_storage, extension>, false>;
    };

    template <class chunk_storage, class extension>
    class xchunked_array: public xaccessible<xchunked_array<chunk_storage, extension>>,
                          public xiterable<xchunked_array<chunk_storage, extension>>,
                          public xcontainer_semantic<xchunked_array<chunk_storage, extension>>,
                          public extension
    {
    public:

        using chunk_storage_type = chunk_storage;
        using chunk_type = typename chunk_storage::value_type;
        using const_reference = typename chunk_type::const_reference;
        using reference = typename chunk_type::reference;
        using self_type = xchunked_array<chunk_storage, extension>;
        using semantic_base = xcontainer_semantic<self_type>;
        using iterable_base = xconst_iterable<self_type>;
        using const_stepper = typename iterable_base::const_stepper;
        using stepper = typename iterable_base::stepper;
        using inner_types = xcontainer_inner_types<self_type>;
        using size_type = typename inner_types::size_type;
        using storage_type = typename inner_types::storage_type;
        using value_type = typename storage_type::value_type;
        using pointer = value_type*;
        using const_pointer = const value_type*;
        using difference_type = std::ptrdiff_t;
        using shape_type = typename chunk_type::shape_type;
        using temporary_type = typename inner_types::temporary_type;
        using bool_load_type = xt::bool_load_type<value_type>;
        static constexpr layout_type static_layout = layout_type::dynamic;
        static constexpr bool contiguous_layout = false;

        template <class S, class U = typename std::enable_if<is_chunk_store_manager<chunk_storage>::value>>
        xchunked_array(S&& shape, S&& chunk_shape, const std::string& directory, std::size_t pool_size = 1);

        template <class S, class U = typename std::enable_if<!is_chunk_store_manager<chunk_storage>::value>>
        xchunked_array(S&& shape, S&& chunk_shape);

        ~xchunked_array() = default;

        xchunked_array(const xchunked_array&) = default;
        xchunked_array& operator=(const xchunked_array&) = default;

        xchunked_array(xchunked_array&&) = default;
        xchunked_array& operator=(xchunked_array&&) = default;

        template <class E>
        xchunked_array(const xexpression<E>& e);

        template <class E, class S>
        xchunked_array(const xexpression<E>& e, S&& chunk_shape);

        template <class E>
        xchunked_array& operator=(const xexpression<E>& e);

        template <class E>
        void assign(const xexpression<E>& e);

        const shape_type& shape() const noexcept;
        layout_type layout() const noexcept;
        bool is_contiguous() const noexcept;

        template <class... Idxs>
        reference operator()(Idxs... idxs);

        template <class... Idxs>
        const_reference operator()(Idxs... idxs) const;

        template <class It>
        reference element(It first, It last);

        template <class It>
        const_reference element(It first, It last) const;

        template <class S>
        bool broadcast_shape(S& s, bool reuse_cache = false) const;

        template <class S>
        bool has_linear_assign(const S& strides) const noexcept;

        template <class S>
        stepper stepper_begin(const S& shape) noexcept;
        template <class S>
        stepper stepper_end(const S& shape, layout_type) noexcept;

        template <class S>
        const_stepper stepper_begin(const S& shape) const noexcept;
        template <class S>
        const_stepper stepper_end(const S& shape, layout_type) const noexcept;

        const shape_type& chunk_shape() const;
        chunk_storage_type& chunks();
        const chunk_storage_type& chunks() const;

    private:

        template <class... Idxs>
        using indexes_type = std::pair<std::array<std::size_t, sizeof...(Idxs)>, std::array<std::size_t, sizeof...(Idxs)>>;

        template <class... Idxs>
        using chunk_indexes_type = std::array<std::pair<std::size_t, std::size_t>, sizeof...(Idxs)>;

        template <std::size_t N>
        using static_indexes_type = std::pair<std::array<std::size_t, N>, std::array<std::size_t, N>>;

        using dynamic_indexes_type = std::pair<std::vector<std::size_t>, std::vector<std::size_t>>;

        template <class S1, class S2>
        void resize(S1&& shape, S2&& chunk_shape);

        template <class... Idxs>
        indexes_type<Idxs...> get_indexes(Idxs... idxs) const;

        template <class Idx>
        std::pair<std::size_t, std::size_t> get_chunk_indexes_in_dimension(std::size_t dim, Idx idx) const;

        template <std::size_t... dims, class... Idxs>
        chunk_indexes_type<Idxs...> get_chunk_indexes(std::index_sequence<dims...>, Idxs... idxs) const;

        template <class T, std::size_t N>
        static_indexes_type<N> unpack(const std::array<T, N> &arr) const;

        template <class It>
        dynamic_indexes_type get_indexes_dynamic(It first, It last) const;

        shape_type m_shape;
        shape_type m_chunk_shape;
        chunk_storage_type m_chunks;
    };

    template<class E>
    constexpr bool is_chunked(const xexpression<E>& e);

    /*******************************
     * chunk_helper implementation *
     *******************************/

    namespace detail
    {
        // Workaround for VS2015
        template <class E>
        using try_chunk_shape = decltype(std::declval<E>().chunk_shape());

        template <class E, template <class> class OP, class = void>
        struct chunk_helper_impl
        {
            using is_chunked = std::false_type;
            static const auto& chunk_shape(const xexpression<E>& e)
            {
                return e.derived_cast().shape();
            }
        };

        template <class E, template <class> class OP>
        struct chunk_helper_impl<E, OP, void_t<OP<E>>>
        {
            using is_chunked = std::true_type;
            static const auto& chunk_shape(const xexpression<E>& e)
            {
                return e.derived_cast().chunk_shape();
            }
        };

        template <class E>
        using chunk_helper = chunk_helper_impl<E, try_chunk_shape>;
    }

    template<class E>
    constexpr bool is_chunked(const xexpression<E>&)
    {
        using return_type = typename detail::chunk_helper<E>::is_chunked;
        return return_type::value;
    }

    /*********************************
     * xchunked_array implementation *
     *********************************/

    template <class CS, class EX>
    template <class S, class U>
    inline xchunked_array<CS, EX>::xchunked_array(S&& shape, S&& chunk_shape, const std::string& directory, std::size_t pool_size)
    {
        resize(std::forward<S>(shape), std::forward<S>(chunk_shape));
        m_chunks.set_directory(directory);
        m_chunks.set_pool_size(pool_size);
    }

    template <class CS, class EX>
    template <class S, class U>
    inline xchunked_array<CS, EX>::xchunked_array(S&& shape, S&& chunk_shape)
    {
        resize(std::forward<S>(shape), std::forward<S>(chunk_shape));
    }

    template <class CS, class EX>
    template <class E>
    inline xchunked_array<CS, EX>::xchunked_array(const xexpression<E>& e)
        : xchunked_array(e, detail::chunk_helper<E>::chunk_shape(e))
    {
    }

    template <class CS, class EX>
    template <class E, class S>
    inline xchunked_array<CS, EX>::xchunked_array(const xexpression<E>& e, S&& chunk_shape)
    {
        resize(e.derived_cast().shape(), std::forward<S>(chunk_shape));
        assign(e);
    }

    template <class CS, class EX>
    template <class E>
    inline void xchunked_array<CS, EX>::assign(const xexpression<E>& e)
    {
        xstrided_slice_vector sv(m_chunk_shape.size());  // element slice corresponding to chunk
        std::transform(m_chunk_shape.begin(), m_chunk_shape.end(), sv.begin(),
                       [](auto size) { return range(0, size); });
        shape_type ic(this->dimension());  // index of chunk, initialized to 0...
        size_type ci = 0;
        for (auto& chunk: m_chunks)
        {
            noalias(chunk) = strided_view(e.derived_cast(), sv);
            bool last_chunk = ci == m_chunks.size() - 1;
            if (!last_chunk)
            {
                size_type di = this->dimension() - 1;
                while (true)
                {
                    if (ic[di] + 1 == m_chunks.shape()[di])
                    {
                        ic[di] = 0;
                        sv[di] = range(0, m_chunk_shape[di]);
                        if (di == 0)
                        {
                            break;
                        }
                        else
                        {
                            di--;
                        }
                    }
                    else
                    {
                        ic[di] += 1;
                        sv[di] = range(ic[di] * m_chunk_shape[di], (ic[di] + 1) * m_chunk_shape[di]);
                        break;
                    }
                }
            }
            ++ci;
        }
    }

    template <class CS, class EX>
    template <class E>
    inline auto xchunked_array<CS, EX>::operator=(const xexpression<E>& e) -> self_type&
    {
        assign(e);
        return *this;
    }

    template <class CS, class EX>
    inline auto xchunked_array<CS, EX>::shape() const noexcept -> const shape_type&
    {
        return m_shape;
    }

    template <class CS, class EX>
    inline auto xchunked_array<CS, EX>::layout() const noexcept -> layout_type
    {
        return static_layout;
    }

    template <class CS, class EX>
    inline bool xchunked_array<CS, EX>::is_contiguous() const noexcept
    {
        return false;
    }

    template <class CS, class EX>
    template <class... Idxs>
    inline auto xchunked_array<CS, EX>::operator()(Idxs... idxs) -> reference
    {
        auto ii = get_indexes(idxs...);
        auto& chunk = m_chunks.element(ii.first.cbegin(), ii.first.cend());
        return chunk.element(ii.second.cbegin(), ii.second.cend());
    }

    template <class CS, class EX>
    template <class... Idxs>
    inline auto xchunked_array<CS, EX>::operator()(Idxs... idxs) const -> const_reference
    {
        auto ii = get_indexes(idxs...);
        auto& chunk = m_chunks.element(ii.first.cbegin(), ii.first.cend());
        return chunk.element(ii.second.cbegin(), ii.second.cend());
    }

    template <class CS, class EX>
    template <class It>
    inline auto xchunked_array<CS, EX>::element(It first, It last) -> reference
    {
        auto ii = get_indexes_dynamic(first, last);
        auto& chunk = m_chunks.element(ii.first.begin(), ii.first.end());
        return chunk.element(ii.second.begin(), ii.second.end());
    }

    template <class CS, class EX>
    template <class It>
    inline auto xchunked_array<CS, EX>::element(It first, It last) const -> const_reference
    {
        auto ii = get_indexes_dynamic(first, last);
        auto& chunk = m_chunks.element(ii.first.begin(), ii.first.end());
        return chunk.element(ii.second.begin(), ii.second.end());
    }

    template <class CS, class EX>
    template <class S>
    inline bool xchunked_array<CS, EX>::broadcast_shape(S& s, bool) const
    {
        return xt::broadcast_shape(shape(), s);
    }

    template <class CS, class EX>
    template <class S>
    inline bool xchunked_array<CS, EX>::has_linear_assign(const S& strides) const noexcept
    {
        return false;
    }

    template <class CS, class EX>
    template <class S>
    inline auto xchunked_array<CS, EX>::stepper_begin(const S& shape) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(this, offset);
    }

    template <class CS, class EX>
    template <class S>
    inline auto xchunked_array<CS, EX>::stepper_end(const S& shape, layout_type) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(this, offset, true);
    }

    template <class CS, class EX>
    template <class S>
    inline auto xchunked_array<CS, EX>::stepper_begin(const S& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset);
    }

    template <class CS, class EX>
    template <class S>
    inline auto xchunked_array<CS, EX>::stepper_end(const S& shape, layout_type) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset, true);
    }

    template <class CS, class EX>
    inline auto xchunked_array<CS, EX>::chunks() -> chunk_storage_type&
    {
        return m_chunks;
    }

    template <class CS, class EX>
    inline auto xchunked_array<CS, EX>::chunks() const -> const chunk_storage_type&
    {
        return m_chunks;
    }

    template <class CS, class EX>
    inline auto xchunked_array<CS, EX>::chunk_shape() const -> const shape_type&
    {
        return m_chunk_shape;
    }

    template <class CS, class EX>
    template <class S1, class S2>
    inline void xchunked_array<CS, EX>::resize(S1&& shape, S2&& chunk_shape)
    {
        // compute chunk number in each dimension (shape_of_chunks)
        std::vector<std::size_t> shape_of_chunks(shape.size());
        std::transform
        (
            shape.cbegin(), shape.cend(),
            chunk_shape.cbegin(),
            shape_of_chunks.begin(),
            [](auto s, auto cs)
            {
                std::size_t cn = s / cs;
                if (s % cs > 0)
                    cn += std::size_t(1); // edge_chunk
                return cn;
            }
        );

        // resize the chunk container
        m_chunks.resize(shape_of_chunks);
        // resize each chunk
        for (auto& c: m_chunks)
        {
            c.resize(chunk_shape);
        }

        m_shape = xtl::forward_sequence<shape_type, S1>(shape);
        m_chunk_shape = xtl::forward_sequence<shape_type, S2>(chunk_shape);
    }

    template <class CS, class EX>
    template <class... Idxs>
    inline auto xchunked_array<CS, EX>::get_indexes(Idxs... idxs) const -> indexes_type<Idxs...>
    {
        auto chunk_indexes_packed = get_chunk_indexes(std::make_index_sequence<sizeof...(Idxs)>(), idxs...);
        return unpack(chunk_indexes_packed);
    }

    template <class CS, class EX>
    template <class Idx>
    inline std::pair<std::size_t, std::size_t> xchunked_array<CS, EX>::get_chunk_indexes_in_dimension(std::size_t dim, Idx idx) const
    {
        std::size_t index_of_chunk = static_cast<size_t>(idx) / m_chunk_shape[dim];
        std::size_t index_in_chunk = static_cast<size_t>(idx) - index_of_chunk * m_chunk_shape[dim];
        return std::make_pair(index_of_chunk, index_in_chunk);
    }

    template <class CS, class EX>
    template <std::size_t... dims, class... Idxs>
    inline auto xchunked_array<CS, EX>::get_chunk_indexes(std::index_sequence<dims...>, Idxs... idxs) const
        -> chunk_indexes_type<Idxs...>
    {
        chunk_indexes_type<Idxs...> chunk_indexes = {{get_chunk_indexes_in_dimension(dims, idxs)...}};
        return chunk_indexes;
    }

    template <class CS, class EX>
    template <class T, std::size_t N>
    inline auto xchunked_array<CS, EX>::unpack(const std::array<T, N> &arr) const -> static_indexes_type<N>
    {
        std::array<std::size_t, N> arr0;
        std::array<std::size_t, N> arr1;
        for (std::size_t i = 0; i < N; ++i)
        {
            arr0[i] = std::get<0>(arr[i]);
            arr1[i] = std::get<1>(arr[i]);
        }
        return std::make_pair(arr0, arr1);
    }

    template <class CS, class EX>
    template <class It>
    inline auto xchunked_array<CS, EX>::get_indexes_dynamic(It first, It last) const -> dynamic_indexes_type
    {
        auto size = static_cast<std::size_t>(std::distance(first, last));
        std::vector<std::size_t> indexes_of_chunk(size);
        std::vector<std::size_t> indexes_in_chunk(size);
        for (std::size_t dim = 0; dim < size; ++dim)
        {
            auto chunk_index = get_chunk_indexes_in_dimension(dim, *first++);
            indexes_of_chunk[dim] = chunk_index.first;
            indexes_in_chunk[dim] = chunk_index.second;
        }
        return std::make_pair(indexes_of_chunk, indexes_in_chunk);
    }
}

#endif
