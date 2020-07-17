#ifndef XTENSOR_CHUNKED_ARRAY_HPP
#define XTENSOR_CHUNKED_ARRAY_HPP

#include <vector>
#include <array>

#include "xarray.hpp"
#include "xnoalias.hpp"
#include "xstrided_view.hpp"

namespace xt
{
    namespace detail
    {
        // Workaround for VS2015
        template <class E>
        using try_chunk_shape = decltype(std::declval<E>().chunk_shape());

        template <class E, template <class> class OP, class = void>
        struct chunk_helper_impl
        {
            static const auto& chunk_shape(const xexpression<E>& e)
            {
                return e.derived_cast().shape();
            }
            using is_chunked = std::false_type;
        };

        template <class E, template <class> class OP>
        struct chunk_helper_impl<E, OP, void_t<OP<E>>>
        {
            static const auto& chunk_shape(const xexpression<E>& e)
            {
                return e.derived_cast().chunk_shape();
            }
            using is_chunked = std::true_type;
        };

        template <class E>
        using chunk_helper = chunk_helper_impl<E, try_chunk_shape>;
    }

    template<class E>
    constexpr bool is_chunked(const xexpression<E>& e)
    {
        using return_type = typename detail::chunk_helper<E>::is_chunked;
        return return_type::value;
    }

    template <class chunk_storage>
    class xchunked_array;

    template <class chunk_storage>
    struct xcontainer_inner_types<xchunked_array<chunk_storage>>
    {
        using chunk_type = typename chunk_storage::value_type;
        using const_reference = typename chunk_type::const_reference;
        using reference = typename chunk_type::reference;
        using size_type = std::size_t;
        using storage_type = chunk_type;
        using temporary_type = xchunked_array<chunk_storage>;
    };

    template <class chunk_storage>
    struct xiterable_inner_types<xchunked_array<chunk_storage>>
    {
        using chunk_type = typename chunk_storage::value_type;
        using inner_shape_type = typename chunk_type::shape_type;
        using const_stepper = xindexed_stepper<xchunked_array<chunk_storage>, true>;
        using stepper = xindexed_stepper<xchunked_array<chunk_storage>, false>;
    };

    template <class chunk_storage>
    class xchunked_array: public xaccessible<xchunked_array<chunk_storage>>,
                          public xiterable<xchunked_array<chunk_storage>>,
                          public xcontainer_semantic<xchunked_array<chunk_storage>>
    {
    public:

        using chunk_type = typename chunk_storage::value_type;
        using const_reference = typename chunk_type::const_reference;
        using reference = typename chunk_type::reference;
        using self_type = xchunked_array<chunk_storage>;
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

        template <class O>
        const_stepper stepper_begin(const O& shape) const noexcept;
        template <class O>
        const_stepper stepper_end(const O& shape, layout_type) const noexcept;

        template <class O>
        stepper stepper_begin(const O& shape) noexcept;
        template <class O>
        stepper stepper_end(const O& shape, layout_type) noexcept;

        const shape_type& shape() const
        {
            return m_shape;
        }

        inline layout_type layout() const noexcept
        {
            return static_layout;
        }

        inline bool is_contiguous() const noexcept
        {
            return false;
        }

        template <class... Idxs>
        inline const_reference operator()(Idxs... idxs) const
        {
            auto ii = get_indexes(idxs...);
            auto& chunk = m_chunks.element(ii.first.cbegin(), ii.first.cend());
            return chunk.element(ii.second.cbegin(), ii.second.cend());
        }

        template <class... Idxs>
        inline reference operator()(Idxs... idxs)
        {
            auto ii = get_indexes(idxs...);
            auto& chunk = m_chunks.element(ii.first.cbegin(), ii.first.cend());
            return chunk.element(ii.second.cbegin(), ii.second.cend());
        }

        template <class S>
        xchunked_array(S shape, S chunk_shape)
        {
            resize(shape, chunk_shape);
        }

        xchunked_array(const xchunked_array&) = default;
        xchunked_array& operator=(const xchunked_array&) = default;

        xchunked_array(xchunked_array&&) = default;
        xchunked_array& operator=(xchunked_array&&) = default;

        template <class E, class S>
        xchunked_array(const xexpression<E>& e, S chunk_shape)
        {
            const auto& shape = e.derived_cast().shape();
            resize_container(m_shape, shape.size());
            std::copy(shape.begin(), shape.end(), m_shape.begin());
            m_chunk_shape = chunk_shape;
            resize(m_shape, m_chunk_shape);
            shape_type ic(dimension());  // index of chunk, initialized to 0...
            xstrided_slice_vector sv;  // element slice corresponding to chunk
            size_t di = 0;  // dimension index
            // initialize slice
            for (auto i: ic)
            {
                sv.push_back(range(0, m_chunk_shape[di]));
                di++;
            }
            size_t ci = 0;
            for (auto& chunk: m_chunks)
            {
                noalias(chunk) = strided_view(e.derived_cast(), sv);
                bool last_chunk = ci == m_chunks.size() - 1;
                if (!last_chunk)
                {
                    di = dimension() - 1;
                    while (true)
                    {
                        if (ic[di] + 1 == m_chunks.shape()[di])
                        {
                            ic[di] = 0;
                            sv[di] = range(0, m_chunk_shape[di]);
                            if (di == 0)
                                break;
                            else
                                di--;

                        }
                        else
                        {
                            ic[di] += 1;
                            sv[di] = range(ic[di] * m_chunk_shape[di], (ic[di] + 1) * m_chunk_shape[di]);
                            break;
                        }
                    }
                }
                ci += 1;
            }
        }

        template <class E>
        xchunked_array(const xexpression<E>& e)
        {
            const auto& chunk_shape = detail::chunk_helper<E>::chunk_shape(e);
            *this = xchunked_array<chunk_storage>(e, chunk_shape);
        }

        template <class E>
        self_type& operator=(const xexpression<E>& e)
        {
            return semantic_base::operator=(e);
        }

        reference operator[](const xindex& index)
        {
            reference el = element(index.cbegin(), index.cend());
            return el;
        }

        const_reference operator[](const xindex& index) const
        {
            const_reference const_el = element(index.cbegin(), index.cend());
            return const_el;
        }

        template <class It>
        inline reference element(It first, It last)
        {
            auto ii = get_indexes_dynamic(first, last);
            auto& chunk = m_chunks.element(ii.first.begin(), ii.first.end());
            return chunk.element(ii.second.begin(), ii.second.end());
        }

        template <class It>
        inline const_reference element(It first, It last) const
        {
            auto ii = get_indexes_dynamic(first, last);
            auto& chunk = m_chunks.element(ii.first.begin(), ii.first.end());
            return chunk.element(ii.second.begin(), ii.second.end());
        }

        size_type dimension() const
        {
            return shape().size();
        }

        shape_type chunk_shape() const
        {
            return m_chunk_shape;
        }

        template <class S>
        bool broadcast_shape(S& s, bool reuse_cache = false) const
        {
            return xt::broadcast_shape(shape(), s);
        }

    private:

        chunk_storage m_chunks;
        shape_type m_shape;
        shape_type m_chunk_shape;

        template <class S>
        void resize(S& shape, S& chunk_shape)
        {
            // compute chunk number in each dimension (shape_of_chunks)
            std::vector<size_t> shape_of_chunks(shape.size());
            size_t di = 0;
            for (auto s: shape)
            {
                size_t cn = s / chunk_shape[di];
                if (s % chunk_shape[di] > 0)
                    cn += 1;  // edge chunk
                shape_of_chunks[di] = cn;
                di++;
            }
            // resize the xarray of chunks
            m_chunks.resize(shape_of_chunks);
            // resize each chunk
            for (auto& c: m_chunks)
                c.resize(chunk_shape);
            m_shape = shape;
            m_chunk_shape = chunk_shape;
        }

        template <class... Idxs>
        inline std::pair<std::array<size_t, sizeof...(Idxs)>, std::array<size_t, sizeof...(Idxs)>> get_indexes(Idxs... idxs) const
        {
            auto chunk_indexes_packed = get_chunk_indexes(std::make_index_sequence<sizeof...(Idxs)>(), idxs...);
            auto chunk_indexes = unpack(chunk_indexes_packed);
            auto indexes_of_chunk = chunk_indexes.first;
            auto indexes_in_chunk = chunk_indexes.second;
            return std::make_pair(indexes_of_chunk, indexes_in_chunk);
        }

        template <class Idx>
        std::pair<size_t, size_t> get_chunk_indexes_in_dimension(size_t dim, Idx idx) const
        {
            size_t index_of_chunk = idx / m_chunk_shape[dim];
            size_t index_in_chunk = idx - index_of_chunk * m_chunk_shape[dim];
            return std::make_pair(index_of_chunk, index_in_chunk);
        }

        template <size_t... dims, class... Idxs>
        std::array<std::pair<size_t, size_t>, sizeof...(Idxs)>
        get_chunk_indexes(std::index_sequence<dims...>, Idxs... idxs) const
        {
            std::array<std::pair<size_t, size_t>, sizeof...(Idxs)> chunk_indexes = {{get_chunk_indexes_in_dimension(dims, idxs)...}};
            return chunk_indexes;
        }

        template <class T, std::size_t N>
        std::pair<std::array<size_t, N>, std::array<size_t, N>> unpack(std::array<T, N> &arr) const
        {
            std::array<size_t, N> arr0;
            std::array<size_t, N> arr1;
            for (size_t i = 0; i < N; ++i)
            {
                arr0[i] = std::get<0>(arr[i]);
                arr1[i] = std::get<1>(arr[i]);
            }
            return std::make_pair(arr0, arr1);
        }

        template <class It>
        inline std::pair<std::vector<size_t>, std::vector<size_t>> get_indexes_dynamic(It first, It last) const
        {
            std::vector<size_t> indexes_of_chunk;
            std::vector<size_t> indexes_in_chunk;
            std::pair<size_t, size_t> chunk_index;
            size_t dim = 0;
            for (auto it = first; it != last; ++it)
            {
                chunk_index = get_chunk_indexes_in_dimension(dim, *it);
                indexes_of_chunk.push_back(chunk_index.first);
                indexes_in_chunk.push_back(chunk_index.second);
                dim++;
            }
            return std::make_pair(indexes_of_chunk, indexes_in_chunk);
        }

        template <class S>
        bool is_trivial_broadcast(const S& str) const noexcept
        {
            return false;
        }
    };

    template <class chunk_storage>
    template <class O>
    inline auto xchunked_array<chunk_storage>::stepper_begin(const O& shape) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset);
    }

    template <class chunk_storage>
    template <class O>
    inline auto xchunked_array<chunk_storage>::stepper_end(const O& shape, layout_type) const noexcept -> const_stepper
    {
        size_type offset = shape.size() - this->dimension();
        return const_stepper(this, offset, true);
    }

    template <class chunk_storage>
    template <class O>
    inline auto xchunked_array<chunk_storage>::stepper_begin(const O& shape) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(this, offset);
    }

    template <class chunk_storage>
    template <class O>
    inline auto xchunked_array<chunk_storage>::stepper_end(const O& shape, layout_type) noexcept -> stepper
    {
        size_type offset = shape.size() - this->dimension();
        return stepper(this, offset, true);
    }
}

#endif
