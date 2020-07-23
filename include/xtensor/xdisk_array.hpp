#ifndef XTENSOR_DISK_ARRAY_HPP
#define XTENSOR_DISK_ARRAY_HPP

#include "xarray.hpp"
#include "xcsv.hpp"

namespace xt
{
    template <class T>
    class xdisk_reference
    {
    public:

        xdisk_reference(T& value, bool* array_dirty)
        {
            m_pvalue = &value;
            m_array_dirty = array_dirty;
        }

        operator const T() const
        {
            return *m_pvalue;
        }

        operator T()
        {
            return *m_pvalue;
        }

        T operator=(const T value)
        {
            if (value != *m_pvalue)
            {
                *m_array_dirty = true;
                *m_pvalue = value;
            }
            return *m_pvalue;
        }

        friend std::ostream& operator<<(std::ostream& os, const xdisk_reference<T>& obj)
        {
            return os << *obj.m_pvalue;
        }

    private:

        T* m_pvalue;
        bool* m_array_dirty;

    };

    template <class T>
    class xdisk_array;

    template <class T>
    struct xcontainer_inner_types<xdisk_array<T>>
    {
        using const_reference = const T&;
        using reference = T&;
        using size_type = std::size_t;
        using storage_type = T;
    };

    template <class EC>
    struct xiterable_inner_types<xdisk_array<EC>>
    {
        using inner_shape_type = std::vector<size_t>;
        using const_stepper = xindexed_stepper<xdisk_array<EC>, true>;
        using stepper = xindexed_stepper<xdisk_array<EC>, false>;
    };

    template <class T>
    class xdisk_array: public xaccessible<xdisk_array<T>>,
                       public xiterable<xdisk_array<T>>
    {
    public:

        using const_reference = const T&;
        using reference = xdisk_reference<T>;
        using shape_type = std::vector<size_t>;
        using self_type = xdisk_array<T>;
        using inner_types = xcontainer_inner_types<self_type>;
        using size_type = typename inner_types::size_type;
        using storage_type = typename inner_types::storage_type;
        using value_type = storage_type;

        xdisk_array() {}

        xdisk_array(xarray<T>& array, bool& array_dirty)
        {
            m_array = &array;
            m_array_dirty = &array_dirty;
        }

        template <class S>
        void resize(S& shape)
        {
            m_array->resize(shape);
            *m_array = broadcast(0, shape);
        }

        template <class... Idxs>
        inline reference operator()(Idxs... idxs)
        {
            auto index = get_indexes(idxs...);
            return reference(m_array->element(index.cbegin(), index.cend()), m_array_dirty);
        }

        template <class... Idxs>
        inline const_reference operator()(Idxs... idxs) const
        {
            auto index = get_indexes(idxs...);
            return m_array->element(index.cbegin(), index.cend());
        }

        reference operator[](const xindex& index)
        {
            return reference(m_array->element(index.cbegin(), index.cend()), m_array_dirty);
        }

        const_reference operator[](const xindex& index) const
        {
            return m_array->element(index.cbegin(), index.cend());
        }

        template <class It>
        inline reference element(It first, It last)
        {
            return reference(m_array->element(first, last), m_array_dirty);
        }

        template <class It>
        inline const_reference element(It first, It last) const
        {
            return m_array->element(first, last);
        }

    private:

        xarray<T>* m_array;
        bool* m_array_dirty;

        template <class... Idxs>
        inline std::array<size_t, sizeof...(Idxs)> get_indexes(Idxs... idxs) const
        {
            std::array<size_t, sizeof...(Idxs)> indexes = {{idxs...}};
            return indexes;
        }
    };
}

#endif
