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

        xdisk_reference(T& value)
        {
            m_pvalue = &value;
        }

        operator T()
        {
            return *m_pvalue;
        }

        T operator=(const T value)
        {
            *m_pvalue = value;
            dump_csv(*m_out_file, *m_array);
            return *m_pvalue;
        }

        friend std::ostream& operator<<(std::ostream& os, const xdisk_reference<T>& obj)
        {
            return os << *obj.m_pvalue;
        }

        static std::ifstream* m_in_file;
        static std::ofstream* m_out_file;
        const static xarray<T>* m_array;

    private:

        T* m_pvalue;
    };

    template <class T>
    std::ifstream* xdisk_reference<T>::m_in_file = NULL;

    template <class T>
    std::ofstream* xdisk_reference<T>::m_out_file = NULL;

    template <class T>
    const xarray<T>* xdisk_reference<T>::m_array = NULL;

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

        xdisk_array(xarray<T>& arr, std::ifstream& in_file, std::ofstream& out_file)
        {
            m_array = &arr;
            m_in_file = &in_file;
            m_out_file = &out_file;
            reference::m_array = &arr;
            reference::m_in_file = &in_file;
            reference::m_out_file = &out_file;
        }

        template <class S>
        void resize(S& shape)
        {
            m_array->resize(shape);
        }

        template <class... Idxs>
        inline reference operator()(Idxs... idxs)
        {
            auto i = get_indexes(idxs...);
            return m_array->element(i.cbegin(), i.cend());
        }

        template <class... Idxs>
        inline const_reference operator()(Idxs... idxs) const
        {
            auto index = get_indexes(idxs...);
            return m_array->element(index.cbegin(), index.cend());
        }

        reference operator[](const xindex& index)
        {
            reference el = m_array->element(index.cbegin(), index.cend());
            return el;
        }

        const_reference operator[](const xindex& index) const
        {
            const_reference const_el = m_array->element(index.cbegin(), index.cend());
            return const_el;
        }

        template <class It>
        inline reference element(It first, It last)
        {
            reference el = m_array->element(first, last);
            return el;
        }

        template <class It>
        inline const_reference element(It first, It last) const
        {
            const_reference const_el = m_array->element(first, last);
            return const_el;
        }

    private:

        xarray<T>* m_array;
        std::ifstream* m_in_file;
        std::ofstream* m_out_file;

        template <class... Idxs>
        inline std::array<size_t, sizeof...(Idxs)> get_indexes(Idxs... idxs) const
        {
            std::array<size_t, sizeof...(Idxs)> indexes = {{idxs...}};
            return indexes;
        }
    };
}

#endif
