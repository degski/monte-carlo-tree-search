
// MIT License
//
// Copyright (c) 2018, 2019, 2020 degski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <cassert> // assert
#include <cstdint> // std::intptr_t
#include <cstddef> // std::size_t
#include <cstring> // std::memcpy

#include <algorithm>
#include <array>
#include <type_traits>

namespace ma {

namespace detail {
// Integer LogN.
template<int Base, typename T, typename sfinae = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr T iLog ( const T n_, const T p_ = T ( 0 ) ) noexcept {
    return n_ < Base ? p_ : iLog<Base, T, sfinae> ( n_ / Base, p_ + 1 );
}

// Integer Log2.
template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr T ilog2 ( const T n_ ) noexcept {

    return iLog<2, T> ( n_ );
}

template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr T next_power_2 ( const T n_ ) noexcept {
    return n_ > 2 ? T ( 1 ) << ( ilog2<T> ( n_ - 1 ) + 1 ) : n_;
}

template<typename T, typename = std::enable_if_t<std::conjunction_v<std::is_integral<T>, std::is_unsigned<T>>>>
constexpr bool is_power_2 ( const T n_ ) noexcept {
    return n_ and not( n_ & ( n_ - 1 ) );
}

} // namespace detail

template<typename T, std::intptr_t I, std::intptr_t BaseI = 0,
         typename = std::enable_if_t<std::is_default_constructible<T>::value, T>>
class alignas ( sizeof ( T ) * I > 32 ? 64 : sizeof ( T ) ) Vector {

    public:
    using value_type    = T;
    using pointer       = value_type *;
    using const_pointer = value_type const *;

    using reference       = value_type &;
    using const_reference = value_type const &;
    using rv_reference    = value_type &&;

    using size_type       = std::intptr_t;
    using difference_type = std::make_signed<size_type>;

    using iterator               = pointer;
    using const_iterator         = const_pointer;
    using reverse_iterator       = pointer;
    using const_reverse_iterator = const_pointer;

    using extents_type = std::array<std::intptr_t, 1>;

    private:
    T m_data[ I ];

    public:
    constexpr Vector ( ) noexcept : m_data{ T{} } {}
    constexpr Vector ( Vector const & o_, std::enable_if_t<std::is_copy_constructible<T>::value> * = nullptr ) noexcept {
        if constexpr ( std::is_arithmetic<T>::value ) {
            std::memcpy ( &*begin ( ), &*o_.begin ( ), sizeof ( *this ) );
        }
        else {
            std::copy ( o_.begin ( ), o_.end ( ), begin ( ) );
        }
    }
    constexpr Vector ( Vector && ) noexcept = delete;
    template<typename... Args>
    constexpr Vector ( Args... a_ ) noexcept : m_data{ std::forward<Args> ( a_ )... } {}
    explicit constexpr Vector ( T const & value_, std::enable_if_t<std::is_copy_constructible<T>::value> * = nullptr ) noexcept {
        std::fill ( begin ( ), end ( ), value_ );
    }

    [[nodiscard]] constexpr std::enable_if_t<std::is_copy_assignable<T>::value, Vector &> operator= ( Vector const & rhs_ ) {
        if constexpr ( std::is_arithmetic<T>::value ) {
            std::memcpy ( &*begin ( ), &*rhs_.begin ( ), sizeof ( *this ) );
        }
        else {
            std::copy ( rhs_.begin ( ), rhs_.end ( ), begin ( ) );
        }
    }
    [[nodiscard]] constexpr Vector & operator= ( Vector && ) noexcept = delete;

    constexpr void clear ( ) noexcept {
        if constexpr ( std::is_arithmetic<T>::value ) {
            std::memset ( this, 0, sizeof ( *this ) );
        }
        else {
            std::fill ( begin ( ), end ( ), T{ } );
        }
    }

    [[nodiscard]] constexpr const_iterator begin ( ) const noexcept { return const_iterator{ m_data }; }
    [[nodiscard]] constexpr const_iterator cbegin ( ) const noexcept { return begin ( ); }
    [[nodiscard]] constexpr iterator begin ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).begin ( ) ); }

    [[nodiscard]] constexpr const_iterator end ( ) const noexcept { return const_iterator{ m_data + ( I ) }; }
    [[nodiscard]] constexpr const_iterator cend ( ) const noexcept { return end ( ); }
    [[nodiscard]] constexpr iterator end ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).end ( ) ); }

    [[nodiscard]] constexpr const_iterator rbegin ( ) const noexcept { return const_iterator{ m_data + ( I - size_type{ 1 } ) }; }
    [[nodiscard]] constexpr const_iterator crbegin ( ) const noexcept { return rbegin ( ); }
    [[nodiscard]] constexpr iterator rbegin ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).rbegin ( ) ); }

    [[nodiscard]] constexpr const_iterator rend ( ) const noexcept { return const_iterator{ m_data - size_type{ 1 } }; }
    [[nodiscard]] constexpr const_iterator crend ( ) const noexcept { return rend ( ); }
    [[nodiscard]] constexpr iterator rend ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).rend ( ) ); }

    [[nodiscard]] constexpr value_type & operator( ) ( size_type i_ ) noexcept { return at ( i_ ); }
    [[nodiscard]] constexpr value_type operator( ) ( size_type i_ ) const noexcept { return at ( i_ ); }

    [[nodiscard]] constexpr T & at ( size_type const i_ ) noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        return ( m_data - BaseI )[ i_ ];
    }

    [[nodiscard]] constexpr T at ( size_type const i_ ) const noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        return ( m_data - BaseI )[ i_ ];
    }

    [[nodiscard]] constexpr T & at_r ( size_type const i_ ) noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        return ( m_data + I - 1 + BaseI )[ -i_ ];
    }

    [[nodiscard]] constexpr T at_r ( size_type const i_ ) const noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        return ( m_data + I - 1 + BaseI )[ -i_ ];
    }

    [[nodiscard]] constexpr pointer data ( ) noexcept { return m_data; }
    [[nodiscard]] constexpr const_pointer data ( ) const noexcept { return m_data; }

    [[nodiscard]] static constexpr std::size_t size ( ) noexcept { return I; }
    [[nodiscard]] static constexpr std::size_t capacity ( ) noexcept { return size ( ); }
    [[nodiscard]] static constexpr extents_type extents ( ) noexcept { return extents_type{ I }; }
};

template<typename T, std::intptr_t I, std::intptr_t J, std::intptr_t BaseI = 0, std::intptr_t BaseJ = 0,
         typename = std::enable_if_t<std::is_default_constructible<T>::value, T>>
class alignas ( sizeof ( T ) * I * J > 32 ? 64 : sizeof ( T ) ) Matrix {

    public:
    using value_type    = T;
    using pointer       = value_type *;
    using const_pointer = value_type const *;

    using reference       = value_type &;
    using const_reference = value_type const &;
    using rv_reference    = value_type &&;

    using size_type       = std::intptr_t;
    using difference_type = std::make_signed<size_type>;

    using iterator               = pointer;
    using const_iterator         = const_pointer;
    using reverse_iterator       = pointer;
    using const_reverse_iterator = const_pointer;

    using extents_type = std::array<std::intptr_t, 2>;

    private:
    T m_data[ I * J ];

    public:
    constexpr Matrix ( ) noexcept : m_data{ T{} } {}
    constexpr Matrix ( Matrix const & o_, std::enable_if_t<std::is_copy_constructible<T>::value> * = nullptr ) noexcept {
        if constexpr ( std::is_arithmetic<T>::value ) {
            std::memcpy ( &*begin ( ), &*o_.begin ( ), sizeof ( *this ) );
        }
        else {
            std::copy ( o_.begin ( ), o_.end ( ), begin ( ) );
        }
    }
    constexpr Matrix ( Matrix && ) noexcept = delete;
    template<typename... Args>
    constexpr Matrix ( Args... a_ ) noexcept : m_data{ std::forward<Args> ( a_ )... } {}
    explicit constexpr Matrix ( T const & value_, std::enable_if_t<std::is_copy_constructible<T>::value> * = nullptr ) noexcept {
        std::fill ( begin ( ), end ( ), value_ );
    }

    [[nodiscard]] constexpr std::enable_if_t<std::is_copy_assignable<T>::value, Matrix &> operator= ( Matrix const & rhs_ ) {
        if constexpr ( std::is_arithmetic<T>::value ) {
            std::memcpy ( &*begin ( ), &*rhs_.begin ( ), sizeof ( *this ) );
        }
        else {
            std::copy ( rhs_.begin ( ), rhs_.end ( ), begin ( ) );
        }
    }
    [[nodiscard]] constexpr Matrix & operator= ( Matrix && ) noexcept = delete;

    constexpr void clear ( ) noexcept {
        if constexpr ( std::is_arithmetic<T>::value ) {
            std::memset ( this, 0, sizeof ( *this ) );
        }
        else {
            std::fill ( begin ( ), end ( ), T{ } );
        }
    }

    [[nodiscard]] constexpr const_iterator begin ( ) const noexcept { return const_iterator{ m_data }; }
    [[nodiscard]] constexpr const_iterator cbegin ( ) const noexcept { return begin ( ); }
    [[nodiscard]] constexpr iterator begin ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).begin ( ) ); }

    [[nodiscard]] constexpr const_iterator end ( ) const noexcept { return const_iterator{ m_data + ( I * J ) }; }
    [[nodiscard]] constexpr const_iterator cend ( ) const noexcept { return end ( ); }
    [[nodiscard]] iterator end ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).end ( ) ); }

    [[nodiscard]] constexpr const_iterator rbegin ( ) const noexcept {
        return const_iterator{ m_data + ( I * J - size_type{ 1 } ) };
    }
    [[nodiscard]] constexpr const_iterator crbegin ( ) const noexcept { return rbegin ( ); }
    [[nodiscard]] constexpr iterator rbegin ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).rbegin ( ) ); }

    [[nodiscard]] constexpr const_iterator rend ( ) const noexcept { return const_iterator{ m_data - size_type{ 1 } }; }
    [[nodiscard]] constexpr const_iterator crend ( ) const noexcept { return rend ( ); }
    [[nodiscard]] constexpr iterator rend ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).rend ( ) ); }

    [[nodiscard]] constexpr value_type & operator( ) ( size_type i_, size_type j_ ) noexcept { return at ( i_, j_ ); }
    [[nodiscard]] constexpr value_type operator( ) ( size_type i_, size_type j_ ) const noexcept { return at ( i_, j_ ); }

    [[nodiscard]] constexpr T const & ref ( size_type const i_, size_type const j_ ) const noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        assert ( j_ >= BaseJ );
        assert ( j_ < J + BaseJ );
        return ( m_data - BaseJ - BaseI * J )[ j_ + i_ * J ];
    }

    [[nodiscard]] constexpr T & at ( size_type const i_, size_type const j_ ) noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        assert ( j_ >= BaseJ );
        assert ( j_ < J + BaseJ );
        return ( m_data - BaseJ - BaseI * J )[ j_ + i_ * J ];
    }

    [[nodiscard]] constexpr T at ( size_type const i_, size_type const j_ ) const noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        assert ( j_ >= BaseJ );
        assert ( j_ < J + BaseJ );
        return ( m_data - BaseJ - BaseI * J )[ j_ + i_ * J ];
    }

    // Mirror the matrix coordinates.
    [[nodiscard]] constexpr T const & ref_r ( size_type const i_, size_type const j_ ) noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        assert ( j_ >= BaseJ );
        assert ( j_ < J + BaseJ );
        return ( m_data + I * J - 1 + BaseJ + BaseI * J )[ -j_ - i_ * J ];
    }

    // Mirror the matrix coordinates.
    [[nodiscard]] constexpr T & at_r ( size_type const i_, size_type const j_ ) noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        assert ( j_ >= BaseJ );
        assert ( j_ < J + BaseJ );
        return ( m_data + I * J - 1 + BaseJ + BaseI * J )[ -j_ - i_ * J ];
    }

    // Mirror the matrix coordinates.
    [[nodiscard]] constexpr T at_r ( size_type const i_, size_type const j_ ) const noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        assert ( j_ >= BaseJ );
        assert ( j_ < J + BaseJ );
        return ( m_data + I * J - 1 + BaseJ + BaseI * J )[ -j_ - i_ * J ];
    }

    [[nodiscard]] constexpr pointer data ( ) noexcept { return m_data; }
    [[nodiscard]] constexpr const_pointer data ( ) const noexcept { return m_data; }

    [[nodiscard]] static constexpr std::size_t size ( ) noexcept { return I * J; }
    [[nodiscard]] static constexpr std::size_t capacity ( ) noexcept { return size ( ); }
    [[nodiscard]] static constexpr extents_type extents ( ) noexcept { return extents_type{ I, J }; }
}; // namespace ma

template<typename T, std::intptr_t I, std::intptr_t J, std::intptr_t BaseI = 0, std::intptr_t BaseJ = 0,
         typename = std::enable_if_t<std::is_default_constructible<T>::value, T>>
using MatrixRM = Matrix<T, I, J, BaseI, BaseJ>;

template<typename T, std::intptr_t J, std::intptr_t I, std::intptr_t BaseJ = 0, std::intptr_t BaseI = 0,
         typename = std::enable_if_t<std::is_default_constructible<T>::value, T>>
using MatrixCM = Matrix<T, J, I, BaseJ, BaseI>;

template<typename T, std::intptr_t I, std::intptr_t J, std::intptr_t K, std::intptr_t BaseI = 0, std::intptr_t BaseJ = 0,
         std::intptr_t BaseK = 0, typename = std::enable_if_t<std::is_default_constructible<T>::value, T>>
class alignas ( sizeof ( T ) * I * J * K > 32 ? 64 : sizeof ( T ) ) Cube {

    public:
    using value_type    = T;
    using pointer       = value_type *;
    using const_pointer = value_type const *;

    using reference       = value_type &;
    using const_reference = value_type const &;
    using rv_reference    = value_type &&;

    using size_type       = std::intptr_t;
    using difference_type = std::make_signed<size_type>;

    using iterator               = pointer;
    using const_iterator         = const_pointer;
    using reverse_iterator       = pointer;
    using const_reverse_iterator = const_pointer;

    using extents_type = std::array<std::intptr_t, 3>;

    private:
    T m_data[ I * J * K ];

    public:
    constexpr Cube ( ) noexcept : m_data{ T{} } {}
    constexpr Cube ( Cube const & o_, std::enable_if_t<std::is_copy_constructible<T>::value> * = nullptr ) noexcept {
        if constexpr ( std::is_arithmetic<T>::value ) {
            std::memcpy ( &*begin ( ), &*o_.begin ( ), sizeof ( *this ) );
        }
        else {
            std::copy ( o_.begin ( ), o_.end ( ), begin ( ) );
        }
    }
    constexpr Cube ( Cube && ) noexcept = delete;
    template<typename... Args>
    constexpr Cube ( Args... a_ ) noexcept : m_data{ std::forward<Args> ( a_ )... } {}
    explicit constexpr Cube ( T const & value_, std::enable_if_t<std::is_copy_constructible<T>::value> * = nullptr ) noexcept {
        std::fill ( begin ( ), end ( ), value_ );
    }

    [[nodiscard]] constexpr std::enable_if_t<std::is_copy_assignable<T>::value, Cube &> operator= ( Cube const & rhs_ ) {
        if constexpr ( std::is_arithmetic<T>::value ) {
            std::memcpy ( &*begin ( ), &*rhs_.begin ( ), sizeof ( *this ) );
        }
        else {
            std::copy ( rhs_.begin ( ), rhs_.end ( ), begin ( ) );
        }
    }
    [[nodiscard]] constexpr Cube & operator= ( Cube && ) noexcept = delete;

    constexpr void clear ( ) noexcept {
        if constexpr ( std::is_arithmetic<T>::value ) {
            std::memset ( this, 0, sizeof ( *this ) );
        }
        else {
            std::fill ( begin ( ), end ( ), T{ } );
        }
    }

    [[nodiscard]] constexpr const_iterator begin ( ) const noexcept { return const_iterator{ m_data }; }
    [[nodiscard]] constexpr const_iterator cbegin ( ) const noexcept { return begin ( ); }
    [[nodiscard]] constexpr iterator begin ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).begin ( ) ); }

    [[nodiscard]] constexpr const_iterator end ( ) const noexcept { return const_iterator{ m_data + ( I * J * K ) }; }
    [[nodiscard]] constexpr const_iterator cend ( ) const noexcept { return end ( ); }
    [[nodiscard]] constexpr iterator end ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).end ( ) ); }

    [[nodiscard]] constexpr const_iterator rbegin ( ) const noexcept {
        return const_iterator{ m_data + ( I * J * K - size_type{ 1 } ) };
    }
    [[nodiscard]] constexpr const_iterator crbegin ( ) const noexcept { return rbegin ( ); }
    [[nodiscard]] constexpr iterator rbegin ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).rbegin ( ) ); }

    [[nodiscard]] constexpr const_iterator rend ( ) const noexcept { return const_iterator{ m_data - size_type{ 1 } }; }
    [[nodiscard]] constexpr const_iterator crend ( ) const noexcept { return rend ( ); }
    [[nodiscard]] constexpr iterator rend ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).rend ( ) ); }

    [[nodiscard]] constexpr value_type & operator( ) ( size_type i_, size_type j_, size_type k_ ) noexcept {
        return at ( i_, j_, k_ );
    }
    [[nodiscard]] constexpr value_type operator( ) ( size_type i_, size_type j_, size_type k_ ) const noexcept {
        return at ( i_, j_, k_ );
    }

    [[nodiscard]] constexpr T & at ( size_type const i_, size_type const j_, size_type const k_ ) noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        assert ( j_ >= BaseJ );
        assert ( j_ < J + BaseJ );
        assert ( k_ >= BaseK );
        assert ( k_ < K + BaseK );
        return ( m_data + K * ( -BaseJ - BaseI * J ) - BaseK )[ K * ( j_ + i_ * J ) + k_ ];
    }

    [[nodiscard]] constexpr T at ( size_type const i_, size_type const j_, size_type const k_ ) const noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        assert ( j_ >= BaseJ );
        assert ( j_ < J + BaseJ );
        assert ( k_ >= BaseK );
        assert ( k_ < K + BaseK );
        return ( m_data + K * ( -BaseJ - BaseI * J ) - BaseK )[ K * ( j_ + i_ * J ) + k_ ];
    }

    [[nodiscard]] constexpr T & at_r ( size_type const i_, size_type const j_, size_type const k_ ) noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        assert ( j_ >= BaseJ );
        assert ( j_ < J + BaseJ );
        assert ( k_ >= BaseK );
        assert ( k_ < K + BaseK );
        return ( m_data + I * J * K - 1 + BaseJ * K + BaseI * J * K + BaseK )[ K * ( -j_ - i_ * J ) - k_ ];
    }

    [[nodiscard]] constexpr T at_r ( size_type const i_, size_type const j_, size_type const k_ ) const noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        assert ( j_ >= BaseJ );
        assert ( j_ < J + BaseJ );
        assert ( k_ >= BaseK );
        assert ( k_ < K + BaseK );
        return ( m_data + I * J * K - 1 + BaseJ * K + BaseI * J * K + BaseK )[ K * ( -j_ - i_ * J ) - k_ ];
    }

    [[nodiscard]] constexpr pointer data ( ) noexcept { return m_data; }
    [[nodiscard]] constexpr const_pointer data ( ) const noexcept { return m_data; }

    [[nodiscard]] static constexpr std::size_t size ( ) noexcept { return I * J * K; }
    [[nodiscard]] static constexpr std::size_t capacity ( ) noexcept { return size ( ); }
    [[nodiscard]] static constexpr extents_type extents ( ) noexcept { return extents_type{ I, J, K }; }
};

template<typename T, std::intptr_t I, std::intptr_t J, std::intptr_t K, std::intptr_t L, std::intptr_t BaseI = 0,
         std::intptr_t BaseJ = 0, std::intptr_t BaseK = 0, std::intptr_t BaseL = 0,
         typename = std::enable_if_t<std::is_default_constructible<T>::value, T>>
class alignas ( sizeof ( T ) * I * J * K * L > 32 ? 64 : sizeof ( T ) ) HyperCube {

    public:
    using value_type    = T;
    using pointer       = value_type *;
    using const_pointer = value_type const *;

    using reference       = value_type &;
    using const_reference = value_type const &;
    using rv_reference    = value_type &&;

    using size_type       = std::intptr_t;
    using difference_type = std::make_signed<size_type>;

    using iterator               = pointer;
    using const_iterator         = const_pointer;
    using reverse_iterator       = pointer;
    using const_reverse_iterator = const_pointer;

    using extents_type = std::array<std::intptr_t, 4>;

    private:
    T m_data[ I * J * K * L ];

    public:
    constexpr HyperCube ( ) noexcept : m_data{ T{} } {}
    constexpr HyperCube ( HyperCube const & o_, std::enable_if_t<std::is_copy_constructible<T>::value> * = nullptr ) noexcept {
        if constexpr ( std::is_arithmetic<T>::value ) {
            std::memcpy ( &*begin ( ), &*o_.begin ( ), sizeof ( *this ) );
        }
        else {
            std::copy ( o_.begin ( ), o_.end ( ), begin ( ) );
        }
    }
    constexpr HyperCube ( HyperCube && ) noexcept = delete;
    template<typename... Args>
    constexpr HyperCube ( Args... a_ ) noexcept : m_data{ std::forward<Args> ( a_ )... } {}
    explicit constexpr HyperCube ( T const & value_, std::enable_if_t<std::is_copy_constructible<T>::value> * = nullptr ) noexcept {
        std::fill ( begin ( ), end ( ), value_ );
    }

    [[nodiscard]] constexpr std::enable_if_t<std::is_copy_assignable<T>::value, HyperCube &> operator= ( HyperCube const & rhs_ ) {
        if constexpr ( std::is_arithmetic<T>::value ) {
            std::memcpy ( &*begin ( ), &*rhs_.begin ( ), sizeof ( *this ) );
        }
        else {
            std::copy ( rhs_.begin ( ), rhs_.end ( ), begin ( ) );
        }
    }
    [[nodiscard]] constexpr HyperCube & operator= ( HyperCube && ) noexcept = delete;

    constexpr void clear ( ) noexcept {
        if constexpr ( std::is_arithmetic<T>::value ) {
            std::memset ( this, 0, sizeof ( *this ) );
        }
        else {
            std::fill ( begin ( ), end ( ), T{ } );
        }
    }

    [[nodiscard]] constexpr const_iterator begin ( ) const noexcept { return const_iterator{ m_data }; }
    [[nodiscard]] constexpr const_iterator cbegin ( ) const noexcept { return begin ( ); }
    [[nodiscard]] constexpr iterator begin ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).begin ( ) ); }

    [[nodiscard]] constexpr const_iterator end ( ) const noexcept { return const_iterator{ m_data + ( I * J * K * L ) }; }
    [[nodiscard]] constexpr const_iterator cend ( ) const noexcept { return end ( ); }
    [[nodiscard]] constexpr iterator end ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).end ( ) ); }

    [[nodiscard]] constexpr const_iterator rbegin ( ) const noexcept {
        return const_iterator{ m_data + ( I * J * K * L - size_type{ 1 } ) };
    }
    [[nodiscard]] constexpr const_iterator crbegin ( ) const noexcept { return rbegin ( ); }
    [[nodiscard]] constexpr iterator rbegin ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).rbegin ( ) ); }

    [[nodiscard]] constexpr const_iterator rend ( ) const noexcept { return const_iterator{ m_data - size_type{ 1 } }; }
    [[nodiscard]] constexpr const_iterator crend ( ) const noexcept { return rend ( ); }
    [[nodiscard]] constexpr iterator rend ( ) noexcept { return const_cast<iterator> ( std::as_const ( *this ).rend ( ) ); }

    [[nodiscard]] constexpr value_type & operator( ) ( size_type i_, size_type j_, size_type k_, size_type l_ ) noexcept {
        return at ( i_, j_, k_, l_ );
    }
    [[nodiscard]] constexpr value_type operator( ) ( size_type i_, size_type j_, size_type k_, size_type l_ ) const noexcept {
        return at ( i_, j_, k_, l_ );
    }

    [[nodiscard]] constexpr T & at ( size_type const i_, size_type const j_, size_type const k_, size_type const l_ ) noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        assert ( j_ >= BaseJ );
        assert ( j_ < J + BaseJ );
        assert ( k_ >= BaseK );
        assert ( k_ < K + BaseK );
        assert ( l_ >= BaseL );
        assert ( l_ < L + BaseL );
        return ( m_data + L * ( K * ( -BaseJ - BaseI * J ) - BaseK ) - BaseL )[ L * ( K * ( j_ + i_ * J ) + k_ ) + l_ ];
    }

    [[nodiscard]] constexpr T at ( size_type const i_, size_type const j_, size_type const k_, size_type const l_ ) const noexcept {
        assert ( i_ >= BaseI );
        assert ( i_ < I + BaseI );
        assert ( j_ >= BaseJ );
        assert ( j_ < J + BaseJ );
        assert ( k_ >= BaseK );
        assert ( k_ < K + BaseK );
        assert ( l_ >= BaseL );
        assert ( l_ < L + BaseL );
        return ( m_data + L * ( K * ( -BaseJ - BaseI * J ) - BaseK ) - BaseL )[ L * ( K * ( j_ + i_ * J ) + k_ ) + l_ ];
    }

    [[nodiscard]] constexpr pointer data ( ) noexcept { return m_data; }
    [[nodiscard]] constexpr const_pointer data ( ) const noexcept { return m_data; }

    [[nodiscard]] static constexpr std::size_t size ( ) noexcept { return I * J * K * L; }
    [[nodiscard]] static constexpr std::size_t capacity ( ) noexcept { return size ( ); }
    [[nodiscard]] static constexpr extents_type extents ( ) noexcept { return extents_type{ I, J, K, L }; }
};
}; // namespace ma
