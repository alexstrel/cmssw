#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_alpakaReducer_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaReducer_h

#include "HeterogeneousCore/AlpakaInterface/interface/blas/BlasHelpers.h"
#include "HeterogeneousCore/AlpakaInterface/interface/blas/AtomicType.h"

namespace cms::alpakatools {
  namespace reduce {

    template <std::size_t Size, std::size_t Alignment>
    struct alignas(Alignment) SmemBuffer {
      std::byte data[Size];

      template <typename T>
      ALPAKA_FN_ACC T* as() noexcept {
        static_assert(alignof(T) <= Alignment);
        static_assert(sizeof(T) <= Size);

        return reinterpret_cast<T*>(data);
      }

      template <typename T>
      ALPAKA_FN_ACC T const* as() const noexcept {
        static_assert(alignof(T) <= Alignment);
        static_assert(sizeof(T) <= Size);

        return reinterpret_cast<T const*>(data);
      }
    };

    using namespace cms::alpakatools::blas;

    template <typename T>
      requires std::is_arithmetic_v<T>
    constexpr T zero() {
      return static_cast<T>(0);
    }

    template <typename T>
      requires extfloat::ExtendedFloatType<T>
    constexpr T zero() {
      return extfloat::ExtendedFloat<T>(static_cast<T>(0.));
    }

    template <typename T, std::int32_t N>
    constexpr VecArray<T, N> zero() {
      VecArray<T, N> v;
      CMS_UNROLL_LOOP
      for (std::int32_t i = 0; i < N; i++)
        v[i] = zero<T>();
      return v;
    }

    template <typename T, typename U>
      requires std::is_arithmetic_v<T> && std::is_arithmetic_v<U>
    constexpr T set(U x) {
      return static_cast<T>(x);
    }

    template <typename T, typename U>
      requires extfloat::ExtendedFloatType<T> && std::is_arithmetic_v<U>
    constexpr T set(U x) {
      return extfloat::ExtendedFloat<T>(x);
    }

    template <typename T, typename U, std::int32_t N>
    constexpr VecArray<T, N> set(VecArray<U, N> const& x) {
      VecArray<T, N> v;
      CMS_UNROLL_LOOP
      for (std::int32_t i = 0; i < N; ++i)
        v[i] = set<T>(x[i]);
      return v;
    }

    template <typename T>
    struct Set {
      template <typename U>
        requires std::is_arithmetic_v<T> && std::is_arithmetic_v<U>
      ALPAKA_FN_HOST_ACC constexpr auto operator()(U const& x) const noexcept -> T {
        return static_cast<T>(x);
      }

      template <typename U>
        requires extfloat::ExtendedFloatType<T> && std::is_floating_point_v<U>
      ALPAKA_FN_HOST_ACC constexpr auto operator()(U const& x) const noexcept -> T {
        return T{x};
      }

      template <typename U>
        requires extfloat::ExtendedFloatType<T> && std::same_as<std::remove_cvref_t<U>, T>
      ALPAKA_FN_HOST_ACC constexpr auto operator()(U const& x) const noexcept -> T {
        return x;
      }

      template <typename U, std::int32_t N>
        requires std::is_arithmetic_v<T> && std::is_arithmetic_v<U>
      ALPAKA_FN_HOST_ACC constexpr auto operator()(VecArray<U, N> const& x) const noexcept -> VecArray<T, N> {
        VecArray<T, N> result;

        CMS_UNROLL_LOOP
        for (std::int32_t i = 0; i < N; ++i) {
          result[i] = (*this)(x[i]);
        }

        return result;
      }
    };

    template <std::floating_point T>
    constexpr auto HighComponent(T const& x) noexcept {
      return x;
    }

    template <extfloat::ExtendedFloatType T>
    constexpr auto HighComponent(T const& x) noexcept {
      return x.HighComponent();
    }

    template <std::floating_point T>
    constexpr auto LowComponent(T const& x) noexcept {
      return static_cast<T>(0);
    }

    template <extfloat::ExtendedFloatType T>
    constexpr auto LowComponent(T const& x) noexcept {
      return x.LowComponent();
    }

    template <std::floating_point T>
    constexpr auto ReducerResult(T const& x) noexcept {
      return x;
    }

    template <extfloat::ExtendedFloatType T>
    constexpr auto ReducerResult(T const& x) noexcept {
      return x.Aggregate();
    }

    /**
      plus reducer, used for conventional sum reductions
    */
    template <typename T>
    struct plus {
      static constexpr bool use_augmented_reduction = extfloat::is_extended_float<std::remove_cvref_t<T>>::value;

      using reduce_t = T;
      using reducer_t = plus<reduce_t>;

      static constexpr reduce_t apply(reduce_t a, reduce_t b) { return a + b; }
      constexpr reduce_t operator()(reduce_t a, reduce_t b) const { return apply(a, b); }
    };

  }  // namespace reduce
}  // namespace cms::alpakatools

#endif
