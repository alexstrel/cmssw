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

    using namespace cms::alpakablasintrinsics;

    template <typename T>
      requires std::is_arithmetic_v<T>
    constexpr T zero() {
      return static_cast<T>(0);
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

    template <typename T, typename U, std::int32_t N>
    constexpr VecArray<T, N> set(VecArray<U, N> const& x) {
      VecArray<T, N> v;
      CMS_UNROLL_LOOP
      for (std::int32_t i = 0; i < N; ++i)
        v[i] = set<T>(x[i]);
      return v;
    }

    template <std::floating_point T>
    using kahan_accumulator = VecArray<T, 2>;

    template <typename T>
    constexpr T& accum(kahan_accumulator<T>& x) {
      return x[0];
    }

    template <typename T>
    constexpr T const& accum(kahan_accumulator<T> const& x) {
      return x[0];
    }

    template <typename T>
    constexpr T& compensation(kahan_accumulator<T>& x) {
      return x[1];
    }

    template <typename T>
    constexpr T const& compensation(kahan_accumulator<T> const& x) {
      return x[1];
    }

    template <std::floating_point T>
    constexpr auto two_sum(const T a, const T b) {
      const T s = warp::add_rn(a, b);

      const T a_prime = warp::sub_rn(s, b);
      const T b_prime = warp::sub_rn(s, a_prime);

      const T delta_a = warp::sub_rn(a, a_prime);
      const T delta_b = warp::sub_rn(b, b_prime);

      const T t = warp::add_rn(delta_a, delta_b);

      return kahan_accumulator<T>{s, t};
    }

    template <std::floating_point T>
    constexpr auto fast_two_sum(const T a, const T b) {
      const T s = warp::add_rn(a, b);
      const T z = warp::sub_rn(s, a);
      const T t = warp::sub_rn(b, z);

      return kahan_accumulator<T>{s, t};
    }

    template <typename T, bool use_compensated_sum>
    constexpr decltype(auto) result(T const& x) noexcept {
      if constexpr (use_compensated_sum) {
        return accum(x) + compensation(x);
      } else {
        return x;  //nop
      }
    }

    /**
      plus reducer, used for conventional sum reductions
    */

    template <typename T, bool use_compensated_sum_ = false>
    struct plus {
      static constexpr bool use_compensated_sum = use_compensated_sum_;

      using reduce_t = T;
      using reducer_t = plus<reduce_t, use_compensated_sum>;

      static constexpr reduce_t apply(reduce_t a, reduce_t b) {
        return a + b;  // assume it's defined for VecArrays..
      }
      constexpr reduce_t operator()(reduce_t a, reduce_t b) const { return apply(a, b); }

      // for improved sloppy reducer only:
      using atomic_t = cms::alpakatools::atomic_type_t<reduce_t>;
      using kahan_atomic_t = kahan_accumulator<atomic_t>;

      static constexpr kahan_atomic_t apply(atomic_t a, atomic_t b)
        requires(use_compensated_sum)
      {
        return two_sum(a, b);
      }
      constexpr kahan_atomic_t operator()(atomic_t a, atomic_t b) const
        requires(use_compensated_sum)
      {
        return apply(a, b);
      }

      static constexpr kahan_atomic_t apply(kahan_atomic_t a, kahan_atomic_t b)
        requires(use_compensated_sum)
      {
        // Add high-order components:
        const kahan_atomic_t hi_sum = two_sum(accum(a), accum(b));
        // Add low-order components:
        const kahan_atomic_t lo_sum = two_sum(compensation(a), compensation(b));

        compensation(hi_sum) = warp::add_rn(componsation(hi_sum), accum(lo_sum));

        // Renormalize hi_sum:
        const kahan_atomic_t v = fast_two_sum(accum(hi_sum), compensation(hi_sum));

        compensation(lo_sum) = warp::add_rn(componsation(lo_sum), compensation(v));

        // Renormalize lo_sum:
        const kahan_atomic_t z = fast_two_sum(accum(v), compensation(lo_sum));

        return z;
      }
      constexpr kahan_atomic_t operator()(kahan_atomic_t a, kahan_atomic_t b) const
        requires(use_compensated_sum)
      {
        return apply(a, b);
      }
    };

  }  // namespace reduce
}  // namespace cms::alpakatools

#endif
