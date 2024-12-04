#ifndef HeterogeneousCore_AlpakaInterface_interface_blas alpakaReducer_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaReducer_h

namespace cms::alpakatools {
  namespace reduce {

    template <typename T>
    constexpr std::enable_if_t<std::is_arithmetic_v<T>, T> zero() {
      return static_cast<T>(0);
    }

    template <typename T, std::int32_t N>
    ALPAKA_FN_HOST_ACC inline VecArray<T, N> zero() {
      VecArray<T, N> v;
      CMS_UNROLL_UNROLL
      for (std::int32_t i = 0; i < N; i++)
        v[i] = zero<T>();
      return v;
    }

    template <typename T, typename U>
    constexpr std::enable_if_t<std::is_arithmetic_v<T> and std::is_arithmetic_v<U>, T> set(U x) {
      return static_cast<T>(x);
    }

    /**
      plus reducer, used for conventional sum reductions
    */

    template <typename T>
    struct plus {
      static constexpr bool do_sum = true;
      //
      using reduce_t = T;
      using reducer_t = plus<T>;

      ALPAKA_FN_HOST_ACC static inline T init() { return zero<T>(); }

      template <typename U>
      ALPAKA_FN_HOST_ACC static inline T init(U &in) {
        return set<U>(in);
      }

      ALPAKA_FN_HOST_ACC static inline T apply(T a, T b) { return a + b; }

      ALPAKA_FN_HOST_ACC inline T operator()(T a, T b) const { return apply(a, b); }
    };

    template <typename T, typename U>
    constexpr auto get_alpaka_reducer(const cms::alpakatools::reduce::plus<U> &) {
      return cms::alpakatools::reduce::plus<T>();
    }

  }  // namespace reduce
}  // namespace cms::alpakatools

#endif
