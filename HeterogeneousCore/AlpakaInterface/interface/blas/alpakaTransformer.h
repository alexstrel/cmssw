#ifndef HeterogeneousCore_AlpakaInterface_interface_blas_alpakaTransformer_h
#define HeterogeneousCore_AlpakaInterface_interface_blas_alpakaTransformer_h

#include "HeterogeneousCore/AlpakaInterface/interface/VecArray.h"

/**
 *  Collection of the transform kernels
 */

namespace cms::alpakatools {
  namespace transform {
    template <typename T>
    concept NonBoolIntegral = std::integral<std::remove_cvref_t<T>> && !std::same_as<std::remove_cvref_t<T>, bool>;

    template <typename... T>
    concept SameIntegralType = (NonBoolIntegral<T> && ...);

    template <typename... T>
    concept SameFloatingPointType = (std::floating_point<std::remove_cvref_t<T>> && ...);

    template <typename... T>
    concept SameArithmeticType = sizeof...(T) > 0 && (SameIntegralType<T...> || SameFloatingPointType<T...>);

    template <typename... T>
      requires SameArithmeticType<T...>
    struct sloppy_precision_selector;

    template <typename T>
    struct sloppy_precision_selector<T> {
      using type = std::remove_cvref_t<T>;
    };

    template <typename T, typename U, typename... V>
      requires SameArithmeticType<T, U, V...>
    struct sloppy_precision_selector<T, U, V...> {
    private:
      using t = std::remove_cvref_t<T>;
      using u = std::remove_cvref_t<U>;

      using tu = std::conditional_t<(std::numeric_limits<t>::digits <= std::numeric_limits<u>::digits), t, u>;

    public:
      using type = typename sloppy_precision_selector<tu, V...>::type;
    };

    template <typename... T>
      requires SameArithmeticType<T...>
    using sloppy_precision_t = typename sloppy_precision_selector<T...>::type;

    template <int n, bool use_sloppy = false>
    struct xpy_batched {
      template <typename Tx, typename Ty>
        requires VecArrayType<cms::alpakatools::VecArray<Tx *, n>> && VecArrayType<cms::alpakatools::VecArray<Ty *, n>>
      ALPAKA_FN_ACC void operator()(cms::alpakatools::VecArray<Tx *, n> const &x,
                                    cms::alpakatools::VecArray<Ty *, n> const &y,
                                    int i,
                                    int j,
                                    int k) {
        using x_t = std::remove_cvref_t<Tx>;
        using y_t = std::remove_cvref_t<Ty>;

        using compute_t = std::conditional_t<use_sloppy, sloppy_precision_t<x_t, y_t>, y_t>;

        const compute_t x_ = static_cast<compute_t>(x[k][i]);
        const compute_t y_ = static_cast<compute_t>(y[k][i]);

        y[k][i] = static_cast<y_t>(x_ + y_);
      }
      constexpr int flops() const { return 2; }  //! flops per element
    };

    template <int n, bool use_sloppy = false>
    struct xmy_batched {
      template <typename Tx, typename Ty>
        requires VecArrayType<cms::alpakatools::VecArray<Tx *, n>> && VecArrayType<cms::alpakatools::VecArray<Ty *, n>>
      ALPAKA_FN_ACC void operator()(cms::alpakatools::VecArray<Tx *, n> const &x,
                                    cms::alpakatools::VecArray<Ty *, n> const &y,
                                    int i,
                                    int j,
                                    int k) {
        using x_t = std::remove_cvref_t<Tx>;
        using y_t = std::remove_cvref_t<Ty>;

        using compute_t = std::conditional_t<use_sloppy, sloppy_precision_t<x_t, y_t>, y_t>;

        const compute_t x_ = static_cast<compute_t>(x[k][i]);
        const compute_t y_ = static_cast<compute_t>(y[k][i]);

        y[k][i] = static_cast<y_t>(x_ - y_);
      }
      constexpr int flops() const { return 2; }  //! flops per element
    };

    template <typename data_t, int n, bool use_sloppy = false>
    struct axpy_batched {
      const data_t a;

      axpy_batched(const data_t &a) : a(a) {}

      template <alpaka::concepts::Acc TAcc, typename Tx, typename Ty>
        requires VecArrayType<cms::alpakatools::VecArray<Tx *, n>> && VecArrayType<cms::alpakatools::VecArray<Ty *, n>>
      ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                    cms::alpakatools::VecArray<Tx *, n> const &x,
                                    cms::alpakatools::VecArray<Ty *, n> const &y,
                                    int i,
                                    int j,
                                    int k) {
        using x_t = std::remove_cvref_t<Tx>;
        using y_t = std::remove_cvref_t<Ty>;

        using compute_t = std::conditional_t<use_sloppy, sloppy_precision_t<x_t, y_t, data_t>, y_t>;

        const compute_t x_ = static_cast<compute_t>(x[k][i]);
        const compute_t y_ = static_cast<compute_t>(y[k][i]);
        const compute_t a_ = static_cast<compute_t>(a);

        const compute_t res = alpaka::math::fma(acc, a_, x_, y_);

        y[k][i] = static_cast<y_t>(res);
      }
      constexpr int flops() const { return 3; }  //! flops per element
    };

    template <typename data_t, int n, bool use_sloppy = false>
    struct xpay_batched {
      const data_t a;

      xpay_batched(const data_t &a) : a(a) {}

      template <alpaka::concepts::Acc TAcc, typename Tx, typename Ty>
        requires VecArrayType<cms::alpakatools::VecArray<Tx *, n>> && VecArrayType<cms::alpakatools::VecArray<Ty *, n>>
      ALPAKA_FN_ACC void operator()(TAcc const &acc,
                                    cms::alpakatools::VecArray<Tx *, n> const &x,
                                    cms::alpakatools::VecArray<Ty *, n> const &y,
                                    int i,
                                    int j,
                                    int k) {
        using x_t = std::remove_cvref_t<Tx>;
        using y_t = std::remove_cvref_t<Ty>;

        using compute_t = std::conditional_t<use_sloppy, sloppy_precision_t<x_t, y_t, data_t>, y_t>;

        const compute_t x_ = static_cast<compute_t>(x[k][i]);
        const compute_t y_ = static_cast<compute_t>(y[k][i]);
        const compute_t a_ = static_cast<compute_t>(a);

        const compute_t res = alpaka::math::fma(acc, a_, y_, x_);

        y[k][i] = static_cast<y_t>(res);
      }
      constexpr int flops() const { return 3; }  //! flops per element
    };

    template <typename data_t, int n, bool use_sloppy = false>
    struct axpynorm2_batched {
      const data_t a;

      axpynorm2_batched(const data_t &a) : a(a) {}

      template <alpaka::concepts::Acc TAcc, typename Tx, typename Ty>
        requires VecArrayType<cms::alpakatools::VecArray<Tx *, n>> && VecArrayType<cms::alpakatools::VecArray<Ty *, n>>
      ALPAKA_FN_ACC auto operator()(TAcc const &acc,
                                    cms::alpakatools::VecArray<Tx *, n> const &x,
                                    cms::alpakatools::VecArray<Ty *, n> const &y,
                                    int i,
                                    int j,
                                    int k) {
        using x_t = std::remove_cvref_t<Tx>;
        using y_t = std::remove_cvref_t<Ty>;

        using compute_t = std::conditional_t<use_sloppy, sloppy_precision_t<x_t, y_t, data_t>, y_t>;

        const compute_t x_ = static_cast<compute_t>(x[k][i]);
        const compute_t y_ = static_cast<compute_t>(y[k][i]);
        const compute_t a_ = static_cast<compute_t>(a);

        const compute_t res = alpaka::math::fma(acc, a_, x_, y_);
        y[k][i] = static_cast<y_t>(res);

        return static_cast<compute_t>(res * res);
      }
      constexpr int flops() const { return 4; }  //! flops per element
    };

  }  // namespace transform
}  // namespace cms::alpakatools

#endif
