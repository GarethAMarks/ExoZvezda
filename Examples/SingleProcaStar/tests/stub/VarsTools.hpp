// Minimal stand-in for GRChombo's VarsTools.hpp so that ProcaField.hpp compiles without Chombo (test only)
#ifndef VARSTOOLS_HPP_
#define VARSTOOLS_HPP_
#include "Tensor.hpp"
template <int a, int b> struct GRInterval
{
};
namespace VarsTools
{
template <class F, class I, class T> void define_enum_mapping(F, I, T &) {}
} // namespace VarsTools
#endif
