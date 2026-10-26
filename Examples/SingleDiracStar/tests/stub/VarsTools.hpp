// Minimal stand-in for GRChombo's VarsTools.hpp so that DiracField.hpp compiles without Chombo (test only)
#ifndef VARSTOOLS_HPP_
#define VARSTOOLS_HPP_
#include "Tensor.hpp"
namespace VarsTools
{
template <class F, class T> void define_enum_mapping(F, int, T &) {}
} // namespace VarsTools
#endif
