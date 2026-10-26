# -*- Mode: Makefile -*-
# Single-box unit test of the 3D Dirac code (see box_test_dirac.cpp). Set CHOMBO_HOME and GRCHOMBO_SOURCE as for the
# examples.

DIM = 3

export EXOZVEZDA_SOURCE = $(shell pwd)/../../../../Source

ebase := box_test_dirac

LibNames := AMRTimeDependent AMRTools BoxTools

# the example's UserVariables.hpp / DiagnosticVariables.hpp
XTRACPPFLAGS += -I../..

src_dirs := $(GRCHOMBO_SOURCE)/utils \
            $(GRCHOMBO_SOURCE)/simd  \
            $(GRCHOMBO_SOURCE)/CCZ4  \
            $(GRCHOMBO_SOURCE)/Matter  \
            $(GRCHOMBO_SOURCE)/BoxUtils  \
            $(GRCHOMBO_SOURCE)/GRChomboCore  \
            $(GRCHOMBO_SOURCE)/AMRInterpolator  \
            $(EXOZVEZDA_SOURCE)/InitialConditions/DiracStars \
            $(EXOZVEZDA_SOURCE)/InitialConditions/BosonStars/UnequalMassFix \
            $(EXOZVEZDA_SOURCE)/Matter

include $(CHOMBO_HOME)/mk/Make.test
