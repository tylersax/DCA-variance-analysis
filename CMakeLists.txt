################################################################################
# Standalone top-level project for the symmetrization variance-reduction demo.
#
# Inverts the usual nesting: this project is the top level and pulls the DCA
# source in as a subdirectory, so **DCA itself needs no modification** -- no
# add_subdirectory hook, no edit to its CMakeLists.txt. Point it at a DCA
# checkout with -DDCA_SOURCE_DIR=<path>.
#
# Note: add_subdirectory gives DCA its own variable scope, so DCA_LIBS defined
# in there is NOT visible here. CMake *targets*, however, are global -- so we
# link the target names directly (list mirrored from DCA's DCA_LIBS).
################################################################################
cmake_minimum_required(VERSION 3.21)
project(symmetry_variance_demo LANGUAGES C CXX)

set(DCA_SOURCE_DIR "" CACHE PATH "Path to the DCA source tree")
if(NOT DCA_SOURCE_DIR OR NOT EXISTS "${DCA_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Set -DDCA_SOURCE_DIR=<path to DCA source checkout>")
endif()

add_subdirectory(${DCA_SOURCE_DIR} ${CMAKE_BINARY_DIR}/dca)

# DCA finds MPI inside its own scope, so MPI_FOUND / MPI_CXX_INCLUDE_DIRS are not visible here.
# Find it again in this scope (cached, so this is nearly free) to get the include path.
find_package(MPI)
# Likewise DCA_EXTERNAL_LIBS (the system BLAS/LAPACK) lives in DCA's scope. DCA's own `lapack`
# target is just its internal wrapper, so the actual kernels must be linked here.
find_package(LAPACK REQUIRED)

# Mirror of DCA's DCA_LIBS (targets are global; the variable is not).
set(DEMO_LIBS
  git_version modules timer cmake_options ctaux ctint ss_ct_hyb g0_interpolation
  function_transform gaussian_quadrature nfft tetrahedron_mesh random
  parallel_util dca_algorithms json dca_hdf5 enumerations profiling function
  convert_domains cluster_domains quantum_domains time_and_frequency_domains
  signals coarsegraining parallel_mpi_concurrency parallel_no_concurrency
  parallel_stdthread lapack models)

foreach(_tgt ctaux_variance_demo_on ctaux_variance_demo_off
             fe_as_variance_demo_on fe_as_variance_demo_off)
  add_executable(${_tgt} ${CMAKE_CURRENT_SOURCE_DIR}/variance_demo/${_tgt}.cpp)
  # DCA sets its include paths with include_directories() inside its own scope, which does not
  # propagate up to this parent project -- so list them explicitly. ${CMAKE_BINARY_DIR}/dca/include
  # holds the CMake-generated config headers (threading.hpp, haves_defines.hpp, ...).
  target_include_directories(${_tgt} PRIVATE
    ${DCA_SOURCE_DIR}
    ${DCA_SOURCE_DIR}/include
    ${DCA_SOURCE_DIR}/src/util
    ${DCA_SOURCE_DIR}/libs/simplex_gm_rule
    ${CMAKE_BINARY_DIR}/include        # DCA generates its config headers against CMAKE_BINARY_DIR
    ${CMAKE_BINARY_DIR}/dca/include
    ${CMAKE_CURRENT_SOURCE_DIR}/variance_demo)
  target_link_libraries(${_tgt} PUBLIC ${DEMO_LIBS} ${LAPACK_LIBRARIES} ${BLAS_LIBRARIES})
  if (MPI_FOUND)
    target_link_libraries(${_tgt} PRIVATE ${MPI_C_LIBRARIES})
    target_include_directories(${_tgt} PRIVATE ${MPI_CXX_INCLUDE_DIRS})
  endif()
endforeach()
