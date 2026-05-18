#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

modules=${MODULE_LIST:-""}
mpiexec_executable=${MPIEXEC_EXECUTABLE:-"srun"}
# If using flux, append "run" after the flux executable path
if [[ "${mpiexec_executable}" == "flux" ]]
then
    mpiexec_executable="$(which ${mpiexec_executable}) run"
    flux jobs
    flux resource list
else
    mpiexec_executable="$(which ${mpiexec_executable})"
fi

mpiexec_preflags=${MPIEXEC_PREFLAGS:-""}
host=$(hostname)
build_type=${BUILD_TYPE:-"Debug"}
# note: toolchain file here is relative to repo root dir
toolchain_file=${TOOLCHAIN_FILE:-"../cmake/toolchains/default-toss_4_x86_64_ib-librom-dev.cmake"}
use_mfem=${USE_MFEM:-"Off"}
use_gslib=${MFEM_USE_GSLIB:-"Off"}
librom_flags=${LIBROM_FLAGS:-""}

basehost=${host//[[:digit:]]/}

echo ${host}

build_dir=build_${host}_${CI_PIPELINE_ID}_$(date +%F_%H_%M_%S)

if [[ -n ${modules} ]]
then
    module load ${modules}
fi

# ---- setup googletest ----
DEPS_DIR="${CI_BUILDS_DIR}/${basehost}_deps"
echo $DEPS_DIR
if [[ ! -d "${DEPS_DIR}" ]]; then
    mkdir ${DEPS_DIR} && cd ${DEPS_DIR}
    git clone https://github.com/google/googletest
    cd googletest && mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=./ .. && make && make install
    cd ${CI_PROJECT_DIR}
fi
PATH=${DEPS_DIR}/googletest:${DEPS_DIR}/googletest/build:$PATH
echo ${PATH}
# --------------------------

#./scripts/compile.sh -d -r

mkdir ${build_dir}
cd ${build_dir}
pwd

cmake -DCMAKE_TOOLCHAIN_FILE=${toolchain_file} \
      -DCMAKE_BUILD_TYPE=${build_type} \
      -DMPIEXEC_EXECUTABLE=${mpiexec_executable} \
      -DMPIEXEC_PREFLAGS="${mpiexec_preflags}" \
      -DUSE_MFEM=${use_mfem} \
      -DMFEM_USE_GSLIB=${use_gslib} \
      -DENABLE_TESTS=ON \
      -DLIBROM_FLAGS="${librom_flags}" ..

make -j

ctest -VV --output-on-failure
