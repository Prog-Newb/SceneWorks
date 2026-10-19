# Machine-specific overrides for set-scene-env.ps1.
#
# Copy this file to set-scene-env.local.ps1 (which is gitignored) and point it
# at wherever your ROCm pieces actually live. set-scene-env.ps1 sources it
# automatically before it starts looking for anything itself.
#
# Only set what you need - everything here is optional, and anything you leave
# out is discovered from the usual locations.

# The ROCm runtime: the directory literally named _rocm_sdk_core. It ships in
# AMD's rocm-sdk-core wheel, so it is often inside some virtualenv's
# site-packages, or wherever you unpacked the wheel.
# $env:SCENEWORKS_HIP_PATH = "D:\rocm\_rocm_sdk_core"

# The ROCm devel SDK - headers and import libraries, needed only to build.
# $env:SCENEWORKS_ROCM_PATH = "D:\rocm\_rocm_sdk_devel"

# The gfx target to compile kernels for. Kernels are compiled at runtime and
# cached per architecture, so this is a preference, not a lock.
# $env:SCENEWORKS_GPU_TARGETS = "gfx1100"

# LLVM install root, if libclang is somewhere unusual and bindgen cannot find it.
# $env:SCENEWORKS_LLVM_PATH = "D:\LLVM"

# Set to 1 if you want the env script to also cd into the candle checkout.
# $env:SCENEWORKS_ENV_CHDIR = "1"
