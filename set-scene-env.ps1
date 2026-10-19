# Sets up the environment for an AMD/ROCm build of SceneWorks.
#
# Dot-source it so the variables survive into your shell:
#
#     . .\set-scene-env.ps1
#
# Every path is discovered or overridable; nothing is hardcoded to one machine.
# Override any of these before dot-sourcing to take control:
#
#   SCENEWORKS_HIP_PATH    the ROCm runtime (the _rocm_sdk_core directory)
#   SCENEWORKS_ROCM_PATH   the ROCm devel SDK (headers/libs, build time only)
#   SCENEWORKS_GPU_TARGETS the gfx target(s) to build kernels for
#   SCENEWORKS_LLVM_PATH   LLVM install root (for libclang / bindgen)
#   SCENEWORKS_ENV_CHDIR   set to 1 to also cd into the candle checkout

$ErrorActionPreference = 'Stop'

# The script lives at the repository root, so the repo needs no hardcoded path.
$repo = $PSScriptRoot
if (-not $repo) { $repo = (Get-Location).Path }

# Machine-specific overrides live in an untracked file beside this one, so your
# own paths never end up in a commit. Copy set-scene-env.local.example.ps1 to
# set-scene-env.local.ps1 and edit it.
$localOverride = Join-Path $repo 'set-scene-env.local.ps1'
if (Test-Path -LiteralPath $localOverride) { . $localOverride }

function Resolve-FirstExisting {
    param([string[]] $Candidates)
    foreach ($c in $Candidates) {
        if ($c -and (Test-Path -LiteralPath $c)) { return (Resolve-Path -LiteralPath $c).Path }
    }
    return $null
}

# ---- HIP runtime (_rocm_sdk_core) -------------------------------------------
# Needed at runtime. Shipped either by a real HIP SDK install or by AMD's
# rocm-sdk-core wheel, which is just a zip of DLLs and can be unpacked anywhere.
$hip = Resolve-FirstExisting @(
    $env:SCENEWORKS_HIP_PATH
    (Join-Path $repo 'rocm-runtime\_rocm_sdk_core')
    (Join-Path $repo '.venv\Lib\site-packages\_rocm_sdk_core')
    "$env:LOCALAPPDATA\Programs\rocm-sdk\_rocm_sdk_core"
    'C:\Program Files\AMD\ROCm\_rocm_sdk_core'
)
if (-not $hip) {
    throw @"
Could not locate the ROCm runtime (_rocm_sdk_core).

Set SCENEWORKS_HIP_PATH to it, or unpack AMD's wheels into $repo\rocm-runtime:

    pip download --no-deps --find-links https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/ `
        rocm-sdk-core==7.2.1 rocm-sdk-libraries-custom==7.2.1

They are plain win_amd64 wheels (zip archives of DLLs), so no venv is required.

This matters: with these DLLs missing the worker exits instantly with no stderr
and no listener (Windows STATUS_DLL_NOT_FOUND prints nothing), which looks like
"the binary does nothing" rather than a missing dependency.
"@
}
$env:HIP_PATH = $hip

# ---- ROCm devel SDK (headers + import libs, build time only) ----------------
$rocm = Resolve-FirstExisting @(
    $env:SCENEWORKS_ROCM_PATH
    (Join-Path $repo 'rocm-sdk\_rocm_sdk_devel')
    $env:ROCM_PATH
)
if ($rocm) {
    $env:ROCM_PATH = $rocm
} else {
    Write-Warning "ROCm devel SDK not found - set SCENEWORKS_ROCM_PATH if the build needs headers."
}

# ---- Target architecture ----------------------------------------------------
# Kernels are compiled at runtime and cached per architecture, so this is a
# build-time preference rather than a lock: one binary serves any ROCm gfx target.
$env:GPU_TARGETS = if ($env:SCENEWORKS_GPU_TARGETS) { $env:SCENEWORKS_GPU_TARGETS } else { 'gfx1100' }

# ---- HIP toolchain selection ------------------------------------------------
# candle-rocm-kernels resolves the target architecture at runtime: this env var
# first, then an option_env! fallback baked in at build time, then rocminfo.
# Setting it here makes the build stamp in a sensible default.
$env:CANDLE_ROCM_ARCH = ($env:GPU_TARGETS -split '[;,]')[0]
$env:HIP_PLATFORM = 'amd'
$hipClang = Join-Path $env:HIP_PATH 'lib\llvm\bin'
if (Test-Path -LiteralPath $hipClang) { $env:HIP_CLANG_PATH = $hipClang }

# ---- LLVM / libclang for bindgen -------------------------------------------
# Derive the LLVM root from clang on PATH, if it is there. Guarded: with
# $ErrorActionPreference = 'Stop', Split-Path on a null source is fatal.
$clangRoot = $null
$clangCmd = Get-Command clang.exe -ErrorAction SilentlyContinue
if ($clangCmd -and $clangCmd.Source) {
    $clangRoot = Split-Path -Parent (Split-Path -Parent $clangCmd.Source)
}

$llvmRoot = Resolve-FirstExisting @(
    $env:SCENEWORKS_LLVM_PATH
    $clangRoot
    'C:\Program Files\LLVM'
    "$env:LOCALAPPDATA\Programs\LLVM"
)
if ($llvmRoot) {
    $env:LIBCLANG_PATH = Join-Path $llvmRoot 'bin'
    # Take the highest installed clang version rather than pinning one.
    $res = Get-ChildItem (Join-Path $llvmRoot 'lib\clang') -Directory -ErrorAction SilentlyContinue |
        Sort-Object { [int]($_.Name -split '\.')[0] } -Descending | Select-Object -First 1
    if ($res) {
        $env:BINDGEN_EXTRA_CLANG_ARGS_x86_64_pc_windows_msvc =
            "-resource-dir=`"$($res.FullName)`" --target=x86_64-pc-windows-msvc"
    }
} else {
    Write-Warning "LLVM not found - set SCENEWORKS_LLVM_PATH if bindgen fails on libclang."
}

# ---- PATH -------------------------------------------------------------------
$prepend = @(
    (Join-Path $env:HIP_PATH 'bin')
    (Join-Path $env:HIP_PATH 'lib\llvm\bin')
    $(if ($env:ROCM_PATH) { Join-Path $env:ROCM_PATH 'bin' })
    $(if ($env:ROCM_PATH) { Join-Path $env:ROCM_PATH 'lib' })
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

$existing = $env:PATH -split ';'
$new = @($prepend | Where-Object { $existing -notcontains $_ })
if ($new.Count) { $env:PATH = ($new -join ';') + ';' + $env:PATH }

# ---- Summary ----------------------------------------------------------------
Write-Host "ROCm build environment ready:"
Write-Host "  HIP_PATH     $env:HIP_PATH"
Write-Host "  ROCM_PATH    $(if ($env:ROCM_PATH) { $env:ROCM_PATH } else { '(not set)' })"
Write-Host "  GPU_TARGETS  $env:GPU_TARGETS"
Write-Host "  LIBCLANG     $(if ($env:LIBCLANG_PATH) { $env:LIBCLANG_PATH } else { '(not set)' })"

# Optional convenience: cd into the candle checkout. Off by default so the
# script only sets environment and has no surprising side effects.
if ($env:SCENEWORKS_ENV_CHDIR -eq '1') {
    $candle = Join-Path $repo 'candle-rocm-test\candle'
    if (Test-Path -LiteralPath $candle) { Set-Location -LiteralPath $candle }
}
