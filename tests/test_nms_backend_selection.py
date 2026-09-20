from __future__ import annotations

import importlib.machinery
import types

import pytest

from kwimage.algo import algo_nms


@pytest.mark.parametrize('num', [1, 50, 250, 1000])
def test_auto_ndarray_prefers_rust_cpu(num):
    valid = {'rust_cpu', 'cython_cpu', 'cython_gpu', 'numpy'}
    got = algo_nms._heuristic_auto_nms_impl('ndarray', num, valid)
    assert got == 'rust_cpu'


@pytest.mark.parametrize('code', ['tensor0', 'tensor', 'ndarray'])
@pytest.mark.parametrize('num', [1, 50, 250, 1000])
def test_auto_never_selects_legacy_gpu(code, num):
    valid = {'cython_gpu', 'numpy'}
    got = algo_nms._heuristic_auto_nms_impl(code, num, valid)
    assert got == 'numpy'


def test_auto_cuda_tensor_keeps_torchvision_on_device():
    valid = {'torchvision', 'rust_cpu', 'cython_gpu', 'numpy'}
    got = algo_nms._heuristic_auto_nms_impl('tensor0', 1000, valid)
    assert got == 'torchvision'


def test_native_extension_detection_distinguishes_python_stub():
    python_stub = types.SimpleNamespace(__file__='/tmp/gpu_nms.py')
    assert not algo_nms._is_compiled_extension_module(python_stub)

    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    native_module = types.SimpleNamespace(__file__='/tmp/gpu_nms' + suffix)
    assert algo_nms._is_compiled_extension_module(native_module)


def test_rust_cpu_backend_name_and_legacy_fallback():
    rust_shim = types.SimpleNamespace(
        backend_metadata=lambda: {'kind': 'rust'},
    )
    legacy_module = types.SimpleNamespace()
    assert algo_nms._cpu_nms_backend_name(rust_shim) == 'rust_cpu'
    assert algo_nms._cpu_nms_backend_name(legacy_module) == 'cython_cpu'


def test_cython_cpu_alias_resolves_to_rust():
    impls = algo_nms._NMS_Impls()
    impls._aliases['cython_cpu'] = 'rust_cpu'
    assert impls._resolve_alias('cython_cpu') == 'rust_cpu'
    assert impls._resolve_alias('rust_cpu') == 'rust_cpu'


def test_kwimage_ext_rust_shim_is_not_mistaken_for_gpu_backend():
    cpu_nms = pytest.importorskip('kwimage_ext.algo._nms_backend.cpu_nms')
    metadata = getattr(cpu_nms, 'backend_metadata', lambda: {})()
    if metadata.get('kind') != 'rust':
        pytest.skip('installed kwimage_ext is not using the Rust CPU backend')

    gpu_nms = pytest.importorskip('kwimage_ext.algo._nms_backend.gpu_nms')
    assert algo_nms._cpu_nms_backend_name(cpu_nms) == 'rust_cpu'
    assert not algo_nms._is_compiled_extension_module(gpu_nms)
