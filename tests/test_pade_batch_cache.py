"""Tests for deduplicated batched Padé coefficient generation."""

import importlib

import cupy
import numpy

from cupyram import CuPyRAM


def test_compute_pade_batch_caches_unique_parameter_sets(monkeypatch):
    cupyram_module = importlib.import_module("cupyram.cupyram")
    calls = []

    def fake_compute_pade_coefficients(freq, c0, np_pade, ns, dr, ip):
        calls.append((float(freq), float(c0), np_pade, ns, float(dr), ip))
        marker = float(freq) + float(c0) + ip
        return (
            numpy.full(np_pade, marker, dtype=numpy.complex128),
            numpy.full(np_pade, -marker, dtype=numpy.complex128),
        )

    monkeypatch.setattr(
        cupyram_module,
        "compute_pade_coefficients",
        fake_compute_pade_coefficients,
    )

    model = CuPyRAM.__new__(CuPyRAM)
    model._np = 3
    model._ns = 1
    model._dr = 50.0
    model._freqs = numpy.array([75.0, 100.0])
    model._c0_array = numpy.array([1500.0, 1500.0, 1480.0])
    model._n_freq = 2
    model._total_batch = 6
    model._pade_coefficient_cache = {}

    pd1, pd2 = model._compute_pade_batch(ip=1)
    first_call_count = len(calls)
    pd1_again, pd2_again = model._compute_pade_batch(ip=1)

    assert first_call_count == 4
    assert len(calls) == first_call_count
    numpy.testing.assert_array_equal(cupy.asnumpy(pd1), cupy.asnumpy(pd1_again))
    numpy.testing.assert_array_equal(cupy.asnumpy(pd2), cupy.asnumpy(pd2_again))

    expected_markers = numpy.array(
        [1576.0, 1601.0, 1576.0, 1601.0, 1556.0, 1581.0]
    )
    numpy.testing.assert_array_equal(cupy.asnumpy(pd1)[0], expected_markers)
    numpy.testing.assert_array_equal(cupy.asnumpy(pd2)[0], -expected_markers)
