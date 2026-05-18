"""Testes do módulo de pré-processamento."""

import numpy as np
import pytest

from libras.utils.mediapipe_holistic import (
    REGION_SLICES, TOTAL_DIM, normalize_sequence, parse_class_name,
)


class TestParseClassName:
    def test_simple(self):
        assert parse_class_name("Abacaxi_Articulador1.mp4") == "Abacaxi"

    def test_with_spaces(self):
        assert parse_class_name("À noite toda_Articulador2.mp4") == "À noite toda"

    def test_without_articulador(self):
        assert parse_class_name("OutroFormato.mp4") == "OutroFormato"


class TestNormalizeSequence:
    def test_padding(self):
        frames = np.ones((10, TOTAL_DIM))
        result = normalize_sequence(frames, target=30)
        assert result.shape == (30, TOTAL_DIM)

    def test_truncation(self):
        frames = np.ones((50, TOTAL_DIM))
        result = normalize_sequence(frames, target=30)
        assert result.shape == (30, TOTAL_DIM)

    def test_empty(self):
        frames = np.array([]).reshape(0, TOTAL_DIM)
        result = normalize_sequence(frames, target=30)
        assert result.shape == (30, TOTAL_DIM)
        assert (result == 0).all()


class TestRegionSlices:
    def test_total_dimension(self):
        last_end = max(end for _, end in REGION_SLICES.values())
        assert last_end == TOTAL_DIM

    def test_no_overlap(self):
        slices = sorted(REGION_SLICES.values())
        for (_, end_a), (start_b, _) in zip(slices, slices[1:]):
            assert end_a == start_b
