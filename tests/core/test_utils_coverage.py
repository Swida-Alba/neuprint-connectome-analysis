"""Coverage tests for the lightweight ``utils`` modules and shared options.

Targets uncovered branches in:
- ``utils/color_utils.py``  : parsing edge cases, output formats, palettes
- ``utils/report_utils.py`` : ``img2pptx`` aggregation + layout paths
- ``utils/neuron_filter.py``: legacy parsing + vectorized operator internals
- ``visualization_options.py``: FAFB/local-release detection

All tests are hermetic: no network, no cache reads, file I/O confined to
pytest ``tmp_path``. Synthetic images are generated with Pillow.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.color_utils import (  # noqa: E402
    color_has_explicit_alpha,
    color_to_hex,
    color_to_rgba_string,
    darken_color,
    extract_rgb_tuple,
    extract_rgba_tuple,
    generate_color_palette,
    interpolate_colors,
    is_dark_color,
    lighten_color,
    set_alpha,
    standardize_color,
    standardize_color_list,
)
from utils.neuron_filter import NeuronFilter, parse_neuron_query  # noqa: E402
from visualization_options import (  # noqa: E402
    _is_flywire_family,
    default_analysis_skeleton_mesh_simplification,
    default_skeleton_tab_simplification,
)


# ---------------------------------------------------------------------------
# color_utils: parsing edge cases + output formats
# ---------------------------------------------------------------------------
class TestColorParsingEdgeCases:
    def test_none_raises(self):
        with pytest.raises(ValueError):
            standardize_color(None)

    def test_auto_passthrough(self):
        assert standardize_color('auto') == 'auto'

    def test_transparent_named(self):
        assert extract_rgba_tuple('transparent') == (0, 0, 0, 0.0)

    def test_invalid_hex_raises(self):
        with pytest.raises(ValueError):
            standardize_color('#12345')

    def test_invalid_rgb_string_raises(self):
        with pytest.raises(ValueError):
            standardize_color('rgb(255, 0)')

    def test_non_rgb_function_string_raises(self):
        with pytest.raises(ValueError):
            standardize_color('notafunction(1,2,3)')

    def test_invalid_hsl_raises(self):
        with pytest.raises(ValueError):
            standardize_color('hsl(120, 100%)')

    def test_hsl_saturation_out_of_range_raises(self):
        with pytest.raises(ValueError):
            standardize_color('hsl(120, 150%, 50%)')

    def test_hsl_hue_units(self):
        # turn / rad / deg suffixes all parse to the same hue family
        turn = extract_rgb_tuple('hsl(0.5turn, 100%, 50%)')
        deg = extract_rgb_tuple('hsl(180deg, 100%, 50%)')
        rad = extract_rgb_tuple('hsl(3.14159rad, 100%, 50%)')
        assert turn == deg
        assert rad[2] > rad[0]  # cyan-ish: blue dominates

    def test_tuple_wrong_length_raises(self):
        with pytest.raises(ValueError):
            standardize_color((255, 0))

    def test_tuple_alpha_none_raises(self):
        with pytest.raises(ValueError):
            standardize_color((255, 0, 0, None))

    def test_numeric_scalar_grayscale_float(self):
        assert extract_rgb_tuple(0.5) == (128, 128, 128)

    def test_numeric_scalar_grayscale_int(self):
        assert extract_rgb_tuple(200) == (200, 200, 200)

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError):
            standardize_color(object())

    def test_alpha_invalid_percent_string(self):
        with pytest.raises(ValueError):
            standardize_color('red', default_alpha='not-a-number%')

    def test_alpha_invalid_string(self):
        with pytest.raises(ValueError):
            standardize_color('red', default_alpha='abc')

    def test_alpha_non_numeric(self):
        with pytest.raises(ValueError):
            standardize_color('red', default_alpha=object())


class TestColorOutputFormats:
    def test_rgb_format(self):
        assert standardize_color('red', output_format='rgb') == 'rgb(255, 0, 0)'

    def test_hex_format(self):
        assert standardize_color('red', output_format='hex') == '#ff0000'

    def test_hex_alpha_format(self):
        assert standardize_color(
            'rgba(255, 0, 0, 0.5)', output_format='hex_alpha') == '#ff000080'

    def test_tuple_normalized_format(self):
        r, g, b, a = standardize_color('red', output_format='tuple_normalized')
        assert r == pytest.approx(1.0)
        assert g == pytest.approx(0.0)
        assert a == pytest.approx(1.0)

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError):
            standardize_color('red', output_format='bogus')


class TestColorListHelpers:
    def test_none_returns_empty(self):
        assert standardize_color_list(None) == []

    def test_single_string_wraps(self):
        assert standardize_color_list('red') == ['rgba(255, 0, 0, 1.0)']

    def test_non_sequence_raises(self):
        with pytest.raises(ValueError):
            standardize_color_list(12345)

    def test_empty_returns_empty(self):
        assert standardize_color_list([]) == []

    def test_invalid_entry_skipped_with_warning(self):
        with pytest.warns(UserWarning):
            result = standardize_color_list(['red', 'not-a-color-xyz'])
        assert result == ['rgba(255, 0, 0, 1.0)']

    def test_extract_rgb_tuple(self):
        assert extract_rgb_tuple('#00ff00') == (0, 255, 0)

    def test_color_to_hex(self):
        assert color_to_hex((255, 128, 0)) == '#ff8000'

    def test_color_to_rgba_string_override_alpha(self):
        assert color_to_rgba_string('red', alpha=0.5) == 'rgba(255, 0, 0, 0.5)'

    def test_color_to_rgba_string_inherit_alpha(self):
        assert color_to_rgba_string('rgba(10, 20, 30, 0.25)') == \
            'rgba(10, 20, 30, 0.25)'


class TestExplicitAlphaDetection:
    def test_transparent_has_alpha(self):
        assert color_has_explicit_alpha('transparent') is True

    def test_bare_hex_8_digit_has_alpha(self):
        assert color_has_explicit_alpha('ff000080') is True

    def test_bare_hex_6_digit_no_alpha(self):
        assert color_has_explicit_alpha('ff0000') is False

    def test_mixed_list_with_alpha_entry(self):
        assert color_has_explicit_alpha(['#ff0000', '#00ff0080']) is True

    def test_non_color_value(self):
        assert color_has_explicit_alpha(None) is False


class TestColorTransforms:
    def test_is_dark_black_white(self):
        assert is_dark_color('black') is True
        assert is_dark_color('white') is False

    def test_is_dark_auto(self):
        assert is_dark_color('auto') is False

    def test_is_dark_unparseable_defaults_light(self):
        assert is_dark_color('not-a-color-xyz') is False

    def test_darken(self):
        assert darken_color('red', 0.5) == 'rgba(127, 0, 0, 1.0)'

    def test_lighten(self):
        assert lighten_color('red', 0.5) == 'rgba(255, 127, 127, 1.0)'

    def test_set_alpha(self):
        assert set_alpha('red', 0.5) == 'rgba(255, 0, 0, 0.5)'


class TestInterpolateAndPalette:
    def test_interpolate_zero(self):
        assert interpolate_colors(['red'], 0) == []

    def test_interpolate_empty_source(self):
        assert interpolate_colors([], 3) == ['rgba(128, 128, 128, 1.0)'] * 3

    def test_interpolate_single_color(self):
        assert interpolate_colors(['red', 'blue'], 1) == ['rgba(255, 0, 0, 1.0)']

    def test_interpolate_gradient(self):
        result = interpolate_colors(['red', 'blue'], 5)
        assert len(result) == 5
        assert result[0] == 'rgba(255, 0, 0, 1.000)'
        assert result[-1] == 'rgba(0, 0, 255, 1.000)'

    def test_generate_palette_zero(self):
        assert generate_color_palette(0) == []

    def test_generate_palette_category10(self):
        result = generate_color_palette(5, 'category10')
        assert len(result) == 5
        assert all(c.startswith('rgba(') for c in result)

    def test_generate_palette_viridis(self):
        result = generate_color_palette(4, 'viridis')
        assert len(result) == 4

    def test_generate_palette_fallback_hue(self):
        # An unknown palette falls through to the hue-based fallback.
        result = generate_color_palette(6, 'not-a-real-palette-xyz')
        assert len(result) == 6
        assert all(c.startswith('rgba(') for c in result)


# ---------------------------------------------------------------------------
# report_utils: img2pptx
# ---------------------------------------------------------------------------
def _make_images(folder, count=3, prefix='img', size=(64, 48)):
    from PIL import Image
    paths = []
    for i in range(count):
        img = Image.new('RGB', size, color=((i * 60) % 255, 100, 200))
        path = Path(folder) / f'{prefix}_{i}.png'
        img.save(path)
        paths.append(str(path))
    return paths


class TestImg2Pptx:
    def test_aggregate_directory(self, tmp_path):
        from utils.report_utils import img2pptx
        _make_images(tmp_path, count=5)
        out = img2pptx(str(tmp_path), images_per_slide=(2, 2))
        assert Path(out).exists()
        assert out.endswith('.pptx')

    def test_aggregate_list_of_images(self, tmp_path):
        from utils.report_utils import img2pptx
        paths = _make_images(tmp_path, count=4)
        out_path = tmp_path / 'from_list.pptx'
        out = img2pptx(paths, output_pptx=str(out_path), images_per_slide=(2, 1))
        assert out == str(out_path)
        assert out_path.exists()

    def test_single_image_file(self, tmp_path):
        from utils.report_utils import img2pptx
        paths = _make_images(tmp_path, count=1)
        out = img2pptx(paths[0])
        assert Path(out).exists()

    def test_dark_background_autofont(self, tmp_path):
        from utils.report_utils import img2pptx
        _make_images(tmp_path, count=2)
        out = img2pptx(
            str(tmp_path),
            background_color='black',
            slide_title='Slide {page}',
            slide_size='standard',
            label_position='above',
        )
        assert Path(out).exists()

    def test_background_hex_and_overlay_label(self, tmp_path):
        from utils.report_utils import img2pptx
        _make_images(tmp_path, count=2)
        out = img2pptx(
            str(tmp_path),
            background_color='#102030',
            label_position='overlay',
            slide_size='a4',
            font_color=(1.0, 1.0, 1.0),
        )
        assert Path(out).exists()

    def test_subfolders_grouped(self, tmp_path):
        from utils.report_utils import img2pptx
        sub = tmp_path / 'subA'
        sub.mkdir()
        _make_images(tmp_path, count=2, prefix='root')
        _make_images(sub, count=2, prefix='child')
        out = img2pptx(
            str(tmp_path),
            include_subfolders=True,
            group_by_subfolder=True,
            slide_title='{subfolder} page {page}',
        )
        assert Path(out).exists()

    def test_missing_pdf_raises(self, tmp_path):
        from utils.report_utils import img2pptx
        with pytest.raises(FileNotFoundError):
            img2pptx(str(tmp_path / 'nope.pdf'))

    def test_empty_directory_raises(self, tmp_path):
        from utils.report_utils import img2pptx
        empty = tmp_path / 'empty'
        empty.mkdir()
        with pytest.raises(ValueError):
            img2pptx(str(empty))

    def test_missing_single_file_raises(self, tmp_path):
        from utils.report_utils import img2pptx
        with pytest.raises(FileNotFoundError):
            img2pptx(str(tmp_path / 'missing_image.png'))

    def test_list_all_missing_raises(self, tmp_path):
        from utils.report_utils import img2pptx
        with pytest.raises(ValueError):
            img2pptx([str(tmp_path / 'a.png'), str(tmp_path / 'b.png')])

    def test_pdf_to_pptx(self, tmp_path):
        """Convert a tiny generated PDF to PPTX (uses PyMuPDF)."""
        fitz = pytest.importorskip('fitz')
        from utils.report_utils import img2pptx
        pdf_path = tmp_path / 'doc.pdf'
        doc = fitz.open()
        for _ in range(2):
            doc.new_page(width=200, height=150)
        doc.save(str(pdf_path))
        doc.close()
        out = img2pptx(str(pdf_path), slide_title='Page {page}')
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# neuron_filter
# ---------------------------------------------------------------------------
@pytest.fixture
def neuron_df():
    return pd.DataFrame({
        'type': ['aMe12', 'aMe12', 'DN1p', 'Mi1'],
        'instance': ['aMe12_R', 'aMe12_L', 'DN1p_R', 'Mi1'],
        'bodyId': [101, 102, 103, 104],
    })


class TestNeuronFilterParsing:
    def test_empty_list_matches_all(self, neuron_df):
        f = parse_neuron_query([])
        assert f.match_all
        assert len(f.apply(neuron_df)) == 4

    def test_digit_string_is_bodyid(self, neuron_df):
        f = parse_neuron_query(['101'])
        assert f.get_bodyIds(neuron_df) == [101]

    def test_unknown_dict_operator_skipped(self, neuron_df):
        f = parse_neuron_query({'bogus_op': 'x'})
        assert f.filter_spec == {}
        assert f.describe() == 'No filter'

    def test_describe_single_and_multi(self):
        assert parse_neuron_query({'contains': 'DN'}).describe() == \
            "contains='DN'"
        multi = parse_neuron_query({'startswith': ['aMe', 'Mi']})
        assert 'startswith=' in multi.describe()

    def test_repr_and_bool(self):
        f = parse_neuron_query({'contains': 'DN'})
        assert bool(f) is True
        assert 'NeuronFilter(' in repr(f)
        assert bool(parse_neuron_query(None)) is False


class TestNeuronFilterInternals:
    """Directly exercise the per-cell and vectorized operator helpers."""

    def test_match_value_all_operators(self):
        f = NeuronFilter()
        assert f._match_value('aMe12', 'exact', ['aMe12']) is True
        assert f._match_value('aMe12', 'exact', ['other']) is False
        assert f._match_value('aMe12', 'contains', ['Me1']) is True
        assert f._match_value('aMe12', 'not_contains', ['zz']) is True
        assert f._match_value('aMe12', 'startswith', ['aMe']) is True
        assert f._match_value('aMe12', 'endswith', ['12']) is True
        assert f._match_value('aMe12', 'regex', [r'^aMe']) is True
        assert f._match_value('aMe12', 'not_regex', [r'^DN']) is True
        # NaN / None never match
        assert f._match_value(None, 'contains', ['x']) is False
        assert f._match_value(float('nan'), 'contains', ['x']) is False
        # invalid regex falls back to exact
        assert f._match_value('a[', 'regex', ['a[']) is True
        assert f._match_value('a[', 'not_regex', ['a[']) is False
        # unknown operator -> False
        assert f._match_value('aMe12', 'nope', ['aMe12']) is False

    def test_apply_operator_int_bodyids(self, neuron_df):
        f = NeuronFilter()
        mask = f._apply_operator(neuron_df, 'exact', [101, 103])
        assert mask.sum() == 2

    def test_apply_operator_int_no_bodyid_column(self):
        f = NeuronFilter()
        df = pd.DataFrame({'type': ['a', 'b']})
        mask = f._apply_operator(df, 'exact', [101])
        assert mask.sum() == 0

    def test_apply_operator_column_each_operator(self, neuron_df):
        f = NeuronFilter()
        series = neuron_df['type']
        assert f._apply_operator_column(
            series, 'contains', ['aMe'], ['aMe']).sum() == 2
        assert f._apply_operator_column(
            series, 'not_contains', ['aMe'], ['aMe']).sum() == 2
        assert f._apply_operator_column(
            series, 'startswith', ['Mi'], ['Mi']).sum() == 1
        assert f._apply_operator_column(
            series, 'endswith', ['1p'], ['1p']).sum() == 1
        assert f._apply_operator_column(
            series, 'regex', [r'^aMe'], [r'^aMe']).sum() == 2
        assert f._apply_operator_column(
            series, 'not_regex', [r'^aMe'], [r'^aMe']).sum() == 2
        # empty patterns -> all False
        assert f._apply_operator_column(series, 'contains', [], []).sum() == 0
        # invalid regex falls back to exact equality
        assert f._apply_operator_column(
            series, 'regex', ['aMe12'], ['aMe12']).sum() >= 1

    def test_apply_operator_string_across_columns(self, neuron_df):
        f = NeuronFilter()
        mask = f._apply_operator(neuron_df, 'contains', ['_R'])
        assert mask.sum() == 2

    def test_get_search_columns(self, neuron_df):
        f = NeuronFilter()
        cols = f._get_search_columns(neuron_df)
        assert 'type' in cols


class TestNeuronFilterAccessors:
    def test_get_types(self, neuron_df):
        f = parse_neuron_query({'contains': 'aMe'})
        assert set(f.get_types(neuron_df)) == {'aMe12'}

    def test_get_types_no_column(self):
        f = parse_neuron_query(None)
        df = pd.DataFrame({'bodyId': [1]})
        assert f.get_types(df) == []

    def test_get_instances(self, neuron_df):
        f = parse_neuron_query({'contains': 'aMe'})
        assert set(f.get_instances(neuron_df)) == {'aMe12_R', 'aMe12_L'}

    def test_get_instances_no_column(self):
        f = parse_neuron_query(None)
        df = pd.DataFrame({'bodyId': [1]})
        assert f.get_instances(df) == []

    def test_get_bodyids_without_bodyid_column(self):
        f = parse_neuron_query(None)
        df = pd.DataFrame({'type': ['a'], 'some_id': [7]})
        # No bodyId column and no recognized alias -> empty list.
        assert f.get_bodyIds(df) == []


# ---------------------------------------------------------------------------
# visualization_options
# ---------------------------------------------------------------------------
class TestVisualizationOptions:
    def test_flywire_family_detection(self):
        assert _is_flywire_family('flywire_FAFB_v783') is True
        assert _is_flywire_family('something_fafb') is True
        # BANC is a standalone local release, not a FlyWire-family value.
        assert _is_flywire_family('banc_v888') is False
        assert _is_flywire_family('hemibrain:v1.2.1') is False
        assert _is_flywire_family(None) is False

    def test_skeleton_tab_defaults(self):
        assert default_skeleton_tab_simplification('x', 'fast') == 0.90
        assert default_skeleton_tab_simplification('x', 'direct') == 0.90
        assert default_skeleton_tab_simplification('x', 'fine') == 0.95
        assert default_skeleton_tab_simplification('x', None) == 0.90

    def test_skeleton_tab_defaults_banc_uses_full_res_default(self):
        # The BANC slider default stays the standard FAFB-style 90%: it only
        # ever drives full-resolution sources, while L2 tubes skip decimation
        # structurally in the BANC render processor. A resolution-dependent
        # 0.0 default would leave full-res-only neurons (they exist, e.g.
        # one l-LNv) undecimated in the default L2 render mode.
        assert default_skeleton_tab_simplification(
            'banc_v888', 'fast') == 0.90
        assert default_skeleton_tab_simplification(
            'banc_v626', 'fine') == 0.95
        assert default_skeleton_tab_simplification(
            'flywire_BANC_v888', 'fast') == 0.90

    def test_analysis_defaults(self):
        assert default_analysis_skeleton_mesh_simplification('x', 'fine') == 0.95
        assert default_analysis_skeleton_mesh_simplification('x', 'fast') == 0.90
        assert default_analysis_skeleton_mesh_simplification('x', None) == 0.95
