"""Regenerate the valid-bridge network visualization (§9I).

Renders ``BRIDGE_SOURCE_MAP`` — the declarative licensing table of the
cross-dataset type mapper — as an interactive HTML network:
``outputs/type_mapping/source_map_network.html``.  The HTML embeds the
licensing rules and links back to ``docs/AUTO_TYPE_MAPPING.md``
("Valid bridge source map"), so the visualization, the code constant,
and the docs all describe the same map.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comparison.mapping_visualization import render_source_map_network_html


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / "type_mapping" / "source_map_network.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    html = render_source_map_network_html()
    if not html:
        print("vispath unavailable — nothing rendered", file=sys.stderr)
        return 1
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
