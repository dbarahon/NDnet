#!/usr/bin/env python3
"""Plot the canonical CloudNet ground-site locations used by NDnet.

The site inventory is read from the NDnet site diagnostics JSONL.  Site
membership is defined independently from the plotted map extent: the canonical
evaluation set is obtained first, then every retained site is shown in a broad
domain panel and the dense European subset is repeated in a zoom panel.

The canonical set excludes Invercargill and Payerne because they do not
contribute usable site-evaluation profiles in NDnet_site_test.nc, and excludes
Soverato intentionally because of its very limited sampling.

Examples
--------
Default canonical map::

    python plots/site_locations.py

Leave additional sites out::

    python plots/site_locations.py --exclude-site eriswil cluj

Use another diagnostics file and write PNG instead of PDF::

    python plots/site_locations.py --input /path/to/sites.jsonl \
        --output Cloudnet_site_locations.png
"""

import argparse
import json
import os

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# =====================================================================
# Defaults
# =====================================================================

DEFAULT_JSONL = (
    "/panfs/ccds02/nobackup/people/dbarahon/NDnet/data/dailies/"
    "Cloudnet_sites_multiyear/site_timeprofile_diagnostics.jsonl"
)
DEFAULT_OUTPUT = "Cloudnet_site_locations.pdf"

# Canonical evaluation-set exclusions. Command-line --exclude-site entries
# are added to this tuple rather than replacing it. Matching is
# case-insensitive.
EXCLUDE_SITES = (
    "invercargill",
    "payerne",
    "soverato",
)

FIGSIZE = (7.2, 4.8)
DPI = 400

# The broad panel contains the complete canonical site set.
FULL_LON_MIN = -80.0
FULL_LON_MAX = 70.0
FULL_LAT_MIN = -80.0
FULL_LAT_MAX = 85.0

# The second panel is a visualization zoom only. It does not define site
# membership and therefore does not alter the canonical set.
EU_LON_MIN = -12.0
EU_LON_MAX = 35.0
EU_LAT_MIN = 34.0
EU_LAT_MAX = 70.0

FULL_LON_TICKS = [-60, -30, 0, 30, 60]
FULL_LAT_TICKS = [-60, -30, 0, 30, 60]

EU_LON_TICKS = [-10, 0, 10, 20, 30]
EU_LAT_TICKS = [40, 50, 60, 70]

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "p", "*")


# =====================================================================
# Journal style
# =====================================================================

def set_journal_style():
    """Matplotlib settings consistent with the NDnet journal figures."""
    mpl.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.4,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


# =====================================================================
# Input and site handling
# =====================================================================

def normalize_longitude(lon):
    """Normalize longitude to the [-180, 180) interval."""
    return ((float(lon) + 180.0) % 360.0) - 180.0


def display_name(name):
    """Make lower-case JSON site identifiers easier to read in the legend."""
    return str(name).replace("_", " ").strip().title()


def read_sites(path):
    """Read and validate unique site locations from a JSON-lines file."""
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Site diagnostics JSONL does not exist:\n{path}")

    sites_by_name = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}: {exc}"
                ) from exc

            missing = [key for key in ("site", "lat", "lon") if key not in record]
            if missing:
                raise KeyError(
                    f"Line {line_number} of {path} is missing fields: {missing}"
                )

            name = str(record["site"]).strip()
            if not name:
                raise ValueError(f"Empty site name on line {line_number} of {path}")

            lat = float(record["lat"])
            lon = normalize_longitude(record["lon"])

            if not np.isfinite(lat) or not np.isfinite(lon):
                raise ValueError(
                    f"Non-finite location for site '{name}' on line {line_number}."
                )
            if not -90.0 <= lat <= 90.0:
                raise ValueError(
                    f"Latitude outside [-90, 90] for site '{name}': {lat}"
                )

            key = name.casefold()
            if key in sites_by_name:
                previous = sites_by_name[key]
                if not (
                    np.isclose(previous["lat"], lat, atol=1.0e-5)
                    and np.isclose(previous["lon"], lon, atol=1.0e-5)
                ):
                    raise ValueError(
                        f"Site '{name}' occurs more than once with different locations."
                    )
                continue

            sites_by_name[key] = {
                "site": name,
                "lat": lat,
                "lon": lon,
            }

    sites = list(sites_by_name.values())
    if not sites:
        raise ValueError(f"No valid sites were found in:\n{path}")

    sites.sort(key=lambda item: item["site"].casefold())
    return sites


def apply_exclusions(sites, exclusions):
    """Remove exact site-name matches, case-insensitively."""
    excluded = {
        str(name).strip().casefold()
        for name in exclusions
        if str(name).strip()
    }
    selected = [s for s in sites if s["site"].casefold() not in excluded]

    available = {s["site"].casefold(): s["site"] for s in sites}
    unknown = sorted(name for name in excluded if name not in available)
    if unknown:
        print(
            "[WARNING] exclusion names not present in the JSONL: "
            + ", ".join(unknown)
        )

    if not selected:
        raise ValueError("All sites were excluded; nothing remains to plot.")

    return selected


def assign_site_styles(sites):
    """Assign one stable color/marker combination to every canonical site."""
    nsite = len(sites)
    cmap = plt.get_cmap("turbo")
    colors = cmap(np.linspace(0.04, 0.96, nsite))

    styled = []
    for index, (site, color) in enumerate(zip(sites, colors)):
        item = dict(site)
        item["color"] = color
        item["marker"] = MARKERS[index % len(MARKERS)]
        styled.append(item)

    return styled


def sites_in_extent(sites, lon_min, lon_max, lat_min, lat_max):
    """Return sites inside a rectangular plotting extent."""
    return [
        s
        for s in sites
        if (
            lon_min <= s["lon"] <= lon_max
            and lat_min <= s["lat"] <= lat_max
        )
    ]


# =====================================================================
# Plotting helpers
# =====================================================================

def add_geography(ax):
    """Add restrained journal-style geographic context."""
    ax.add_feature(
        cfeature.LAND,
        facecolor="0.94",
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.OCEAN,
        facecolor="white",
        edgecolor="none",
        zorder=0,
    )
    ax.add_feature(
        cfeature.LAKES,
        facecolor="white",
        edgecolor="0.70",
        linewidth=0.25,
        zorder=1,
    )
    ax.coastlines(linewidth=0.55, color="0.25", zorder=2)
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.30,
        edgecolor="0.45",
        zorder=2,
    )


def configure_ticks(ax, lon_ticks, lat_ticks, proj):
    """Apply formatted geographic ticks."""
    ax.set_xticks(lon_ticks, crs=proj)
    ax.set_yticks(lat_ticks, crs=proj)
    ax.xaxis.set_major_formatter(
        LongitudeFormatter(
            zero_direction_label=True,
            dateline_direction_label=True,
        )
    )
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=7.0,
        pad=1.5,
    )


def plot_site_symbols(ax, sites, proj, markersize=5.0):
    """Plot styled site symbols on one map axis."""
    for site in sites:
        ax.plot(
            site["lon"],
            site["lat"],
            linestyle="None",
            marker=site["marker"],
            markersize=markersize,
            markerfacecolor=site["color"],
            markeredgecolor=site["color"],
            markeredgewidth=0.6,
            transform=proj,
            zorder=5,
        )


def build_legend_handles(sites):
    """Construct legend handles matching the site symbols."""
    handles = []
    for site in sites:
        handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=site["marker"],
                markersize=4.7,
                markerfacecolor=site["color"],
                markeredgecolor=site["color"],
                markeredgewidth=0.6,
                color=site["color"],
                label=display_name(site["site"]),
            )
        )
    return handles


# =====================================================================
# Plotting
# =====================================================================

def plot_sites(sites, output, dpi=DPI):
    """Draw the full canonical map plus a European zoom panel."""
    set_journal_style()

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=FIGSIZE)

    # Full-domain panel, European zoom, and a narrow right-hand legend column.
    ax_full = fig.add_axes(
        [0.055, 0.12, 0.43, 0.79],
        projection=proj,
    )
    ax_eu = fig.add_axes(
        [0.515, 0.12, 0.275, 0.79],
        projection=proj,
    )

    # -----------------------------------------------------------------
    # (a) Full canonical network
    # -----------------------------------------------------------------
    ax_full.set_extent(
        [FULL_LON_MIN, FULL_LON_MAX, FULL_LAT_MIN, FULL_LAT_MAX],
        crs=proj,
    )
    add_geography(ax_full)
    configure_ticks(
        ax_full,
        FULL_LON_TICKS,
        FULL_LAT_TICKS,
        proj,
    )
    plot_site_symbols(ax_full, sites, proj, markersize=5.0)

    ax_full.set_title(
        "(a) Canonical site network",
        loc="left",
        fontsize=7.5,
        pad=3.0,
    )

    # Show exactly what geographic region is magnified in panel (b).
    zoom_box = Rectangle(
        (EU_LON_MIN, EU_LAT_MIN),
        EU_LON_MAX - EU_LON_MIN,
        EU_LAT_MAX - EU_LAT_MIN,
        fill=False,
        linewidth=0.8,
        linestyle="--",
        edgecolor="0.25",
        transform=proj,
        zorder=4,
    )
    ax_full.add_patch(zoom_box)

    # -----------------------------------------------------------------
    # (b) Europe zoom
    # -----------------------------------------------------------------
    eu_sites = sites_in_extent(
        sites,
        EU_LON_MIN,
        EU_LON_MAX,
        EU_LAT_MIN,
        EU_LAT_MAX,
    )

    ax_eu.set_extent(
        [EU_LON_MIN, EU_LON_MAX, EU_LAT_MIN, EU_LAT_MAX],
        crs=proj,
    )
    add_geography(ax_eu)
    configure_ticks(
        ax_eu,
        EU_LON_TICKS,
        EU_LAT_TICKS,
        proj,
    )
    plot_site_symbols(ax_eu, eu_sites, proj, markersize=5.3)

    ax_eu.set_title(
        "(b) European zoom",
        loc="left",
        fontsize=7.5,
        pad=3.0,
    )

    # -----------------------------------------------------------------
    # External legend
    # -----------------------------------------------------------------
    legend_handles = build_legend_handles(sites)
    legend = fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.805, 0.515),
        ncol=1,
        frameon=False,
        handletextpad=0.40,
        columnspacing=0.8,
        borderaxespad=0.0,
        labelspacing=0.28,
        numpoints=1,
        fontsize=6.2,
    )

    # Match each legend label color to the corresponding map symbol.
    for text, handle in zip(legend.get_texts(), legend_handles):
        text.set_color(handle.get_color())

    output = os.path.abspath(
        os.path.expanduser(os.path.expandvars(output))
    )
    outdir = os.path.dirname(output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    fig.savefig(output, dpi=dpi)
    plt.close(fig)

    print(f"Wrote:\n  {output}")
    print(f"Canonical sites plotted in panel (a): {len(sites)}")
    print(f"Sites repeated in European zoom: {len(eu_sites)}")


# =====================================================================
# Main
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the canonical NDnet/CloudNet ground-site network with "
            "a full-domain panel and a European zoom."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_JSONL,
        help=f"Site diagnostics JSONL. Default: {DEFAULT_JSONL}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output PDF/PNG/SVG path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--exclude-site",
        nargs="+",
        default=[],
        metavar="SITE",
        help=(
            "One or more additional exact site names to omit. These are "
            "combined with the canonical EXCLUDE_SITES tuple."
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DPI,
        help=f"Raster output resolution. Default: {DPI}",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    sites = read_sites(args.input)

    exclusions = list(EXCLUDE_SITES) + list(args.exclude_site)
    sites = apply_exclusions(sites, exclusions)
    sites = assign_site_styles(sites)

    print(f"Sites in input catalog: {len(read_sites(args.input))}")
    print(f"Canonical sites after exclusions: {len(sites)}")
    print("Canonical exclusions: " + ", ".join(EXCLUDE_SITES))
    if args.exclude_site:
        print("Additional exclusions: " + ", ".join(args.exclude_site))

    plot_sites(
        sites=sites,
        output=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
