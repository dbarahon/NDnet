#!/usr/bin/env python3
"""Plot Cloudnet ground-site locations from the NDnet site diagnostics JSONL.

The JSONL file is expected to contain one object per line with, at minimum,
site, lat and lon fields. The default map is restricted to 30 W--70 E because
the network is concentrated in Europe and the eastern Atlantic; sites outside
that longitude range are omitted and reported on stdout. Each retained site is
assigned a stable alphabetical letter ID after exclusions and domain filtering.
The map shows a colored marker and matching colored letter for each site; full
site names are kept in an external legend to avoid crowding the map.

Examples
--------
Default input and all sites::

    python plots/site_locations.py

Leave selected sites out::

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
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D

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

# Permanent exclusions can be set here. Command-line --exclude-site entries
# are added to this tuple rather than replacing it. Matching is case-insensitive.
EXCLUDE_SITES = ()

FIGSIZE = (7.2, 5.0)
DPI = 400

# The site network is concentrated in Europe and the eastern Atlantic.
# Restricting the longitude range makes individual sites identifiable while
# retaining the full meridional range of the network.
DOMAIN_LON_MIN = -30.0
DOMAIN_LON_MAX = 70.0
DOMAIN_LAT_MIN = -80.0
DOMAIN_LAT_MAX = 85.0

LON_TICKS = [-30, -10, 10, 30, 50, 70]
LAT_TICKS = [-60, -30, 0, 30, 60]

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
            "legend.fontsize": 6.8,
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

def panel_letter(index):
    """Convert 0,1,... into a,b,...,z,aa,ab,..."""
    result = ""
    i = int(index)
    while True:
        result = chr(ord("a") + (i % 26)) + result
        i = i // 26 - 1
        if i < 0:
            break
    return result


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

            sites_by_name[key] = {"site": name, "lat": lat, "lon": lon}

    sites = list(sites_by_name.values())
    if not sites:
        raise ValueError(f"No valid sites were found in:\n{path}")

    sites.sort(key=lambda item: item["site"].casefold())
    return sites


def apply_exclusions(sites, exclusions):
    """Remove exact site-name matches, case-insensitively."""
    excluded = {str(name).strip().casefold() for name in exclusions if str(name).strip()}
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


def apply_domain_filter(sites):
    """Keep sites within the plotted longitude window.

    Sites outside 30 W--70 E are omitted from both the map and legend and are
    returned separately so the script reports them explicitly for the caption.
    """
    selected = []
    omitted = []

    for site in sites:
        if DOMAIN_LON_MIN <= site["lon"] <= DOMAIN_LON_MAX:
            selected.append(site)
        else:
            omitted.append(site)

    if not selected:
        raise ValueError("No sites fall within the configured map domain.")

    return selected, omitted


def assign_site_styles(sites):
    """Assign letter, color and marker identifiers to sites."""
    nsite = len(sites)
    cmap = plt.get_cmap("turbo")
    # Avoid the darkest extreme endpoints while spanning the full palette.
    colors = cmap(np.linspace(0.04, 0.96, nsite))

    styled = []
    for index, (site, color) in enumerate(zip(sites, colors)):
        item = dict(site)
        item["letter"] = panel_letter(index)
        item["color"] = color
        item["marker"] = MARKERS[index % len(MARKERS)]
        styled.append(item)
    return styled


# =====================================================================
# Plotting
# =====================================================================

def plot_sites(sites, output, dpi=DPI):
    """Draw the journal-style global site-location map."""
    set_journal_style()

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    # Reserve the right portion of the figure for the external legend.
    fig.subplots_adjust(left=0.075, right=0.64, bottom=0.10, top=0.98)

    ax.set_extent(
        [DOMAIN_LON_MIN, DOMAIN_LON_MAX, DOMAIN_LAT_MIN, DOMAIN_LAT_MAX],
        crs=proj,
    )
    ax.add_feature(cfeature.LAND, facecolor="0.94", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAKES, facecolor="white", edgecolor="0.70", linewidth=0.25)
    ax.coastlines(linewidth=0.55, color="0.25", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.30, edgecolor="0.45", zorder=2)

    ax.set_xticks(LON_TICKS, crs=proj)
    ax.set_yticks(LAT_TICKS, crs=proj)
    ax.xaxis.set_major_formatter(
        LongitudeFormatter(zero_direction_label=True, dateline_direction_label=True)
    )
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(axis="both", which="major", labelsize=7.0, pad=1.5)

    legend_handles = []
    for site in sites:
        ax.plot(
            site["lon"],
            site["lat"],
            linestyle="None",
            marker=site["marker"],
            markersize=5.2,
            markerfacecolor=site["color"],
            markeredgecolor=site["color"],
            markeredgewidth=0.6,
            transform=proj,
            zorder=5,
        )

        # Keep the map uncluttered by drawing only compact letter IDs on-map.
        # A white halo preserves legibility over coastlines and neighboring sites.
        txt = ax.annotate(
            site["letter"],
            xy=(site["lon"], site["lat"]),
            xycoords="data",
            xytext=(3.2, 3.2),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
            color=site["color"],
            clip_on=True,
            zorder=6,
        )
        txt.set_path_effects(
            [path_effects.withStroke(linewidth=1.5, foreground="white")]
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=site["marker"],
                markersize=4.8,
                markerfacecolor=site["color"],
                markeredgecolor=site["color"],
                markeredgewidth=0.6,
                color=site["color"],
                label=f"{site['letter']}  {display_name(site['site'])}",
            )
        )

    legend_columns = 2 if len(sites) > 18 else 1
    legend = ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.50),
        bbox_transform=ax.transAxes,
        ncol=legend_columns,
        frameon=False,
        handletextpad=0.45,
        columnspacing=0.9,
        borderaxespad=0.0,
        labelspacing=0.35,
        numpoints=1,
    )

    # The legend labels use the same color as their corresponding map symbols.
    for text, handle in zip(legend.get_texts(), legend_handles):
        text.set_color(handle.get_color())

    output = os.path.abspath(os.path.expanduser(os.path.expandvars(output)))
    outdir = os.path.dirname(output)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    print(f"Wrote:\n  {output}")


# =====================================================================
# Main
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot NDnet/Cloudnet ground-site locations from JSONL."
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
            "One or more exact site names to omit. May be combined with the "
            "EXCLUDE_SITES tuple near the top of the script."
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
    sites, domain_omitted = apply_domain_filter(sites)
    sites = assign_site_styles(sites)

    print(f"Sites plotted: {len(sites)}")
    if exclusions:
        print("Exclusions requested: " + ", ".join(exclusions))
    if domain_omitted:
        print(
            "Outside plotted longitude range "
            f"({abs(DOMAIN_LON_MIN):g} W--{DOMAIN_LON_MAX:g} E): "
            + ", ".join(site["site"] for site in domain_omitted)
        )

    plot_sites(sites, args.output, dpi=args.dpi)


if __name__ == "__main__":
    main()
