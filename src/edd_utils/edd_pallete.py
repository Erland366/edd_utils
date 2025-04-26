def register_edd_pallete():
    import matplotlib
    import matplotlib.colors as mcolors
    import seaborn as sns

    seq_colors_hex = [
        "#e0f2f7", "#cce8e3", "#9bc4bc", "#6b8e8a", "#405d5b",
    ]
    seq_cmap_name = "edd_sequential_gb" # Renamed for clarity under 'edd' umbrella

    div_colors_hex = [
        "#405d5b", "#9bc4bc", "#f0f0f0", "#fdbb84", "#fc8d59",
    ]
    div_cmap_name = "edd_diverging_gb_orange" # Renamed

    qual_colors_hex = [
        "#a6cee3", "#b2df8a", "#fdbf6f", "#cab2d6", "#ffff99",
        "#fb9a99", "#e3a1c2", "#bdbdbd",
    ]
    qual_palette = sns.color_palette(qual_colors_hex)
    qual_palette_name = "edd_qualitative_pastel" # Name for reference

    cyc_colors_hex = [
        "#405d5b", "#7baca5", "#cce8e3", "#a1c9d5", "#6b8e8a",
    ]
    cyc_cmap_name = "edd_cyclic_gb"


    sequential_cmap = mcolors.LinearSegmentedColormap.from_list(seq_cmap_name, seq_colors_hex)
    diverging_cmap = mcolors.LinearSegmentedColormap.from_list(div_cmap_name, div_colors_hex)
    cyclic_cmap = mcolors.LinearSegmentedColormap.from_list(cyc_cmap_name, cyc_colors_hex)

    def register_all():
        """Registers all continuous colormaps defined in this module."""
        cmaps_to_register = {
            seq_cmap_name: sequential_cmap,
            div_cmap_name: diverging_cmap,
            cyc_cmap_name: cyclic_cmap,
        }
        registered_count = 0
        skipped_count = 0

        for name, cmap in cmaps_to_register.items():
            try:
                if name not in matplotlib.colormaps:
                    matplotlib.colormaps.register(cmap=cmap)
                    registered_count += 1
                else:
                    skipped_count +=1

            except Exception as e:
                # Catch potential issues during registration
                print(f"Warning: Could not register colormap '{name}'. Error: {e}")

        if registered_count > 0:
            print(f"Registered {registered_count} custom 'edd' colormaps.")
        if skipped_count > 0:
            # This is normal if the module is re-imported in the same session
            # print(f"Skipped registration for {skipped_count} 'edd' colormaps (already registered).")
            pass


    register_all()

    n_continuous = 11
    seq_palette_list = sns.color_palette(seq_cmap_name, n_colors=n_continuous)
    div_palette_list = sns.color_palette(div_cmap_name, n_colors=n_continuous)
    cyc_palette_list = sns.color_palette(cyc_cmap_name, n_colors=n_continuous)

    print("Custom 'edd' palettes module loaded and colormaps registered.")

def register_edd_style():
    import matplotlib.pyplot as plt
    import matplotlib as mpl


    modern_font = 'Helvetica Neue'  # or 'Arial', 'Lato', 'Roboto'

    try:
        global qual_palette
        color_cycle = qual_palette
    except AttributeError:
        print("Warning: Could not load 'qual_palette' from edd_palettes.")
        print("Using default Matplotlib color cycle instead.")
        # Fallback to a default mpl cycle if edd_palettes isn't found/correct
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


    # Define the rcParams dictionary for the 'edd_modern' style
    edd_modern_style = {
        # --- Font Settings ---
        "font.family": "sans-serif",
        "font.sans-serif": [modern_font],
        "font.size": 12,             # Base font size
        "axes.titlesize": 16,        # Title font size
        "axes.labelsize": 14,        # Axis label font size
        "xtick.labelsize": 11,       # X-tick label size
        "ytick.labelsize": 11,       # Y-tick label size
        "legend.fontsize": 11,       # Legend font size
        "figure.titlesize": 18,      # Figure suptitle size

        # --- Color Settings ---
        "axes.prop_cycle": plt.cycler(color=color_cycle), # Use your qualitative palette
        "axes.facecolor": "white",   # Background color of the axes area
        "figure.facecolor": "white", # Background color of the figure area
        "axes.edgecolor": "lightgray", # Color of the axes border lines (spines)
        "xtick.color": "dimgray",    # Color of the x-tick marks and labels
        "ytick.color": "dimgray",    # Color of the y-tick marks and labels
        "axes.labelcolor": "dimgray", # Color of the axis labels
        "axes.titlecolor": "black",   # Color of the axes title

        # --- Line and Marker Settings ---
        "lines.linewidth": 2.0,      # Default line width
        "lines.markersize": 6,       # Default marker size

        # --- Grid Settings ---
        "axes.grid": True,           # Enable grid
        "grid.color": "lightgray",   # Grid line color
        "grid.linestyle": "--",      # Grid line style
        "grid.linewidth": 0.7,       # Grid line width
        "grid.alpha": 0.7,           # Grid line transparency

        # --- Axes Spine Settings (Modern Look - remove top/right) ---
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,

        # --- Tick Settings ---
        "xtick.direction": "out",    # Ticks point outwards
        "ytick.direction": "out",
        "xtick.major.size": 5,       # Length of major ticks
        "xtick.minor.size": 3,       # Length of minor ticks
        "ytick.major.size": 5,
        "ytick.minor.size": 3,
        "xtick.major.width": 0.8,    # Width of major ticks
        "xtick.minor.width": 0.6,    # Width of minor ticks
        "ytick.major.width": 0.8,
        "ytick.minor.width": 0.6,
        "xtick.major.pad": 7,        # Distance from axis to tick label
        "ytick.major.pad": 7,

        # --- Figure Layout Settings ---
        "figure.figsize": (8, 6),    # Default figure size in inches (adjust as needed)
        "figure.dpi": 100,           # Figure resolution
        # Tighter layout often looks better
        "figure.autolayout": True,   # Automatically adjust subplot params for tight layout
        # Or use constrained layout (newer alternative):
        # "figure.constrained_layout.use": True,

        # --- Legend Settings ---
        "legend.frameon": False,     # No frame around the legend
        "legend.loc": "best",        # Default legend location
        "legend.borderaxespad": 0.5, # Padding between legend and axes borders
    }

    # --- Function to Apply the Style ---
    def apply_style():
        """Applies the 'edd_modern' style settings to Matplotlib."""
        try:
            plt.style.use(edd_modern_style)
            print("Applied 'edd_modern' plotting style.")
        except Exception as e:
            print(f"Error applying 'edd_modern' style: {e}")
            print("Using default Matplotlib settings.")

    apply_style()