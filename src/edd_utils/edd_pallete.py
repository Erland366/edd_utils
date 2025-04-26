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

