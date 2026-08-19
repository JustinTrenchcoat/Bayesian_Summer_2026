############################################
# Replication Plotting Functions
############################################
# Line Plots
def MSE_vs_Warmup(W_c_df, W_n_df,title):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    common = dict(
        logx=True,
        logy=True,
        legend=False,
        ax=ax,
        ylabel="MSE",
        x="Warmup Length")
    ax = W_c_df.plot(
        y="Avg MSE",
        title=title,
        linestyle="-",
        color="orange",
        **common)
    W_c_df.plot(
        y="Best MSE",
        linestyle="--",
        color="orange",
        **common)
    W_c_df.plot(
        y="Worst MSE",
        linestyle="--",
        color="orange",
        **common)
    W_n_df.plot(
        y="Avg MSE",
        linestyle="-",
        color="black",
        **common)
    W_n_df.plot(
        y="Worst MSE",
        linestyle="--",
        color="black",
        **common)
    W_n_df.plot(
        y="Best MSE",
        linestyle="--",
        color="black",
        **common)
    ax.legend(handles=[
        Line2D([0], [0], color="orange", lw=2, label="Constrained"),
        Line2D([0], [0], color="black", lw=2, label="Naive")])

# Scatter Plots
def MSE_vs_Rhat(df, title, naive, bound, threshold, num_subchains):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    ax.scatter(
        df["Rhat"] - 1,
        df["MSE"],
        color="blue",
        s=35,
        alpha=0.4,
    )

    ax.set_yscale("log")

    if not naive:
        ax.set_xscale("log")

    ax.axhline(bound[0], color="black", linestyle="--")
    ax.axhline(bound[1], color="black", linestyle="--")
    ax.axhline(1 / num_subchains, color="black")
    ax.axvline(threshold, color="blue", linestyle="--")

    ax.set_xlabel(r"$\widehat{R}_{\nu}-1$")
    ax.set_ylabel("Scaled squared error")

    suffix = "Constrained" if not naive else "Naive"
    ax.set_title(f"{title} - {suffix}")

    plt.tight_layout()
    plt.show()

############################################
# Enhanced Plotting Functions
############################################
def MSE_vs_Warmup_Jumbo(tfp_c_df, tfp_n_df, 
                        bjx_c_df, bjx_n_df,
                        pf_df, title):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    colors = {
           "TFP":"orange",
           "BlackJAX":"black",
           "PathFinder":"blue"}
    linestyles = {
           "Constrained":"-",
           "Naive":"--"}
    common = dict(
           logx=True,
           logy=True,
           legend=False,
           ax=ax,
           x="Warmup Length")
    def plot_with_band(df, color, linestyle, label, hatch):
           ax.plot(
               df["Warmup Length"],
               df["Avg MSE"],
               color = color,
               linestyle = linestyle,
               linewidth = 2,
               label = label)
           ax.fill_between(
               df["Warmup Length"],
               df["Best MSE"],
               df["Worst MSE"],
               color = color,
               alpha = 0.15,
               hatch = hatch)
    # TFP implementation:
    plot_with_band(
        tfp_c_df,
        colors["TFP"],
        linestyles["Constrained"],
        "TFP-Constrained",
        hatch = None)
    plot_with_band(
        tfp_n_df,
        colors["TFP"],
        linestyles["Naive"],
        "TFP-Naive",
        hatch = "///")
    # BlackJAX implementation
    plot_with_band(
        bjx_c_df,
        colors["BlackJAX"],
        linestyles["Constrained"],
        "BlackJAX-Constrained",
        hatch = None)
    plot_with_band(
        bjx_n_df,
        colors["BlackJAX"],
        linestyles["Naive"],
        "BlackJAX-Naive",
        hatch = "xxx")
    plot_with_band(
        pf_df,
        colors["PathFinder"],
        '-',
        "PathFinder Initialization",
        hatch = None)
    ax.set_title(title)
    ax.set_ylabel("MSE")
    ax.set_xlabel("Warmup Length")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.legend()
    plt.show()
    
# Color Coded Scatter Plots
def MSE_vs_Rhat_color(tfp_c_df, tfp_n_df,bjx_c_df, bjx_n_df, pf_df,
                      title, bound, threshold,num_subchains, color_choice):
     fig, axes = plt.subplots(2,3, figsize=(25,10),dpi=150,
                              sharex=True, sharey = True)
     axes = axes.flatten()
     dfs=[tfp_c_df, tfp_n_df, pf_df,
          bjx_c_df, bjx_n_df, pf_df]
     titles =["TFP - Constrained","TFP - Naive","PathFinder",
              "BlackJAX - Constrained","BlackJAX - Naive","PathFinder"]
     if color_choice == "Dimension":
         vmin = min(df["Dimension"].min() for df in dfs)
         vmax = max(df["Dimension"].max() for df in dfs)

         for ax, df, panel_title in zip(axes, dfs, titles):
               sc = ax.scatter(
                    df["Rhat"] - 1,
                    df["MSE"],
                    c = df["Dimension"],
                    cmap = "viridis",
                    vmin = vmin,
                    vmax = vmax,
                    s=35,
                    alpha = 0.4
               )

               ax.set_title(panel_title)
         cbar = fig.colorbar(sc, ax = axes, shrink=0.8)
         cbar.set_label("Dimension")

         if vmax - vmin <= 20:
               tick_values = np.arange(vmin, vmax +1)
         else:
               tick_values = np.linspace(vmin , vmax, 5). round().astype(int)
         cbar.set_ticks(tick_values)
          
     elif color_choice== "Warmup Length":
          warmups = np.sort(
               np.unique(np.concatenate(
                    [df["Warmup Length"].unique() for df in dfs]
               ))
          )
          cmap = plt.get_cmap("RdYlBu_r")
          color_map = {
               w:cmap(x)
               for w, x in zip(warmups, np.linspace(0,1,len(warmups)))
          }  

          for ax, df, panel_title in zip(axes, dfs, titles):
               groups = df.groupby("Warmup Length")

               for warmup, subset in groups:
                    ax.scatter(
                         subset["Rhat"] -1,
                         subset["MSE"],
                         color = color_map[warmup],
                         s=35,
                         alpha=0.4)
               ax.set_title(panel_title)
          handles = [Line2D(
               [0],[0],marker="o", color = color_map[warmup],
               linestyle = "", markersize = 7, label=str(warmup)) for warmup in warmups]
          fig.legend(
               handles = handles,
               title = "Warmup Length",
               loc = "center right",
               bbox_to_anchor = (1.02, 0.5),
               ncol = 1,
               fontsize = 9,
          )
     else:
          raise ValueError("color_choice must be either \"Dimension\" or \"Warmup Length\"!")
     for ax in axes:
          ax.set_yscale("log")
          ax.set_xscale("log")

          ax.axhline(
               bound[0], color="black",linestyle="--"
          )

          ax.axhline(
               bound[1], color="black",linestyle="--"
          )

          ax.axhline(
            1 / num_subchains,color="black"
            )
          ax.axvline(
            threshold, color="blue",linestyle="--"
            )
          
          ax.set_xlabel(r"$\widehat{R}_{\nu}-1$")

          ax.tick_params(
               axis="both",
               which = "both",
               labelbottom = True,
               labelleft = True
          )
     axes[0].set_ylabel("MSE")

     fig.suptitle(title)
     if color_choice == "Warmup Length":
          plt.subplots_adjust(right=0.88)
     plt.show()