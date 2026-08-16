def plot_top_dimension_failures(
    df,
    bound,
    threshold,
    warmup_groups,
    criterion,
    top_n,
    figsize_per_subplot=(5, 6),
):

    df = df.copy()

    mse_lower, mse_upper = bound

    df["Rhat Failure"] = (
        df["Rhat"] - 1 > threshold
    )

    df["MSE Failure"] = (
        (df["MSE"] < mse_lower) |
        (df["MSE"] > mse_upper)
    )

    if criterion == "Rhat":

        failure_column = "Rhat Failure"
        criterion_label = "Nested R-hat"

    elif criterion == "MSE":

        failure_column = "MSE Failure"
        criterion_label = "MSE"

    else:

        raise ValueError(
            "criterion must be either 'Rhat' or 'MSE'"
        )

    n_groups = len(warmup_groups)

    fig, axes = plt.subplots(
        1,
        n_groups,
        figsize=(
            figsize_per_subplot[0] * n_groups,
            figsize_per_subplot[1],
        ),
        squeeze=False,
        dpi=150,
    )

    axes = axes.ravel()

    results = {}

    for ax, (group_name, (warmup_min, warmup_max)) in zip(
        axes,
        warmup_groups.items()
    ):

        # Select warmup lengths belonging to this group
        subset = df[
            df["Warmup Length"].between(
                warmup_min,
                warmup_max
            )
        ].copy()

        if subset.empty:

            ax.set_title(group_name)

            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

            ax.axis("off")

            continue

        frequency = (
            subset
            .groupby("Dimension")[failure_column]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        top_dimensions = (
            frequency
            .head(top_n)
            .iloc[::-1]
        )

        # Store results before plotting
        results[group_name] = top_dimensions

        ranks = range(
            len(top_dimensions),
            0,
            -1,
        )

        ranked_labels = [
            f"{rank}: {dimension}"
            for rank, dimension in zip(
                ranks,
                top_dimensions.index,
            )
        ]

        y = np.arange(
            len(top_dimensions)
        )

        ax.barh(
            y,
            top_dimensions.values,
        )

        ax.set_yticks(y)

        ax.set_yticklabels(
            ranked_labels
        )

        ax.set_ylabel(
            "Rank: Dimension"
        )

        ax.set_xlabel(
            "Failure Frequency"
        )

        ax.set_title(
            f"Warmup Length: {group_name}"
        )

    fig.suptitle(
        f"Top {top_n} Dimensions by "
        f"{criterion_label} Failure Frequency",
        y=1.02,
    )

    plt.tight_layout()

    plt.show()

    return results