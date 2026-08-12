def _reduce_variance_interval(x, axis=None, biased=True, keepdims=False):
    # ddof=0 is biased variance (N), ddof=1 is unbiased variance (N-1)
    ddof = 0 if biased else 1
    return jnp.var(x, axis=axis, ddof=ddof, keepdims=keepdims)

def nested_rhat_constrained(result_state, num_super_chains,idx):
    # since we use only N=1, W_k is reduced to 0

    num_sub_chains = result_state.shape[0] // num_super_chains
    num_dimensions = result_state.shape[1]

    chain_states = result_state.reshape(1, -1, num_sub_chains, num_dimensions)
    # chain_states.shape = (1,16,128,2)
    # f_bar 1*k is:
    mean_subchain = jnp.mean(chain_states, axis=2)
    # mean_subchain.shape =(1,16,2)

    # f_bar **K is:
    mean_superchain = jnp.mean(mean_subchain, axis=1)
    # mean_superchain.shape = (1,2)

    variance_chain = _reduce_variance_interval(chain_states, axis=2, biased=False)
    # print(variance_chain.shape) # (1,16,2)
    W = jnp.mean(variance_chain, axis=1)
    # print(f"W dim: {W.shape}") # (1,2)
    B = _reduce_variance_interval(mean_subchain, axis=1, biased=False) # variance of between super chain

    r_hat = jnp.sqrt(1+B/W)[:,idx]
    return r_hat

##############################################
# ArviZ implementations
###############################################
# chain_ids = np.repeat(
#     np.arange(0, num_super_chains),
#     num_chains_short // num_super_chains
# )


# test = new_states_1.position[:, 0][:,None]
# _rhat_nested(test,chain_ids)

# from src/arviz_stats/base/diagnostics.py
def _rhat_nested(ary, superchain_ids):
        ary = np.asarray(ary)
        nchains, niterations = ary.shape

        # Check that all chains are assigned a superchain
        if len(superchain_ids) != nchains:
            raise ValueError("Length of superchain_ids not equal to number of chains")

        # Check that superchains have equal length
        superchain_counts = np.bincount(superchain_ids)
        nchains_per_superchain = np.max(superchain_counts)

        if nchains_per_superchain != np.min(superchain_counts):
            raise ValueError("Number of chains per superchain is not the same for each superchain")

        superchains = np.unique(superchain_ids)

        # Compute chain means and variances
        chain_mean = np.mean(ary, axis=1)
        chain_var = np.var(ary, axis=1, ddof=1)

        # mean of superchains calculated by only including specified chains
        # (equation 4 in Margossian et al. 2024)
        superchain_mean = np.array([np.mean(chain_mean[superchain_ids == k]) for k in superchains])

        # between-chain variance estimate (Bhat_k in equation 7 in Margossian et al. 2024)
        if nchains_per_superchain == 1:
            var_between_chain = np.zeros(len(superchains))
        else:
            var_between_chain = np.array(
                [np.var(chain_mean[superchain_ids == k], ddof=1) for k in superchains]
            )

        #  within-chain variance estimate (What_k in equation 7 in Margossian et al. 2024)
        if niterations == 1:
            var_within_chain = np.zeros(len(superchains))
        else:
            var_within_chain = np.array(
                [np.mean(chain_var[np.where(superchain_ids == k)[0]]) for k in superchains]
            )

        # between-superchain variance (Bhat_nu in equation 6 in Margossian et al. 2024)
        var_between_superchain = np.var(superchain_mean, ddof=1)

        # within-superchain variance (What_nu in equation 7 in Margossian et al. 2024)
        var_within_superchain = np.mean(var_within_chain + var_between_chain)

        # nested Rhat (Rhat_nu in equation 8 in Margossian et al. 2024)
        return np.sqrt(1 + var_between_superchain / var_within_superchain)

def kernel_setup(warmup_length,
                 num_total_chains, num_super_chains,
                 naive,initialize_fn, randomKey,
                 target_log_prob_fn,init_step_size):
    key, init_key = random.split(randomKey)
    
    if naive:
      initial_position = initialize_fn((num_total_chains,), init_key)
    else:
        num_sub_chains = num_total_chains//num_super_chains
        initial_position_super = initialize_fn((num_super_chains,), init_key)
        initial_position = jnp.repeat(initial_position_super,num_sub_chains,axis=0)

    warmup = blackjax.chees_adaptation(
        target_log_prob_fn,num_chains=num_total_chains,
        target_acceptance_rate=0.75)
    optimizer = optax.adam(learning_rate=0.001)
    key_warmup, key_sample = random.split(key)
    (last_states, parameters), _= warmup.run(
      key_warmup,
      initial_position,
      init_step_size,
      optimizer,
      warmup_length,)
    sample_keys = random.split(key_sample, num_total_chains)
    kernel = blackjax.dhmc(target_log_prob_fn, **parameters).step
    return kernel, sample_keys, last_states

def mse_calculation(result, mean_benchmark,
                    var_benchmark, mse_list):
    mc_mean = result.mean(axis=0)
    squared_error = (mc_mean - mean_benchmark)**2
    factor = 1/var_benchmark
    factored_sq_err = factor*squared_error
    mse = (factored_sq_err).mean()
    mse_list.append(mse)
    return mse_list, factored_sq_err

def simulation(warmup_length,num_total_chains, num_super_chains,
               naive,initialize_fn, randomKeys,
               target_log_prob_fn,init_step_size,
               repitition,R_hat_list,MSE_list,
               true_mean,true_var):
     result_mse = []
     for sim in range(repitition):
        kernel, sample_keys, last_states = kernel_setup(warmup_length,
                                                        num_total_chains, num_super_chains,
                                                        naive,initialize_fn, randomKeys[sim],
                                                        target_log_prob_fn,init_step_size)
        sample_states, info = jax.vmap(kernel)(sample_keys, last_states)
        samples = sample_states.position
        dims = samples.shape[1]
        result_mse, factored_sq_err = mse_calculation(samples,true_mean,true_var,result_mse)

        for dim in range(dims):
            rhat = nested_rhat_constrained(samples, num_super_chains, dim)
            R_hat_list.append({
                "Warmup Length": warmup_length,
                "Iteration":sim,
                "Dimension": dim,
                "Rhat": rhat[-1],
                "MSE":factored_sq_err[dim]})
        del kernel, sample_keys, last_states, sample_states, info, samples
        gc.collect()
     mse_list = np.array(result_mse)
     mse_best = mse_list.min(axis=0)
     mse_worst = mse_list.max(axis=0)
     avg_mse = mse_list.mean(axis=0)
     MSE_list.append({
            "Warmup Length": warmup_length,
            "Avg MSE": avg_mse,
            "Best MSE": mse_best,
            "Worst MSE": mse_worst
        })
     if naive:
         print(f"Naive initialization. Warmup Length: {warmup_length}; mean of MSE is: {avg_mse}")
     else:
         print(f"Constrained initialization. Warmup Length: {warmup_length}; mean of MSE is: {avg_mse}")
     del result_mse
     gc.collect()
     jax.clear_caches()

############################################
# Plotting Functions
############################################
def MSE_vs_Warmup(constrained_df, naive_df, title):
  fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

  common = dict(
    logx=True,
    logy=True,
    legend=False,
    ax=ax,
    ylabel="MSE",
    x="Warmup Length",
    )
  ax = constrained_df.plot(
    y="Avg MSE",
    title=title,
    linestyle="-",
    color="orange",
    **common
  )
  constrained_df.plot(
    y="Best MSE",
    linestyle="--",
    color="orange",
    **common
    )
  constrained_df.plot(
    y="Worst MSE",
    linestyle="--",
    color="orange",
    **common)
  naive_df.plot(
    y="Avg MSE",
    linestyle="-",
    color="black",
    **common)
  naive_df.plot(
    y="Worst MSE",
    linestyle="--",
    color="black",
    **common)
  naive_df.plot(
    y="Best MSE",
    linestyle="--",
    color="black",
    **common)
  ax.legend(handles=[
    Line2D([0], [0], color="orange", lw=2, label="Constrained"),
    Line2D([0], [0], color="black", lw=2, label="Naive")])



def MSE_vs_Rhat(
    df,
    title,
    naive,
    bound,
    threshold,
    num_subchains,
    ColorDimension=False,
):

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)

    if ColorDimension:
        ###################################################
        # Color by Dimension (continuous)
        ###################################################

        sc = ax.scatter(
            df["Rhat"]-1,
            df["MSE"],
            c=df["Dimension"],
            cmap="viridis",      # or "turbo"
            s=35,
            alpha=0.4,
        )

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Dimension")

    else:
        ###################################################
        # Color by Warmup Length (discrete)
        ###################################################

        cmap = plt.get_cmap("turbo")

        warmups = np.sort(df["Warmup Length"].unique())

        color_map = {
            w: cmap(x)
            for w, x in zip(
                warmups,
                np.linspace(0, 1, len(warmups))
            )
        }

        groups = df.groupby("Warmup Length")

        for warmup, subset in groups:
            ax.scatter(
                subset["Rhat"]-1,
                subset["MSE"],
                color=color_map[warmup],
                s=35,
                alpha=0.4,
            )

        handles = [
            Line2D(
                [0], [0],
                marker="o",
                color=color,
                linestyle="",
                markersize=8,
                label=str(warmup),
            )
            for warmup, color in color_map.items()
        ]

        ax.legend(
            handles=handles,
            title="Warmup Length",
            loc="lower right",
            ncol=2,          # helpful if you have many warmups
            fontsize=9,
        )

    ###################################################
    # Common formatting
    ###################################################

    ax.set_yscale("log")

    if not naive:
        ax.set_xscale("log")

    ax.axhline(bound[0], color="black", linestyle="--")
    ax.axhline(bound[1], color="black", linestyle="--")
    ax.axhline(1 / num_subchains, color="black")
    ax.axvline(threshold, color="blue", linestyle="--")

    ax.set_xlabel(r"$\widehat{R}_{\nu}-1$")
    ax.set_ylabel("MSE")

    suffix = "Constrained" if not naive else "Naive"
    ax.set_title(f"{title} - {suffix}")

    plt.tight_layout()
    plt.show()