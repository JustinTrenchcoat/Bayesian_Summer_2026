def kernel_setup(warmup_length, sampling_length,
                 num_sub_chains,num_super_chains,
                 naive, initialize_fn,ranKey,
                 target_log_prob_fn, init_step_size):
  num_warmup_short, num_sampling_short = warmup_length, sampling_length
  total_samples_short = num_warmup_short + num_sampling_short

  kernel_short = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn, init_step_size, 1)
  kernel_short = tfp.experimental.mcmc.GradientBasedTrajectoryLengthAdaptation(kernel_short, num_warmup_short)
  kernel_short = tfp.mcmc.DualAveragingStepSizeAdaptation(
      kernel_short, num_warmup_short, target_accept_prob = 0.75,  #0.75,
      reduce_fn = tfp.math.reduce_log_harmonic_mean_exp)

  if (naive):
    # initialize each chain at a different location
    initial_state = initialize_fn((num_sub_chains,),key=ranKey)
    initial_state_super = initial_state

  else:
    # Chains within a super chain are all initialized at the same location
    initial_state_super = initialize_fn((num_super_chains,), key=ranKey)
    initial_state = jnp.repeat(initial_state_super, num_sub_chains // num_super_chains,
                            axis = 0)
  return kernel_short, initial_state, total_samples_short, initial_state_super


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

# add a switch for recording the states
def simulation(keys, initialize_fn,
               warmup_length,sampling_length,
               naive, repitition,record_states,
               MSE_list, R_Hat_list,
               num_dim, state_list,
               mean_benchmark,var_benchmark,
               num_super_chains,num_sub_chains,
               target_log_prob_fn, init_step_size):
  result_mse = []
  # very long warmup phase might result in memory issue.
  # if record_states = True we do record when warmup < 300
  # if record_states = False, we skip trace
  if not record_states:
     skip_trace = True
  else:
     skip_trace = (warmup_length > 300)

  for sim in range(repitition):
    kernel_short, initial_state, total_samples_short, initial_state_super = kernel_setup(warmup_length,sampling_length,
                                                            num_sub_chains, num_super_chains,
                                                            naive, initialize_fn,keys[sim],
                                                            target_log_prob_fn, init_step_size)
    # print(initial_state.shape)
    # print(f"unique vals in initial state: {jnp.unique(initial_state[:,0])}")
    # print(f"unique vals in initial state: {jnp.unique(initial_state[:,1])}")
    # print(f"initial state super chain: {initial_state_super}")
    if not skip_trace:
      result = tfp.mcmc.sample_chain(
        total_samples_short, initial_state, kernel = kernel_short,
        seed =keys[sim], trace_fn=None)
      result_with_init = jnp.concatenate([initial_state[None, :, :], result],axis=0)
      result_short = result[-1,:,:]
      state_record = {
            "Warmup Length": warmup_length,
            "Iteration": sim,
            "States":np.asarray(result_with_init),
            "Initial Position":np.asarray(initial_state_super)
          }
    else:
      result = []
      result_with_init = []
      result_short = tfp.mcmc.sample_chain(
         total_samples_short, initial_state, kernel = kernel_short,
         seed =keys[sim], trace_fn=None)[-1,:,:]
      #  if not record_states:
      state_record = {}
      #  else:
      #     state_record = {
      #       "Warmup Length": warmup_length,
      #       "Iteration": sim,
      #       "States":[],
      #       "Initial Position":[]
      #     }
    state_list.append(state_record)
    # result_short_shape is (2048, num_dim)
    # print(f"result full shape: {result.shape}")
    # print(f"result with init shape: {result_with_init.shape}")
    # print(f"result_short shape: {result_short.shape}")

    # the f bar
    mc_mean = result_short.mean(axis=0)
    squared_error = (mc_mean - mean_benchmark)**2
    # the factor was in fact 1/var
    factor = 1/var_benchmark
    # sq_err for all dimensions: a vector with dimension=num_dim
    factored_sq_err = factor*squared_error
    # avg over all dimension
    mse = (factored_sq_err).mean()
    result_mse.append(mse)

    for dim in range(num_dim):
      new_r_hat = nested_rhat_constrained(result_short, num_super_chains, dim)
      R_Hat_list.append({
         "Warmup Length": warmup_length,
         "Iteration":sim,
         "Dimension": dim,
         "Rhat": new_r_hat[-1],
         "MSE":factored_sq_err[dim]
      })
    del state_record
    del result_short
    del result
    del kernel_short
    del initial_state
    del initial_state_super
    del total_samples_short
    del result_with_init
    gc.collect()

  # calculation time:
  result_mse = np.array(result_mse)
  result_mse_best = result_mse.min(axis=0)
  result_mse_worst = result_mse.max(axis=0)
  # avg on all simulations
  mean_mse = result_mse.mean(axis=0)

  MSE_list.append({"Warmup Length":warmup_length,
                   "Avg MSE": mean_mse,
                   "Best MSE": result_mse_best,
                   "Worst MSE": result_mse_worst})
  if naive:
    print(f"Naive initialization. Warmup Length: {warmup_length}; mean of MSE is: {mean_mse}")
  else:
    print(f"Constrained initialization. Warmup Length: {warmup_length}; mean of MSE is: {mean_mse}")

  del result_mse
  gc.collect()
  jax.clear_caches()



#########################################################################################################

# plotting functions:
# the plotting function would be based on dfs returned by the simulation
def trace_plot(state_df, iteration, warmup_length,
               DimIdx1, DimIdx2,
               show_chain_num, naive):
  #          state_c_df[(state_c_df['Warmup Length']==10) & (state_c_df['Iteration'] == 0)]['States']
    states = state_df.loc[
        (state_df['Warmup Length'] == warmup_length) &
        (state_df['Iteration'] == iteration),
        'States'
        ].iloc[0]
  # with shape of (W+2, num_subchain, dim)
    initial_pos = states[0]
    num_super_chain = initial_pos.shape[0]
    chains = np.random.choice(states.shape[1], show_chain_num, replace=False)

    plt.figure(figsize=(10, 8), dpi=150)

    for chain in chains:
        trajectory = states[:, chain, :]
        # with shape (W+2, dim)
        plt.plot(trajectory[:, DimIdx1], trajectory[:, DimIdx2], alpha=0.3, lw=1)
        plt.scatter(trajectory[-1, DimIdx1], trajectory[-1, DimIdx2], s=20)

    if not naive:
        plt.scatter(
            initial_pos[:, DimIdx1],
            initial_pos[:, DimIdx2],
            color="black",
            marker="^",
            s=40,
        )
        plt.title(
           f"Iteration: {iteration}, Traceplot of Dim {DimIdx1} and Dim {DimIdx2}, "
           f"{show_chain_num} chains, constrained initialization."
           f"Warmup Length: {warmup_length}"
        )
    else:
       plt.title(
          f"Iteration: {iteration}, Traceplot of Dim {DimIdx1} and Dim {DimIdx2}, "
          f"{show_chain_num} chains, naive initialization."
          f"Warmup Length: {warmup_length}"
       )
    plt.xlabel(f"Dimension {DimIdx1}")
    plt.ylabel(f"Dimension {DimIdx2}")

    handles = [
    Line2D(
        [0], [0],
        marker="o",
        color="black",
        linestyle="",
        markersize=6,
        label="Endpoint"
    )]
    if not naive:
      handles.append(
        Line2D(
            [0], [0],
            marker="^",
            color="black",
            linestyle="",
            markersize=8,
            label="Starting Point"
        ))

    plt.legend(handles=handles, loc="best")
    plt.show()

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