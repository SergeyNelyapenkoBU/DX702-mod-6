# Potential outcomes from the prompt
# Treated items: observed factuals are Y(1); given counterfactuals are Y(0)
Y1_treated = [3, 4]
Y0_treated = [0, 2]

# Untreated items: observed factuals are Y(0); given counterfactuals are Y(1)
Y0_untreated = [1, 2]
Y1_untreated = [-3, 4]

# Put all units in a common order: (treated #1, treated #2, untreated #1, untreated #2)
Y1 = Y1_treated + Y1_untreated      # [3, 4, -3, 4]
Y0 = Y0_treated + Y0_untreated      # [0, 2,  1, 2]

# Observed (factual) outcomes under the actual assignment:
# first two were treated -> observe Y1; last two were untreated -> observe Y0
Y_obs = [Y1[0], Y1[1], Y0[2], Y0[3]]

# Optimal policy outcome for each unit = max(Y1, Y0)
Y_opt = [max(a, b) for a, b in zip(Y1, Y0)]

# "Optimal treatment effect" as total improvement from observed to optimal policy
total_improvement = sum(y_opt - y_o for y_opt, y_o in zip(Y_opt, Y_obs))

# Also show per-unit average improvement for reference
avg_improvement = total_improvement / len(Y_obs)

print("Total improvement (optimal policy - observed):", total_improvement)  # -> 2
print("Average improvement per unit:", avg_improvement)                      # -> 0.5
