function formulate_oil(model, n_layers, n_neurons)

    #### CREATE JUMP MODEL ####
    wfp_jump = Model(() -> Gurobi.Optimizer(ENV))
    set_silent(wfp_jump)

    # variables and constraints (6g) and (6i)
    @variable(wfp_jump, F[i=union(Nd, Nt, Ns), j=union(Nd, Nt, Ns), k=K] >= 0) # moved commodities in tons

    # no transports that are not listed or transports from the node to itself
    for i in union(Nd, Nt, Ns), j in union(Nd, Nt, Ns)
        if haskey(trans_cost, (i, j)) == false || i == j
            @constraint(wfp_jump, [k=K], F[i, j, k] <= 0)
        end
    end

    @variable(wfp_jump, x[k=K] >= 0) # amount of commodities in the final diet
    @variable(wfp_jump, y >= t) # palatability

    # objective (6a)
    procurement_costs = @expression(wfp_jump, sum(proc_cost[i][k]*F[i, j, k] for i in Ns, j in union(Nt, Nd), k in K))
    transport_costs = @expression(wfp_jump, sum(get(trans_cost, (i, j), 0)*F[i, j, k] for i in union(Ns, Nt), j in union(Nt, Nd), k in K))

    @objective(wfp_jump, Min, procurement_costs + transport_costs)

    @constraint(wfp_jump, x["Salt"] == 5) # (6e)
    @constraint(wfp_jump, x["Sugar"] == 20) # (6f)

    @constraint(wfp_jump, [l=L], sum(nutval[k][l] * x[k] for k in K) >= nutreq[l]) # (6d)
    @constraint(wfp_jump, [i=Nd, k=K], sum(1e6 * F[j, i, k] for j in union(Ns, Nt)) == x[k] * demand[i]) # (6c)

    @constraint(wfp_jump, [i=Nt, k=K], sum(F[i, j, k] for j in union(Nd, Nt)) == sum(F[j, i, k] for j in union(Ns, Nt))) # (6b)

    @variable(wfp_jump, h_hat)
    @variable(wfp_jump, x_kilo[k=K])
    @constraint(wfp_jump, [k=K], x_kilo[k] == 0.01 * x[k])

    if model == "ICNN"
        # ICNN formulation
        ICNN_incorporate!(wfp_jump, "models/palatability_ICNN_$(n_layers)_$(n_neurons).json", h_hat, x_kilo...)
        @constraint(wfp_jump, y == -h_hat)
    elseif model == "NN"
        # NN formulation
        NN_incorporate!(wfp_jump, "models/palatability_NN_$(n_layers)_$(n_neurons).json", y, x_kilo...; U_in=ones(25), L_in=zeros(25))
    end

    return wfp_jump
end