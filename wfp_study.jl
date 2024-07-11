using XLSX

data = XLSX.readxlsx("data/Syria_instance.xlsx")

Nd = data["Demand"][:][2:4, 1] # delivery nodes
Nt = data["Transhippers"][:][2:5] # transshipment nodes
Ns = data["Suppliers"][:][2:6] # source nodes

K = data["nutr_val"][:][2:end, 1] # commodities
L = data["nutr_val"][:][1, 2:end] # nutrients

demand = Dict([row[1] => row[2] for row in eachrow(data["Demand"][:][2:4, :])]) # demand (people*days) at each location
t = 0.5 # minimum palatability

nutreq = Dict([col[1] => col[2] for col in eachcol(data["nutr_req"][:][1:2, 2:end])]) # nutrition requirements
nutval = Dict()
[nutval[row[1]] = Dict([L[i] => nutrient/100 for (i, nutrient) in enumerate(row[2:end])]) for row in eachrow(data["nutr_val"][:][2:end, :])] # nutritional value [commodity][nutrient] per gram

proc_cost = Dict()
[proc_cost[row[1]] = Dict([K[i] => cost for (i, cost) in enumerate(row[2:end])]) for row in eachrow(data["FoodCost"][:][2:end, :])] # $ per ton for [source][commodity]

trans_cost = Dict()
# [trans_cost[row[1], row[2]] = row[3] + row[4]*row[5] for row in eachrow(data["EdgesCost"][:][2:end, :])] # $ per ton for [source, destination] (same price for every commodity)
[trans_cost[row[1], row[2]] = row[5] for row in eachrow(data["EdgesCost"][:][2:end, :])] # $ per ton for [source, destination] (same price for every commodity)

using JuMP
using Gurobi
using JSON

include("ICNN_to_LP.jl")

wfp_jump = Model(Gurobi.Optimizer)
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

# ICNN formulation
ICNN_formulate!(wfp_jump, "models/palatability_ICNN_negated.json", h_hat, x_kilo...)
@constraint(wfp_jump, y == -h_hat)

# NN formulation
NN_formulate!(wfp_jump, "models/palatability_NN_small.json", y, x_kilo...; U_in=ones(25), L_in=zeros(25))

objective_function(wfp_jump)

unset_silent(wfp_jump)
optimize!(wfp_jump)
solution_summary(wfp_jump)

for s in F.axes[1], d in F.axes[2], f in F.axes[3]
    if value(F[s, d, f]) != 0# && f == "Beans"
        printstyled(f; color=:red)
        print(" FROM ")
        printstyled(s; color=:green)
        print(" TO ")
        printstyled(d; color=:blue)
        print(" AT ")
        printstyled("$(round(value(F[s, d, f]), digits=3)) TONS"; color=:yellow)
        println()
    end
end

filter(pair -> pair[2] > 0, map(food -> food => value(x[food]), x.axes[1]))

objective_value(wfp_jump) - value(h_hat)
println("PROCUREMENT COST: $(sum(proc_cost[i][k]*value(F[i, j, k]) for i in Ns, j in union(Nt, Nd), k in K))")
println("TRANSPORT COST: $(sum(get(trans_cost, (i, j), 0)*value(F[i, j, k]) for i in union(Ns, Nt), j in union(Nt, Nd), k in K))")

function check_ICNN(optimizer, filepath, output_value, input_values...; show_output=true, nonnegated=-1)
    
    in_values = [val for val in input_values]

    icnn = Model()
    set_optimizer(icnn, optimizer)
    set_silent(icnn)
    @objective(icnn, Max, 0)

    @variable(icnn, inputs[1:length(in_values)])
    @variable(icnn, output)

    ICNN_formulate!(icnn, filepath, output, inputs...)
    icnn_value = nonnegated * forward_pass_ICNN!(icnn, in_values, output, inputs...)

    show_output && println("Output should be: $output_value")
    show_output && println("ICNN output with given input: $icnn_value")
    
    if icnn_value ≈ output_value
        show_output && println("ICNN output matches full problem\n")
        return true
    else
        show_output && @warn "ICNN output does not match"
        return false
    end
end

check_ICNN(Gurobi.Optimizer, "models/palatability_ICNN_negated.json", value(y), value.(x_kilo)...)