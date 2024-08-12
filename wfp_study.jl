using XLSX
using JuMP
using Gurobi
using JSON
using Gogeta
using Plots

#### LOAD DATA ####
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
[trans_cost[row[1], row[2]] = row[5] for row in eachrow(data["EdgesCost"][:][2:end, :])] # $ per ton for [source, destination] (same price for every commodity)

#### FORMULATION ####
ENV = Gurobi.Env()
for model in ["NN", "ICNN"], layers in [1, 2], neurons in [10, 10, 50, 100]

    include("wfp_util.jl")

    form_time = @elapsed begin
        wfp_jump = formulate_oil(model, layers, neurons)
    end

    # unset_silent(wfp_jump)
    opt_time = @elapsed optimize!(wfp_jump)

    file = open("results/palatability_$(model)s.txt", "a")
    write(file, "\nLAYERS: $layers, NEURONS: $neurons, ")
    write(file, "FORMULATION TIME: $form_time, ")
    write(file, "OPTIMIZATION TIME: $opt_time, ")
    write(file, "COST: $(objective_value(wfp_jump) - value(wfp_jump[:h_hat])), ")
    write(file, "PALATABILITY: $(value(wfp_jump[:y]))")
    close(file)

    ## Food basket
    plot = bar(wfp_jump[:x].axes[1], value.(wfp_jump[:x]).data, xticks=(1:25, wfp_jump[:x].axes[1]), xrotation=45, ylabel="Amount (grams per day)", label=false, title="Optimal food basket with $(model)_$(layers)_$(neurons)", ylims=[0, 300], bottom_margin=10Plots.mm)
    savefig(plot, "results/Optimal food basket with $(model)_$(layers)_$(neurons)")
end














filter(pair -> pair[2] > 0, map(food -> food => value(wfp_jump[:x][food]), wfp_jump[:x].axes[1]))

for s in wfp_jump[:F].axes[1], d in wfp_jump[:F].axes[2], f in wfp_jump[:F].axes[3]
    if value(wfp_jump[:F][s, d, f]) != 0# && f == "Beans"
        printstyled(f; color=:red)
        print(" FROM ")
        printstyled(s; color=:green)
        print(" TO ")
        printstyled(d; color=:blue)
        print(" AT ")
        printstyled("$(round(value(wfp_jump[:F][s, d, f]), digits=3)) TONS"; color=:yellow)
        println()
    end
end

# check_ICNN(Gurobi.Optimizer, "models/palatability_ICNN_negated.json", value(y), value.(x_kilo)...; negated=true)
println("PROCUREMENT COST: $(sum(proc_cost[i][k]*value(F[i, j, k]) for i in Ns, j in union(Nt, Nd), k in K))")
println("TRANSPORT COST: $(sum(get(trans_cost, (i, j), 0)*value(F[i, j, k]) for i in union(Ns, Nt), j in union(Nt, Nd), k in K))")

## Food prices

bar(collect(keys(proc_cost["Amman S"])), map(p -> p > 1e5 ? 0 : p, collect(values(proc_cost["Amman S"]))), xticks=(1:25, keys(proc_cost["Amman S"])), xrotation=45, ylabel="Price (\$ per ton)", label=false)

bar(collect(keys(proc_cost["Hassakeh S"])), map(p -> p > 1e5 ? 0 : p, collect(values(proc_cost["Hassakeh S"]))), xticks=(1:25, keys(proc_cost["Hassakeh S"])), xrotation=45, label="Price")

bar(collect(keys(proc_cost["Homs S"])), map(p -> p > 1e5 ? 0 : p, collect(values(proc_cost["Homs S"]))), xticks=(1:25, keys(proc_cost["Homs S"])), xrotation=45, label="Price")

bar(collect(keys(proc_cost["Gaziantep S"])), map(p -> p > 1e5 ? 0 : p, collect(values(proc_cost["Gaziantep S"]))), xticks=(1:25, keys(proc_cost["Gaziantep S"])), xrotation=45, label="Price")

bar(collect(keys(proc_cost["Dayr_Az_Zor S"])), map(p -> p > 1e5 ? 0 : p, collect(values(proc_cost["Dayr_Az_Zor S"]))), xticks=(1:25, keys(proc_cost["Dayr_Az_Zor S"])), xrotation=45, label="Price")