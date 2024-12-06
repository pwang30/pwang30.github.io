# Author: Peng Wang       from Technical University of Madrid (UPM)
# Supervisor: Luis Badesa
# This code is a simple example of an         UC-constrained optimization problem in DC power system,
# intended to learn how to model, then solve an     DC power system           optimization problem, in JuMP. 
# 03.Dec.2024

#----------------------------------Pkg Introduction----------------------------------
import Pkg
#Pkg.add("Ipopt")
#Pkg.build("CPLEX")
#ENV["CPLEX_STUDIO_BINARIES"] = "/Users/kl/Documents/cplex/CPLEX_Studio2211/cplex/bin/arm64_osx/"
#Pkg.add("CPLEX")
using JuMP, CPLEX, CSV, DataFrames,LinearAlgebra, XLSX # JuMP for optimization, CPLEX as the solver



#----------------------------------DC System Data Introduction----------------------------------
df = DataFrame(CSV.File("/Users/kl/Desktop/Julia_test/New folder/Loadcurves.csv"))          # The path to your system data can be easily changed, find the path in the property
loadcurve =df[:,:]  
df = DataFrame(CSV.File( "/Users/kl/Desktop/Julia_test/New folder/UnitPara.csv") ) 
paragen=df[:,:]
df = DataFrame(CSV.File( "/Users/kl/Desktop/Julia_test/New folder/NetworkPara.csv" )) 
netpara=df[:,:]


#----------------------------------Parameters Define----------------------------------
branch_num = size(netpara, 1)                # Number of branches in the network

PL_max = netpara[:, 6]                       # Maximum power in lines
PL_min = netpara[:, 7]                       # Minimum power in lines

limit = paragen[:, 3:4]                      # Limits for output of agents  # Upper limit is in [:, 3], Lower limit is in [:, 4]

cost_coff = paragen[:, 5:7]                  # Cost cofficients for generators (per unit value)      [:, 5]: a, [:, 6]: b, [:, 7]: c

price = 100.0; cost_coff .*= price           # Price conversion to rated value


lasttime = paragen[:, 9]                     # Generators state duration (hours)
Rud = paragen[:, 8]                          # Ramp rates
H = paragen[:, 10]                           # Startup costs
J = paragen[:, 11]                           # Shutdown costs

u0 = [1, 1, 1, 1, 1, 1]                      # Initial states of generators

gennum = size(paragen, 1)                    # Number of generators
numnodes = size(loadcurve, 1)-1              # Number of buses
T = size(loadcurve, 2) -1                    # Operation cycle (hours)

PL=loadcurve[size(loadcurve, 1),2:end]       # Total load (demand power) of all buses at each unit time (hours)   
                                             # Why 2:end, cause the index 1 is another index number, not power value!!!

slack_bus = 26                               # Balance bus number

hp = 5 /100                                  # Hot Reserve factor



#----------------------------------Model Define---------------------------------- 
model = Model(CPLEX.Optimizer)                                                                          # JuMP model

#-----------Decision Variables-----------
@variable(model, u[1:gennum, 1:T], Bin)                                                                 # State variables of generators
@variable(model, p[1:gennum, 1:T] >= 0)                                                                 # Output of each generator
@variable(model, costH[1:gennum, 1:T] >= 0)                                                             # Startup costs
@variable(model, costJ[1:gennum, 1:T] >= 0)                                                             # Shutdown costs
@variable(model, sum_PowerPTDF[1:T, 1:branch_num, 1:numnodes])                                          # The total output power transferred from the generators

for t in 1:T                                                                                            # Load=demand, equality constraint
    @constraint(model, sum(p[:, t]) == PL[t])
    #@constraint(model, 0.98*PL[t]<= sum(p[:, t]) )
end


for t in 1:T, i in 1:gennum                                                                             # Generators output upper and lower limit constraints
    @constraint(model, u[i, t] * limit[i, 2] <= p[i, t] )
    @constraint(model, p[i, t] <= u[i, t] * limit[i, 1])
end


for t in 2:T, i in 1:gennum                                                                             # Generators climbing constraints
    @constraint(model, p[i, t - 1] - p[i, t] <= Rud[i] * u[i, t] + (1 - u[i, t]) * (limit[i, 2] + limit[i, 1]) / 2)
    @constraint(model, p[i, t] - p[i, t - 1] <= Rud[i] * u[i, t - 1] + (1 - u[i, t - 1]) * (limit[i, 2] + limit[i, 1]) / 2)
end


for t in 1:T                                                                                            # Hot reserve constraints
    @constraint(model, sum(u[:, t] .* limit[:, 1] - p[:, t]) >= hp * PL[t])
end


for t in 2:T, i in 1:gennum                                                                             # Startup time constraints / The indicator is 1 to start
    indicator = u[i, t] - u[i, t - 1]
    range = t:min(T, t + lasttime[i] - 1)
    for r in range
        @constraint(model, u[i, r] >= indicator)
    end
end

for t in 2:T, i in 1:gennum                                                                             # Shutdown time constraints /  1-indicator is 0 to stop
    indicator = u[i, t-1] - u[i, t]
    range = t:min(T, t + lasttime[i] - 1)
    for r in range
        @constraint(model, u[i, r] <= 1-indicator)
    end
end


for i in 1:gennum                                                                                       # Startup-Shutdown cost constraints
    for t in 2:T
        @constraint(model, costH[i, t] >= H[i] * (u[i, t] - u[i, t-1]))                                 # Startup cost constraints
        @constraint(model, costJ[i, t] >= J[i] * (u[i, t-1] - u[i, t]))                                 # Shutdown cost constraints
    end
    @constraint(model, costH[i, 1] >= H[i] * (u[i, 1] - u0[i]))                                         # Startup cost in the initial state (first hour: 01:00)
    @constraint(model, costJ[i, 1] >= J[i] * (u0[i] - u[i, 1]))                                         # Shutdown cost in the initial state (first hour: 01:00)
end


#-----------DC Power Flow Constraints-----------
netpara[:, 4] .= 1 ./ netpara[:, 4]                                                                     # The inverse of reactance is susceptance
Y = zeros(numnodes, numnodes)                                                                           # Initialize the admittance matrix

for k in 1:branch_num                                                                                   #  Calculate the admittance matrix
    i = netpara[k, 2]                                                                                   # Start bus
    j = netpara[k, 3]                                                                                   # End bus
    Y[i, j] = -netpara[k, 4]                                                                            # Off-diagonal elements in the admittance matrix
    Y[j, i] = Y[i, j]                                                                                   # Symmetry
end

for k in 1:numnodes
    Y[k, k] = -sum(Y[k, :])                                                                             # Diagonal elements in the admittance matrix
end

                                                                                   
Y = vcat(Y[1:slack_bus-1, :], Y[slack_bus+1:end, :])                                                    # Delete the row where the slack_bus is located
Y = hcat(Y[:, 1:slack_bus-1], Y[:, slack_bus+1:end])                                                    # Delete the column where the slack_bus is located


X = inv(Y)                                                                                              # X is the inverse matrix of the admittance matrix

# The matrix value of the slack_bus is introduced again. 
# According to the DC power flow definition,    ΔΘ=ΧΔP,     the slack_bus angle is always 0, 
# so all Xs involving the slack_bus are 0.

row = zeros(1,numnodes - 1)                                                                             # Insert row with all 0
X = vcat(X[1:slack_bus-1, :], row, X[slack_bus:numnodes-1, :])                                          # Insert row with all 0
column = zeros(numnodes,1)                                                                              # Insert column with all 0
X = hcat(X[:, 1:slack_bus-1], column, X[:, slack_bus:numnodes-1])                                       # Insert column with all 0


# Power Transfer Distribution Factor (PTDF) Calculation 
G = zeros(branch_num, numnodes)                                                                         # PTDF power transfer matrix initialization
for k in 1:branch_num
    m = netpara[k, 2]                                                                                   # Start bus
    n = netpara[k, 3]                                                                                   # End bus
    xk = netpara[k, 4]                                                                                  # Reactance value of branch k
    for i in 1:numnodes
        G[k, i] = (X[m, i] - X[n, i]) * xk                                                              # PTDF
    end
end

power_gen = paragen[:, 2]                                                                               # Generators corresponding buses
sum_nodePTDF = zeros(T, branch_num)                                                                     # Output power transfer from load buses


for t in 1:T                                                                                            # Calculate Power Transfer 
    for k in 1:branch_num
        for i in 1:gennum
            @constraint(model, sum_PowerPTDF[t, k, i] == G[k, power_gen[i, 1]] * p[i, t])               # Generator output power (total) transfer to the line
        end
        for i in 1:numnodes
            sum_nodePTDF[t, k] = sum_nodePTDF[t, k] + G[k, i] * loadcurve[i, t+1]                       # Power transfer from load buses to line
        end
        @constraint(model, PL_min[k, 1] <= (sum(sum_PowerPTDF[t, k, :]) - sum_nodePTDF[t, k]))          # Branch capacity constraints for min
        @constraint(model, (sum(sum_PowerPTDF[t, k, :]) - sum_nodePTDF[t, k]) <= PL_max[k, 1])          # Branch capacity constraints for max
    end
end


# Objective function: coal consumption cost + startup cost + shutdown cost
@objective(model, Min,  
sum(cost_coff[i, 1] * p[i, t]^2+cost_coff[i, 2] * p[i, t] + cost_coff[i, 3] * u[i, t] + costH[i, t] + costJ[i, t] for i = 1:gennum, t = 1:T))   



#----------------------------------Model Solving---------------------------------- 
# Run!
# set_attribute(model, "display/verblevel", 0)
# set_attribute(model, "limits/gap", 0.0289)
# set_time_limit_sec(model, 700.0)
optimize!(model)



#----------------------------------Print Result---------------------------------- 
# Print result
p = JuMP.value.(p)
u = JuMP.value.(u)
