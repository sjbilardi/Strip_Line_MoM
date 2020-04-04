"""
    Method of Moments (MoM) Analysis of Strip Lines
                        ECEN 5154
                       Homework  3

Code developed by Sergei Bilardi (2020-04-03)

Methods based off:

- Title:  Analytical and Computational Methods in Electromagnetics
- Author: Ramesh Garg
- Year:   2008

**Things I noticed:**

- I recommend performing a numerical integration of
  Eq. 11.84 rather than using the analytical result of
  the integral in Eq. 11.85. I've noticed that Eq. 11.85
  did not contain the edge effects that are shown in Fig. 11.7.

- Also, note that the values in `α` are the charge densities
  for each segment on the strip. You should multiply each
  value of `α` by `hₓ` to conver to total charge and sum
  them up to get the total charge of the strip. Not doing
  this caused me to obtain a value of 2.39 Ohms for Z₀
  instead of 47.88 Ohms which is much closer to the analytical
  value of 48.51 Ohms, calculated using Eq. 11.91. Therefore,
  Eq. 11.88:

                          Q = ∑α             (11.88)

  should become:

                         Q = hₓ∑α
"""
function homework_info()
    println("Sergei Bilardi; ECEN 5154; Homework 3; Code in Julia")
end


using PyPlot
using LinearAlgebra
using DifferentialEquations
import PhysicalConstants.CODATA2018: ε_0, μ_0, c_0
pygui(true)


"""
    G!(du, u, parms, x′)

Part of the Green's function to integrate in
`l()` below.
"""
function G!(du, u, parms, x′)
    n = parms[1]
    b = parms[2]
    xₚ = parms[3]
    du[1] = exp(-n*π*abs(xₚ-x′)/b)
end


"""
    l(xp, xm, N)

The value for the (p, m) element of the matrix `L`
with `N` terms.
"""
function l(xₚ, xₘ, N, b, hₓ, εᵣ)
    lpm = 0.
    for n in 1:N
        if n%2 > 0
            parms = [n, b, xₚ]
            prob = ODEProblem(G!, [0.], (xₘ-hₓ/2, xₘ+hₓ/2), parms)
            sol = solve(prob, Tsit5(); adaptive=false, dt=hₓ/5)
            lpm += sol[end][1]/(n*π*ε_0.val*εᵣ)
        end
    end
    return lpm
end


"""
    calcxₘ(m)

Calculate where on the x-axis `xₘ` lies given `m`.
"""
function calcxₘ(m, w, hₓ)
    return -w/2 + hₓ*(m-1/2)
end


"""
    calcl(M)

Calculate the `L` matrix for the relationship

[L][α] = [1]

for the strip line problem (11.2.3) in Ch. 11 of the book
`Analytical and Computational Methods in Electromagnetics`
by `Ramesh Garg` (2008).
"""
function calcl(N, w, b, εᵣ)
    hₓ = w/N
    L = zeros(N,N)
    x = zeros(N)
    for p in 1:N
        xₚ = calcxₘ(p, w, hₓ)
        x[p] = xₚ
        for m in 1:N
            xₘ = calcxₘ(m, w, hₓ)
            L[p,m] = l(xₚ, xₘ, N, b, hₓ, εᵣ)
        end
    end
    return (L, x)
end


"""
    calc_K_over_K′(w, b)

Calculate the ratio of K and K′ for use when calculating
the theoretical Z₀ of the strip line.
"""
function calc_K_over_K′(w_over_b)
    k = sech(π*w_over_b/2)
    if (k >= 1/sqrt(2)) && (k <= 1)
        return log(2*(1+sqrt(k))/(1-sqrt(k)))/π
    elseif (k >= 0) && (k <= 1/sqrt(2))
        k′ = tanh(π*w_over_b/2)
        return π/log(2*(1+sqrt(k′))/(1-sqrt(k′)))
    else
        error("k not within required boundaries.")
    end
end


# Problem 11.2.3
# Use the same parameters as used in
# the problem:
#
#   w/b = 1.5
#   εᵣ = 1
#   V₀ = 1 Volt
#   N = 30 segments
#
# Define problem parameters
V₀ = 1  # Volt
w_over_b = 1.5
b = 1
w = w_over_b*b
εᵣ = 1
N = 30
# Calculate segment size
hₓ = w/N
# Construct the NxN matrix `L` and
# generate the vector containing the
# positions in the x-direction of
# each segment on the strip for which
# the coefficients (charge distributions)
# in the vector `α` correspond to.
L, x = calcl(N, w, b, εᵣ)
# Generate the vector containing the
# potentials on each section of the
# strip. The potentials on the strip
# are the same, `V₀`, or 1 Volt.
A = V₀*ones(N)
# Solve for the coefficients, α, also
# known as the charge distribution on
# each segment on the strip.
α = L\A
# Obtain the total charge of the strip
Q = sum(α)*hₓ
# Obtain the capacitance per unit length
# of the strip. Since `V₀` is 1 Volt,
# C₀ = Q.
C₀ = Q/V₀
# Calculate the phase velocity in the
# transmission line. v = c if the strip
# is in a vacuum, where εᵣ = 1.
v = 1/sqrt(μ_0.val*ε_0.val*εᵣ)
# Calculate the characteristic impedance,
# Z₀, of the strip.
Ẑ₀ = 1/(C₀*v)
# Calculate the analytical characteristic
# impendance of the strip using Eq. 11.91
# from the book.
#
#      29.976π  K(k)
# Z₀ = ------- -----
#        √εᵣ   K′(k)
#
# where K(k)/K′(k) is defined in Eq. 4.91
# in the book. Refer to the function
# `calc_K_over_K′ above.
K_over_K′ = calc_K_over_K′(w_over_b)
Z₀ = 29.976*π*K_over_K′/sqrt(εᵣ)
# Generate a figure similar to the one shown
# in Fig. 11.7 in Ch. 11 of the book.
fig = figure(figsize=(5,5))
scale = 11
α_scaled = α.*(10^scale)
plot(x, α_scaled, "k:")
plot(x, α_scaled, "k.")
xlabel("x/b")
ylabel("Charge Density (1×10^$(-scale))")
xlim([floor(x[1], digits=1), ceil(x[end], digits=1)])
ylim([floor(minimum(α_scaled), digits=0)-1,
      ceil(maximum(α_scaled), digits=0)])
messg = string("Strip Line MoM Analysis\n",
               "W/b = $(w_over_b), εᵣ = $(εᵣ), ",
               "V₀ = $(V₀) Volt, N = $(N), M = $(N)\n",
               "Estimated Z₀ = $(round(Ẑ₀, digits=2)) ",
               "Ohm\nAnalytical Z₀ = $(round(Z₀, digits=2)) Ohm")
title(messg)
fig.tight_layout()
