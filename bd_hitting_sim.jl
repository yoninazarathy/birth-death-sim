using Random, Distributions

function bd_rv( λ_f = (x) -> 1.5, 
                μ_f = (x) -> x>0 ? 2.0 : 0.0, 
                stop_crit = (x_prev, x) -> (x_prev == 0 && x == 1),
                reward_f = (s, x) -> s, 
                ; 
                max_T = Inf, 
                x0 = 1)
    t = 0.0
    x_prev = -1 # non-existing state 
    x = x0
    reward = 0.0

    while !stop_crit(x_prev, x)
        λ, μ = λ_f(x), μ_f(x)
        r, p = λ + μ, λ/(λ + μ)
        s = rand(Exponential(1/r))
        reward += reward_f(s, x)
        t += s
        if t > max_T
            @info "Passed max time"
            return reward
        end
        x_prev, x = x, ((rand() < p) ? x+1 : x-1)
    end

    return reward
end

function pi_prob_mm1(k; λ = 1.5, μ = 2)
    ρ = λ/μ
    return (1-ρ)*ρ^k
end

function pi_prob_cum_mm1(k; λ = 1.5, μ = 2) 
    return sum(pi_prob_mm1(i) for i in 0:k)
end

function pi_prob_mm∞(k; λ = 1.5, μ = 2)
    ρ = λ/μ
    return pdf(Poisson(ρ), k)
end

function pi_prob_cum_mm∞(k; λ = 1.5, μ = 2) 
    return sum(pi_prob_mm∞(i) for i in 0:k)
end

function dd_m2(pi_prob, pi_prob_cum, λ₀)
    return (2/(pi_prob(0)*λ₀))*(1/λ₀ + sum([(1-pi_prob_cum(k))^2/(pi_prob(k) * λ₀) for k in 0:100]))
end


println("M/M/1")
monte_carlo_x2 = mean([bd_rv()^2 for _ in 1:10^6])
dd_x2 =  dd_m2(pi_prob_mm1, pi_prob_cum_mm1, 1.5)

@show monte_carlo_x2
@show dd_x2;

println("M/M/∞")
monte_carlo_x2 = mean([bd_rv((x) -> 1.5,  (x) -> x>0 ? x*2.0 : 0.0)^2 for _ in 1:10^6])
dd_x2 =  dd_m2(pi_prob_mm∞, pi_prob_cum_mm∞, 1.5)

@show monte_carlo_x2
@show dd_x2;
