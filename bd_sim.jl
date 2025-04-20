using Random, Distributions

function bd_rv( λ_f = (x) -> 1.0, 
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

# bd_rv()
mean([bd_rv()^2 for _ in 1:10^7])