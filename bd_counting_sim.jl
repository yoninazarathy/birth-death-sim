using Random, Distributions

function bd_count( λ_f = (x) -> 1.0, 
                   μ_f = (x) -> x>0 ? 1.0*(1+1.5/(x+1)) : 0.0, 
                   q_minus_f = (x) -> x>0 ? 0.0 : 0.0,
                   q_plus_f = (x) -> 1.0, 
                ; 
                max_T = 10^5, 
                x0 = 0)
    t = 0.0
    x = x0
    reward = 0.0

    while t ≤ max_T
        λ, μ = λ_f(x), μ_f(x)
        r, p = λ + μ, λ/(λ + μ)
        s = rand(Exponential(1/r))
        t += s
        if rand() < p
            reward += rand() < q_plus_f(x)
            x+=1
        else
            reward += rand() < q_minus_f(x)
            x-=1
        end
    end

    return reward
end

# println("M/M/1")
# data = [bd_count() for _ in 1:10^4]
# @show var(data)/mean(data)

data = [bd_count() for _ in 1:10^4]
@show var(data)/mean(data)
