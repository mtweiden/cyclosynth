import cyclosynth
from timeit import default_timer as timer

epsilon = 1e-8
start_time = timer()
r = cyclosynth.synthesize_u3(1.0472, 2.7577, 5.3947, epsilon, sqrt_t=True)
duration = timer() - start_time

if r:                    # None if nothing was found within epsilon
    print(r.gates)       # gate string, e.g. "HTSHt..." — lowercase = dagger (t = T†)
    print(r.t_count)     # also .q_count, .cost, .lde
    print(r.distance)    # diamond distance to the target, < epsilon
    print(duration)
