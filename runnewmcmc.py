from newmcmc import *

start_time = time.time()
nbmc = NbMC(0.0001, 5, 1, "test.csv","res", "./")
nbmc.set_prior_params(1, 0.0001, 1, 0.0001)
out = "./result"
nbmc.run_model(200, 100, 1, True)
nbmc.model_comp(200, 100, 1)
end_time = time.time() - start_time
print end_time
