#parameters this should be transferred to the init part of a python class
n = 100 #number of observations
n_best = 30 #number of best solutions to keep
p_parent = 0.13 #proportion of population to be selected as parents
p_good = 0.8#proportion of good parents from the selected parents

n_parent = n_best*p_parent//1 #number of parents to select
n_good = n_parent*p_good//1   #number of good parents to select
n_bad = -1*(n_parent*(1-p_good)//-1)  #number of bad parents to select

def generate_initial_population(n_best,n,k=2):
  """
  Generate n_best individuals
  Returns: a numpy array of shape pop. size x k dim x n observations
  """
  popu = []
  popu = rng.random(size=(n_best,k,n))

  return popu


