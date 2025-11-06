def generate_initial_population(n_best,n,k=2):
  """
  Generate n_best individuals
  Returns: list of individuals, each
  """
  popu = []
  for i in range(n_best):
    tind = rng.random(size=(k,n)).tolist()
    popu.append(tind)

  return popu
