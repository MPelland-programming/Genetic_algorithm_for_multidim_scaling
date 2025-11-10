import numpy as np
#parameters this should be transferred to the init part of a python class
n = 10 #number of observations
n_best = 30 #number of best solutions to keep
p_parent = 0.13 #proportion of population to be selected as parents
p_good = 0.8#proportion of good parents from the selected parents

n_parent = n_best*p_parent//1 #number of parents to select
n_good = n_parent*p_good//1   #number of good parents to select
n_bad = -1*(n_parent*(1-p_good)//-1)  #number of bad parents to select

#Create random number generator
seed = 6112025

rng = np.random.default_rng(seed=seed)


def gen_diss_mat(n,rng):
  """
  This function will generate a dissimilarity matrix n x n to mimic the expected
  input for the algorithm.
  When creating a python class, add a test flag that will turn this on.
  Returns: n x n array of randomly generate numbers between 0 and 1.
  """
  return rng.random(size=(n,n))

def get_upper_tri(mat):
  """
  Takes the dissimilarity matrix and extracts the upper triangle
  Retunrs: 1D array of lenght n(n-1)
  """
  siz = mat.shape[1]
  vec = np.empty(siz*(siz-1)//2)

  cc = 0

  for ii in range(0,siz):
    for jj in range(ii+1,siz):
      vec[cc] = mat[ii,jj]
      cc += 1

  return vec

def gen_initial_population(n_best,n,k=2):
  """
  Generate n_best individuals
  Returns: a numpy array of shape pop. size x k dim x n observations
  """
  return rng.random(size=(n_best,k,n))


def calc_dist(coord):
  """
  Given a set of coodinates, find the euclidian distance between all points.
  input: matrix of size k_dim x n_observations
  """
  dim, nn = coord.shape
  dist = np.empty(nn * (nn - 1) // 2)
  cc = 0

  for ii in range(nn):  # iterate through observation
    for jj in range(ii + 1, nn):  # iterate through observations to compare to.
      sum = 0

      for kk in range(dim):
        sum += (coord[kk, ii] - coord[kk, jj]) ** 2

      dist[cc] = np.sqrt(sum)
      cc += 1
  return dist

def calc_error(dissim,distance):
  """
  calculate differences between the distances and the dissimilarity matrix
  input: two arrays of size n(n-1)/2
  """
  nn = dissim.shape[-1]
  error = 0
  for ii in range(nn):
    error += abs(dissim[ii]-distance[ii])

  return error

def evaluate_ind(diss, ind_mat):
  """
  Evaluate individuals.
  inputs:
      ind_mat: array of shape number of individuals to eval x k dim x n observations
      diss: an array of length n*(n-1)/2 of dissimilarity

  return: an array of shape n_ind*n_ind-1/2 with the eval results
  """
  nind, kk, nn = ind_mat.shape
  error = np.empty(nind)
  dist = np.empty(nn * (nn - 1) // 2)

  cc = 0
  for hh in range(nind):  # iterate throug individuals to eval
    dist = calc_dist(np.squeeze(ind_mat[hh, :, :]))
    error[cc] = calc_error(diss, dist)
    cc += 1

  return error

def selection(qual_idx,n_parent,n_good):
  """
  inputs
    qual_idx : vector of indices of solution sorted in worsening order.
    n_parent: integer specifying the number of parents to take
    n_good: integer specifying the number of good parents among the parents.
  outputs: list of indices for the parents to pass to the crossover step
            in random order.
  """
  le = qual_idx.shape[0]
  all_list = list(range(n_good)) #############################################fix this with a loop?
  bad_list = list(range(n_good,le))
  np.random.shuffle(bad_list)

  return all_list[0:n_good] + bad_list[0:n_parent-n_good]

# Main algorithm
diss_mat = gen_diss_mat(n,rng)#add test flag when creating class
diss_vec = get_upper_tri(diss_mat) # shortens the dissimilarity matrix by getting its upper triangle vector.

initpop = gen_initial_population(n_best,n)

ind_error = evaluate_ind(diss_vec,initpop)