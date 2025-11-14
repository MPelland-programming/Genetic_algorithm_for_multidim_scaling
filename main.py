import numpy as np

#On genere les nombres aleatoire
seed = 6112025

rng = np.random.default_rng(seed=seed)



# Paramètres du problème
n = 40 # nombre d'observations
K = 4 # dimension de l'espace de représentation
pop_size = 100 # taille de la population

p_parent = 0.20 #proportion of population to be selected as parents
p_good = 0.8#proportion of good parents from the selected parents

n_generations = 100          # nombre de générations de l'AG
mutation_rate = 0.66         # probabilité de muter une coordonnée
mutation_scale = 0.1        # amplitude de la mutation



def generate_dissimilarity_matrix(n,rng):
  """
  This function will generate a dissimilarity matrix n x n to mimic the expected
  input for the algorithm.
  When creating a python class, add a test flag that will turn this on.
  Returns: n x n array of randomly generate numbers between 0 and 1.
  """
  return rng.random(size=(n,n))


#Function to vetorize a matrix

def get_upper_triangle(mat):
  """
  Takes a matrix and extracts the upper triangle
  Retunrs: 1D array of lenght n(n-1)
  """
  n = mat.shape[1]
  vec = np.empty(n*(n-1)//2)

  cc = 0

  for ii in range(0,n):
    for jj in range(ii+1,n):
      vec[cc] = mat[ii,jj]
      cc += 1

  return vec

def generate_initial_population(pop_size, n, K, rng):
    """
    Génère pop_size solutions, chacune étant une matrice (n, K)
    de coordonnées.
    """
    population = []
    for _ in range(pop_size):
        coords = rng.uniform(size=(n, K))
        population.append(coords)
    return population



def compute_distance_vec(indiv):
  """
  Given a set of coodinates, find the euclidian distance between all points.
  input: matrix of size  n_observations x k dims
  """
  n,K = indiv.shape

  dist = np.empty(n*(n-1)//2)
  cc = 0

  for ii in range(n):       #iterate through observation
    for jj in range(ii+1,n):  #iterate through observations to compare to.
      sum = 0

      for dd in range(K):
        sum += (indiv[ii,dd]-indiv[jj,dd])**2

      dist[cc]= np.sqrt(sum)
      cc += 1
  return dist


def objective_function(indiv, dissimilarity,power = 2):
    """
    calcule Z = sum_{i<j} (delta_ij - d_ij)^2
    inputs: indiv: an individuals with coordinates
    """
    n = dissimilarity.shape[-1]

    distance = compute_distance_vec(indiv) #On calcule le vecteur des distances
    Z = 0.0

    for ii in range(n):
      Z += (dissimilarity[ii]-distance[ii])**power

    return Z  #La fonction renvoie la valeur totale de l’erreur Z.

def evaluate_sample(sample, dissimilarity):
    """
    évalue chaque solution de l'echantillon.
    Elle applique la fonction objective_function()
    à toutes les solutions de la population.
    """
    return [objective_function(indiv, dissimilarity) for indiv in sample]


def argsort_Z(Z):
  """
  prend un liste de valeurs z et retourne les indices pour obtenir les valeurs en ordre croissant.
  """
  return np.argsort(Z)  #####################################################################################Changer pour un tri a bulles maison si possible

def indiv_selection(sorted_idx,n_parent,n_good,rng):
  """
  inputs
    sorted_idx : list of indices of solution sorted in worsening order.
    n_parent: integer specifying the number of parents to take
    n_good: integer specifying the number of good parents among the parents.
  outputs: list of pairs of indices for the parents, in random order, to pass
  to the crossover step
  """
  pop_size = len(sorted_idx)
  all_list = list(range(n_good))               #idx of top of the list (best of).
  bad_list = list(range(n_good,pop_size))
  rng.shuffle(bad_list)                         #idx of others (not necessarily top)

  pos_list = all_list[0:n_good] + bad_list[0:n_parent-n_good]  #put top idx and other together
  rng.shuffle(pos_list)                                         #randomly shuffle the order of idx

  parent_list = sorted_idx[pos_list]

  return parent_list


def crossover_half_split_columns(parent1, parent2):
    """
    Croisement déterministe : la première moitié des coordonnées (axes)
    vient du parent 1, l'autre moitié du parent 2. comme K=3 p1=1 et p2=2
    """
    n, K = parent1.shape #On récupère les dimensions de la matrice parent1
    child = np.empty((n, K))  # matrice vide (non initialisée) de même taille que les parents

    split_point = K // 2 #point de coupure entre les coordonnées provenant du P1 et du P2.
    child[:, :split_point] = parent1[:, :split_point]  #copie les coordonnées de la première moitié des axes (colonnes) du parent 1 vers l’enfant
    child[:, split_point:] = parent2[:, split_point:]

    return child.copy()

def mutate_by_columns(indiv, mutation_rate, mutation_scale,rng):
    """
    Mutation structurée : ajoute du bruit seulement sur certaines colonnes (axes).
    """
    n, K = indiv.shape
    #mutated = indiv.copy()######################################################################### I removed because a copy is already made in generation


    for j in range(0, K):
      if rng.random() < mutation_rate:
        noise = rng.normal(loc=0.0, scale=mutation_scale, size=n)
        indiv[:, j] += noise

    return indiv

def generate_children(population
                      , parent_list
                      , mutation_rate
                      , mutation_scale
                      ,rng):

  children_list = []

  for ii in range(0,len(parent_list),2):
    parent1 =  population[parent_list[ii]]
    parent2 = population[parent_list[ii+1]]
    tempchild = crossover_half_split_columns(parent1, parent2)
    tempchild = mutate_by_columns(tempchild, mutation_rate, mutation_scale, rng)

    children_list.append(tempchild)

  return children_list

def create_next_generation(population,
                           Z_list,
                           dissimilarity,
                           mutation_rate=0.1,
                           mutation_scale=0.1,
                           rng="none"):

    pop_size = len(population)

    # trier par score croissant
    sorted_idx = argsort_Z(Z_list)


    # selection des parents
    parent_list = indiv_selection(sorted_idx,n_parent,n_good,rng)

    # creation des enfants avec mutations
    children_list = generate_children(population, parent_list, mutation_rate, mutation_scale,rng)

    # Remplacement: selection semi aleatoire des parents a enlever pour faire
    # place aux enfants.
    #death_list = indiv_selection(sorted_idx[::-1],n_parent//2,n_good//2,rng)     #on prend l'envers de la liste pour avoir du pire au meilleur. ################ utiliser le sort maison
    death_list = indiv_selection(sorted_idx[::-1], n_parent//2, n_good//2, rng)

    for ii in sorted(death_list, reverse = True): ################################################################################################################ utiliser le sort maison.
      population.pop(ii)

    for ii in children_list:
      population.append(ii)

    return population


# 1. Génération de la matrice de dissimilarités
dissimilarity_mat = generate_dissimilarity_matrix(n, rng)
dissimilarity = get_upper_triangle(dissimilarity_mat)

# 2. Population initiale
population = generate_initial_population(pop_size, n, K, rng)

best_scores = []  # pour suivre la convergence

for gen in range(n_generations):
    # 3. Évaluation de la population actuelle
    Z_list = evaluate_sample(population, dissimilarity)

    # 4. Sauvegarde du meilleur score de cette génération
    best_Z = min(Z_list)
    best_scores.append(best_Z)

    # (optionnel) afficher la progression
    print(f"Génération {gen:03d} - meilleur Z = {best_Z:.4f}")

    # 5. Création de la nouvelle population
    population = create_next_generation(
                           population,
                           Z_list,
                           dissimilarity,
                           mutation_rate=0.1,
                           mutation_scale=0.1,
                           rng=rng)

# 6. Évaluation finale pour récupérer la meilleure solution
final_scores = evaluate_sample(population, dissimilarity)
best_idx = int(np.argmin(final_scores))
best_solution = population[best_idx]
best_Z = final_scores[best_idx]

# print("Meilleure valeur de Z trouvée :", best_Z)
# print("Forme des coordonnées :", best_solution.shape)