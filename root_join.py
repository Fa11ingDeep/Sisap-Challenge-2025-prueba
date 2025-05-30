import numpy as np
import math
import h5py
import csv
import os
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import heapq
from multiprocessing import Pool, freeze_support, Lock
import pickle
from pathlib import Path
import pandas as pd
from datasets import DATASETS, prepare, get_fn
import argparse
import shutil
import ast

def cos_sim(v1, v2):
    """
    Calculates the cosine similarity between two normalized vectors v1 and v2.

    Cosine similarity measures the similarity between two vectors in a feature space
    by comparing the angle between them. The returned value ranges from -1 (completely
    opposite) to 1 (identical), with 0 indicating no similarity.

    Note:
    Both v1 and v2 should be normalized to unit length (i.e., their magnitude should be 1)
    before calling this function.

    Parameters:
    v1 (array-like): First normalized vector.
    v2 (array-like): Second normalized vector.

    Returns:
    float: The cosine similarity between v1 and v2.
    """
    return  1-(np.dot(v1,v2))


def getCenters(data, c):
  """
    Selects a subset of centers randomly from the dataset.

    This function takes a dataset and a parameter c, then randomly selects
    approximately c * sqrt(n) indices as centers, where n is the number of vectors
    in the data. It returns the total number of vectors, the data points not chosen
    as centers, and the chosen centers as (index, vector) tuples.

    Parameters:
    data (numpy.ndarray): The dataset, expected to be a 2D array with shape (n, d),
                          where n is the number of vectors and d is their dimensionality.
    c (float): Scaling factor for the number of centers, used in the formula c * sqrt(n).

    Returns:
    tuple:
        n (int): Number of vectors in the dataset.
        newData (list of tuples): Data points not selected as centers, each as (index, vector).
        centers (list of tuples): Selected centers, each as (index, vector).
    """
  n, d = data.shape  # Get the amount of vectors from the datas.
  centers = []  # List to store chosen centers.
  newData = []  # List to store data points not chosen as centers.
  idx = np.random.choice(n, size=math.floor(c * math.sqrt(n)), replace=False)  # choose randomly c*sqrt(n) indices.
  # Separate data points into centers and newData.
  for i, vector in enumerate(data):
      point = (int(i), vector)
      if i in idx:
           centers.append(point)
      else:
           newData.append(point)
  return n, newData, centers


def makeGroups(n, data, centers, metric_fn, c, size):
    """
    Groups `n` data points into clusters based on proximity to initial centers, 
    with optional group size extension and radius tracking.

    Parameters
    ----------
    n : int
        Total number of data points to group.
    data : list of tuple
        List of tuples (id, vector) representing data points.
    centers : list of tuple
        List of initial centers, each a tuple (id, vector).
    metric_fn : callable
        Function used to calculate the distance between vectors.
    c : float
        Proportion constant used to determine initial group capacity.
    size : int
        Reference size for computing the maximum group size.

    Returns
    -------
    groups : dict
        Dictionary where each key is a group ID (int) and each value is a tuple:
        ([((id, vector), [nearest_centers])], radius, farthest_point)
        - The list contains the group points and the two closest neighboring centers.
        - `radius` is the greatest distance from the center to any point in the group.
        - `farthest_point` is the point at that maximum distance (or None if not set).

    Notes
    -----
    - The function assigns each point to the nearest available group center,
      ensuring that no group exceeds the maximum allowed size.
    - If necessary, group sizes are extended to accommodate all points.
    - In some cases, points may be swapped to maintain balance and radius constraints.
    - Each group retains a list of the two nearest other centers for future reference.
    """

    maxSize = math.floor(c * size) # Maximum size of a group.
    groups = {} # Dictionary to store the groups.

    # Calculate the next closest centers for each selected center.
    for i in range(len(centers)):
        center_dist = []
        for j in range(len(centers)):
            if i != j:
                center_dist.append((j, metric_fn(centers[i][1], centers[j][1])))
        center_dist.sort(key=lambda x: x[1])  # Sort according the distances.

        next_centers = [c for c, _ in heapq.nsmallest(2, center_dist, key=lambda x: x[1])] # Use a heap, because we only need the 2 closest centers.
        groups[i] = ([((centers[i]),next_centers)], -1, None)  # ([((id,vector), prox_centers)], radius, furtherst_point) The second element (-1) is the radius.
    # Add the points to the groups
    for h in range((maxSize - 1) * len(centers)):
        id_point = data[h]
        dists = []
        for group_id, values in groups.items():  # In points, the first point is always the center of the group.
            datas = values[0][0]
            idx, point = id_point
            id_points= datas[0]
            points = id_points[1]
            dists.append((metric_fn(point, points), group_id, idx))  # Append the distances with the group_id.

        dists.sort(key=lambda x: x[0])  # Sort the distances.
        k = 0
        while True:
            dist1, group_id1, id = dists[k]
            if len(groups[group_id1][0]) < maxSize:  # If the group size is less than the maximum size.
                dists.remove(dists[k])  # Remove the point.
                next_group_list = []  # List of centers.
                for l in range(2):  # Get the centers and add the first two centers to the list.
                    dist_temp, group_temp, id = dists[l]
                    next_group_list.append(group_temp)
                groups[group_id1][0].append(((id, point),next_group_list))  # Append the new point with the new data.
                if dist1 > groups[group_id1][1]:  # If the new distance is greater than the current radius.
                    groups[group_id1]=(groups[group_id1][0],dist1,(id,point))  # Update the radius and the farthest point.
                break 
            else:  # If the group is full.
                k += 1  # Move to the next closest group.
        

    # Expand the grups, if required.
    x = math.ceil((n - maxSize * len(centers)) / len(centers))
    
    if x != 0:
        extended_size_group = maxSize + x
        extended_point_without_groups = len(centers) * extended_size_group
        assert extended_point_without_groups >= n
        o = ((maxSize - 1) * len(centers))  # Index of the first point that was not inserted.
        newPending = []
        while o < len(data):
            id_point_extended = data[o]
            dists_extended = []
            for group_id, values in groups.items():  # In points, the first point is always the center of the group.
                datas = values[0][0]
                idx, point = id_point_extended
                id_points= datas[0]
                points = id_points[1]
                dists_extended.append((metric_fn(point, points), group_id, idx))  # Append the distances with the group_id.

            dists_extended.sort(key=lambda x: x[0])
            p = 0
            while True:
                dist1, group_id1, id = dists_extended[p]
                
                if len(groups[group_id1][0]) < extended_size_group:  # If the group size is less than the maximum size.
                    dists_extended.remove(dists_extended[p])  # Remove the point.
                    next_group_list = []  # List of centers.
                    for u in range(2):  # Get the centers and add the first two centers to the list.
                        dist_temp, group_temp, id = dists_extended[u]
                        next_group_list.append(group_temp)
                    groups[group_id1][0].append(((id, point),next_group_list)) # Append the new point with the new data.
                    if dist1 > groups[group_id1][1]:  # If the new distance is greater than the current radius.
                        groups[group_id1]=(groups[group_id1][0],dist1,(id,point))   # Update the radius and the farthest point.
                    o += 1
                    break 
                else:  # If the group is full.
                    if(dist1<groups[group_id1][1]): # If the distance id less than the radius, insert the new point anyway.
                        if(groups[group_id1][2]!=None):
                            newPending.append(groups[group_id1][2])# Add the farthest point as pending.
                            groups[group_id1] = ([item for item in groups[group_id1][0] if item[0][0] != groups[group_id1][2][0]],groups[group_id1][1],groups[group_id1][2])
                        dists_extended.remove(dists_extended[p])
                        next_group_list=[] # List of centers.
                        for p in range(2): # Get the centers and add the first two centers to the list.
                            dist_temp,group_temp,id=dists_extended[p]
                            next_group_list.append(group_temp)
                        groups[group_id1][0].append(((id, point),next_group_list))   # Append the new point with the new data.
                        groups[group_id1]=(groups[group_id1][0],-10000,None) # Reset radius to a negative number and farthest point to None, since we can swap points only once.
                        o+=1
                        break 
                    else:# If the group is full.
                        p+=1 # Move to the next closest group.

        # Processing pending points.
        for id_point in newPending:
            dists_pend = []
            
            for group_id, values in groups.items():  # In points, the first point is always the center of the group.
                datas = values[0][0]
                idx, point = id_point
                id_points = datas[0]
                points = id_points[1]
                dists_pend.append((metric_fn(point, points), group_id, idx))   # Append the distances with the group_id.

            dists_pend.sort(key=lambda x: x[0])
            l = 0
            while True:
                dist1, group_id1, id = dists_pend[l]
                
                if len(groups[group_id1][0]) < extended_size_group:  # If the group size is less than the extended maximum size.
                    dists_pend.remove(dists_pend[l])  # Remove the point.
                    next_group_list = []  # List of centers.
                    for u in range(2):  # Get the centers and add the first two centers to the list.
                        dist_temp, group_temp, id = dists_pend[u]
                        next_group_list.append(group_temp)
                    groups[group_id1][0].append(((id, point),next_group_list))   # Append the new point with the new data.
                    if (dist1 > groups[group_id1][1] and (groups[group_id1][1] >-10000)):  # If the new distance exceeds the current radius and a swap hasn't occurred.
                        groups[group_id1]=(groups[group_id1][0],dist1,(id,point))  # Update the radius and the farthest point.
                    break 
                else:  # If the group is full.
                    l += 1  # Move to the next closest group.
    return groups





def get_knn(k,e,target,metric_fn):
    """
    Compute the k-nearest neighbors (k-NN) of a given element from a set of target elements using a custom distance function.

    Parameters:
        k (int): The number of nearest neighbors to return.
        e (np.ndarray): The query element for which the nearest neighbors are to be found.
        target (List[Tuple[int, np.ndarray]]): A list of tuples where each tuple contains an identifier (int)
                                               and a data point (np.ndarray).
        metric_fn (Callable[[np.ndarray, np.ndarray], float]): A function that computes the distance between two vectors.

    Returns:
        Tuple[List[int], List[float]]: 
            - A list of the identifiers of the k nearest neighbors of the element `e`, sorted by increasing distance.
            - A corresponding list of their distances.
    """
    temp=[] # Temporary array to store the distances.
    target=[item for item in target if not np.array_equal(item[1], e)] # Remove the element from the target.
    for element in target:
        id_element=element[0]
        point_element=element[1]
        dist=metric_fn(e,point_element) # For each element, compute the distance.
        temp.append((dist,id_element)) # Append the tuple (dist, id).
    k_nearest = heapq.nsmallest(k, temp, key=lambda x: x[0]) # Sort the distances.
    return  [int(x[1]) for x in k_nearest],[int(x[0]) for x in k_nearest] # Return the indices and distances of the k nearest elements.


def load_pickle_group(group_id, output_dir, lock):
    """
    Load a pickled group file from the specified directory in a thread/process-safe manner.

    This function attempts to load a pickle (.pkl) file corresponding to a specific group ID.
    It uses a synchronization lock to ensure that only one process or thread accesses the file
    at a time, which is important in concurrent or parallel environments.

    Parameters:
    ----------
    group_id : int
        The identifier of the group whose data should be loaded.
    output_dir : str
        The directory where group pickle files are stored.
    lock : multiprocessing.Lock or threading.Lock
        A synchronization lock used to control access to the file.

    Returns:
    -------
    Any
        The data loaded from the pickle file if successful.
        Returns None if the file does not exist, if there is a permission issue,
        or if another exception occurs during the loading process.
    """
    group_file_path = os.path.join(output_dir, f"group_{group_id}.pkl")

    # Check if the file exists
    if not os.path.exists(group_file_path):
        print(f"The file for group {group_id} does not exist at {group_file_path}.")
        return None

    try:
        # Use a Lock to ensure that only one process accesses the file at a time.
        with lock:
            with open(group_file_path, mode="rb") as grupo_file:
                grupo_data = pickle.load(grupo_file)
        return grupo_data
    except PermissionError:
        print(f"Permission denied when trying to read the file {group_file_path}.")
        return None
    except Exception as e:
        print(f"Error while loading the file {group_file_path}: {e}")
        return None


lock=Lock() # Global lock

def process_group_parallel(args):
    """
    Process a group of elements in parallel to compute k-nearest neighbors (k-NN)
    and write the results to a CSV file in batches.

    This function is intended to be run in parallel (e.g., with multiprocessing). It 
    computes the k-nearest neighbors for each element in a group using a provided 
    distance metric. It also extends the search to include points from the two closest 
    neighboring groups (based on precomputed proximity).

    Args:
        args (tuple): A tuple containing the following elements:
            - group_id (int): Identifier of the group being processed.
            - group (tuple): A tuple where the first element is a list of elements in the group.
              Each element is a tuple containing:
                - ((id, vector), [neighboring_group_ids])
            - k (int): Number of nearest neighbors to compute.
            - metric_fn (callable): Function that takes two vectors and returns a distance metric.
            - batch_size (int): Number of records to write at once to reduce I/O operations.
            - folder_path (str): Path to the directory where group pickle files are stored.
            - fname (str): Prefix for the output CSV filename.

    Returns:
        tuple: A tuple (group_id, duration), where:
            - group_id (int): Identifier of the processed group.
            - duration (float): Time taken to process the group in seconds.
    """
    group_id, group, k, metric_fn, batch_size, folder_path, fname= args # Unpack the arguments.
    result_batch = []
    print(f"Processing group {group_id}.")
    start = time.time()

    temp_output_file = f"{folder_path}/{fname}part{group_id}.csv"

    # Compute k-NN for each element in the group.
    for element in group[0]:
        target = [elem[0] for elem in group[0]]
        id_e = element[0][0]
        point_e = element[0][1]
        nearest_groups = [elem for elem in element[1]]

        for _ in range(2): ## Get points from the two nearest neighboring groups.
            if nearest_groups:
                next_g = nearest_groups.pop(0)
                target += [elem[0] for elem in load_pickle_group(next_g, folder_path,lock)[1]]

        # The following block is useful in the original root_join implementation:
        #while len(target) < k and nearest_groups:
        #    next_g = nearest_groups.pop(0)
        #    target += [elem[1] for elem in load_pickle_group(next_g, folder_path,lock)[1]]

        # Save the result in the temporary array.
        knns=get_knn(k, point_e, target, metric_fn)
        result_batch.append([int(id_e), knns[0], knns[1]])

        # Write results if the size of result_batch is greater than or equal to batch_size.
        if len(result_batch) >= batch_size:
            with open(temp_output_file, mode='a', newline='') as f_out:
                writer = csv.writer(f_out)
                writer.writerows(result_batch)
            result_batch.clear()
    # Write the remaining contents of result_batch, if any.
    if result_batch:
        with open(temp_output_file, mode='a', newline='') as f_out:
            writer = csv.writer(f_out)
            writer.writerows(result_batch)
        result_batch.clear()

    duration = time.time() - start
    print(f"Finished processing group {group_id}.")
    return group_id, duration



def save_pickle_group(groups, output_dir):
    """
    Saves each group of data as a separate .pkl file in the specified output directory.

    Each group is serialized using the pickle module and saved with a filename format
    of 'group_<group_id>.pkl'. The function ensures that the output directory exists,
    creating it if necessary.

    Parameters:
    ----------
    groups : dict
        A dictionary where each key is a group ID and each value is a tuple:
        (group_points, radius, furthest_point). Only group_points and group_id are saved.
    
    output_dir : str
        The path to the directory where the group files should be saved.

    Returns:
    -------
    None
    """
    # Create the directory if it does not exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # For each group, generate an individual .pkl file
    for group_id, (group_points, radius, furthest_point) in groups.items():
        group_file_path = os.path.join(output_dir, f"group_{group_id}.pkl")

        # Open the file in write-binary mode.
        with open(group_file_path, mode="wb") as grupo_file:
            # Serialize the tuple.
            pickle.dump((group_id,group_points), grupo_file)

        print(f"Archivo .pkl para el grupo {group_id} guardado en {group_file_path}")




def self_sim_join(data, c1, c2, k, metric_fn, folder_path, fname):
    """
    Perform a self-similarity join on a dataset using a clustering-based approach.

    This function partitions the dataset into groups based on proximity to selected centers.
    It then performs a k-nearest neighbor search in parallel within and across nearby groups,
    aggregating the results into a single CSV file.

    Parameters:
    -----------
    data : numpy.ndarray
        A 2D array where each row represents a data point.
    c1 : int or float
        A parameter that controls the number or selection strategy of cluster centers.
    c2 : int or float
        A parameter that determines the grouping radius or group density.
    k : int
        The number of nearest neighbors to retrieve for each data point.
    metric_fn : callable
        A function that computes the distance or similarity between two data points.
    folder_path : str
        The directory path where intermediate and final output files will be saved.
    fname : str
        The base name for the output files.

    Steps:
    ------
    1. Select cluster centers from the dataset using `getCenters`.
    2. Create groups of data points based on their proximity to the centers using `makeGroups`.
    3. Persist the grouped data to disk using `save_pickle_group`.
    4. Prepare tasks for multiprocessing and distribute them using `process_group_parallel` to perform
       local k-NN searches.
    5. Merge the partial k-NN results from all groups into a single CSV file.
    6. Clean up temporary group and result files after processing.
    """
    batch_size = 300000
    n, d = data.shape
    

    # Choose the centers
    inicio_gc = time.time()
    length, newData, centers = getCenters(data, c1)
    fin_gc = time.time()
    tiempo_ejecucion_gc = fin_gc - inicio_gc

    os.makedirs(folder_path, exist_ok=True)

    with open(f'{folder_path}/tiempo_gc.csv', mode='a', newline='') as file_gc:
        writer = csv.writer(file_gc)
        writer.writerow(['tiempo'])
        writer.writerow([tiempo_ejecucion_gc])

    # Create the groups
    inicio_mg = time.time()
    groups = makeGroups(length, newData, centers, metric_fn, c2, math.sqrt(n))
    save_pickle_group(groups,f'{folder_path}')
    fin_mg = time.time()
    tiempo_ejecucion_mg = fin_mg - inicio_mg
    

    with open(f'{folder_path}/tiempo_mg.csv', mode='a', newline='') as file_mg:
        writer = csv.writer(file_mg)
        writer.writerow(['tiempo'])
        writer.writerow([tiempo_ejecucion_mg])
    # Create shared objects between processes.
    print("begin self_join")
    args_list = [
        (group_id, group, k, metric_fn, batch_size, folder_path, fname)
        for group_id, group in groups.items()
    ]

    #num_cores = max(8, os.cpu_count())
    # Initialize the processes.
    num_cores=8
    with Pool(processes=num_cores) as pool:
        tiempos = pool.map(process_group_parallel, args_list)
    print("finish self_join")

    # Concatenate partial results.
    print("Concatenating results...")
    output_file = f"{folder_path}/{fname}.csv"
    with open(output_file, mode='w', newline='') as final_out:
        writer = csv.writer(final_out)
        writer.writerow(['id', 'knns', 'dists'])

        for group_id in groups.keys():
            temp_file = f"{folder_path}/{fname}part{group_id}.csv"
            if os.path.exists(temp_file):
                with open(temp_file, mode='r', newline='') as f_in:
                    reader = csv.reader(f_in)
                    writer.writerows(reader)
                os.remove(temp_file) 

    # Remove temporary group pickle files.
    for group_id in groups.keys():
            temp_file_1 = f"{folder_path}/group_{group_id}.pkl"
            if os.path.exists(temp_file_1):
                os.remove(temp_file_1)

    # Save processing times by group.
    with open(f"{folder_path}/tiempo_g.csv", mode='w', newline='') as file_g:
        writer = csv.writer(file_g)
        writer.writerow(['grupo', 'tiempo'])
        for gid, tiempo in tiempos:
            writer.writerow([gid, tiempo])

def store_results(dst, algo, dataset, task, D, I, buildtime, querytime, params):
    """
    Stores the results of a nearest neighbors algorithm into an HDF5 file.

    Parameters:
    ----------
    dst : str or Path
        Path to the output HDF5 file.
    algo : str
        Name of the algorithm used.
    dataset : str
        Name or identifier of the dataset.
    task : str
        Task type (e.g., "knn", "similarity-join", etc.).
    D : np.ndarray
        Array containing the distances between query and neighbor vectors.
    I : np.ndarray
        Array containing the indices of nearest neighbors.
    buildtime : float
        Time taken to build the index or prepare the algorithm.
    querytime : float
        Time taken to perform the query.
    params : dict or str
        Parameters used in the algorithm (can be serialized).
    """
    os.makedirs(Path(dst).parent, exist_ok=True)

    try:
        if I.shape == D.shape:
            #print(f"Data type of I: {I.dtype}")
            #print(f"Data type of D: {D.dtype}")

            with h5py.File(dst, 'w') as f:
                f.attrs['algo'] = algo
                f.attrs['dataset'] = dataset
                f.attrs['task'] = task
                f.attrs['buildtime'] = buildtime
                f.attrs['querytime'] = querytime
                f.attrs['params'] = params

                f.create_dataset('knns', I.shape, dtype=I.dtype)[:] = I
                f.create_dataset('dists', D.shape, dtype=D.dtype)[:] = D
        else:
            print(f"Error: Shapes of 'I' and 'D' do not match. I: {I.shape}, D: {D.shape}")
    except Exception as e:
        print(f"An error occurred while saving the HDF5 file: {e}")

# Conversion functions with error handling.
def safe_literal_eval(val, dtype):
    """
    Safely evaluates a string representation of a Python literal and converts it into a NumPy array 
    with the specified data type.

    This function uses `ast.literal_eval` to safely parse the input string into a Python literal (e.g., list or tuple)
    and then converts the result into a NumPy array of the given dtype. If parsing or conversion fails,
    it returns an empty NumPy array.

    Parameters:
        val (str): The string to be safely evaluated and converted.
        dtype (data-type): The desired NumPy data type for the resulting array.

    Returns:
        np.ndarray: A NumPy array with the specified dtype if successful, otherwise an empty array.
    """
    try:
        return np.array(ast.literal_eval(val), dtype=dtype)  # Convert to an array with the specified data type.
    except (ValueError, SyntaxError) as e:
        print(f"Error while processing value: {val} with error: {e}")
        return np.array([])
    
def run(dataset, task, k):

    print(f'Running {task} on {dataset}')

    # Create fold and filename to store the temporary results.
    folder_path=os.path.join("temporary/", dataset, task)
    fname=f"root_join_{dataset}_{task}"

    os.makedirs(folder_path, exist_ok=True)
    # Prepare the dataset.
    prepare(dataset, task)

    # Load the datas.
    fn, _ = get_fn(dataset, task)
    f = h5py.File(fn)
    data = np.array(DATASETS[dataset][task]['data'](f))
    f.close()
    
    # dimentionarity reduction.
    ini_dim_red=time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=0.8)
    X_pca = pca.fit_transform(X_scaled)
    print(f"pca_dim is: {X_pca.shape[1]}")
    fin_dim_red = time.time()
    d_dim_red = fin_dim_red - ini_dim_red

    ini_global = time.time()
    # Perform self-similarity join.
    self_sim_join(X_pca, 1, 1, k, cos_sim,folder_path,fname)
    # read the CSV using pandas
    df = pd.read_csv(f"{folder_path}/{fname}.csv")
    # Sort the DataFrame by 'id'.
    df = df.sort_values(by='id')
    # Clean possible whitespace and quotes.
    df['knns'] = df['knns'].str.strip().str.replace('"', '')
    df['dists'] = df['dists'].str.strip().str.replace('"', '')
    
    # Convert the 'knns' and 'dists' columns from texts to lists.
    knns = df['knns'].apply(lambda x: safe_literal_eval(x, dtype=int))
    dists = df['dists'].apply(lambda x: safe_literal_eval(x, dtype=float))
    # Convert to Numpy matrices.
    I = np.vstack(knns)
    D = np.vstack(dists)
    
    fin_global = time.time()
    total_global = fin_global - ini_global
    try:
        # Store the final results.
        store_results(os.path.join("results/", dataset, task, f"root_join.h5"), 'Root_Join', 'gooaq', 'task2', D, I, d_dim_red, total_global, f'root_join_params: 1,1; PCA params: 0.8')
        # Remove temporary results folder.
        #shutil.rmtree("temporary")
    except Exception as e:
        print(f"Error while saving the file: {e}")

if __name__ == "__main__":
    freeze_support()
    try:
        
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--task",
            choices=['task2'],
            default='task2'
        )
        parser.add_argument(
            '--dataset',
            choices=DATASETS.keys(),
            default='gooaq'
        )
        
        args = parser.parse_args()
        run(args.dataset, args.task, DATASETS[args.dataset][args.task]['k'])

        print("Process completed successfully.")

    except Exception as e:
        import traceback
        print("An error occurred during execution:")
        traceback.print_exc()