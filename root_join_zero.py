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
from memory_profiler import memory_usage

lock = None 

def init_pool(l):
    global lock
    lock = l

def cos_sim(v1, v2):
    return  1-(np.dot(v1,v2))


def getCenters(data, c):

  n, d = data.shape  # Get the amount of vectors from the datas.
  centers = []  # List to store chosen centers.
  newData = []  # List to store data points not chosen as centers.
  idx = np.random.choice(np.arange(1, n + 1), size=math.floor(c * math.sqrt(n)), replace=False) # choose randomly c*sqrt(n) indices.
  # Separate data points into centers and newData.
  for i, vector in enumerate(data, start=1):
      point = (int(i), vector)
      if i in idx:
           centers.append(point)
      else:
           newData.append(point)
  return n, newData, centers


def makeGroups(n, data, centers, metric_fn, c, size):

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
                for l in range(0):  # Get the centers and add the first two centers to the list.
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
                    for u in range(0):  # Get the centers and add the first two centers to the list.
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
                        for p in range(0): # Get the centers and add the first two centers to the list.
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
                    for u in range(0):  # Get the centers and add the first two centers to the list.
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

    temp=[] # Temporary array to store the distances.
    target=[item for item in target if not np.array_equal(item[1], e)] # Remove the element from the target.
    for element in target:
        id_element=element[0]
        point_element=element[1]
        dist=metric_fn(e,point_element) # For each element, compute the distance.
        temp.append((dist,id_element)) # Append the tuple (dist, id).
    k_nearest = heapq.nsmallest(k, temp, key=lambda x: x[0]) # Sort the distances.
    return  [int(x[1]) for x in k_nearest],[float(x[0]) for x in k_nearest] # Return the indices and distances of the k nearest elements.


def load_pickle_group(group_id, output_dir, lock):

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


def process_group_parallel(args):

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
        #nearest_groups = [elem for elem in element[1]]

        #for _ in range(2): # Get points from the two nearest neighboring groups.
        #    if nearest_groups:
        #        next_g = nearest_groups.pop(0)
        #        target += [elem[0] for elem in load_pickle_group(next_g, folder_path,lock)[1]]

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

        print(f".pkl file for group {group_id} saved in {group_file_path}")




def self_sim_join(data, c1, c2, k, metric_fn, folder_path, fname):

    batch_size = 200000
    n, d = data.shape
    

    # Choose the centers
    print("begin get_centers")
    length, newData, centers = getCenters(data, c1)

    os.makedirs(folder_path, exist_ok=True)


    # Create the groups
    print("begin make_Groups")
    begin_make_group=time.time()
    groups = makeGroups(length, newData, centers, metric_fn, c2, math.sqrt(n))
    end_make_group=time.time()
    print("begin save_Groups")
    save_pickle_group(groups,f'{folder_path}')

    # Create shared objects between processes.
    print("begin self_join")
    args_list = [
        (group_id, group, k, metric_fn, batch_size, folder_path, fname)
        for group_id, group in groups.items()
    ]

    # Initialize the processes.
    begin_queries=time.time()
    lock = Lock() 
    num_cores=os.cpu_count()-2
    with Pool(processes=num_cores, initializer=init_pool, initargs=(lock,)) as pool:
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
    end_queries=time.time()
    return (end_make_group-begin_make_group), (end_queries-begin_queries)

def store_results(dst, algo, dataset, task, D, I, buildtime, querytime, params):

    os.makedirs(Path(dst).parent, exist_ok=True)

    try:
        if I.shape == D.shape:

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

    try:
        return np.array(ast.literal_eval(val), dtype=dtype)  # Convert to an array with the specified data type.
    except (ValueError, SyntaxError) as e:
        print(f"Error while processing value: {val} with error: {e}")
        return np.array([])

def dim_reduction(data):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=0.8)
    X_pca = pca.fit_transform(X_scaled)
    print(f"pca_dim is: {X_pca.shape[1]}")
    return X_pca

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
    mem_usage, X_pca = memory_usage((dim_reduction, (data,)), retval=True, max_usage=True, max_iterations=1)
    fin_dim_red = time.time()
    d_dim_red = fin_dim_red - ini_dim_red

    # Perform self-similarity join.
    mem_join_usage, join_result = memory_usage((self_sim_join, (X_pca, 1, 1, k, cos_sim, folder_path, fname)), retval=True, max_usage=True, max_iterations=1)
    make_groups_time, queries_time = join_result
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
    
    try:
        # Store the final results.
        store_results(os.path.join("results/", dataset, task, f"root_join.h5"), 'Root_Join', 'gooaq', 'task2', D, I, d_dim_red+make_groups_time, queries_time, f'root_join_params: 1,1; PCA params: 0.8; zero; Max memory used during dim reduction: {mem_usage} MiB; Max memory used during self_sim_join: {mem_join_usage} MiB')
        # Remove temporary results folder.
        shutil.rmtree("temporary")
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