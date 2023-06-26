# Dataloading and Downsampling
import os
import trimesh
import json
import numpy as np
import pandas as pd
import open3d as o3d
import vedo
from collections import Counter

# Loading Neural Networks
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.utils import to_categorical
from keras.metrics import MeanIoU

# Metrics and visualization
import pyvista as pv
import sklearn.metrics as metrics
from nltk import ConfusionMatrix

# Post-processing
from sklearn.cluster import DBSCAN
from statistics import mode
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import KDTree
from scipy import stats

#### Mesh Methods ####
def remove_extra_face(ds_mesh, face_labels, face_count=10000):
    if len(ds_mesh.faces) != face_count:
        face_labels[np.where(face_labels==0)[0][0]] = 88
        removal_mask = face_labels != 88
        ds_mesh.update_faces(removal_mask)
        face_labels = face_labels[removal_mask]

    updated_ds_mesh = ds_mesh
    return updated_ds_mesh, face_labels


def get_inputs(mesh, face_labels, face_count=10000):
    """mesh should be already read by trimesh before inputting here"""
    all_adj_indices = []
    mesh, face_labels = remove_extra_face(mesh,face_labels,face_count=face_count)
    all_faces = mesh.triangles
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    adjacency = mesh.face_adjacency
    
    corner_vectors = np.hstack((all_faces[:,0]-face_centers,
                          all_faces[:,1]-face_centers,
                           all_faces[:,2]-face_centers))

    count = 0
    for i in face_centers:
        filtered = adjacency[(adjacency==count).any(axis=1)]
        if filtered.shape[0] != 0:
            adj_faces = np.unique(np.concatenate(filtered))
            adj_faces = adj_faces[adj_faces != count]    
        else:
            adj_faces = np.empty((0,2),dtype=int)
        count += 1
        all_adj_indices.append(adj_faces)
        
    X_input = np.hstack((face_centers,face_normals,corner_vectors))
    P_input = face_centers
    
    return X_input, P_input, face_labels, all_adj_indices
    
  
def generate_save_simplifed_mesh(filepath, new_name, face_count = 10000, file_type="obj"): # new_name is a filepath with new name.
    # O3D decimation ori face_count
    mesh = o3d.io.read_triangle_mesh(filepath)
    
    # Manually transforms original mesh to be aligned with the MICCAI dataset
    #R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    #mesh = mesh.rotate(R)
    #mesh = mesh.translate((0, 0, -95))
    
    simplified_mesh_ori = mesh.simplify_quadric_decimation(face_count)
    triangle_count_ori = len(simplified_mesh_ori.triangles)
    
    if triangle_count_ori != face_count:
        # vedo decimation ori face_count
        mesh_vedo =  vedo.load(filepath)
        simplified_mesh_vedo = mesh_vedo.decimate(face_count/mesh_vedo.ncells)
        triangle_count_vedo = len(simplified_mesh_vedo.faces()) 
        
        if triangle_count_vedo == face_count:
            new_name = f"{new_name}_{str(triangle_count_vedo)}.{file_type}"
            simplified_mesh_vedo.write(new_name)
        else:
            # O3D decimation +1 face_count
            simplified_mesh = mesh.simplify_quadric_decimation(face_count+1)
            triangle_count = len(simplified_mesh.triangles)
            if triangle_count != face_count:
                # Vedo decimation +1 face_count
                simplified_mesh_vedo = mesh_vedo.decimate(face_count+1/mesh_vedo.ncells)
                triangle_count_vedo = len(simplified_mesh_vedo.faces())
                if triangle_count_vedo == face_count:
                    new_name = f"{new_name}_{str(triangle_count_vedo)}.{file_type}"
                    simplified_mesh_vedo.write(new_name)
                else:
                    new_name = f"{new_name}_{str(triangle_count)}.{file_type}"
                    o3d.io.write_triangle_mesh(new_name, simplified_mesh)
    else:
        new_name = f"{new_name}_{str(triangle_count_ori)}.{file_type}"
        o3d.io.write_triangle_mesh(new_name, simplified_mesh_ori)

    return new_name


#### Label Methods ####
def fdi_to_instance(fdi_labels, activate=True):
    """applicable to both original and downsampled"""
    if activate:
        conversion = [0] + [i for i in range(11,19)] + [i for i in range(21,29)] + [i for i in range(31,39)] + [i for i in range(41,49)]
        conversion = [fdi_labels == j for j in conversion]
        
        values = [0] + [x for x in range(1,17)]*2
        
        instances = np.select(conversion,values,fdi_labels)
    else:
        instances = fdi_labels
    return instances
    
    
def load_labels(label_filepath,file_type='json',label_format="fdi",header=None):
    """please use fdi labels only"""
    
    label_format_ref = {"fdi":{"extract":"labels",
                                    "activate":True},
                    "instance":{"extract":"instances",
                               "activate":False}}
    
    if file_type == "json":
        with open(label_filepath, "r") as labels:
            ground_truth_dict = json.load(labels)
            fdi_labels = np.asarray(ground_truth_dict[label_format_ref[label_format]["extract"]])

    elif file_type == "xlsx":
        fdi_labels = pd.read_excel(label_filepath)
        fdi_labels = np.asarray(fdi_labels["Face Labels"].to_list())
    
    elif file_type == "csv" or file_type == "txt":
        fdi_labels = pd.read_csv(label_filepath,header=header)
        fdi_labels = fdi_labels[0].values
        #fdi_labels.loc[fdi_labels[0] != 0, 0] -= 20
    instances = fdi_to_instance(fdi_labels,activate=label_format_ref[label_format]["activate"])
    return instances
    
    
    
class Label_Transfer:
    def __init__(self, labels):
        self.labels = labels

        
    def ori_to_ds(self, ori_mesh, ds_mesh):        
        """original vertex to downsampled face label transfer"""
        #Rx = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
        #ori_mesh.apply_transform(Rx)
        #ori_mesh.apply_translation([0, 0, -95])
        kdt = ori_mesh.kdtree # trimesh object
        dist, idx = kdt.query(ds_mesh.triangles) # also trimesh object
        transferred_labels = self.labels[idx]
        face_labels = np.asarray([Counter(transferred_labels[x]).most_common(1)[0][0] if len(Counter(transferred_labels[x]))<3 else transferred_labels[x][np.argmin(dist[x])] for x in range(len(transferred_labels))])
        
        return face_labels

    
    def ds_to_ori(self, ori_mesh, ds_fc):
        """downsampled face label to original face label transfer"""
        kdt = KDTree(ds_fc) # np vector
        dist, idx = kdt.query(ori_mesh.triangles_center) # trimesh object
        transferred_labels = self.labels[idx]

        return transferred_labels


    def vertex_to_face(self, ori_mesh_files):
        """original or downsampled vertext= to face"""
        faces = ori_mesh_files.faces
        v_labels = self.labels[faces]
        f_labels = np.asarray([Counter(v_labels[x]).most_common(1)[0][0] if len(Counter(v_labels[x]))<3 else 0 for x in range(len(v_labels))])

        return f_labels
        
        
def calculate_offset_label(P_inputs_list, tooth_centroids, labels_list, delta = 6.0, face_count=10000):
    labels_list = np.asarray(labels_list, dtype=int)
    offset_labels = np.zeros(P_inputs_list.shape)
    
    mask_zero = labels_list == 0
    offset_labels[mask_zero]
    
    mask_positive = labels_list>0
    offset_labels[mask_positive] = (tooth_centroids[labels_list[mask_positive]-1] - P_inputs_list[mask_positive])/delta

    return offset_labels
    
    
def generate_offset_labels(P_inputs_list, labels_list, delta=6.0, face_count=10000):
    """P_inputs and labels are not converted to list yet for this function"""
    # Initialization. 
    # tc = tooth centroid
    tc_list = np.zeros((16,3))
    count_list = np.zeros((16,1))
    
    labels_array = np.asarray(labels_list)
    P_inputs_array = np.asarray(P_inputs_list)
    
    mask_nonzero = labels_array != 0
    
    np.add.at(tc_list, labels_array[mask_nonzero] - 1, P_inputs_array[mask_nonzero])
    np.add.at(count_list, labels_array[mask_nonzero] - 1, 1)
    
    tc_vectors = np.divide(tc_list, count_list, where=count_list != 0)
    
    offset_labels = calculate_offset_label(P_inputs_array, tc_vectors, labels_list, delta=delta, face_count=face_count)
    
    return offset_labels
    

#### Visualization ####
def visualize_mesh(mesh, labels, name):
    pv_mesh = pv.PolyData(mesh)
    plotter = pv.Plotter()
    
    plotter.add_mesh(pv_mesh,cmap="tab20b",show_scalar_bar=False,scalars=labels)
    plotter.show(title=name)
    
 
#### Tensor Methods #####
def convert_array_to_tensor(inputs,face_count=10000):
    stacked_array = np.stack(inputs, axis=0)
    stacked_array = tf.convert_to_tensor(stacked_array,dtype=tf.float64)
    return stacked_array
    
 
#### Post-Processing ####
def dbscan_clustering(model_pred_sc, model_pred_labels, minpoints=30,epsilon=1.05):
    dbsc = DBSCAN(eps = epsilon, min_samples = minpoints).fit(model_pred_sc)
    labels = dbsc.labels_ #cluster labels for each datapoint
    upd_labels = labels.copy()
    unique_labels = list(set(labels))[:-1] # [:-1] is to remove the -1 noise cluster
    
    for c in unique_labels:
        clus_idxs = np.where(labels==c)[0]
        mpl_clus = model_pred_labels[clus_idxs] # finding the predicted labels of all the facets clustered c
        mode_label = stats.mode(mpl_clus)[0] # majority predicted label within cluster
        upd_labels[clus_idxs] = mode_label

    # for -1 noise clusters, maintain the original predicted labels
    clus_idxs_neg = np.where(labels==-1)[0]
    upd_labels[clus_idxs_neg] = model_pred_labels[clus_idxs_neg]
    
    return upd_labels
    
    
def label_optimization(cvl, semantic_pred, X_pred_inputs, all_adj_indices, target_index):  
    def custom_one_hot_encode(values,num_classes=17):
        encoded = np.zeros((len(values), num_classes))
        
        for i, val in enumerate(values):
            if val != 0:
                encoded[i, val] = 1
        
        return encoded
    
    def cal_Eu_pi_li(cvl, target_labels): # clus_viz_labels are labels after DBSCAN
        lbl_P = np.argmax(target_labels,axis=1) # predicted labels from model
        mask_equal = cvl == lbl_P
        cvl[~mask_equal] = cvl[~mask_equal]*0 # 0 when cluster != predicted
        OHE_cvl = custom_one_hot_encode(cvl)
        
        pij_numerator = target_labels + 2*OHE_cvl # 2 is sigma in formula, each probability is added with 2*OHE_cvl, 0 when not equal, 1 when equal 
        sum_pij = np.sum(pij_numerator,axis=1, keepdims=True) # denominator of formula
        pij = pij_numerator/sum_pij
        
        Eu_pi_li = -np.log(pij)
        return Eu_pi_li

        
    def cal_Es_pi_pj_li_lj(X_pred_inputs, target_labels, model_pred_labels, all_adj_indices, target_index=0):
        target_X = np.asarray(X_pred_inputs[target_index])
        target_normals = target_X[:,3:6]
        target_fcs = target_X[:,0:3]
        target_adjs = np.asarray(all_adj_indices[target_index],dtype=object)
        outcome = []
        mpl = model_pred_labels.reshape(-1,1)

        encoder = OneHotEncoder(categories=[range(17)],sparse=False)
        OHE_pred_labels = encoder.fit_transform(mpl)

        for idx in range(len(target_adjs)): # iterate through all faces
            ta = target_adjs[idx]
            face_outcome = np.zeros(OHE_pred_labels[idx].shape)
            for ajx in ta: # iterate through all the adjacents in the face
                combined_mask = OHE_pred_labels[idx] != OHE_pred_labels[ajx] # != used so that results return a 0 when equal    
                diff_label = np.where(combined_mask != 0)
                if len(diff_label[0]) >=1:
                    normal_dots = np.around(np.dot(target_normals[idx],target_normals[ajx]),decimals=10)
                    theta = np.arccos(normal_dots)
                    phi = np.linalg.norm(target_fcs[idx] - target_fcs[ajx],ord=2)
                    B_ij = 1 + abs(normal_dots)
                    
                    concavity = np.dot((target_fcs[ajx]-target_fcs[idx]),target_normals[idx]) # (pb - pa).na
                    
                    if concavity<= 0: # convex
                        result = -math.log(theta/math.pi) * phi
                    elif concavity > 0: # concave
                        result = -B_ij * math.log((2*math.pi - theta)/math.pi) * phi
                    temp_array = OHE_pred_labels[ajx] * result

                else:
                    temp_array = np.zeros(OHE_pred_labels[ajx].shape)

                face_outcome += temp_array
            outcome.append(face_outcome)
        outcome = np.asarray(outcome)
        return outcome

    
    target_labels = np.asarray(semantic_pred[target_index]) # set of prediction probabilities
    model_pred_labels = np.argmax(target_labels,axis=1)
    
    Eu_pi_li =  cal_Eu_pi_li(cvl,target_labels)
    Es_pi_pj_li_lj = cal_Es_pi_pj_li_lj(X_pred_inputs, target_labels, model_pred_labels,all_adj_indices,target_index=target_index)

    final_labels = np.argmin(Eu_pi_li + 2*Es_pi_pj_li_lj,axis=1)
    
    return final_labels
    
    
    
    
#### Previous Post-Processing Methods ####
def old_post_processing(model_pred_sc, model_pred_labels, minpoints=30,epsilon=1.05):
		print("Performing post-processing...")
		# DBSCAN
		# Finding the epsilon using NN with minpoints
		neighbors = NearestNeighbors(n_neighbors=minpoints)
		neighbors_fit = neighbors.fit(model_pred_sc)
		distances, indices = neighbors_fit.kneighbors(model_pred_sc)
		distances = np.sort(distances, axis=0)
		distances = distances[:,1]
		plt.plot(distances)

		dbsc = DBSCAN(eps = epsilon, min_samples = minpoints).fit(model_pred_sc)

		labels = dbsc.labels_ #cluster labels for each datapoint
		n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
		n_noise_ = list(labels).count(-1)
		
		clus = {}
		clus_ind = {}
		clus_dict = {}
		clus_viz_labels = []

		for i in range(len(set(labels))):
		    clus[i] = [] # creating clusters in dictionary as empty lists except for class -1
		    clus_ind[i]=[]

		for j in range(len(clus)):
		    clus_dict[j] = (clus[j],clus_ind[j])

		for i, label in enumerate(labels):
			if label in clus_dict:
				cluster, index = clus_dict[label]
				cluster.append(model_pred_sc[i])
				index.append(i)
			clus_viz_labels.append(label+1 if label in clus_dict else 0)
		
 
		# Renaming Cluster Labels
		c_labels = {} # c here means cumulative

		for i in range(len(set(labels))):
			if i<len(set(labels))-1:
				c_labels[i] = []
			else:
				c_labels[-1] = []

		for i, label in enumerate(labels):
			if label in c_labels:
				pred_labels = c_labels[label]
				pred_labels.append(model_pred_labels[i])

		label_to_update = []

		for c in c_labels:
		    mode_label = mode(c_labels[c])
		    label_to_update.append((c,mode_label))

		new_lbl_dict = {}

		for cluster_label, majority_pred_label in label_to_update:
		    new_lbl_dict[cluster_label] = majority_pred_label

		for lb in range(len(clus_viz_labels)):
		    clus_viz_labels[lb] = new_lbl_dict[clus_viz_labels[lb]-1]
		
		# Rectifying Labels in Mesh Model
		final_labels = model_pred_labels.copy()
		for u in range(len(final_labels)):
			if labels[u] >= 0:
				final_labels[u] = new_lbl_dict[labels[u]]
			else:
				continue
		print("Post-processing completed")
		print() 
		return final_labels

def old_label_optimization(cvl, semantic_pred, X_pred_inputs, adj_indices, target_index):  
    def cal_Eu_pi_li(cvl, target_labels): # clus_viz_labels are labels after DBSCAN
        lbl_P = np.argmax(target_labels,axis=1) # predicted labels from model
        mask_equal = cvl == lbl_P
        cvl[~mask_equal] = cvl[~mask_equal]*0 # 0 when cluster != predicted
        cvl_reshaped = cvl.reshape(-1,1)
        
        encoder = OneHotEncoder(categories=[range(17)],sparse=False)
        OHE_cvl = encoder.fit_transform(cvl_reshaped) # creating 17-dimension arrays where value = 1 when clus = labelling
        
        pij_numerator = target_labels + 2*OHE_cvl # 2 is sigma in formula, each probability is added with 2*OHE_cvl, 0 when not equal, 1 when equal 
        sum_pij = np.sum(pij_numerator,axis=1, keepdims=True) # denominator of formula
        pij = pij_numerator/sum_pij
        
        Eu_pi_li = -np.log(pij)
        return Eu_pi_li, pij
        
    def cal_Es_pi_pj_li_lj(X_pred_inputs, model_pred_labels, target_index=target_index):
        target_X = np.asarray(X_pred_inputs[target_index])
        target_normals = target_X[:,3:6]
        target_fcs = target_X[:,0:3]
        target_adjs = adj_indices[target_index]
        sum_results = []
    
        for idxs in range(len(target_adjs)):
            final_result = 0
            for j in target_adjs[idxs]:
                normal_dots = np.around(np.dot(target_normals[idxs],target_normals[j]),decimals=5)
                theta = np.arccos(normal_dots)
                # theta = np.degrees(theta)
                phi = np.linalg.norm(target_fcs[idxs] - target_fcs[j],ord=2)
                B_ij = 1 + abs(normal_dots)
        
                if model_pred_labels[idxs] == model_pred_labels[j]:
                    result = 0
                    
                elif model_pred_labels[idxs] != model_pred_labels[j] and 0 < theta < np.pi:
                    result = -math.log(theta/math.pi) * phi
                
                elif model_pred_labels[idxs] != model_pred_labels[j] and 180 < theta < 360:
                    result = -B_ij * math.log(theta/math.pi) * phi
        
                final_result += result
        
            sum_results.append(np.asarray(final_result))
        sum_results = np.asarray(sum_results)
        sum_results = np.expand_dims(sum_results,axis=1)

        return sum_results
    
    target_labels = np.asarray(semantic_pred[target_index]) # set of prediction probabilities
    model_pred_labels = np.argmax(target_labels,axis=1)
    
    Eu_pi_li,sum_pij =  cal_Eu_pi_li(cvl,target_labels)
    
    Es_pi_pj_li_lj = cal_Es_pi_pj_li_lj(X_pred_inputs, model_pred_labels, target_index=target_index)
    
    final_labels = np.argmin(np.sum([Eu_pi_li + 2*Es_pi_pj_li_lj],axis=0),axis=1)

    return final_labels,sum_pij  


    
if __name__ == "main":
    pass