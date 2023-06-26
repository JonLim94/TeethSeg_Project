# Utils
from py_functions import *

# Data Loading Modules
import os
import trimesh
import numpy as np
import pandas as pd

# Loading Neural Networks
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from Teeth_GNN_model import *

# Metrics and visualization
import sklearn.metrics as metrics
from nltk import ConfusionMatrix


if __name__ == '__main__':	
    current_folder = os.getcwd()
    files = os.listdir(f"{current_folder}/Files_for_prediction")
    X_inputs, P_inputs, file_names, all_adj_indices, ori_mesh_files = [[] for _ in range(5)]

    for i in range(len(files)):
        print()
        print(f"loading {files[i]}....")
        # Downsampling
        mesh_filepath = f"{current_folder}/Files_for_prediction/{files[i]}"
        name = files[i].replace(".obj","")
        new_name = f"{current_folder}/Downsampled_files/{name}"
        new_filepath = generate_save_simplifed_mesh(mesh_filepath, new_name, face_count = 10000, file_type="obj")
        
        # Reading
        ori_mesh = trimesh.load(mesh_filepath,file_type="obj")
        ds_mesh = trimesh.load(new_filepath,file_type="obj")

        # Generating Inputs and Labels
        ds_label = np.array([0 for _ in range(len(ds_mesh.triangles_center))])
        X_input, P_input, ds_label, adj_indices = get_inputs(ds_mesh, ds_label)
        offset_label = generate_offset_labels(P_input,ds_label)
        
        # Appending
        X_inputs.append(X_input)
        P_inputs.append(P_input)
        all_adj_indices.append(adj_indices)
        file_names.append(files[i])
        ori_mesh_files.append([ori_mesh,ds_label])
        print(f"{files[i]} is successfully loaded")

    # Converting splitted lists to tensors
    X_inputs = convert_array_to_tensor(X_inputs)
    P_inputs = convert_array_to_tensor(P_inputs)
    
    ### Load model and weights ###
    #model = createModel()    
    model_path = f"{current_folder}/Trained_NN_Model/Model_Training_39_Both Jaw_100 Epochs.hdf5"
    #model.load_weights(model_path)
    model = model = load_model(model_path,
                 custom_objects={'KNNLayer':KNNLayer,
                              "EdgeFeatureLayer":EdgeFeatureLayer,
                              "Offset_Loss":Offset_Loss},
                 compile=False)

    semantic_pred, offset_pred = model.predict([X_inputs,P_inputs],batch_size=2)

    model_pred_labels = np.argmax(semantic_pred,axis=2)
    model_fc = np.asarray(P_inputs)
    model_pred_offsets = np.asarray(offset_pred)
    model_pred_sc = model_fc + 6*model_pred_offsets

    # Visualization Questions
    viz_q = 20
    while viz_q ==20:
        print()
        viz_a = input("Would you like to visualize all files (before and after post-processing)? (y/n) ")
        if viz_a.lower() =="y":
            viz_q = True
        elif viz_a.lower() =="n":
            viz_q = False

			
    ##### Visualization and Metrics #####
    for target_index in range(len(semantic_pred)):
        #### Post-processing ####
        updated_labels = dbscan_clustering(model_pred_sc[target_index], model_pred_labels[target_index], minpoints=30,epsilon=1.05)
        #updated_labels = label_optimization(db_labels,semantic_pred,X_inputs,all_adj_indices,target_index=target_index)
        
        #### Mapping Back ####
        final_labels = Label_Transfer(updated_labels).ds_to_ori(ori_mesh_files[target_index][0], model_fc[target_index])


        #### Visualization ####
        ## Before post-processing ##
        if viz_q == True:
            visualize_mesh(model_fc[target_index],model_pred_labels[target_index],name=f"Before post-processing: {file_names[target_index]}")
            visualize_mesh(model_fc[target_index],updated_labels,name=f"After post-processing: {file_names[target_index]}")
            visualize_mesh(ori_mesh_files[target_index][0].triangles_center,final_labels,name=f"Final Output (Original Mesh): {file_names[target_index]}")
        
        else:
            print("Predicted labels are saved in the Predictions folder")
            
        
        #### Labels Export ####
        ## Downsampled Mesh ##
        pred_df = pd.DataFrame({"Label (Before Post-processing)":model_pred_labels[target_index],
                               "Label (After Post-processing)": updated_labels})
        csv_name = f"{current_folder}/Predictions/{file_names[target_index].replace('.obj','')} Label Predictions (Downsampled).csv"
        pred_df.to_csv(csv_name,index=False,sep='\n')

        ## Mapped Back to Original Mesh ##
        pred_df = pd.DataFrame({"Final Labels":final_labels})
        csv_name = f"{current_folder}/Predictions/{file_names[target_index].replace('.obj','')} Label Predictions (Final).csv"
        pred_df.to_csv(csv_name,index=False,sep='\n')