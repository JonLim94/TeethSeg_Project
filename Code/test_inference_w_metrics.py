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
#import pyvista as pv
import sklearn.metrics as metrics
from nltk import ConfusionMatrix


if __name__ == '__main__':	
    current_folder = os.getcwd()
    files = os.listdir(f"{current_folder}/Files_for_prediction")
    X_inputs, P_inputs, semantic_labels, offset_labels, file_names, all_adj_indices, ori_mesh_files = [[] for _ in range(7)]

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
        label_files = os.listdir(f"{current_folder}/Labels/")
        label_filepath = f"{current_folder}/Labels/{label_files[i]}"
        ori_label = load_labels(label_filepath,file_type='json',label_format="fdi")
        ds_label = Label_Transfer(ori_label).ori_to_ds(ori_mesh, ds_mesh)
        X_input, P_input, ds_label, adj_indices = get_inputs(ds_mesh, ds_label)
        offset_label = generate_offset_labels(P_input,ds_label)
        
        # Appending
        X_inputs.append(X_input)
        P_inputs.append(P_input)
        semantic_labels.append(ds_label)
        offset_labels.append(offset_label)
        all_adj_indices.append(adj_indices)
        file_names.append(files[i])
        ori_mesh_files.append([ori_mesh,ori_label])
        print(f"{files[i]} is successfully loaded")

    # Converting splitted lists to tensors
    X_inputs = convert_array_to_tensor(X_inputs)
    P_inputs = convert_array_to_tensor(P_inputs)
    semantic_pred_labels = convert_array_to_tensor(semantic_labels)
    offset_pred_labels = convert_array_to_tensor(offset_labels)
	
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
        ground_truth_labels = Label_Transfer(ori_mesh_files[target_index][1]).vertex_to_face(ori_mesh_files[target_index][0])


        #### Visualization ####
        ## Before post-processing ##
        if viz_q == True:
            visualize_mesh(model_fc[target_index],model_pred_labels[target_index],name=f"Before post-processing: {file_names[target_index]}")
            visualize_mesh(model_fc[target_index],updated_labels,name=f"After post-processing: {file_names[target_index]}")
            visualize_mesh(ori_mesh_files[target_index][0].triangles_center,final_labels,name=f"Final Output (Original Mesh): {file_names[target_index]}")
        
        else:
            print("Predicted labels are saved in the Predictions folder")
        
        
        #### Metrics ####
        ## Ground Truth (Downsampled) ##
        label_true = tf.convert_to_tensor(semantic_labels[target_index], dtype=tf.float32)
        
        ## Metrics before post-processing ##
        label_pred = tf.convert_to_tensor(model_pred_labels[target_index], dtype=tf.float32)
        pred_MIoU = MeanIoU(num_classes =17)(label_true,label_pred)
        print(f"Mean IoU (before post-processing) for {file_names[target_index]}: {pred_MIoU.numpy()}")
        print("Accuracy before Post-processing:",metrics.accuracy_score(semantic_labels[target_index],model_pred_labels[target_index]))
        #print("Confusion Matrix before Post-processing:")
        #print(ConfusionMatrix(semantic_labels[target_index],model_pred_labels[target_index]))
        print()

        ## Metrics after post-processing ##
        label_updated = tf.convert_to_tensor(updated_labels, dtype=tf.float32)
        MIoU_afterPP = MeanIoU(num_classes =17)(label_true,label_updated)                                  
        print(f"Mean IoU (after post-processing) for {file_names[target_index]}: {MIoU_afterPP.numpy()}")
        print("Accuracy after Post-processing:",metrics.accuracy_score(semantic_labels[target_index],updated_labels))
        #print("Confusion Matrix after Post-processing:")
        #print(ConfusionMatrix(semantic_labels[target_index],updated_labels))
        print()

        ## Final Metrics ##
        label_ground_truth = tf.convert_to_tensor(ground_truth_labels, dtype=tf.float32) # Ground Truth (Ori Face Centroids)
        label_final = tf.convert_to_tensor(final_labels, dtype=tf.float32)
        final_MIoU = MeanIoU(num_classes =17)(label_ground_truth,label_final)
        print("Final Mean IoU:",final_MIoU.numpy())
        print("Final Accuracy:",metrics.accuracy_score(ground_truth_labels,final_labels))
        #print("Final Confusion Matrix:")
        #print(ConfusionMatrix(ground_truth_labels,final_labels))
        print()

        
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