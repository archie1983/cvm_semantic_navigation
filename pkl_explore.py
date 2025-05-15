import glob
import pickle
import numpy as np
from scene_description import SceneDescription
import shutil, os
from room_type import RoomType

def find_observed_point_by_pose(pose, room_points):
    #(pos, rot) = pose # ((10.75, 1.57599937915802, 1.0), (30.000003814697266, 0.0, 0))
    result = None
    for rp in room_points:
        if rp['point_pose'] == pose:
            result = rp
            return result
    return result

def process_scene_files():
    #pkl_store = "pkl_explore/scene_descr_train_*.pkl"
    pkl_store = "experiment_data/pkl_LLAMA/*.pkl"
    #pkl_store = "experiment_data/pkl_MOONDREAM_one_word/*.pkl"
    #pkl_store = "experiment_data/pkl_LLAMA/scene_descr_train_10.pkl"
    #pkl_store = "experiment_data/pkl_MOONDREAM_one_word/scene_descr_train_10.pkl"
    #pkl_store = "experiment_data/pkl_CHAMELEON/*.pkl"
    #pkl_store = "experiment_data/pkl_CHAMELEON_p_cot_6lbl_img_middle/scene_descr_train_55.pkl"
    pkl_store = "experiment_data/pkl_CHAMELEON_p_cot_6lbl_img_middle/*.pkl"

    scene_files = glob.glob(pkl_store) # files showing gemma scenes
    #scene_files2 = glob.glob(pkl_store2)
    all_rp = 0

    min_obj_cnt = 10000
    max_obj_cnt = -10000
    obj_cnts = []
    for scene_f in scene_files:
        f = open(scene_f,'rb')
        scene = pickle.load(f)

        f_name = scene_f.split("/")[2]
        cor_llm_pkl_path = "experiment_data/pkl_LLAMA/" + f_name # corresponding LLM pkl path
        llm_f = open(cor_llm_pkl_path,'rb')
        llm_scene = pickle.load(llm_f)

        llm_room_points = llm_scene.get_all_points()

        print(scene_f)
        #print(scene)

        room_points = scene.get_all_points()
        print(len(room_points))
        #print(room_points[0])

        for i in range(len(room_points)):
            all_rp += 1
            obj_cnt = len(room_points[i]['visible_object_names'])
            print(room_points[i]['visible_object_names'], obj_cnt)
            obj_cnts.append(obj_cnt)

            if (min_obj_cnt > obj_cnt):
                min_obj_cnt = obj_cnt
            if (max_obj_cnt < obj_cnt):
                max_obj_cnt = obj_cnt

            #print(room_points[i]['room_type_llm'].name + " :: " + room_points[i]['room_type_cvm'].name)
            #if room_points[i]["room_type_llm"] == RoomType.NOT_CLASSIFIED or room_points[i]["room_type_llm"] == RoomType.NOT_KNOWN:
            #    print(room_points[i]["room_type_llm"])
            #print(room_points[i].keys())
            #pic_path = room_points[i]['front_view_at_this_point']
            #print(pic_path)
            #pic_path_components = pic_path.split("/")
            #new_path = pic_path_components[0] + "_new/" + pic_path_components[1] + "/" + pic_path_components[2]
            #print(new_path)
            #new_path_folder = pic_path_components[0] + "_new/" + pic_path_components[1] + "/"
            
            # Create the directory where to store experiment data if it doesn't exist
            #if not os.path.exists(new_path_folder):
            #    os.makedirs(new_path_folder)

            #shutil.copyfile(pic_path, new_path)
            #print(room_points[i]['point_pose'])
            #fp = find_observed_point_by_pose(room_points[i]['point_pose'], llm_room_points)
            #print(str(fp['point_pose']))

    print(all_rp, min_obj_cnt, max_obj_cnt, np.mean(obj_cnts), np.median(obj_cnts))

process_scene_files()
