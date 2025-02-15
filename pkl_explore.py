import glob
import pickle
from scene_description import SceneDescription
import shutil, os

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
    #pkl_store = "experiment_data/pkl_LLAMA/*.pkl"
    #pkl_store = "experiment_data/pkl_MOONDREAM_one_word/*.pkl"
    #pkl_store = "experiment_data/pkl_LLAMA/scene_descr_train_10.pkl"
    #pkl_store = "experiment_data/pkl_MOONDREAM_one_word/scene_descr_train_10.pkl"
    pkl_store = "experiment_data/pkl_CHAMELEON/*.pkl"

    scene_files = glob.glob(pkl_store) # files showing gemma scenes
    #scene_files2 = glob.glob(pkl_store2)
    all_rp = 0

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
            #print(room_points[i]['room_type_svc'].name + " :: " + room_points[i]['room_type_cvm'].name)
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

        print(all_rp)

process_scene_files()
