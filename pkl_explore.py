import glob
import pickle
from scene_description import SceneDescription

def process_scene_files():
    #pkl_store = "pkl_explore/scene_descr_train_*.pkl"
    pkl_store = "experiment_data/pkl_CHAMELEON/*.pkl"

    scene_files = glob.glob(pkl_store) # files showing gemma scenes

    for scene_f in scene_files:
        f = open(scene_f,'rb')
        scene = pickle.load(f)
        print(scene_f)
        #print(scene)

        room_points = scene.get_all_points()
        print(len(room_points))
        #print(room_points[0])
        for i in range(len(room_points)):
            #print(room_points[i]['front_view_at_this_point'] + " :: " + room_points[i]['room_type_cvm'].name)
            print(room_points[i].keys())

process_scene_files()
