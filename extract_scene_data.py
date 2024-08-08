from time import time
import pandas as pd
from thortils import (launch_controller,
                      convert_scene_to_grid_map, proper_convert_scene_to_grid_map, proper_convert_scene_to_grid_map_and_poses)
from thortils.scene import SceneDataset
from thortils.utils.visual import GridMapVisualizer
from thortils.agent import thor_reachable_positions
from thortils.grid_map import GridMap

import prior, pickle, glob, os

from llm_room_classifier import LLMRoomClassifier # LLM room classifier
from room_classifier import RoomClassifier # SVC room classifier
from cvm_room_classifier import CVMRoomClassifier # CVM room classifier
from ModelType import ModelType
from room_type import RoomType
from ae_llm import LLMType
from ae_cvm import CVMType
from scene_description import SceneDescription

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scene_data_management import ClassificationMethod, SceneManagement
from ai2_thor_utils import AI2THORUtils

class DataSceneExtractor:
    ##
    # We can override default data storage directory (normally- the name of LLM within experiment_data folder)
    ##
    def __init__(self, llm_type, cvm_type, classification_method_in, data_store_dir = ""):
        self.dataset = None
        # If our point only has these common objects visible, then there's little point
        # to classify, because these are common.
        self.common_objs = {'Wall', 'Doorway', 'Window', 'Floor', 'Doorframe'}

        self.atu = AI2THORUtils()

        self.classification_method = classification_method_in

        if (self.classification_method.llm_required()):
            self.lrc = LLMRoomClassifier(llm_type) # LLM classifier
        if (self.classification_method.svc_required()):
            self.src = RoomClassifier(False, ModelType.HYBRID_AI2_THOR) # SVC classifier
        if (self.classification_method.cvm_required()):
            self.crc = CVMRoomClassifier(cvm_type)
        self.NUMBER_OF_SCENES_IN_BATCH = 7

        self.LLM_TYPE = llm_type.name

        if (data_store_dir == ""):
            self.data_store_dir = "experiment_data/" + "pkl_" + self.LLM_TYPE
        else:
            self.data_store_dir = "experiment_data/" + data_store_dir

        self.scene_mgmt = SceneManagement(self.data_store_dir)

        self.DEBUG = False # A flag of whether we want to debug and go through a room very quickly - only small rooms and only 3 points in each to classify.

        # Create the directory where to store experiment data if it doesn't exist
        if not os.path.exists(self.data_store_dir):
            os.makedirs(self.data_store_dir)

    ##
    # Ground truth functions - data extracted from the actual room and point is
    # tested to belong to the room polygon or not.
    ##
    def is_point_inside_room_ground_truth(self, point_to_test, room_polygon):
        (x, y, z) = point_to_test
        point = Point(x, z)
        polygon = Polygon(room_polygon)
        return polygon.contains(point)
    ##
    # Ground truth functions - data extracted from the actual room and point is
    # tested to belong to the room polygon or not.
    ##
    def what_room_is_point_in_ground_truth(self, rooms, point):
        for room in rooms:
            if self.is_point_inside_room_ground_truth(point, room[1]):
                return RoomType.interpret_label(room[0])
        return RoomType.interpret_label("NONE")
    ##
    # Ground truth functions - data extracted from the actual room and point is
    # tested to belong to the room polygon or not.
    ##
    def get_rooms_ground_truth(self, house):
        rooms = []
        #print(house)
        #print("\n")
        #print(house["rooms"])
        for room in house["rooms"]:
            room_poly = [(corner["x"], corner["z"]) for corner in room["floorPolygon"]]
            #print(room["roomType"] + " # " + str(room["floorPolygon"]))
            #print(room["roomType"] + " ?? " + str(room_poly))
            rooms.append((room["roomType"], room_poly))

        return rooms

    def is_full_house(self, rooms):
        existing_room_names = set()
        for room in rooms:
            rl = room[0].upper()
            if rl == "LIVINGROOM":
                rl = "LIVING ROOM"
            existing_room_names.add(rl)

        return set(RoomType.all_labels()) == existing_room_names

    def getDataSet(self):
        if (self.dataset is None):
            self.dataset = prior.load_dataset("procthor-10k", "439193522244720b86d8c81cde2e51e3a4d150cf")
            #print(self.dataset)
        return self.dataset

    def process_1_batch_of_data_scenes(self):
        ds = self.getDataSet()
        # scene descriptions. Each of which will contain points of its floorplan
        # that were traversed using the proper_convert_scene_to_grid_map_and_poses
        # method
        # If pickle file for our scene description exists, then move on to the next
        # scene, otherwise explore this one
        highest_scene_index = self.scene_mgmt.last_index_extracted()
        print("Highest index explored: " + str(highest_scene_index))
        processed_scenes_in_this_batch = 0

        while processed_scenes_in_this_batch < self.NUMBER_OF_SCENES_IN_BATCH:
            scene_id = "train_" + str(highest_scene_index + 1)
            print("Processing " + scene_id)
            sd = self.ae_process_proctor_scene(scene_id, ds)
            highest_scene_index += 1
            if not sd:
                continue

            processed_scenes_in_this_batch += 1

    ##
    # Load a PROCTHOR scene specified by the scene_id, build map and classify all
    # visited points belonging to one of the semantic room types. Also build a pkl
    # file with the classification result and the visible objects at that point.
    #
    # The scene_id takes form of <dataset>_<scene_number>, where dataset is one of:
    # train, val, test and scene_number is a number of scene in that set.
    # e.g.: "train_3" or "test_10"
    ##
    def ae_process_proctor_scene(self, scene_id, dataset):
        scene_id_split = scene_id.split("_")
        data_set = scene_id_split[0]
        scene_num = int(scene_id_split[1])
        time_records = []  # List to store time records for each position

        print("Loading : " + data_set + "[" + str(scene_num) + "]")

        house = dataset[data_set][scene_num]
        rooms = self.get_rooms_ground_truth(house)

        print("ROOMS:" + str(rooms))

        # If we don't have all 4 rooms types- kitchen, bedroom, living room and bathroom,
        # then skip.
        # for debug only - when we want quick classification of a small room, the flag self.DEBUG must be true.
        if (not self.DEBUG and not self.is_full_house(rooms)) or (self.DEBUG and self.is_full_house(rooms)):
            print("skipping house because it's not full ----------------------- ")
            return False

        controller = launch_controller({"scene": house, "VISIBILITY_DISTANCE": 3.0})

        #grid_map = convert_scene_to_grid_map(controller, floor_plan, 0.25)

        keywords = {'num_stops': 100, 'num_rotates': 8, 'sep': 1.25, 'downsample': True, 'v_angles': [30]}
        (grid_map, observed_pos, observed_front_views) = proper_convert_scene_to_grid_map_and_poses(controller,
                                     floor_cut=0.1,
                                     ceiling_cut=1.0,
                                     scene_id=scene_id,
                                     **keywords)

        # scene description, which will contain points of its floorplan
        # that were traversed using the proper_convert_scene_to_grid_map_and_poses
        # method.
        sd = SceneDescription() # scene description classified with LLM and SVC
        #print("observed_front_views ::::::::::::::::::::::::::::::")
        #print(observed_front_views)

        i = 0
        for pos, objs in observed_pos.items():
            #print(pos)
            objs_at_this_pos = self.atu.get_visible_object_names_from_collection_set(objs)
            img_url = observed_front_views[pos]

            # at first assume that we can use object list to classify
            classify_using_object_list = True

            # No point to classify this point if there are no objects
            # We may not want to classify it using SVC or LLM, but a visual
            # CVM might still be able to classify it.
            if (len(objs_at_this_pos) < 1):
                print("Empty set of objects -- skipping")
                classify_using_object_list = False
            # Similarly with common objects only- SVC and LLM can't help here,
            # but CVM might be able to, so continue, just don't use LLM or SVC.
            if (objs_at_this_pos.issubset(self.common_objs)):
                print("Only common objects -- skipping")
                classify_using_object_list = False

            # initialise result variables
            rt_llm = RoomType.NOT_CLASSIFIED
            rt_svc = RoomType.NOT_CLASSIFIED
            rt_cvm = RoomType.NOT_CLASSIFIED
            svc_elapsed_time = 0
            llm_elapsed_time = 0
            cvm_elapsed_time = 0

            # classify using the appropriate methods
            if (self.classification_method.llm_required() and classify_using_object_list):
                t0 = time()
                rt_llm = self.lrc.classify_room_by_this_object_set(objs_at_this_pos)
                llm_elapsed_time = round(time() - t0, 5)
                #print("llm predict time:", llm_elapsed_time, "s")

            if (self.classification_method.svc_required() and classify_using_object_list):
                t0 = time()
                rt_svc = self.src.classify_room_by_this_object_set(objs_at_this_pos)
                svc_elapsed_time = round(time() - t0, 5)
                #print("svc predict time:", svc_elapsed_time, "s")

            if (self.classification_method.cvm_required()):
                t0 = time()
                rt_cvm = self.crc.classify_room_by_this_image(img_url)
                cvm_elapsed_time = round(time() - t0, 5)
                #print("svc predict time:", svc_elapsed_time, "s")

            rt_gt = self.what_room_is_point_in_ground_truth(rooms, pos[0])

            sd.addPoint(pos, rt_llm, rt_svc, rt_cvm, rt_gt, objs, None, img_url, llm_elapsed_time, svc_elapsed_time, cvm_elapsed_time, "", "")

            time_records.append({
                "Position": pos,
                "LLM Time (s)": llm_elapsed_time,
                "SVC Time (s)": svc_elapsed_time,
                "CVM Time (s)": cvm_elapsed_time
            })

            print(objs_at_this_pos)
            print(rt_svc.name + " ## " + rt_llm.name + " ## " + rt_cvm.name + " ## " + rt_gt.name)

            # For debug only - this would limit the explored points per room to only 3
            if self.DEBUG:
                i+=1
                if i > 3:
                    break

        #print("AE::::::::::::::::::::::;")

        # Data processing and saving as Excel
        df = pd.DataFrame(time_records)
        df.to_excel(self.data_store_dir + f"/timing_data_{scene_id}.xlsx", index=False)

        # store our room points collection into a pickle file
        scene_descr_fname = self.data_store_dir + "/scene_descr_" + scene_id + ".pkl"
        pickle.dump(sd, open(scene_descr_fname, "wb"))

        #print(len(observed_pos))

        #print(floor_plan)
        for y in range(grid_map.length):
            row = []
            for x in range(grid_map.width):
                if (x,y) in grid_map.free_locations:
                    row.append(".")
                else:
                    #assert (x,y) in grid_map.obstacles
                    if (x,y) in grid_map.obstacles:
                        row.append("x")
                    else:
                        row.append("u")
            print("".join(row))

        return sd # return scene description - classified with LLM and with SVC

if __name__ == "__main__":
    #dse = DataSceneExtractor(LLMType.LLAMA, CVMType.MOONDREAM, ClassificationMethod.SVC_CVM)
    dse = DataSceneExtractor(LLMType.LLAMA, CVMType.MOONDREAM, ClassificationMethod.SVC, "data_collection")
    dse.process_1_batch_of_data_scenes()
