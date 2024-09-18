from scene_data_management import ClassificationMethod, SceneManagement
from ae_llm import LLMType
from ae_cvm import CVMType
from llm_room_classifier import LLMRoomClassifier # LLM room classifier
from room_classifier import RoomClassifier # SVC room classifier
from cvm_room_classifier import CVMRoomClassifier # CVM room classifier
from ModelType import ModelType
import pickle, os
from ai2_thor_utils import AI2THORUtils
from scene_description import SceneDescription
from room_type import RoomType

##
# This is a slightly different processor to the one in "process_extracted_scenes.py".
# This one will open the already processed extracted scenes and re-classify all CVM
# outputs because they were interpreted wrongly in the past.
##
class DataSceneProcessor:
    ##
    # We can override default data storage directory (normally- the name of LLM within experiment_data folder)
    ##
    def __init__(self, cvm_type, directory_suffix):
        self.NUMBER_OF_SCENES_IN_BATCH = 25

        self.CVM_TYPE = cvm_type.name
        self.atu = AI2THORUtils()

        ## This is where we'll store old pkl files - the ones that have faulty CVM classification stored
        self.data_store_dir_broken = "experiment_data/" + "pkl_" + self.CVM_TYPE + directory_suffix

        ## This is where we'll store new prepared pkl files - the ones that will have re-classified CVM values stored
        self.data_store_dir_fixed = "experiment_data/" + "pkl_" + self.CVM_TYPE + "_fixed"

        self.scene_mgmt = SceneManagement(self.data_store_dir_fixed)

        self.DEBUG = False # A flag of whether we want to debug and go through scenes quickly - only analyzing some points.

    ##
    # Loads a single specified scene file from the data directory
    ##
    def load_scene_file(self, scene_id):
        scene_f = self.data_store_dir_broken + "/scene_descr_" + str(scene_id) + ".pkl"
        ##
        # Check this scene file. If it doesn't exist, then return None.
        # If it does, then do something with it.
        ##
        try:
            f = open(scene_f,'rb')
        except FileNotFoundError:
            return None

        scene = pickle.load(f)

        return scene

    def store_scene_file_cvm(self, scene_id, sd):
        # store our room points collection into a pickle file
        # Create the directory where to store experiment data if it doesn't exist
        if not os.path.exists(self.data_store_dir_fixed):
            os.makedirs(self.data_store_dir_fixed)

        scene_descr_fname = self.data_store_dir_fixed + "/scene_descr_" + str(scene_id) + ".pkl"
        pickle.dump(sd, open(scene_descr_fname, "wb"))

    ##
    # Process a whole batch of scene files and don't stop until a batch size
    # has been processed.
    ##
    def process_1_batch_of_data_scenes(self):
        # scene descriptions. Each of which will contain points of its floorplan
        # that were traversed using the proper_convert_scene_to_grid_map_and_poses
        # method
        # If pickle file for our scene description exists, then move on to the next
        # scene, otherwise explore this one
        highest_scene_index = self.scene_mgmt.last_index_processed()
        print("Highest index processed: " + str(highest_scene_index))
        processed_scenes_in_this_batch = 0

        while processed_scenes_in_this_batch < self.NUMBER_OF_SCENES_IN_BATCH:
            scene_id = "train_" + str(highest_scene_index + 1)
            print("Processing " + scene_id)
            sd = self.load_scene_file(scene_id)
            highest_scene_index += 1
            if not sd:
                continue

            self.process_scene(sd, scene_id)

            processed_scenes_in_this_batch += 1

    ##
    # Processes a single scene that's already loaded from the data directory.
    ##
    def process_scene(self, scene_data, scene_id):
        points_cnt = 0

        new_sd_with_cvm = SceneDescription() # scene description - as before but now also classified with CVM

        room_points = scene_data.get_all_points()
        for point in room_points:
            # printing out a few already existing data about each point
            img_url = point["front_view_at_this_point"]
            print("################## NEW POINT:################# Image: " + img_url)
            objs_at_this_pos = self.atu.get_visible_object_names_from_collection_csv_unique(point["visible_objects_at_this_point"])
            print("Objects (AI2-THOR): " + objs_at_this_pos)
            print("Room Type GT: " + point["room_type_gt"].name)
            print("Room Type SVC: " + point["room_type_svc"].name)
            points_cnt += 1

            # Now let's try to analyze the pictures with a CVM and see what we would classify them as and
            # what items do we see in each of them.
            cvm_text = point["cvm_text"]
            reclassified_label = RoomType.parse_llm_response(cvm_text, 104)
            print(cvm_text)
            print("Room Type CVM was: " + point["room_type_cvm"].name + ", became: " + reclassified_label.name)


            ## Let's not do items in the picture inference yet- I'm not yet sure how to parse the item list.
            items_in_image = ""
            #items_in_image = self.crc.extract_items_from_this_image(img_url)
            #print("Items in image: " + items_in_image)
            #print("\n")

            new_sd_with_cvm.addPoint(point["point_pose"],
                                    point["room_type_llm"],
                                    point["room_type_svc"],
                                    reclassified_label,
                                    point["room_type_gt"],
                                    point["visible_objects_at_this_point"],
                                    items_in_image,
                                    point["front_view_at_this_point"],
                                    point["elapsed_time_llm"],
                                    point["elapsed_time_svc"],
                                    point["elapsed_time_cvm"],
                                    "",
                                    cvm_text)

            if self.DEBUG and points_cnt >= 3:
                break

        self.store_scene_file_cvm(scene_id, new_sd_with_cvm)

if __name__ == "__main__":
    dsp = DataSceneProcessor(CVMType.CHAMELEON, "_failed_prompt")
    dsp.process_1_batch_of_data_scenes()
