import os, pickle, prior, time, re
from pathlib import Path

from llm_room_classifier import LLMRoomClassifier, LLMType
from room_type import RoomType
from ai2_thor_utils import what_room_is_point_in_ground_truth, get_rooms_ground_truth, convert_pose_set2tuple
from scene_description import SceneDescription, ClassifierType

from thortils import (launch_controller,
                      convert_scene_to_grid_map, proper_convert_scene_to_grid_map, proper_convert_scene_to_grid_map_and_poses)

from thortils.navigation import get_shortest_path_to_object_type, get_shortest_path_to_object
from thortils.agent import thor_reachable_positions, thor_agent_position, thor_agent_pose
from thortils.utils import roundany
from thortils.controller import _resolve
from thortils.object import thor_closest_object_of_type
from ae_robot_simulation_control import RobotNavigationControl

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import copy
from IPython import get_ipython

##
# This class will analyze harvested scene data and ask LLM:
# 1) In which room is the best chance to find the requested object?
# 2) Near which object should we look first?
#
# Based on the answers we then plan a path to the best choice room, best choice
# object.
##
class SemanticPathPlanner:
    def __init__(self, scene_id, llm_type, hyb_data = ""):
        '''

        :param scene_id: ID of the scene we are going to use (e.g. train_55)
        :param llm_type: The LLM type we want (defined in LLMType in ae_llm.py
        :param hyb_data: If we want to hybridize data, then this has to contain VLM data directory
        '''
        self.data_store_dir = "experiment_data"
        self.LLM_TYPE = llm_type.name

        ## figure out where are we running- in terminal or jupyter
        if not self.is_running_in_jupyter():
            matplotlib.use('Agg')

        scene_descr_fname = self.data_store_dir + "/pkl_" + self.LLM_TYPE + "/scene_descr_" + scene_id + ".pkl"

        if os.path.isfile(scene_descr_fname):
            file = open(scene_descr_fname,'rb')
            self.scene_description = pickle.load(file)
            file.close()

            print("Loaded : " + scene_descr_fname + " scene")
        else:
            # if no scenes' data, then nothing to do
            raise Exception("No scenes data file found. Nothing to do.")

        ## If we want data hybridization, then open the VLM data file
        self.scene_description_cvm = None
        if hyb_data != "":
            scene_descr_cvm_fname = self.data_store_dir + "/pkl_" + hyb_data + "/scene_descr_" + scene_id + ".pkl"
            if os.path.isfile(scene_descr_cvm_fname):
                file = open(scene_descr_cvm_fname,'rb')
                self.scene_description_cvm = pickle.load(file)
                file.close()

                print("Loaded : " + scene_descr_cvm_fname + " scene")
            else:
                # if no scenes' data, then nothing to do
                raise Exception("No scenes data file found. Nothing to do.")

        self.scene_id = scene_id
        self.lrc = LLMRoomClassifier(llm_type)
        self.dataset = None
        self.controller = None
        self.rnc = RobotNavigationControl()

        self.ae_load_proctor_scene(self.scene_id)
        self.last_start_position = None
        self.last_goal_position = None

    def getDataSet(self):
        if (self.dataset is None):
            self.dataset = prior.load_dataset("procthor-10k", "439193522244720b86d8c81cde2e51e3a4d150cf")
            #print(self.dataset)
        return self.dataset

    def ae_load_proctor_scene(self, scene_id):
        dataset = self.getDataSet()

        scene_id_split = scene_id.split("_")
        data_set = scene_id_split[0]
        scene_num = int(scene_id_split[1])
        time_records = []  # List to store time records for each position

        print("Loading : " + data_set + "[" + str(scene_num) + "]")

        house = dataset[data_set][scene_num]
        self.rooms = get_rooms_ground_truth(house)

        self.controller = launch_controller({"scene": house, "VISIBILITY_DISTANCE": 3.0, "headless": False})

        self.rnc.set_controller(self.controller)

    def get_path_to_actual_object(self, needed_obj):
        #return get_shortest_path_to_object_type(controller, object_id, start_position, start_rotation, **{"return_plan": return_plan})
        (start_position, start_rotation) = self.rnc.get_agent_pos_and_rotation()

        event = _resolve(self.controller)
        self.last_start_position, _ = thor_agent_pose(event)

        obj = needed_obj #thor_closest_object_of_type(self.controller, object_type)
        #print(obj)
        self.last_goal_position = obj["position"]

        try:
            path = get_shortest_path_to_object(self.controller, obj["objectId"], start_position, start_rotation)
        except ValueError:
            path = None

        return path

    def get_path_to(self, object_type):
        #return get_shortest_path_to_object_type(controller, object_id, start_position, start_rotation, **{"return_plan": return_plan})
        (start_position, start_rotation) = self.rnc.get_agent_pos_and_rotation()

        event = _resolve(self.controller)
        self.last_start_position, _ = thor_agent_pose(event)

        obj = thor_closest_object_of_type(self.controller, object_type)
        #print(obj)
        self.last_goal_position = obj["position"]

        return get_shortest_path_to_object_type(self.controller, object_type, start_position, start_rotation)

    ##
    # Asking LLM to tell us where to look for a bottle of beer
    ##
    def bring_me_a_bottle_of_beer(self):
        work_scene = self.scene_description

        room_to_look_in, full_text = self.lrc.where_to_find_this("A bottle of beer")
        object_names_to_look_at = work_scene.getAllVisibleObjectNamesInThisRoom(ClassifierType.LLM, room_to_look_in)
        print(object_names_to_look_at)
        object_to_look_at, full_text_obj = self.lrc.where_to_look_first("A fresh, cold, unopened bottle of beer", object_names_to_look_at)

        path = self.get_path_to(object_to_look_at)
        #path = self.get_path_to("Fridge")
        #print(str(path))

        return (path, room_to_look_in, object_to_look_at)

    ##
    # Asking LLM to tell us where to look for a bottle of beer
    ##
    def bring_me_a_bottle_of_beer_from_actual_objs(self):
        work_scene = self.scene_description

        room_to_look_in, full_text = self.lrc.where_to_find_this("A bottle of beer")
        object_names_to_look_at = work_scene.getAllVisibleObjectNamesInThisRoom(ClassifierType.LLM, room_to_look_in)
        actual_objects_to_look_at = work_scene.getAllVisibleObjectsInThisRoom(ClassifierType.LLM, room_to_look_in)
        print(object_names_to_look_at)
        object_to_look_at, full_text_obj = self.lrc.where_to_look_first("A fresh, cold, unopened bottle of beer", object_names_to_look_at)

        for obj in actual_objects_to_look_at:
            if obj["objectType"] == object_to_look_at:
                needed_obj = obj
                break

        #path = self.get_path_to(object_to_look_at)
        path = self.get_path_to_actual_object(needed_obj)
        #path = self.get_path_to("Fridge")
        #print(str(path))

        return path

    ##
    # Asking LLM to tell us where to look for a hair pin
    ##
    def bring_me_hair_pin(self):
        work_scene = self.scene_description

        room_to_look_in, full_text = self.lrc.where_to_find_this("Hair pin")
        object_names_to_look_at = work_scene.getAllVisibleObjectNamesInThisRoom(ClassifierType.LLM, room_to_look_in)
        print(object_names_to_look_at)
        object_to_look_at, full_text_obj = self.lrc.where_to_look_first("Hair pin", object_names_to_look_at)

        path = self.get_path_to(object_to_look_at)
        #path = self.get_path_to("Fridge")
        #print(str(path))

        return path

    ##
    # Asking LLM to tell us where to look for an arbitrary object
    ##
    def bring_me_this(self, what_to_bring):
        work_scene = self.scene_description

        room_to_look_in, full_text = self.lrc.where_to_find_this(what_to_bring)
        object_names_to_look_at = work_scene.getAllVisibleObjectNamesInThisRoom(ClassifierType.LLM, room_to_look_in)
        #print(object_names_to_look_at)
        object_to_look_at, full_text_obj = self.lrc.where_to_look_first(what_to_bring, object_names_to_look_at)

        path = self.get_path_to(object_to_look_at)
        #path = self.get_path_to("Fridge")
        #print(str(path))

        return (path, room_to_look_in, object_to_look_at)

    def get_all_points_of_room_type_hyb(self, room_to_look_in):
        '''
        A replacement for SceneDescription.get_all_points_of_room_type for the purposes of room classifier hybridization.
        :param room_to_look_in: Room of interest
        :return: a set of points belonging to given room which is chosen using the hybrid classifier.
        '''
        work_scene = self.scene_description
        vlm_scene = self.scene_description_cvm

        ret_points = []
        for point in vlm_scene.points_of_scene:
            # go through VLM points and immediately look up LLM point at the same pose.
            # If LLM point doesn't exist, then use VLM point. If LLM point does exist,
            # then use LLM point.
            llm_point = self.find_observed_point_by_pose(point['point_pose'], work_scene.points_of_scene)
            if llm_point is None or llm_point["room_type_llm"] == RoomType.NOT_CLASSIFIED or llm_point["room_type_llm"] == RoomType.NOT_KNOWN:
                # If LLM doesn't know, then use VLM's knowledge
                #print("VLM turn")
                if point is not None and point["room_type_cvm"] == room_to_look_in:
                    #print("VLM helped")
                    ret_points.append(point)
            else:
                # If LLM knows the room label, then use LLM's knowledge
                #print("LLM did the job")
                ret_points.append(llm_point)

        return ret_points

    def find_observed_point_by_pose(self, pose, room_points):
        '''
        Looks up a point in the given set by its pose
        :param pose:
        :param room_points:
        :return:
        '''
        #(pos, rot) = pose # ((10.75, 1.57599937915802, 1.0), (30.000003814697266, 0.0, 0))
        result = None
        for rp in room_points:
            if rp['point_pose'] == pose:
                result = rp
                return result
        return result

    def getAllVisibleObjectNamesInThisRoom_hyb(self, rt):
        if self.scene_description_cvm is None:
            return self.scene_description.getAllVisibleObjectNamesInThisRoom(ClassifierType.LLM, rt)
        else:
            points = self.get_all_points_of_room_type_hyb(rt)
            ret_set = set()
            for p in points:
                ret_set = ret_set.union(p["visible_object_names"])

            return ret_set

    def getAllVisibleObjectsInThisRoom_hyb(self, rt):
        if self.scene_description_cvm is None:
            return self.scene_description.getAllVisibleObjectsInThisRoom(ClassifierType.LLM, rt)
        else:
            points = self.get_all_points_of_room_type_hyb(rt)
            ret_list = []
            for p in points:
                # print(str(p["visible_objects_at_this_point"]))
                # ret_list = {*ret_list, *p["visible_objects_at_this_point"]} # join two lists
                ret_list += p["visible_objects_at_this_point"]

            return ret_list

    def bring_me_this_from_actual_objs(self, what_to_bring):
        '''
        Querying the LLM for the full thing. We have an item that we want. First we ask LLM which room to look for it,
        then we ask what item in that room is it going to be next to?
        :param what_to_bring: Item wanted
        :return: a tuple of path, chosen room, chosen object category
        '''
        work_scene = self.scene_description # the earlier processed scene using LLM

        # Query LLM for a room where we need to look for the wanted item (can be RoomType.NOT_KNOWN)
        room_to_look_in, full_text_room = self.lrc.where_to_find_this(what_to_bring)

        if not (room_to_look_in in RoomType.all_options(False)):
            # If the chosen room is not one of: Bathroom, Kitchen, Living Room, Bedroom, then stop here.
            return (None, None, None, None, full_text_room, None)

        # All visible objects (names only) in the room that was selected (can be empty set)
        #object_names_to_look_at = work_scene.getAllVisibleObjectNamesInThisRoom(ClassifierType.LLM, room_to_look_in)
        object_names_to_look_at = self.getAllVisibleObjectNamesInThisRoom_hyb(room_to_look_in)
        if len(object_names_to_look_at) < 1:
            # If there are no object names in the selected room, then stop here.
            return (None, room_to_look_in, None, None, full_text_room, None)

        # A list of all visible actual objects in the selected room (can be empty list)
        #actual_objects_to_look_at = work_scene.getAllVisibleObjectsInThisRoom(ClassifierType.LLM, room_to_look_in)
        actual_objects_to_look_at = self.getAllVisibleObjectsInThisRoom_hyb(room_to_look_in)
        #print(object_names_to_look_at)
        # Query LLM for the object name from object_names_to_look_at list that is closest match for the wanted object
        object_to_look_at, full_text_obj = self.lrc.where_to_look_first(what_to_bring, object_names_to_look_at)

        # Select an actual object based on the chosen object name
        needed_obj = None
        for obj in actual_objects_to_look_at:
            if obj["objectType"] == object_to_look_at:
                needed_obj = obj
                break

        # If an actual object could not be selected, then we can't plan a path
        if needed_obj is not None:
            path = self.get_path_to_actual_object(needed_obj)
            # print(str(path))
        else:
            path = None

        return (path, room_to_look_in, object_to_look_at, needed_obj, full_text_room, full_text_obj)

    ##
    # For display purposes - the top down view of the habitat
    ##
    def get_top_down_frame(self):
        # Setup the top-down camera
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.1 * max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        del pose["orthographicSize"]

        # add the camera to the scene
        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )

        top_down_frame = event.third_party_camera_frames[-1]
        return Image.fromarray(top_down_frame)

    ##
    # Plot a path on the top-down view of the habitat
    ##
    def visualise_path(self, path, store_url = "1.png"):
        grid_size = self.controller.initialization_parameters["gridSize"]

        reachable_positions = [
            tuple(map(lambda x: roundany(x, grid_size), pos))
            for pos in thor_reachable_positions(self.controller)]

        x_max = max([pos[0] for pos in reachable_positions])
        z_max = max([pos[1] for pos in reachable_positions])
        x_min = min([pos[0] for pos in reachable_positions])
        z_min = min([pos[1] for pos in reachable_positions])

        start = self.last_start_position
        goal = self.last_goal_position

        fig, ax = plt.subplots()

        # another way how to plot the path
        #x = [p[0]["x"] for p in path]
        #z = [p[0]["z"] for p in path]
        #ax.scatter(x, z, s=300, c='gray', zorder=1)

        # setting up for the top-down picture of the habitat
        print(str(x_min-grid_size) + " " + str(x_max+grid_size) + " " + str(z_min-grid_size) + " " + str(z_max+grid_size))
        img = self.get_top_down_frame()
        ex_mul = 7
        ax.imshow(img, extent=[x_min-ex_mul*grid_size, x_max+ex_mul*grid_size, z_min-ex_mul*grid_size, z_max+ex_mul*grid_size])

        # set up for the path print
        lim_mul = 4
        ax.set_xlim(x_min-lim_mul*grid_size, x_max+lim_mul*grid_size)
        ax.set_ylim(z_min-lim_mul*grid_size, z_max+lim_mul*grid_size)

        # start pos
        xs = start["x"]
        zs = start["z"]
        ax.scatter([xs], [zs], s=100, c='red', zorder=4)

        # goal
        xg = goal["x"]
        zg = goal["z"]
        ax.scatter([xg], [zg], s=100, c='green', zorder=4)

        # path
        for step in path:
            x = step[0]["x"]
            z = step[0]["z"]
            ax.scatter([x], [z], s=30, zorder=2, c="blue")

        plt.axis('off')

        if self.is_running_in_jupyter():
            plt.show()
        else:
            #plt.savefig(self.habitat_mgmt.get_current_top_view_fname())
            plt.savefig(store_url)

    def get_controller(self):
        return self.controller

    ##
    # Extracts item name from its pickle file
    ##
    def extract_item_name(self, filename):
        """Extracts the item name from the filename (e.g., 'oven_mitts' from '20_oven_mitts_train_55.pkl')."""
        pattern = r"\d+_(.*?)_train_\d+\.pkl"
        match = re.match(pattern, filename)
        if match:
            return match.group(1).replace("_", " ")  # Convert underscores to spaces
        return None

    ##
    # Finds processed items from their pickle files
    ##
    def find_processed_items(self, root_dir="."):
        """Scans subfolders for .pkl files and organizes them by room type."""
        completed_items = {
            "Kitchen": [],
            "Bathroom": [],
            "Living Room": [],
            "Bedroom": []
        }

        for room in completed_items.keys():
            room_dir = Path(root_dir) / room
            if not room_dir.exists():
                continue  # Skip if the subfolder doesn't exist

            for file in room_dir.glob("*.pkl"):
                item_name = self.extract_item_name(file.name)
                if item_name:
                    completed_items[room].append(item_name)

        return completed_items

    ##
    # Test finding some object 100 times.
    # Store both the selected room and the selected object.
    ##
    def test_goal_finding(self, num_times = 100, object_to_find = "bottle of beer", str_expected_room = ""):
        results = []
        progress_pkl = "1.pkl"

        self.test_data_main_dir = self.data_store_dir + "/test_" + self.LLM_TYPE

        if (str_expected_room == ""):
            self.test_data_dir = self.test_data_main_dir
        else:
            # Let's check if we have any work already done and maybe we need to skip this object
            progress_pkl = self.test_data_main_dir + "/completed_items.pkl"
            if os.path.exists(progress_pkl):
                f = open(progress_pkl, 'rb')
                completed_items = pickle.load(f)
            else:
                completed_items = self.find_processed_items(self.test_data_main_dir)

            print("Completed items: ", completed_items)

            # If we have this item for this room type already, then skip
            if (object_to_find in completed_items[str_expected_room]):
                return

            self.test_data_dir = self.test_data_main_dir + "/" + str_expected_room
            expected_room = RoomType.parse_llm_response(str_expected_room, 0, False)

        if not os.path.exists(self.test_data_dir):
            os.makedirs(self.test_data_dir)

        self.results_fname = self.test_data_dir + "/" + str(num_times) + "_" + object_to_find.replace(" ", "_") + "_" + self.scene_id + ".pkl"

        for i in range(num_times):
            qry_start = time.time()
            (path, wanted_room, object_name, actual_object, full_text_room, full_text_obj) = self.bring_me_this_from_actual_objs(object_to_find)
            qry_time_taken = qry_start - time.time()

            # Did we get an actual object to navigate to?
            print("Object selected: ", object_name, " Object located: ", (actual_object is not None), " wanted_room: ", wanted_room)
            room_where_required_object_is = None
            end_point_room = None

            if actual_object is not None and path is not None:
                # what room is the location of the selected object?
                #print((actual_object['position'], actual_object['rotation']))
                convert_pose_set2tuple((actual_object['position'], actual_object['rotation']))[0]
                room_where_required_object_is = what_room_is_point_in_ground_truth(self.rooms, convert_pose_set2tuple(path[-1])[0])
                print("room_where_required_object_is: ", room_where_required_object_is)

            if path is not None:
                # We can only evaluate planned path if it exists
                end_point_room = what_room_is_point_in_ground_truth(self.rooms, convert_pose_set2tuple(path[-1])[0])
                print("END POINT IN: ", end_point_room, " WANTED: ", wanted_room)

                self.gen_path_img_fname = self.test_data_dir + "/" + str(i + 1) + "_of_" + str(
                    num_times) + "_" + object_to_find.replace(" ", "_") + "_" + self.scene_id + ".png"

                #self.visualise_path(path, self.gen_path_img_fname)

            # Now we can evaluate 5 error types.
            # TYPE1: No room was selected, from the available ones (usually due to LLM not giving an interpretable answer)
            # TYPE2: No object was selected from the available ones (usually due to LLM not giving an interpretable answer)
            # TYPE3: Object was selected, but it is in a different room than was expected (usually due to room mis-classification)
            # TYPE4: Path was generated, but it led to a different room than initially expected (usually due to path planning leading to a vicinity of the target, but in a different room)
            # TYPE5: LLM made its decisions and navigated to where it navigated, but it ended up not where we (humans) expected.
            score_for_this_run = 0
            if wanted_room is not None: # test for TYPE1 error
                score_for_this_run += 1
                if expected_room == wanted_room:  # test for TYPE5 error
                    score_for_this_run += 1
                if actual_object is not None: # test for TYPE2 error
                    score_for_this_run += 1
                    if room_where_required_object_is == wanted_room: # test for TYPE3 error
                        score_for_this_run += 1
                        if end_point_room == wanted_room: # test for TYPE4 error
                            score_for_this_run += 1

            print("score_for_this_run: ", score_for_this_run)

            result_dict = {
                "object_to_find": object_to_find,
                "expected_room": expected_room,
                "selected_room": wanted_room,
                "selected_object_name": object_name,
                "room_where_selected_object_is": room_where_required_object_is,
                "path_end_point_room": end_point_room,
                "score_for_this_run": score_for_this_run,
                "qry_time": qry_time_taken,
                "full_text_room": full_text_room,
                "full_text_obj": full_text_obj
            }

            results.append(result_dict)

        if (str_expected_room != ""):
            completed_items[str_expected_room].append(object_to_find)
            pickle.dump(completed_items, open(progress_pkl, "wb"))

        pickle.dump(results, open(self.results_fname, "wb"))
        return results

    ##
    # A way to tell if we're running in jupyter or not
    # If in jupyter, we might want to show matplotlibs,
    # but if in terminal, then we may want to save them.
    ##
    def is_running_in_jupyter(self):
        try:
            return get_ipython() is not None
        except ImportError:
            return False

if __name__ == "__main__":
    #spp = SemanticPathPlanner("train_55", LLMType.LLAMA, "CHAMELEON_p_cot_6lbl_img_middle")
    #spp = SemanticPathPlanner("train_55", LLMType.MISTRAL_4b, "CHAMELEON_p_cot_6lbl_img_middle")
    spp = SemanticPathPlanner("train_55", LLMType.GEMMA, "CHAMELEON_p_cot_6lbl_img_middle")
    #path = spp.bring_me_a_bottle_of_beer()
    #spp.test_goal_finding(2, "A fresh, cold, unopened bottle of beer")
    #spp.test_goal_finding(10, "pocket calculator")
    #spp.visualise_path(path)

    household_items = {
        "Kitchen": [
            "tea kettle", "food processor", "stand mixer", "rolling pin", "baking sheet",
            "pot holders", "oven mitts", "dish towels", "spice rack", "salt shaker",
            "pepper mill", "cutlery tray", "serving platter", "salad spinner", "ice cream scoop",
            "grater", "thermos", "lunch boxes", "water pitcher", "paper towel holder",
            "trash can", "dish soap", "sponge", "scrub brush", "dish drying mat"
        ],
        "Bathroom": [
            "toothbrush holder", "bath towel", "shower curtain",
            "bath mat", "toilet brush", "plunger", "medicine cabinet",
            "cotton swabs", "nail clippers", "razor", "shaving cream", "hair dryer",
            "curling iron", "makeup mirror", "tissue box", "laundry hamper",
            "facial cleanser", "body wash", "bathrobe", "slippers",
            "magnifying mirror", "electric toothbrush", "floss", "mouthwash", "first aid kit"
        ],
        "Living Room": [
            "sofa", "coffee table", "recliner", "floor lamp",
            "throw pillows", "blanket", "bookshelf", "entertainment center",
            "remote control", "candles", "picture frames", "vase", "clock",
            "ottoman", "side table", "fireplace tools", "chess set", "board games",
            "magazine rack", "curtains", "wall art",
            "DVD player", "gaming console", "throw blanket", "floor cushions"
        ],
        "Bedroom": [
            "nightstand", "dresser", "wardrobe",
            "alarm clock", "bedside lamp", "jewelry box", "full-length mirror", "laundry basket",
            "closet organizer", "shoe rack", "hangers", "duvet cover",
            "blackout curtains", "ceiling fan", "reading chair", "vanity table", "perfume bottles",
            "tissue box cover", "memory foam topper", "electric blanket", "sleep mask",
            "ear plugs", "white noise machine", "closet doors", "underbed storage"
        ]
    }

    for item in household_items["Kitchen"]:
        spp.test_goal_finding(100, item, "Kitchen")
    for item in household_items["Bathroom"]:
        spp.test_goal_finding(100, item, "Bathroom")
    for item in household_items["Living Room"]:
        spp.test_goal_finding(100, item, "Living Room")
    for item in household_items["Bedroom"]:
        spp.test_goal_finding(100, item, "Bedroom")
