import os
import pickle
import prior

from llm_room_classifier import LLMRoomClassifier, LLMType
from room_type import RoomType
from scene_description import SceneDescription, ClassifierType

from thortils import (launch_controller,
                      convert_scene_to_grid_map, proper_convert_scene_to_grid_map, proper_convert_scene_to_grid_map_and_poses)

from thortils.navigation import get_shortest_path_to_object_type, get_shortest_path_to_object
from thortils.agent import thor_reachable_positions, thor_agent_position, thor_agent_pose
from thortils.utils import roundany
from thortils.controller import _resolve
from thortils.object import thor_closest_object_of_type
from ae_robot_simulation_control import RobotNavigationControl

import matplotlib.pyplot as plt
from PIL import Image
import copy

##
# This class will analyze harvested scene data and plot a semantic map from the
# already classified points
##
class SemanticMapper:
    def __init__(self, scene_id, llm_type):
        self.data_store_dir = "experiment_data"
        self.LLM_TYPE = llm_type.name

        scene_descr_fname = self.data_store_dir + "/pkl_" + self.LLM_TYPE + "/scene_descr_" + scene_id + ".pkl"
        if os.path.isfile(scene_descr_fname):
            file = open(scene_descr_fname,'rb')
            self.scene_description = pickle.load(file)
            file.close()

            print("Loaded : " + scene_descr_fname + " scene")
        else:
            # if no scenes' data, then nothing to do
            raise Exception("No scenes data file found. Nothing to do.")

        self.scene_id = scene_id
        #self.lrc = LLMRoomClassifier(llm_type)
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

        self.controller = launch_controller({"scene": house, "VISIBILITY_DISTANCE": 3.0})

        self.rnc.set_controller(self.controller)

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
    def visualise_path(self, path):
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
        plt.show()

    def get_controller(self):
        return self.controller

if __name__ == "__main__":
    spp = SemanticMapper("train_55", LLMType.LLAMA)
    spp.get_top_down_frame()
