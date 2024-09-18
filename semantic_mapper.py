import os
import pickle
import prior

from llm_room_classifier import LLMRoomClassifier, LLMType
from room_type import RoomType
from ml_model_type import MLModelType
from classifier_fusion_type import ClassifierFusionType
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
    def __init__(self, scene_id, llm_type, additional_data_store_param = ""):
        self.data_store_dir = "experiment_data"
        self.LLM_TYPE = llm_type
        self.additional_data_store_param = additional_data_store_param

        #self.lrc = LLMRoomClassifier(llm_type)
        self.dataset = None
        self.controller = None
        self.rnc = RobotNavigationControl()

        self.scene_id = scene_id

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

        # Now figure out file names to load for the results
        if self.LLM_TYPE.type_of_model() == MLModelType.LLM:
            scene_descr_fname = self.data_store_dir + "/pkl_" + self.LLM_TYPE.name + "/scene_descr_" + scene_id + ".pkl"
            cor_llm_pkl_path = ""
        elif self.LLM_TYPE.type_of_model() == MLModelType.CVM:
            scene_descr_fname = self.data_store_dir + "/pkl_" + self.LLM_TYPE.name + self.additional_data_store_param + "/scene_descr_" + scene_id + ".pkl"

            # when we're looking at CVM results, we also want to load LLM results so that we can fuse them if desired
            cor_llm_pkl_path = self.data_store_dir + "/pkl_LLAMA/scene_descr_" + scene_id + ".pkl" # corresponding LLM pkl path

        # Now load results of the CVM or LLM classification
        if os.path.isfile(scene_descr_fname):
            file = open(scene_descr_fname,'rb')
            self.scene_description = pickle.load(file)
            file.close()
            print("Loaded : " + scene_descr_fname + " scene")

            # If we loaded CVM results, then we should also have a corresponding LLM results to load
            if os.path.isfile(cor_llm_pkl_path):
                llm_f = open(cor_llm_pkl_path,'rb')
                self.scene_description_llm = pickle.load(llm_f)
                llm_f.close()
                print("Loaded : " + cor_llm_pkl_path + " scene")
        else:
            # if no scenes' data, then nothing to do
            raise Exception("No scenes data file found. Nothing to do.")

        # Now load AI2-THOR scene
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

    def get_controller(self):
        return self.controller

    ##
    # Find a room point in a collection of the points that match the given pose
    ##
    def find_observed_point_by_pose(self, pose, room_points):
        #(pos, rot) = pose # ((10.75, 1.57599937915802, 1.0), (30.000003814697266, 0.0, 0))
        result = None
        for rp in room_points:
            if rp['point_pose'] == pose:
                result = rp
                return result
        return result

    ##
    # Extract all rotations with the given XY position in all room points.
    # It also extracts the classified result and sorts the extracted rotations.
    # @fusion_type - Strategy of how we want to fuse CVM classification results with LLM ones. This is only valid when we
    ##
    def get_all_rotations_of_xy_pose(self, xy_pose, room_points, fusion_type = ClassifierFusionType.NO_FUSION):
        result = []
        room_points = self.scene_description.get_all_points() # get CVM points
        if self.LLM_TYPE.type_of_model() == MLModelType.CVM:
            llm_room_points = self.scene_description_llm.get_all_points() # get corresponding LLM points

        for rp in room_points: # go through all points
            if rp['point_pose'][0] == xy_pose:
                if self.LLM_TYPE.type_of_model() == MLModelType.LLM:
                    result.append((int(rp['point_pose'][1][1]), rp["room_type_llm"])) # and extract yaw rotations along with classification result from the required XY position
                elif self.LLM_TYPE.type_of_model() == MLModelType.CVM:
                    llm_rp = self.find_observed_point_by_pose(rp['point_pose'], llm_room_points)
                    # understand what did LLM classify for this point
                    if llm_rp is None:
                        llm_rp_room_type = RoomType.NOT_CLASSIFIED
                    else:
                        llm_rp_room_type = llm_rp["room_type_llm"]

                    clas_result = rp["room_type_cvm"]

                    if (fusion_type == ClassifierFusionType.LLM_THEN_CVM): # if we want to combine LLM results with CVM - when LLM hasn't got a clue, use CVM result
                        if (llm_rp_room_type != RoomType.NOT_CLASSIFIED and llm_rp_room_type != RoomType.NOT_KNOWN):
                            clas_result = llm_rp_room_type
                    elif (fusion_type == ClassifierFusionType.CVM_THEN_LLM): # if we want to combine LLM results with CVM - when CVM doesn't know, use LLM result
                        if (rp["room_type_cvm"] == RoomType.NOT_CLASSIFIED or rp["room_type_cvm"] == RoomType.NOT_KNOWN):
                            clas_result = llm_rp_room_type

                    result.append((int(rp['point_pose'][1][1]), clas_result)) # and extract yaw rotations along with classification result from the required XY position

        result = self.pad_missing_rotations(result)

        result = sorted(result, key=lambda x: x[0], reverse=True) # sort the result by yaw rotation angle, but we need id reversed because of the way pie plotting works
        return result

    ##
    # Some rotations will not be classified because the object list at that pose contained only common objects or was empty.
    # The classifications will be missing, so let's fill them back in.
    ##
    def pad_missing_rotations(self, classified_rotations):
        padded_classified_rotations = []
        pad_needed = False

        for i in range(0, 360, 45): # there should be degrees of 0, 45, 90, 135, 180, 225, 270, 315. If there ain't, then we pad
            pad_needed = True
            for cr in classified_rotations:
                if cr[0] == i: # if rotation found, then skip the rest
                    pad_needed = False
                    padded_classified_rotations.append(cr)
                    break
            if (pad_needed): # if rotation was not found, then we need to inject it
                padded_classified_rotations.append((i, RoomType.NOT_CLASSIFIED))

        return padded_classified_rotations

    ##
    # Go through all of the room points and process all rotations for each point
    # making it ready to plot on the top down frame. This will return
    # classified_pose_rtns_pairs, which will be used in other functions.
    # @fusion_type - Strategy of how we want to fuse CVM classification results with LLM ones.
    ##
    def prepare_classified_poses_for_processing(self, fusion_type = ClassifierFusionType.NO_FUSION):
        room_points = self.scene_description.get_all_points()
        classified_pose_rtns_pairs = []
        processed_xy_poses = []

        # Go through all of the room points and process all rotations for each point
        for rp in room_points:
            (current_xy, current_rot) = rp['point_pose'] # get current XY position
            if (current_xy in processed_xy_poses): continue
            #print(rp['point_pose'][0])
            classified_rotations = self.get_all_rotations_of_xy_pose(current_xy, room_points, fusion_type)
            #print(classified_rotations) # get all rotations for the given XY position
            #self.plot_semantic_position(classified_rotations)
            processed_xy_poses.append(current_xy) # append the XY position to the collection of positions that we don't want to see anymore
            classified_pose_rtns_pairs.append((current_xy, classified_rotations, rp["room_type_gt"]))

        return classified_pose_rtns_pairs

    ##
    # Create the pie plot of a semantic position with all the segments coloured
    # according to the classifictation.
    ##
    def plot_semantic_position(self, position, sorted_classified_rotations, ax):
        segment_size = 100.0/8.0 # we have 8 directions of view and 100% to cover
        colors = []

        # assemble the colours for the observed classifications
        for rtn in sorted_classified_rotations:
            #print(rtn)
            colors.append(RoomType.colour_of_room(rtn[1]))

        # all pie segments will be of the same size, just different colours
        y = [segment_size, segment_size, segment_size, segment_size, segment_size, segment_size, segment_size, segment_size]
        #plt.pie(y, colors=colors, radius=0.1)
        ax.pie(y, center=(position[0],position[2]), radius=0.7,colors=colors, wedgeprops={'clip_on':True}, frame=False, startangle = (180 - (360.0/8.0)/2.0))
        #plt.show()

    ##
    # Plot a path on the top-down view of the habitat
    ##
    def visualise_map(self, classified_positions, show_directions=True):
        #grid_size = self.controller.initialization_parameters["gridSize"]
        grid_size = self.controller.initialization_parameters["gridSize"]

        reachable_positions = [
            tuple(map(lambda x: roundany(x, grid_size), pos))
            #for pos in thor_reachable_positions(self.controller)]
            for pos in thor_reachable_positions(self.controller)]

        x_max = max([pos[0] for pos in reachable_positions])
        z_max = max([pos[1] for pos in reachable_positions])
        x_min = min([pos[0] for pos in reachable_positions])
        z_min = min([pos[1] for pos in reachable_positions])

        fig, ax = plt.subplots()

        # another way how to plot the path
        #x = [p[0]["x"] for p in path]
        #z = [p[0]["z"] for p in path]
        #ax.scatter(x, z, s=300, c='gray', zorder=1)

        # setting up for the top-down picture of the habitat
        #print(str(x_min-grid_size) + " " + str(x_max+grid_size) + " " + str(z_min-grid_size) + " " + str(z_max+grid_size))
        #img = self.get_top_down_frame()
        img = self.get_top_down_frame()
        ex_mul = 7
        ax.imshow(img, extent=[x_min-ex_mul*grid_size, x_max+ex_mul*grid_size, z_min-ex_mul*grid_size, z_max+ex_mul*grid_size])

        # set up for the path print
        lim_mul = 4
        ax.set_xlim(x_min-lim_mul*grid_size, x_max+lim_mul*grid_size)
        ax.set_ylim(z_min-lim_mul*grid_size, z_max+lim_mul*grid_size)

        if show_directions: # display pie charts with the classified directions
            y_init = ax.get_ylim()
            x_init = ax.get_xlim()

            i = 0
            # map
            for pos_rtns in classified_positions:
                x = pos_rtns[0][0]
                z = pos_rtns[0][2]
                #ax.scatter([x], [z], s=30, zorder=2, c="blue")
                self.plot_semantic_position(pos_rtns[0], pos_rtns[1], ax)
                i+=1
                #if i == 3: break

            ax.set_ylim(y_init)
            ax.set_xlim(x_init)
        else: # display a ground-truth map with the dots corresponding to the correct ground truth room colour
            # map
            for pos_rtns in classified_positions:
                x = pos_rtns[0][0]
                z = pos_rtns[0][2]
                colour = RoomType.colour_of_room(pos_rtns[2])
                ax.scatter([x], [z], s=30, zorder=2, c=colour)
                #plot_semantic_position(pos_rtns[0], pos_rtns[1], ax)

        plt.axis('off')
        plt.show()

    ##
    # @fusion_type - Strategy of how we want to fuse CVM classification results with LLM ones.
    ##
    def display_semantic_map(self, fusion_type = ClassifierFusionType.NO_FUSION):
        classified_poses = self.prepare_classified_poses_for_processing(fusion_type)
        self.visualise_map(classified_poses, False)
        self.visualise_map(classified_poses, True)


if __name__ == "__main__":
    spp = SemanticMapper("train_55", LLMType.LLAMA)
    spp.get_top_down_frame()
    spp.display_semantic_map(ClassifierFusionType.NO_FUSION)
