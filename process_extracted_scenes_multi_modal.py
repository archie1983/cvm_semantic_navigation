from scene_data_management import ClassificationMethod, SceneManagement
from ae_llm import LLMType
from room_type import RoomType
from ae_cvm import CVMType
from llm_room_classifier import LLMRoomClassifier # LLM room classifier
from room_classifier import RoomClassifier # SVC room classifier
from cvm_room_classifier import CVMRoomClassifier # CVM room classifier
from ModelType import ModelType
import pickle, os, torch, time
from ai2_thor_utils import AI2THORUtils
from scene_description import SceneDescription

class DataSceneProcessor:
    ##
    # We can override default data storage directory (normally- the name of LLM within experiment_data folder)
    ##
    def __init__(self, llm_type, cvm_type, classification_method_in, data_store_dir = "", prompt_mode = "p_cot_4lbl", new_llm_type = None):
        self.classification_method = classification_method_in

        if (self.classification_method.llm_required()):
            # if we want to re-classify with a new llm, then load room classifier with the new llm
            if new_llm_type is not None:
                self.lrc = LLMRoomClassifier(new_llm_type) # LLM classifier
            else:
                self.lrc = LLMRoomClassifier(llm_type)  # LLM classifier
        if (self.classification_method.svc_required()):
            self.src = RoomClassifier(False, ModelType.HYBRID_AI2_THOR) # SVC classifier
        if (self.classification_method.cvm_required()):
            self.crc = CVMRoomClassifier(cvm_type, prompt_mode)
        self.NUMBER_OF_SCENES_IN_BATCH = 25

        self.LLM_TYPE = llm_type.name
        if new_llm_type is not None:
            self.NEW_LLM_TYPE = new_llm_type.name
        self.CVM_TYPE = cvm_type.name
        self.atu = AI2THORUtils()
        self.prompt_mode = prompt_mode
        self.common_objs = {'Wall', 'Doorway', 'Window', 'Floor', 'Doorframe'}

        ##
        # We'll read pkl files from here - the ones that already have SVC or LLM classification stored
        ##
        if (data_store_dir == ""):
            self.data_store_dir = "experiment_data/" + "pkl_" + self.LLM_TYPE
        else:
            self.data_store_dir = "experiment_data/" + data_store_dir

        self.data_store_dir_newLLM = None
        if new_llm_type is not None:
            self.data_store_dir_newLLM = "experiment_data/" + "pkl_" + self.NEW_LLM_TYPE

        ## This is where we'll store new prepared pkl files - the ones that will also have CVM classification stored
        self.data_store_dir_cvm = "experiment_data/" + "pkl_" + self.CVM_TYPE

        if (self.classification_method.cvm_required()):
            self.scene_mgmt = SceneManagement(self.data_store_dir_cvm)
        else:
            if self.data_store_dir_newLLM is not None:
                self.scene_mgmt = SceneManagement(self.data_store_dir_newLLM)
            else:
                self.scene_mgmt = SceneManagement(self.data_store_dir)

        self.DEBUG = False # A flag of whether we want to debug and go through scenes quickly - only analyzing some points.

    def set_prompt_type(self, prompt_type):
        if (self.classification_method.llm_required()):
            #self.lrc.set_prompt_type(prompt_type)
            pass
        if (self.classification_method.svc_required()):
            #self.crc.set_prompt_type(prompt_type)
            pass
        if (self.classification_method.cvm_required()):
            self.crc.set_prompt_type(prompt_type)

        self.prompt_mode = prompt_type

    def store_processed_data(self):
        if (self.classification_method.cvm_required()):
            os.rename(self.data_store_dir_cvm, self.data_store_dir_cvm + "_" + self.prompt_mode)

    ##
    # Loads a single specified scene file from the data directory
    ##
    def load_scene_file(self, scene_id):
        scene_f = self.data_store_dir + "/scene_descr_" + str(scene_id) + ".pkl"
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

    def store_scene_file(self, scene_id, sd):
        # store our room points collection into a pickle file
        # Create the directory where to store experiment data if it doesn't exist
        if (self.classification_method.cvm_required()):
            dir_name = self.data_store_dir_cvm
        else:
            if self.data_store_dir_newLLM is not None:
                dir_name = self.data_store_dir_newLLM
            else:
                dir_name = self.data_store_dir


        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        scene_descr_fname = dir_name + "/scene_descr_" + str(scene_id) + ".pkl"
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
        highest_scene_index = self.scene_mgmt.last_index_extracted()
        print("Highest index processed: " + str(highest_scene_index))
        processed_scenes_in_this_batch = self.scene_mgmt.get_num_extracted_scenes()

        while processed_scenes_in_this_batch < self.NUMBER_OF_SCENES_IN_BATCH:
            scene_id = "train_" + str(highest_scene_index + 1)
            print("Processing " + scene_id, " ", processed_scenes_in_this_batch, " ", scene_id)
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

        new_updated_sd = SceneDescription() # scene description - as before but now also classified with CVM or otherwise updated

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

            # at first assume that we can use object list to classify
            classify_using_object_list = True
            classify_using_image = False # use image only when object list is missing

            # No point to classify this point if there are no objects
            # We may not want to classify it using SVC or LLM, but a visual
            # CVM might still be able to classify it.
            if (len(objs_at_this_pos) < 1):
                print("Empty set of objects -- not skipping this time, but classifying with image only")
                classify_using_object_list = False
                classify_using_image = True
            # Similarly with common objects only- SVC and LLM can't help here,
            # but CVM might be able to, so continue, just don't use LLM or SVC.
            #print(objs_at_this_pos)
            objs_at_this_pos = [obj.strip() for obj in objs_at_this_pos.split(",")]
            objs_at_this_pos = set(objs_at_this_pos)
            #objs_at_this_pos = {obj.strip() for obj in objs_at_this_pos}
            if (objs_at_this_pos.issubset(self.common_objs)):
                print("Only common objects -- not skipping this time, but classifying with image too")
                classify_using_object_list = False
                classify_using_image = True

            # initialise result variables
            rt_llm = RoomType.NOT_CLASSIFIED
            rt_svc = RoomType.NOT_CLASSIFIED
            rt_cvm = RoomType.NOT_CLASSIFIED
            svc_elapsed_time = 0
            llm_elapsed_time = 0
            cvm_elapsed_time = 0
            llm_text = ""
            cvm_text = ""

            # classify using the appropriate methods
            if (self.classification_method.llm_required()):
                t0 = time.time()
                
                if classify_using_image and classify_using_object_list:
                    rt_llm, llm_text = self.lrc.classify_room_by_this_object_set_and_pic(obj_set=objs_at_this_pos, img_uri=img_url)
                elif classify_using_object_list:
                    rt_llm, llm_text = self.lrc.classify_room_by_this_object_set_and_pic(obj_set=objs_at_this_pos, img_uri=None)
                else:
                    rt_llm, llm_text = self.lrc.classify_room_by_this_object_set_and_pic(obj_set=None, img_uri=img_url)

                llm_elapsed_time = round(time.time() - t0, 5)
                print("Room Type new LLM: " + rt_llm.name, " llm predict time: ", llm_elapsed_time, " s")

            if (self.classification_method.svc_required() and classify_using_object_list):
                t0 = time.time()
                rt_svc = self.src.classify_room_by_this_object_set(objs_at_this_pos)
                svc_elapsed_time = round(time.time() - t0, 5)
                #print("svc predict time:", svc_elapsed_time, "s")

            if (self.classification_method.cvm_required()):
                # Now let's try to analyze the pictures with a CVM and see what we would classify them as and
                # what items do we see in each of them.
                (rt_cvm, cvm_elapsed_time, cvm_text) = self.crc.classify_room_by_this_image(img_url)
                print("Room Type CVM: " + rt_cvm.name)

            ## Let's not do items in the picture inference yet- I'm not yet sure how to parse the item list.
            items_in_image = ""
            #items_in_image = self.crc.extract_items_from_this_image(img_url)
            #print("Items in image: " + items_in_image)
            #print("\n")

            # # Now take the new data, where we have it or preserve the old data
            # if rt_llm == RoomType.NOT_CLASSIFIED:
            #     rt_llm = point["room_type_llm"]
            #     llm_elapsed_time = point["elapsed_time_llm"]
            #
            # if rt_svc == RoomType.NOT_CLASSIFIED:
            #     rt_svc = point["room_type_svc"]
            #     svc_elapsed_time = point["elapsed_time_svc"]

            new_updated_sd.addPoint(point["point_pose"],
                                    rt_llm,
                                    rt_svc,
                                    rt_cvm,
                                    point["room_type_gt"],
                                    point["visible_objects_at_this_point"],
                                    items_in_image,
                                    point["front_view_at_this_point"],
                                    llm_elapsed_time,
                                    svc_elapsed_time,
                                    cvm_elapsed_time,
                                    llm_text,
                                    cvm_text)

            if self.DEBUG and points_cnt >= 3:
                break

        self.store_scene_file(scene_id, new_updated_sd)

    def _recovery_procedure(self, attempt):
        """Multi-stage recovery for severe CUDA errors"""
        time.sleep(1)  # Cool-down period
        
        # Stage 1: Basic cleanup
        try:
            torch.cuda.empty_cache()
        except RuntimeError:
            pass
        
        # Stage 2: More aggressive reset
        if attempt > 0:
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
        
        # Stage 3: Nuclear option
        if attempt > 1:
            self._hard_reset_cuda()

    def _hard_reset_cuda(self):
        """Last-resort CUDA recovery"""
        print("Performing hard CUDA reset")
        try:
            # Try proper cleanup first
            torch.cuda.empty_cache()
            # Reset Python-side state
            torch.cuda.init()
        except:
            # If all else fails, suggest environment-level reset
            print("""
            CRITICAL: CUDA requires full environment reset.
            Suggestions:
            1. Restart Python process
            2. Run 'nvidia-smi --gpu-reset -i [gpu_id]'
            3. Reboot machine if persistent
            """)
            raise RuntimeError("CUDA device needs hard reset")

if __name__ == "__main__":
    #dse = DataSceneExtractor(LLMType.LLAMA, CVMType.MOONDREAM, ClassificationMethod.SVC_CVM)
    #dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.MOONDREAM, ClassificationMethod.CVM, "data_collection") # 

    #dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_cot_4lbl_img_middle_nf4") # 
    #dsp.process_1_batch_of_data_scenes()
    #dsp.store_processed_data()

    #dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_cot_6lbl_img_middle_nf4") # 
    #dsp.process_1_batch_of_data_scenes()
    #dsp.store_processed_data()

#    dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_cot_0lbl_img_middle_nf4") # 
#    dsp.process_1_batch_of_data_scenes()
#    dsp.store_processed_data()

#    dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_nocot_4lbl_img_middle_nf4") # 
#    dsp.process_1_batch_of_data_scenes()
#    dsp.store_processed_data()

    #dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_nocot_6lbl_img_middle_nf4") # 
#    dsp.set_prompt_type("p_nocot_6lbl_img_middle_nf4")
#    dsp.process_1_batch_of_data_scenes()
#    dsp.store_processed_data()

#    dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_nocot_0lbl_nf4") # 
#    dsp.set_prompt_type("p_nocot_0lbl_nf4")
#    dsp.process_1_batch_of_data_scenes()
#    dsp.store_processed_data()

    runtime_err_count = 0
#    attempt = 0
#    while True:
#        try:
#            dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_cot_4lbl_img_end_nf4") # 
#            dsp.process_1_batch_of_data_scenes()
#            dsp.store_processed_data()
#        except RuntimeError as e:
#            runtime_err_count += 1
#            attempt += 1
#            dsp._recovery_procedure(attempt)
#            continue
#        break

#    print("Runtime Error happened: ", runtime_err_count, " times.")

#    attempt = 0
#    while True:
#        try:
#            dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_cot_6lbl_img_end_nf4") 
#            #dsp.set_prompt_type("p_cot_6lbl_img_end_nf4")
#            dsp.process_1_batch_of_data_scenes()
#            dsp.store_processed_data()
#        except RuntimeError as e:
#            runtime_err_count += 1
#            attempt += 1
#            dsp._recovery_procedure(attempt)
##            dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_cot_6lbl_img_end_nf4")
#            continue
#        break
#
#    print("Runtime Error happened: ", runtime_err_count, " times.")

#    attempt = 0
#    while True:
#        try:
#            dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_cot_0lbl_img_end_nf4")
#            dsp.set_prompt_type("p_cot_0lbl_img_end_nf4")
#            dsp.process_1_batch_of_data_scenes()
#            dsp.store_processed_data()
#        except RuntimeError as e:
#            runtime_err_count += 1
#            attempt += 1
#            dsp._recovery_procedure(attempt)
# #           dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_cot_0lbl_img_end_nf4")
#            continue
#        break

#    print("Runtime Error happened: ", runtime_err_count, " times.")

#    attempt = 0
#    while True:
#        try:
#            dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_nocot_4lbl_img_end_nf4")
#            dsp.set_prompt_type("p_nocot_4lbl_img_end_nf4")
#            dsp.process_1_batch_of_data_scenes()
#            dsp.store_processed_data()
#        except RuntimeError as e:
#            runtime_err_count += 1
#            attempt += 1
#            dsp._recovery_procedure(attempt)
#            dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_nocot_4lbl_img_end_nf4")
#            continue
#        break

    print("Runtime Error happened: ", runtime_err_count, " times.")

    attempt = 0
    while True:
        try:
            #dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_nocot_6lbl_img_end_nf4")
            #dsp.set_prompt_type("p_nocot_6lbl_img_end_nf4")
            dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.LLM, "data_collection",
                                     prompt_mode="na", new_llm_type=LLMType.MINISTRAL_3_3b_instruct_nf4_bnb)

            dsp.process_1_batch_of_data_scenes()
            dsp.store_processed_data()
        except RuntimeError as e:
            runtime_err_count += 1
            attempt += 1
            dsp._recovery_procedure(attempt)
#            dsp = DataSceneProcessor(LLMType.LLAMA, CVMType.CHAMELEON, ClassificationMethod.CVM, "data_collection", "p_nocot_6lbl_img_end_nf4")
            continue
        break

    print("Runtime Error happened: ", runtime_err_count, " times.")
