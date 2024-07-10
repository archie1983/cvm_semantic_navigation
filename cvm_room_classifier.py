import pickle
import os.path

from time import time
import glob

from ae_cvm import CVMControl, CVMType
from room_type import RoomType

##
# Similar to LLMRoomClassifier, but for CVM - Computer Vision Model, e.g. moondream2.
# A class that is sort of a middle man between the CVM that we will use for
# classifying rooms and the data that is being provided.
##
class CVMRoomClassifier:
  def __init__(self):
    self.data_counter = 0
    self.false_cnt = 0
    self.true_cnt = 0

    self.stored_labels_loaded = False

    self.glc = CVMControl(CVMType.MOONDREAM)

  def __init__(self, cvm_type):
    self.data_counter = 0
    self.false_cnt = 0
    self.true_cnt = 0

    self.stored_labels_loaded = False

    self.glc = CVMControl(cvm_type)

    self.piclist = []

    #########################################################

  def load_stored_data(self):
    if (not self.stored_labels_loaded):
        self.stored_labels_loaded = True
        # look into the pictures directory and go through them and then
        # load ready for processing.
        png_files_glob = "pictures_first_view/*.png"

        scene_files = glob.glob(png_files_glob) # scene files

        for file_name in scene_files:
            self.piclist.append(file_name)

  ##
  # Classify room by a given image
  ##
  def classify_room_by_this_image(self, img_url):
      self.glc.initialise_for_ai2_thor_room_classification()
      #print("Analyzing: " + img_url)
      ans = self.glc.classify_room(img_url)

      return ans

  ##
  # Extract visible items from a given image
  ##
  def extract_items_from_this_image(self, img_url):
      #print("Analyzing: " + img_url)
      ans = self.glc.extract_visible_items(img_url)

      return ans

  ##
  # Allows us asking the LLM where to find a given object
  ##
  def where_to_find_this(self, object_name):
      self.glc.construct_room_selector_question(object_name)
      ans = self.glc.get_answer()

      return ans

  def where_to_look_first(self, what_to_look_for, where_to_look):
      objs_to_look_near = ""
      for obj in where_to_look:
          objs_to_look_near += obj + ", "

      objs_to_look_near = objs_to_look_near[:-2]

      self.glc.construct_object_selector_question(what_to_look_for, objs_to_look_near)
      ans = self.glc.get_object_selector_answer(where_to_look)

      print("ANS: " + ans)
      return ans

  def test_classification_on_stored_data(self):
      self.load_stored_data()
      self.glc.initialise_for_ai2_thor_room_classification()
      for i in range(len(self.piclist)):

          # limiting the run to 3 images
          if (i > 2):
              break
          label = self.piclist[i].split("/")[-1].split("_")[-2]
          print("Analyzing: " + self.piclist[i] + " which is " + label + " ## " + RoomType.interpret_label(label).name)
          ans = self.glc.classify_room(self.piclist[i])
          #ans = "aaa"
          print("\n" + str(i) + ") ANS: " + ans.name + " ## " + str(ans.value) + " # " + " @@ " + str(RoomType.interpret_label(label) == ans))

          if (RoomType.interpret_label(label) == ans):
              self.true_cnt += 1
          else:
              self.false_cnt += 1

      print("TRUE CNT: " + str(self.true_cnt) + " :: False CNT: " + str(self.false_cnt))

if __name__ == "__main__":
    rc = CVMRoomClassifier(CVMType.MOONDREAM)
    rc.test_classification_on_stored_data()
