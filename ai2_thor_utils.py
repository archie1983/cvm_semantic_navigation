##
# My own utilities functions for AI2-THOR. I couldn't find analogous functions in Thortils,
# so I wrote my own here. Eventually some of them should probably be pushed to Thortils
# project.
##
class AI2THORUtils:
    def __init__(self):
        pass

    ##
    # Extract visible objects from a collection of objects
    ##
    def get_visible_objects_from_collection(self, objects, print_objects = False):
        visible_objects = []

        for obj in objects:
            if obj['visible']:
                if print_objects:
                    print(obj['objectType'] + " : " + str(obj['position']))
                visible_objects.append(obj)

        return visible_objects

    ##
    # Extract visible objects (but only their names) from a collection of objects.
    # Skip duplicates.
    ##
    def get_visible_object_names_from_collection_set(self, objects):
        objs_at_this_pos = set()

        vis_objs = self.get_visible_objects_from_collection(objects)

        for obj in vis_objs:
            objs_at_this_pos.add(obj['objectType'])

        return objs_at_this_pos

    ##
    # Extract visible objects (but only their names) from a collection of objects.
    # Return as a comma separated list.
    ##
    def get_visible_object_names_from_collection_csv(self, objects):
        objs_at_this_pos = ""

        vis_objs = self.get_visible_objects_from_collection(objects)

        for obj in vis_objs:
            objs_at_this_pos += ", " + obj['objectType']

        if len(objs_at_this_pos) > 2:
            objs_at_this_pos = objs_at_this_pos[2:]

        return objs_at_this_pos

    ##
    # Extract visible objects (but only their names) from a collection of objects.
    # Return as a comma separated list. Skip duplicates.
    ##
    def get_visible_object_names_from_collection_csv_unique(self, objects):
        objs_at_this_pos = ""

        vis_objs = self.get_visible_object_names_from_collection_set(objects)

        for obj in vis_objs:
            objs_at_this_pos += ", " + obj

        if len(objs_at_this_pos) > 2:
            objs_at_this_pos = objs_at_this_pos[2:]

        return objs_at_this_pos
