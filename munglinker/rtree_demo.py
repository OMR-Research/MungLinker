from typing import List, Dict

from muscima.cropobject import CropObject, cropobject_distance
from rtree import index
from rtree.index import Index

rtree_index = index.Index()
rtree_index.add(0, (0, 0, 10, 10), obj="first_object")
rtree_index.add(1, (5, 5, 15, 15), obj="second_object")
rtree_index.add(2, (25, 25, 35, 35), obj="third_object")
rtree_index.add(3, (35, 36, 37, 38), obj="fourth_object")
rtree_index.add(4, (40, 41, 42, 43), obj="fifth_object")
rtree_index.add(5, (37, 40, 50, 55), obj="sixth_object")

for nearest_object in rtree_index.nearest((20, 15), objects=True):
    print(nearest_object.object)

for intersecting_objects in rtree_index.intersection((2, 2, 12, 12), objects=True):
    print(intersecting_objects.object)

for nearest in rtree_index.nearest((0, 0, 10, 10), num_results=2, objects=True):
    print(nearest.object)


def get_close_objects2(cropobjects: List[CropObject], threshold=100) -> Dict[CropObject, List[CropObject]]:
    rtree_index = Index()
    nearest_cropobjects = {}
    for index, cropobject in enumerate(cropobjects):
        top, left, bottom, right = cropobject.bounding_box
        coordinates = (left, top, right, bottom)
        rtree_index.add(index, coordinates, obj=cropobject)

    for cropobject in cropobjects:
        top, left, bottom, right = cropobject.bounding_box
        coordinates = (left, top, right, bottom)
        nearest_samples = rtree_index.nearest(coordinates, num_results=40, objects=True)
        nearest_samples = [nearest_sample.object for nearest_sample in nearest_samples if
                           cropobject_distance(nearest_sample.object, cropobject) < threshold]
        nearest_cropobjects[cropobject] = nearest_samples

    return nearest_cropobjects
