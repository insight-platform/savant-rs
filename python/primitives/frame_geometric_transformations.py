from savant_rs.utils import gen_frame, VideoObjectBBoxTransformation

f = gen_frame()
obj = f.get_object(0)
obj.bbox_ref.xc = 25
obj.bbox_ref.yc = 50
obj.bbox_ref.height = 10
obj.bbox_ref.width = 20
print(obj)

transformations = [
    VideoObjectBBoxTransformation.scale(2.0, 2.0),
    VideoObjectBBoxTransformation.shift(10, 10)
]

f.transform_geometry(transformations)
obj = f.get_object(0)
print(obj)
