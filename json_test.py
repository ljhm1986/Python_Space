import json

#test dictionary
Front_meta = {"net": 
	{"type": "[net]",
	 "batch": 1,
	 "subdivisions": 1,
 	"width": 608,
	 "height": 608,
	 "channels": 3,
	 "momentum": 0.9,
	 "decay": 0.0005,
	 "angle": 0,
	 "saturation": 1.5,
	 "exposure": 1.5,
	 "hue": 0.1,
	 "learning_rate": 0.001,
	 "burn_in": 1000,
 	"max_batches": 500200,
	 "policy": "steps",
	 "steps": "400000,450000",
	 "scales": ".1,.1"},
 "type": "[region]",
 "anchors": [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
 "bias_match": 1,
 "classes": 2,
 "coords": 4,
 "num": 5,
 "softmax": 1,
 "jitter": 0.3,
 "rescore": 1,
 "object_scale": 5,
 "noobject_scale": 1,
 "class_scale": 1,
 "coord_scale": 1,
 "absolute": 1,
 "thresh": 0.1,
 "random": 1,
 "model": "cfg/Front.cfg",
 "inp_size": [608, 608, 3],
 "out_size": [19, 19, 35],
 "name": "Front",
 "labels": ["car", "space"],
 "colors": [[254.0, 254.0, 254], [222.25, 190.5, 127]]
}

#json encoding
json_file1 = json.dumps(Front_meta, indent=4)

print(json_file1)