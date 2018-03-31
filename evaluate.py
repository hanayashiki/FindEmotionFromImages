import numpy as np
import os
from config import *
from single_predict import *

import test_module_VGG_top_and_hist
import test_module_vgg_no_top

table = {
    'amusement': 0,
    'anger': 1,
    'excitement': 2,
    'sadness': 3,
}

def evaluate(model):
    result = np.zeros((4, 4), dtype=np.int32)
    for root, dirs, files in os.walk(test_data_dir):
        for class_name in dirs:
            ans = table[class_name]
            for _, dirs2, files2 in os.walk(os.path.join(root, class_name)):
                for img in files2:
                    prediction = img_predict(model, os.path.join(_, img))
                    print(img)
                    res = np.argmax(prediction)
                    result[res, ans]+=1
    return result

model = test_module_vgg_no_top.getModel()
model.load_weights(test_module_vgg_no_top.checkpoints_dir + test_module_vgg_no_top.model_weight)
print(evaluate(model))