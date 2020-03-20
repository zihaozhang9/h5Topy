import cv2
import numpy as np
import tensorflow as tf

class c_PbToSess(object):
    def __init__(self,PbPath,s_input,s_output):
        self.b_graph = tf.Graph()
        self.b_sess = tf.Session(graph = self.b_graph)
        self.b_GraphDef = tf.GraphDef()
        self.l_tensorShape = []
        with self.b_sess.as_default():
            with self.b_sess.graph.as_default():
                f = open(PbPath, "rb")
                self.b_GraphDef.ParseFromString(f.read())
                tf.import_graph_def(self.b_GraphDef, name="")
                init = tf.global_variables_initializer()
                self.b_sess.run(init)
                self.b_input = self.b_sess.graph.get_tensor_by_name(s_input)
                self.b_output = self.b_sess.graph.get_tensor_by_name(s_output)
    def fun_tensor(self,cv_image,t_imgShape):
        cv_image = cv2.resize(cv_image,t_imgShape)
        img_tensor = cv2.cvtColor(cv_image,cv2.COLOR_BGR2RGB)
        img_tensor = img_tensor.astype(np.float32)
        img_tensor /= 255.
        self.l_tensorShape = [-1 , t_imgShape[0] , t_imgShape[1] , 3]
        return img_tensor
    def fun_predict(self,n_tensor):        
        b_tensorOut = self.b_sess.run(self.b_output, feed_dict={self.b_input:np.reshape(n_tensor, self.l_tensorShape)})
        return b_tensorOut
        
Model_pb='keras_model.pb'
img_path = '000027.jpg'

Model_sess = c_PbToSess(Model_pb,"conv2d_1_input:0","dense_2/Sigmoid:0")
cv_image = cv2.imread(img_path)
img_tensor = Model_sess.fun_tensor(cv_image,(112,112))
img_out_softmax = Model_sess.fun_predict(img_tensor)
prediction_labels = np.argmax(img_out_softmax, axis=1)
#class_pre = Car_Color_map[prediction_labels[0]]
print(prediction_labels)