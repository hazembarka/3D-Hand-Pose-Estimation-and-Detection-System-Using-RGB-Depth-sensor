from django_project.Hand_pose_estimation_part.process import model_setup
from django_project.Object_detectin_part.hand_detection_RGBD import load_inference_graph
  
if __name__ == '__main__':

    ## loading hand detection model
    detection_graph, sess = load_inference_graph()
    dataset_input='nyu'

    #loading hand pose estimation model
    model=model_setup(dataset_input,'django_project\\Hand_pose_estimation_part\\model\\crossInfoNet_NYU_first_training_final.ckpt')
    
    #Running the estimation process, detection model is used in the pose estimation process to crop the area containing a hand
    model.sess_run( detection_graph = detection_graph,detection_sess= sess)
