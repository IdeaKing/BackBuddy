import cv2

from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlFile, ControlButton, ControlText, ControlSlider, ControlPlayer

from src.pose_keypoints.pipeline import keypoints_pipeline

class ComputerVisionAlgorithm(BaseWidget):

    def __init__(
        self, 
        keypoints_model, 
        classifier_model,
        *args, 
        **kwargs
    ):
        super().__init__('Computer vision algorithm example')

        #Definition of the forms fields
        self._videofile = ControlFile('Video')
        self._player = ControlPlayer('Player')
        self._camButton = ControlButton('Camera')
        self._videofile.changed_event = self.__videoFileSelectionEvent
        self._player.process_frame_event = self.__process_frame
        self._camButton.value = self.cameraOn
        #self._outputfile    = ControlText('Results output file')
        #self._threshold     = ControlSlider('Threshold', 114, 0,255)
        #self._blobsize      = ControlSlider('Minimum blob size', 100, 100,2000)
        #self._runbutton     = ControlButton('Run')
        #Define the function that will be called when a file is selected
        #Define the event that will be called when the run button is processed
        #self._runbutton.value       = self.__runEvent
        #Define the event called before showing the image in the player

        #Definition of attributes
        self.keypoints_model = keypoints_model
        self.classifier_model = classifier_model

        #Define the organization of the Form Controls
        self._formset = [
            ('_videofile', #'_outputfile'
            ),
            #'_threshold',
            (#'_blobsize',
             #'_runbutton'
             ),'_player', '_camButton'
        ]


    def __videoFileSelectionEvent(self):
        """
        When the videofile is selected instanciate the video in the player
        """
        self._player.value = self._videofile.value

    def __process_frame(self, frame):
        """
        Do some processing to the frame and return the result frame
        """
        return frame

    def __runEvent(self):
        """
        After setting the best parameters run the full algorithm
        """
        pass

    def cameraOn(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            
            # cv2.imshow('Input', frame)
            # c = cv2.waitKey(1)
            # if c == 27:
            #     break
        # cap.release()

    def process_camera_frame(self, frame):
        keypoints_pipeline(
            image=frame,
            model=self.keypoints_model,
            process_batch_size=1
        )

    def analysis(self, frame):
        """
        Full AI Analysis Function.
        Keypoints Detection -> Classification

        :params frame (numpy.array) image to be processed:
        :returns the classification, and image:
        """
        original_frame = frame
        original_frame_width, original_frame_height = frame.shape[1], frame.shape[0]
        
if __name__ == '__main__':

    from pyforms import start_app
    start_app(ComputerVisionAlgorithm)
