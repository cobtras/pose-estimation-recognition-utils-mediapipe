import cv2
import mediapipe as mp

from .MediaPipePoseNames import MediaPipePoseNames
from pose_estimation_recognition_utils import SkeletonData, SkeletonDataPoint, SkeletonDataPointWithName


class HumanSkeletonExtrator:
    def __init__(self, mode, confidence_value):
        allowed_modes = {"holistic", "pose", "hand"}
        if mode not in allowed_modes:
            raise ValueError(f"Invalid mode: {mode}. Allowed values are: {allowed_modes}")
        self.mode = mode
        self.pose_names = MediaPipePoseNames()

        if mode == "holistic":
            mp_holistic = mp.solutions.holistic

            self.model = mp_holistic.Holistic(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                refine_face_landmarks=True,
                min_detection_confidence=confidence_value,
                min_tracking_confidence=confidence_value
            )
        elif mode == "pose":
            mp_pose = mp.solutions.pose

            self.model = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                enable_segmentation=True,
                min_detection_confidence=confidence_value,
                min_tracking_confidence=confidence_value
            )
        elif mode == "hand":
            mp_hands = mp.solutions.hands

            self.model = mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=confidence_value,
                min_tracking_confidence=confidence_value
            )

    def extract_skeleton_object_from_frame(self, frame, frame_id=0, with_names=False):

        try:
            skeleton_data, right_hand_data, left_hand_data, head_data = self.extract_skeleton(frame)

            skeleton_data_object = self.get_skeleton_data_object(skeleton_data, right_hand_data, left_hand_data,
                                                                 head_data, frame_id, with_names)

            return skeleton_data_object

        except:
            return None

    def extract_skeleton_from_frame(self, frame, frame_id=0, with_names=False):

        try:
            return self.extract_skeleton_object_from_frame(frame, frame_id, with_names).to_json()
        except:
            return None

    def extract_skeleton(self, frame):

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.model.process(frame_rgb)

        if self.mode == "holistic":
            # noinspection PyUnresolvedReferences
            return (results.pose_landmarks,
                    results.right_hand_landmarks,
                    results.left_hand_landmarks,
                    results.face_landmarks)
        elif self.mode == "pose":
            # noinspection PyUnresolvedReferences
            return (results.pose_landmarks,
                    None, None, None)
        elif self.mode == "hand":
            # noinspection PyUnresolvedReferences
            return (None,
                    results.right_hand_landmarks,
                    results.left_hand_landmarks,
                    None)

    def get_skeleton_data_object(self, skeleton_data_pose, skeleton_data_right_hand, skeleton_data_left_hand,
                                 skeleton_data_head, frame, with_name):

        skeleton_data_obj = SkeletonData(frame)

        if skeleton_data_pose is not None:
            for idx, landmark in enumerate(skeleton_data_pose.landmark):
                if with_name:
                    landmark_name = self.pose_names.landmark_names_pose().get(idx, f"Landmark {idx}")
                    skeleton_data_obj.add_data_point(SkeletonDataPointWithName(idx, landmark_name, landmark.x, landmark.y,
                                                                               landmark.z))
                else:
                    skeleton_data_obj.add_data_point(SkeletonDataPoint(idx, landmark.x, landmark.y, landmark.z))

        if skeleton_data_right_hand is not None:
            try:
                for idx, landmark in enumerate(skeleton_data_right_hand.landmark):
                    if with_name:
                        landmark_name = self.pose_names.landmark_names_hand_right().get(idx, f"Landmark {idx}")
                        skeleton_data_obj.add_data_point(
                            SkeletonDataPointWithName((idx + 100), landmark_name, landmark.x, landmark.y, landmark.z))
                    else:
                        skeleton_data_obj.add_data_point(SkeletonDataPoint((idx + 100), landmark.x, landmark.y,
                                                                           landmark.z))
            except:
                for i in range(len(self.pose_names.landmark_names_hand_right)):
                    if with_name:
                        landmark_name = self.pose_names.landmark_names_hand_right().get(i, f"Landmark {i}")
                        skeleton_data_obj.add_data_point(
                            SkeletonDataPointWithName((i + 100), landmark_name, 0, 0, 0))
                    else:
                        skeleton_data_obj.add_data_point(SkeletonDataPoint((i + 100), 0, 0, 0))

        if skeleton_data_left_hand is not None:
            try:
                for idx, landmark in enumerate(skeleton_data_left_hand.landmark):
                    if with_name:
                        landmark_name = self.pose_names.landmark_names_hand_left().get(idx, f"Landmark {idx}")
                        skeleton_data_obj.add_data_point(
                            SkeletonDataPointWithName((idx + 200), landmark_name, landmark.x, landmark.y, landmark.z))
                    else:
                        skeleton_data_obj.add_data_point(SkeletonDataPoint((idx + 200), landmark.x, landmark.y,
                                                                           landmark.z))
            except:
                for i in range(len(self.pose_names.landmark_names_hand_left)):
                    if with_name:
                        landmark_name = self.pose_names.landmark_names_hand_left().get(i, f"Landmark {i}")
                        skeleton_data_obj.add_data_point(
                            SkeletonDataPointWithName((i + 200), landmark_name, 0, 0, 0))
                    else:
                        skeleton_data_obj.add_data_point(SkeletonDataPoint((i + 200), 0, 0, 0))

        if skeleton_data_head is not None:
            try:
                for idx, landmark in enumerate(skeleton_data_head.landmark):
                    if with_name:
                        skeleton_data_obj.add_data_point(
                            SkeletonDataPointWithName((idx + 1000), "Face " + str(idx), landmark.x, landmark.y,
                                                      landmark.z))
                    else:
                        skeleton_data_obj.add_data_point(SkeletonDataPoint((idx + 1000), landmark.x, landmark.y,
                                                                           landmark.z))
            except:
                for i in range(478):
                    if with_name:
                        skeleton_data_obj.add_data_point(
                            SkeletonDataPointWithName((i + 1000), "Face " + str(i), 0, 0, 0))
                    else:
                        skeleton_data_obj.add_data_point(SkeletonDataPoint((i + 1000), 0, 0, 0))

        return skeleton_data_obj
