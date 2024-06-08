import pytest
import cv2
from unittest.mock import patch, MagicMock
from Gesture_Controller import GestureController, Controller, HandRecog, Gest, HLabel

def test_gesture_enum():
    assert Gest.FIST == 0
    assert Gest.PALM == 31
    assert Gest.V_GEST == 33

def test_hlabel_enum():
    assert HLabel.MINOR == 0
    assert HLabel.MAJOR == 1

def test_handrecog_initialization():
    hand_recog = HandRecog(HLabel.MAJOR)
    assert hand_recog.hand_label == HLabel.MAJOR
    assert hand_recog.finger == 0
    assert hand_recog.ori_gesture == Gest.PALM
    assert hand_recog.prev_gesture == Gest.PALM

def test_controller_initialization():
    assert Controller.tx_old == 0
    assert Controller.ty_old == 0
    assert Controller.trial == True
    assert Controller.flag == False
    assert Controller.grabflag == False
    assert Controller.pinchmajorflag == False
    assert Controller.pinchminorflag == False
    assert Controller.pinchstartxcoord == None
    assert Controller.pinchstartycoord == None
    assert Controller.pinchdirectionflag == None
    assert Controller.prevpinchlv == 0
    assert Controller.pinchlv == 0
    assert Controller.framecount == 0
    assert Controller.prev_hand == None
    assert Controller.pinch_threshold == 0.3

def test_gesture_controller_initialization():
    gc = GestureController()
    assert GestureController.gc_mode == 1
    assert GestureController.cap.isOpened()
    assert GestureController.CAM_HEIGHT == GestureController.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    assert GestureController.CAM_WIDTH == GestureController.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

def test_gesture_controller_classify_hands():
    gc = GestureController()
    # Create a mock object for results
    results = type('obj', (object,), {'multi_handedness': None, 'multi_hand_landmarks': None})()
    GestureController.classify_hands(results)
    assert GestureController.hr_major == None
    assert GestureController.hr_minor == None

def test_handrecog_get_signed_dist():
    hand_recog = HandRecog(HLabel.MAJOR)
    hand_recog.hand_result = type('obj', (object,), {'landmark': [type('obj', (object,), {'x': 0.5, 'y': 0.5})(), type('obj', (object,), {'x': 0.4, 'y': 0.4})()]})
    dist = hand_recog.get_signed_dist([0, 1])
    assert dist == pytest.approx(0.1414, 0.001)

def test_handrecog_get_dist():
    hand_recog = HandRecog(HLabel.MAJOR)
    hand_recog.hand_result = type('obj', (object,), {'landmark': [type('obj', (object,), {'x': 0.5, 'y': 0.5})(), type('obj', (object,), {'x': 0.4, 'y': 0.4})()]})
    dist = hand_recog.get_dist([0, 1])
    assert dist == pytest.approx(0.1414, 0.001)

def test_handrecog_get_dz():
    hand_recog = HandRecog(HLabel.MAJOR)
    hand_recog.hand_result = type('obj', (object,), {'landmark': [type('obj', (object,), {'z': 0.5})(), type('obj', (object,), {'z': 0.4})()]})
    dz = hand_recog.get_dz([0, 1])
    assert dz == pytest.approx(0.1, 0.001)

@patch('Gesture_Controller.pyautogui')
def test_controller_get_position(mock_pyautogui):
    mock_pyautogui.size.return_value = (1920, 1080)
    hand_result = type('obj', (object,), {'landmark': [type('obj', (object,), {'x': 0.5, 'y': 0.5})() for _ in range(21)]})
    x, y = Controller.get_position(hand_result)
    sx, sy = mock_pyautogui.size()
    assert 0 <= x <= sx
    assert 0 <= y <= sy
