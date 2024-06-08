import pytest
from Gesture_Controller import HandRecog, HLabel, Gest

class TestHandRecog:
    @pytest.fixture
    def hand_recog(self):
        return HandRecog(HLabel.MINOR)

    def test_set_finger_state(self, hand_recog):
        hand_recog.update_hand_result(None)
        hand_recog.set_finger_state()
        assert hand_recog.finger == 0

    def test_get_gesture(self, hand_recog):
        hand_recog.update_hand_result(None)
        gesture = hand_recog.get_gesture()
        assert gesture == Gest.PALM

    def test_update_hand_result(self, hand_recog):
        hand_recog.update_hand_result("Some hand result")
        assert hand_recog.hand_result == "Some hand result"
