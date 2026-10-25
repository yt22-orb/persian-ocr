"""Run from the repository root: python -m unittest discover -s TESTING/tests -v."""
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from TESTING.preprocessing import enhance_contrast, outcome_from_result, recognize_batch, rotate_view


def rejected(reason='zero detections'):
    return dict(accepted=False, reason=reason, candidates=[])


def accepted(label=1, confidence=.9):
    p = dict(class_id=label, confidence=confidence, box=[1, 2, 3, 4])
    return dict(accepted=True, reason='accepted', prediction=p, candidates=[p])


class QueueEvaluator:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, images):
        self.calls.append(len(images))
        result = next(self.responses)
        assert len(result) == len(images)
        return result


class PreprocessingTest(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (20, 30), 'white')
        self.image.putpixel((4, 5), (30, 40, 50))

    def test_accepted_base_is_final(self):
        base = accepted()
        evaluate = QueueEvaluator([[base]])
        result = recognize_batch([self.image], evaluate)[0]
        self.assertIs(result['outcome'], base)
        self.assertEqual(evaluate.calls, [1])
        self.assertFalse(result['preprocessing']['needs_review'])

    def test_multiple_letters_are_never_retried(self):
        base = rejected('multiple letters')
        evaluate = QueueEvaluator([[base]])
        result = recognize_batch([self.image], evaluate)[0]
        self.assertIs(result['outcome'], base)
        self.assertEqual(evaluate.calls, [1])

    def test_contrast_stops_cascade(self):
        evaluate = QueueEvaluator([[rejected('low confidence')], [accepted(3)]])
        result = recognize_batch([self.image], evaluate)[0]
        self.assertEqual(evaluate.calls, [1, 1])
        self.assertEqual(result['preprocessing']['method'], 'contrast')
        self.assertTrue(result['preprocessing']['needs_review'])

    def test_both_prefers_strongest_accepted_view_and_reuses_zero(self):
        weak = dict(accepted=False, reason='multiple letters',
                    candidates=[dict(class_id=8, confidence=.999)])
        evaluate = QueueEvaluator([[rejected()], [rejected()],
            [accepted(2, .9), weak, accepted(3, .95), accepted(4, .95), rejected(), rejected(), rejected()]])
        result = recognize_batch([self.image], evaluate)[0]
        self.assertEqual(evaluate.calls, [1, 1, 7])
        self.assertEqual(result['preprocessing']['method'], 'both')
        self.assertEqual(result['preprocessing']['angle_degrees'], 135)
        self.assertEqual(result['outcome']['prediction']['class_id'], 3)

    def test_all_failed_keeps_base_rejection(self):
        base = rejected('low confidence')
        evaluate = QueueEvaluator([[base], [rejected()], [rejected()] * 7, [rejected()] * 7])
        result = recognize_batch([self.image], evaluate)[0]
        self.assertIs(result['outcome'], base)
        self.assertEqual(evaluate.calls, [1, 1, 7, 7])
        self.assertEqual(result['preprocessing']['unique_views_requested'], 16)

    def test_disabled_preserves_base(self):
        evaluate = QueueEvaluator([[rejected()]])
        result = recognize_batch([self.image], evaluate, enabled=False)[0]
        self.assertEqual(result['preprocessing']['attempted_methods'], ['base'])

    def test_batch_routes_only_eligible_images(self):
        evaluate = QueueEvaluator([[accepted(), rejected('multiple letters'), rejected()], [accepted(2)]])
        results = recognize_batch([self.image] * 3, evaluate)
        self.assertEqual(evaluate.calls, [3, 1])
        self.assertEqual([r['preprocessing']['method'] for r in results], ['base', 'base', 'contrast'])

    def test_views_do_not_modify_source_or_clip_rotations(self):
        pixels = self.image.tobytes()
        enhanced = enhance_contrast(self.image)
        self.assertEqual(enhanced.mode, 'RGB')
        self.assertEqual(enhanced.size, self.image.size)
        self.assertEqual(rotate_view(self.image, 90).size, (30, 20))
        self.assertGreater(rotate_view(self.image, 45).width, self.image.width)
        self.assertEqual(self.image.tobytes(), pixels)

    def test_uniform_white_stays_blank(self):
        result = enhance_contrast(Image.new('RGB', (30, 30), 'white'))
        self.assertTrue(np.all(np.asarray(result) == 255))

    def test_detector_decisions_match_existing_policy_at_boundaries(self):
        from persian_ocr.inference import decision

        class Values:
            def __init__(self, values):
                self.values = values
            def cpu(self):
                return self
            def tolist(self):
                return self.values

        policy = dict(candidate_floor=.03, acceptance_threshold=.8, nms_iou=.4)
        for confidences in ([], [.01], [.03], [.799999], [.8], [.999], [.99, .04]):
            with self.subTest(confidences=confidences):
                result = SimpleNamespace(boxes=SimpleNamespace(
                    xyxy=Values([[1, 2, 3, 4] for _ in confidences]),
                    conf=Values(confidences), cls=Values([0 for _ in confidences])))
                candidates = [dict(class_id=0, confidence=score, box=[1., 2., 3., 4.]) for score in confidences]
                self.assertEqual(outcome_from_result(result, policy), decision(candidates, policy))


if __name__ == '__main__':
    unittest.main()
