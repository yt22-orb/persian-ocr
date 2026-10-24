import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from serve_ui import PersianOCRHandler


class UIAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), PersianOCRHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/api/predict"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def post(self, body=b"image", content_type="image/png"):
        request = Request(self.url, body, {"Content-Type": content_type}, method="POST")
        try:
            response = urlopen(request)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def test_success_returns_model_result(self):
        result = {"accepted": True, "letter": "ق", "confidence": 0.99,
                  "matches": [{"letter": "ق", "confidence": 0.99}]}
        with patch("serve_ui._predict", return_value=result) as predict:
            status, payload = self.post()
        self.assertEqual(status, 200)
        self.assertEqual(payload, result)
        predict.assert_called_once_with(b"image")

    def test_rejection_uses_validation_status(self):
        with patch("serve_ui._predict", return_value={"accepted": False, "code": "unrecognized"}):
            status, payload = self.post()
        self.assertEqual(status, 422)
        self.assertEqual(payload["code"], "unrecognized")

    def test_model_error_uses_service_unavailable_status(self):
        with patch("serve_ui._predict", side_effect=FileNotFoundError("missing checkpoint")):
            status, payload = self.post()
        self.assertEqual(status, 503)
        self.assertEqual(payload["code"], "model_unavailable")

    def test_non_image_content_is_rejected(self):
        status, payload = self.post(content_type="text/plain")
        self.assertEqual(status, 422)
        self.assertEqual(payload["code"], "unrecognized")


if __name__ == "__main__":
    unittest.main()
