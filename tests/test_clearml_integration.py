import unittest

from clearml_integration import clearml_status


class ClearMLIntegrationTest(unittest.TestCase):
    def test_status_does_not_expose_secret_values(self):
        status = clearml_status()

        self.assertIn("package_installed", status)
        self.assertIn("env_keys", status)
        for value in status["env_keys"].values():
            self.assertIsInstance(value, bool)


if __name__ == "__main__":
    unittest.main()
