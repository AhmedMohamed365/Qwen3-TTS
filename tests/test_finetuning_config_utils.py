import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "finetuning"))

from config_utils import build_finetuned_config, parse_added_languages


class ParseAddedLanguagesTest(unittest.TestCase):
    def test_normalizes_language_names_and_ids(self):
        parsed = parse_added_languages([" Arabic = 2048 ", "urdu=2049"])

        self.assertEqual(parsed, {"arabic": 2048, "urdu": 2049})

    def test_rejects_invalid_spec(self):
        with self.assertRaisesRegex(ValueError, "Expected NAME=ID"):
            parse_added_languages(["arabic"])


class BuildFinetunedConfigTest(unittest.TestCase):
    def test_adds_new_language_while_preserving_existing_ones(self):
        original = {
            "tts_model_type": "base",
            "talker_config": {
                "codec_language_id": {
                    "english": 100,
                    "chinese": 101,
                },
                "spk_id": {
                    "vivian": 3001,
                },
                "spk_is_dialect": {
                    "vivian": False,
                },
            },
        }

        updated = build_finetuned_config(
            config_dict=original,
            speaker_name="speaker_ar",
            added_languages={"arabic": 2048},
        )

        self.assertEqual(updated["tts_model_type"], "custom_voice")
        self.assertEqual(updated["talker_config"]["spk_id"], {"speaker_ar": 3000})
        self.assertEqual(updated["talker_config"]["spk_is_dialect"], {"speaker_ar": False})
        self.assertEqual(
            updated["talker_config"]["codec_language_id"],
            {"english": 100, "chinese": 101, "arabic": 2048},
        )
        self.assertEqual(original["talker_config"]["codec_language_id"], {"english": 100, "chinese": 101})

    def test_keeps_existing_languages_when_no_new_language_is_provided(self):
        original = {
            "tts_model_type": "base",
            "talker_config": {
                "codec_language_id": {
                    "english": 100,
                    "chinese": 101,
                },
            },
        }

        updated = build_finetuned_config(
            config_dict=original,
            speaker_name="speaker_ar",
            added_languages=None,
        )

        self.assertEqual(
            updated["talker_config"]["codec_language_id"],
            {"english": 100, "chinese": 101},
        )


if __name__ == "__main__":
    unittest.main()
