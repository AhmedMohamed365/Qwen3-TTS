import copy


def parse_added_languages(language_specs):
    added_languages = {}
    for spec in language_specs or []:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --add_language value: {spec!r}. Expected NAME=ID, for example arabic=2048."
            )

        language_name, language_id = spec.split("=", 1)
        language_name = language_name.strip().lower()
        language_id = language_id.strip()
        if not language_name:
            raise ValueError(f"Invalid --add_language value: {spec!r}. Language name cannot be empty.")

        try:
            language_id = int(language_id)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --add_language value: {spec!r}. Language id must be an integer."
            ) from exc

        added_languages[language_name] = language_id

    return added_languages


def build_finetuned_config(config_dict, speaker_name, speaker_id=3000, added_languages=None):
    finetuned_config = copy.deepcopy(config_dict)
    finetuned_config["tts_model_type"] = "custom_voice"

    talker_config = finetuned_config.get("talker_config", {})
    talker_config["spk_id"] = {
        speaker_name: speaker_id
    }
    talker_config["spk_is_dialect"] = {
        speaker_name: False
    }

    if added_languages:
        codec_language_id = talker_config.get("codec_language_id", {})
        codec_language_id.update(added_languages)
        talker_config["codec_language_id"] = codec_language_id

    finetuned_config["talker_config"] = talker_config
    return finetuned_config
