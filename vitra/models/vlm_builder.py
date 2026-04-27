import copy
import transformers
import torch

def build_vlm(vlm_config):
    vlm_config = copy.deepcopy(vlm_config)
    model_path = vlm_config.get("pretrained_model_name_or_path")
    model_name = vlm_config.get("name")
    model_type = vlm_config.get("type", "AutoModel")
    if model_name == "paligemma":
        from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            # attn_implementation="eager",
            # revision="bfloat16",
        )
        processor = PaliGemmaProcessor.from_pretrained(model_path)
    elif model_name == "small_paligemma":
        from transformers import (
            GemmaConfig,
            PaliGemmaConfig,
            PaliGemmaForConditionalGeneration,
            PaliGemmaProcessor,
            SiglipVisionConfig,
        )

        processor_path = vlm_config.get("processor_name_or_path", model_path)
        processor = PaliGemmaProcessor.from_pretrained(processor_path)

        vision_cfg = vlm_config.get("vision_config", {})
        text_cfg = vlm_config.get("text_config", {})
        tokenizer_image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        image_token_index = int(vlm_config.get("image_token_index", tokenizer_image_token_id))
        vocab_size = max(
            int(vlm_config.get("vocab_size", len(processor.tokenizer))),
            len(processor.tokenizer),
            image_token_index + 1,
        )
        hidden_size = int(text_cfg.get("hidden_size", vlm_config.get("hidden_size", 768)))

        vision_config = SiglipVisionConfig(
            hidden_size=int(vision_cfg.get("hidden_size", 768)),
            intermediate_size=int(vision_cfg.get("intermediate_size", 3072)),
            num_hidden_layers=int(vision_cfg.get("num_hidden_layers", 12)),
            num_attention_heads=int(vision_cfg.get("num_attention_heads", 12)),
            image_size=int(vision_cfg.get("image_size", 224)),
            patch_size=int(vision_cfg.get("patch_size", 14)),
            projection_dim=int(vision_cfg.get("projection_dim", hidden_size)),
            vision_use_head=bool(vision_cfg.get("vision_use_head", False)),
        )
        text_config = GemmaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=int(text_cfg.get("intermediate_size", 3072)),
            num_hidden_layers=int(text_cfg.get("num_hidden_layers", 10)),
            num_attention_heads=int(text_cfg.get("num_attention_heads", 12)),
            num_key_value_heads=int(text_cfg.get("num_key_value_heads", 4)),
            head_dim=int(text_cfg.get("head_dim", hidden_size // int(text_cfg.get("num_attention_heads", 12)))),
            max_position_embeddings=int(text_cfg.get("max_position_embeddings", 8192)),
            pad_token_id=int(text_cfg.get("pad_token_id", 0)),
            eos_token_id=int(text_cfg.get("eos_token_id", 1)),
            bos_token_id=int(text_cfg.get("bos_token_id", 2)),
            tie_word_embeddings=bool(text_cfg.get("tie_word_embeddings", True)),
        )
        text_config.num_image_tokens = int(vlm_config.get("num_image_tokens", 256))
        config = PaliGemmaConfig(
            vision_config=vision_config.to_dict(),
            text_config=text_config.to_dict(),
            projection_dim=int(vlm_config.get("projection_dim", hidden_size)),
            hidden_size=hidden_size,
            image_token_index=image_token_index,
            vocab_size=vocab_size,
        )
        model = PaliGemmaForConditionalGeneration(config)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

    return processor, model
