#!/usr/bin/env python3
"""
Export Parakeet TDT 1.1B and Canary models to ONNX for use with ONNX Runtime.

This script uses NeMo's export utilities to convert .nemo models to ONNX format.
Run this once to generate ONNX models, then use ONNX Runtime for inference.

Usage:
    python export_parakeet_onnx.py --model parakeet-tdt-1.1b --output-dir ./models/onnx
    python export_parakeet_onnx.py --model canary-1b --output-dir ./models/onnx
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def export_parakeet_tdt(model_name: str, output_dir: str, device: str = "cpu"):
    """Export Parakeet TDT model to ONNX."""
    try:
        import nemo.collections.asr as nemo_asr
        from nemo.utils.export_utils import replace_for_export
    except ImportError as e:
        logger.error(f"NeMo not installed: {e}")
        logger.error("Install with: pip install nemo_toolkit[asr]")
        return False

    logger.info(f"Loading {model_name}...")
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
    asr_model = asr_model.to(device)
    asr_model.eval()

    # Prepare for export - replace modules with ONNX-compatible versions
    logger.info("Preparing model for ONNX export...")
    replace_for_export(asr_model)

    # Create dummy input for tracing
    batch_size = 1
    seq_len = 16000 * 30  # 30 seconds at 16kHz
    dummy_audio = torch.randn(batch_size, seq_len, device=device)
    dummy_length = torch.tensor([seq_len], device=device)

    # Export encoder
    logger.info("Exporting encoder to ONNX...")
    encoder_path = os.path.join(output_dir, "encoder.onnx")
    torch.onnx.export(
        asr_model.encoder,
        (dummy_audio, dummy_length),
        encoder_path,
        input_names=["audio_signal", "length"],
        output_names=["encoded", "encoded_len"],
        dynamic_axes={
            "audio_signal": {0: "batch", 1: "time"},
            "length": {0: "batch"},
            "encoded": {0: "batch", 1: "time"},
            "encoded_len": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info(f"Encoder exported to {encoder_path}")

    # Export decoder + joint (for RNNT decoding)
    # This is more complex due to autoregressive nature
    logger.info("Exporting decoder+joint to ONNX...")
    decoder_joint_path = os.path.join(output_dir, "decoder_joint.onnx")
    
    # Create dummy inputs for decoder+joint
    dummy_encoder_out = torch.randn(batch_size, 100, 1024, device=device)  # encoder output
    dummy_encoder_len = torch.tensor([100], device=device)
    dummy_decoder_in = torch.tensor([[asr_model.decoder.blank_token_id]], device=device)  # blank token
    
    # We need to export the joint network separately
    class DecoderJointWrapper(torch.nn.Module):
        def __init__(self, decoder, joint, encoder_projector):
            super().__init__()
            self.decoder = decoder
            self.joint = joint
            self.encoder_projector = encoder_projector
            
        def forward(self, encoder_output, encoder_length, decoder_input_ids):
            # Project encoder output
            projected_encoder = self.encoder_projector(encoder_output)
            # Run decoder
            decoder_out = self.decoder(decoder_input_ids)
            # Joint network
            logits = self.joint(projected_encoder, decoder_out)
            return logits

    decoder_joint = DecoderJointWrapper(
        asr_model.decoder, 
        asr_model.joint, 
        asr_model.encoder_projector
    )
    
    torch.onnx.export(
        decoder_joint,
        (dummy_encoder_out, dummy_encoder_len, dummy_decoder_in),
        decoder_joint_path,
        input_names=["encoder_output", "encoder_length", "decoder_input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "encoder_output": {0: "batch", 1: "time"},
            "encoder_length": {0: "batch"},
            "decoder_input_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "time", 2: "seq"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    logger.info(f"Decoder+Joint exported to {decoder_joint_path}")

    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer.model")
    # The tokenizer is a SentencePiece model - save it
    if hasattr(asr_model, 'tokenizer') and asr_model.tokenizer is not None:
        asr_model.tokenizer.save(tokenizer_path)
        logger.info(f"Tokenizer saved to {tokenizer_path}")

    # Save model config
    import json
    config = {
        "model_type": "parakeet-tdt",
        "model_name": model_name,
        "sample_rate": 16000,
        "blank_token_id": asr_model.decoder.blank_token_id,
        "vocab_size": asr_model.joint.vocab_size,
        "encoder_dim": 1024,
        "decoder_dim": asr_model.decoder.decoder_hidden_size,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_path}")

    return True


def export_canary(model_name: str, output_dir: str, device: str = "cpu"):
    """Export Canary model to ONNX for translation."""
    try:
        import nemo.collections.asr as nemo_asr
        from nemo.utils.export_utils import replace_for_export
    except ImportError as e:
        logger.error(f"NeMo not installed: {e}")
        return False

    logger.info(f"Loading {model_name}...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    asr_model = asr_model.to(device)
    asr_model.eval()

    logger.info("Preparing model for ONNX export...")
    replace_for_export(asr_model)

    # Canary is a multi-task model (ASR + translation)
    # Export encoder
    batch_size = 1
    seq_len = 16000 * 30
    dummy_audio = torch.randn(batch_size, seq_len, device=device)
    dummy_length = torch.tensor([seq_len], device=device)

    encoder_path = os.path.join(output_dir, "encoder.onnx")
    torch.onnx.export(
        asr_model.encoder,
        (dummy_audio, dummy_length),
        encoder_path,
        input_names=["audio_signal", "length"],
        output_names=["encoded", "encoded_len"],
        dynamic_axes={
            "audio_signal": {0: "batch", 1: "time"},
            "length": {0: "batch"},
            "encoded": {0: "batch", 1: "time"},
            "encoded_len": {0: "batch"},
        },
        opset_version=17,
    )
    logger.info(f"Encoder exported to {encoder_path}")

    # Save tokenizer
    tokenizer_path = os.path.join(output_dir, "tokenizer.model")
    if hasattr(asr_model, 'tokenizer') and asr_model.tokenizer is not None:
        asr_model.tokenizer.save(tokenizer_path)
        logger.info(f"Tokenizer saved to {tokenizer_path}")

    # Save config
    import json
    config = {
        "model_type": "canary",
        "model_name": model_name,
        "sample_rate": 16000,
        "supports_translation": True,
        "supports_asr": True,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Export Parakeet/Canary models to ONNX")
    parser.add_argument("--model", required=True, choices=["parakeet-tdt-1.1b", "parakeet-rnnt-1.1b", "canary-1b"],
                        help="Model to export")
    parser.add_argument("--output-dir", required=True, help="Output directory for ONNX models")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use for export")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    success = False
    if args.model.startswith("parakeet"):
        success = export_parakeet_tdt(args.model, str(output_dir), device)
    elif args.model.startswith("canary"):
        success = export_canary(args.model, str(output_dir), device)
    
    if success:
        logger.info(f"Export completed successfully! Models saved to {output_dir}")
        sys.exit(0)
    else:
        logger.error("Export failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()