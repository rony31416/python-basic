# Finetuning Bangla TTS Model with TTS 0.15.0
# For use with Python 3.11.11 in Jupyter Notebook

# Cell 1: Import necessary libraries
import os
import torch
import numpy as np
import pandas as pd

# Check TTS version
import TTS
print(f"TTS version: {TTS.__version__}")

# Cell 2: Import TTS modules (they have changed in 0.15.0)
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.config import load_config

# The trainer has been moved in 0.15.0
from TTS.trainer import Trainer, TrainerArgs
from TTS.tts.datasets import load_tts_samples

# Cell 3: Define paths (update these to your actual paths)
# Define paths
output_path = "/path/to/save/finetuned/model"  # Change this
model_path = "/path/to/original/model.pth"     # Change this
config_path = "/path/to/original/config.json"  # Change this

# Your custom dataset paths
dataset_root = "/path/to/your/dataset"  # Change this
audio_dir = os.path.join(dataset_root, "wavs")
tsv_file = os.path.join(dataset_root, "audio_text.tsv")

# Cell 4: Custom formatter for TSV data
def custom_formatter(root_path, meta_file, **kwargs):
    """Formats the TSV file data to TTS format"""
    items = []
    try:
        # Try to read the TSV file with different encodings
        try:
            df = pd.read_csv(meta_file, sep='\t', header=None, encoding='utf-8')
        except UnicodeDecodeError:
            # If utf-8 fails, try another encoding
            df = pd.read_csv(meta_file, sep='\t', header=None, encoding='latin1')
            
        print(f"Loaded TSV file with {len(df)} rows and {len(df.columns)} columns")
        
        # Handle the column assignment based on the number of columns
        if len(df.columns) >= 2:
            # Rename the first two columns to 'audio_file' and 'text'
            column_names = ['audio_file', 'text'] + [f'col_{i}' for i in range(2, len(df.columns))]
            df.columns = column_names
        elif len(df.columns) == 1:
            # Try to split the single column that may contain tab-separated values
            print("Found only one column. Checking if it contains tab-separated values...")
            first_row = df.iloc[0, 0]
            if '\t' in str(first_row):
                # Split the single column by tabs
                df = pd.DataFrame([row[0].split('\t') for row in df.values], columns=['audio_file', 'text'])
                print(f"Split successful. New shape: {df.shape}")
            else:
                raise ValueError(f"Single column doesn't contain tabs. First row: {first_row}")
        else:
            raise ValueError(f"Unexpected number of columns: {len(df.columns)}")
        
        speaker_name = "custom_speaker"  # You can customize this
        
        count_valid = 0
        count_invalid = 0
        
        for idx, row in df.iterrows():
            # Get the audio file name and text
            audio_file = str(row['audio_file']).strip()
            text = str(row['text']).strip()
            
            # Make sure audio file has .wav extension
            if not audio_file.endswith('.wav'):
                audio_file = f"{audio_file}.wav"
                
            wav_file = os.path.join(audio_dir, audio_file)
            
            # Print debugging info for the first few rows
            if idx < 5:
                print(f"Row {idx}:")
                print(f"  Audio file: {audio_file}")
                print(f"  Text: {text}")
                print(f"  Full path: {wav_file}")
                print(f"  Exists: {os.path.exists(wav_file)}")
            
            if os.path.exists(wav_file):
                count_valid += 1
                items.append({
                    "text": text, 
                    "audio_file": wav_file, 
                    "speaker_name": speaker_name,
                    "root_path": root_path
                })
            else:
                count_invalid += 1
                if idx < 20:  # Only print first 20 missing files to avoid flooding output
                    print(f"File not found: {wav_file}")
        
        print(f"Total valid files: {count_valid}, missing files: {count_invalid}")
        
        return items
    except Exception as e:
        print(f"Error in formatter: {e}")
        # Return any items we've collected so far
        return items

# Cell 5: Create dataset config and load samples
dataset_config = BaseDatasetConfig(
    meta_file_train=tsv_file,
    path=dataset_root
)

# Debug the TSV file
print(f"Reading TSV file from: {tsv_file}")
try:
    df = pd.read_csv(tsv_file, sep='\t', header=None)
    print(f"TSV file loaded successfully. Number of rows: {len(df)}")
    # Show first few rows to verify format
    if len(df.columns) >= 2:
        df.columns = ['audio_file', 'text'] + [f'col_{i}' for i in range(2, len(df.columns))]
    else:
        print(f"WARNING: TSV file has {len(df.columns)} columns, expected at least 2")
        if len(df.columns) == 1:
            # Try to split the single column if it contains tab characters
            print("Attempting to split the single column...")
            if '\t' in str(df.iloc[0, 0]):
                df = pd.DataFrame([row[0].split('\t') for row in df.values], columns=['audio_file', 'text'])
                print("Split successful.")
    
    print("First 3 rows of the TSV file:")
    print(df.head(3))
except Exception as e:
    print(f"Error reading TSV file: {e}")

# Debug the audio directory
print(f"Audio directory: {audio_dir}")
if os.path.exists(audio_dir):
    audio_files = os.listdir(audio_dir)
    print(f"Audio directory exists. Contains {len(audio_files)} files.")
    print(f"First 5 audio files: {audio_files[:5] if len(audio_files) >= 5 else audio_files}")
else:
    print(f"WARNING: Audio directory {audio_dir} does not exist!")

# Update the formatter to provide more debugging information
def debug_formatter(root_path, meta_file, **kwargs):
    """Debugs the TSV file data loading process with detailed output"""
    items = []
    try:
        df = pd.read_csv(meta_file, sep='\t', header=None)
        print(f"Formatter loaded TSV with shape: {df.shape}")
        
        # Handle column naming based on count
        if len(df.columns) >= 2:
            df.columns = ['audio_file', 'text'] + [f'col_{i}' for i in range(2, len(df.columns))]
        else:
            print(f"WARNING: TSV has only {len(df.columns)} columns!")
            if len(df.columns) == 1:
                # Try to manually split based on tab character
                try:
                    first_row = df.iloc[0, 0]
                    if '\t' in str(first_row):
                        print("Single column contains tabs, attempting to split...")
                        df = pd.DataFrame([row[0].split('\t') for row in df.values], columns=['audio_file', 'text'])
                        print(f"After splitting: {df.shape}")
                    else:
                        print(f"Cannot split row: '{first_row}'")
                except Exception as e:
                    print(f"Error during manual split: {e}")
        
        speaker_name = "custom_speaker"
        
        processed = 0
        skipped = 0
        
        for idx, row in df.iterrows():
            if idx < 5:  # Print first 5 rows for debugging
                print(f"Processing row {idx}: {row.values}")
            
            try:
                # Get audio file name and text based on available columns
                if 'audio_file' in df.columns and 'text' in df.columns:
                    audio_file = row['audio_file']
                    text = row['text']
                elif len(df.columns) >= 2:
                    audio_file = row.iloc[0]
                    text = row.iloc[1]
                else:
                    print(f"Cannot parse row {idx}: {row}")
                    skipped += 1
                    continue
                
                # Make sure audio file has the right extension
                if not audio_file.endswith('.wav'):
                    audio_file = f"{audio_file}.wav"
                
                # Check if audio file exists
                wav_file = os.path.join(audio_dir, audio_file)
                if os.path.exists(wav_file):
                    processed += 1
                    items.append({
                        "text": text, 
                        "audio_file": wav_file, 
                        "speaker_name": speaker_name,
                        "root_path": root_path
                    })
                else:
                    if idx < 5:  # Limit debug output 
                        print(f"Warning: File {wav_file} not found, skipping.")
                    skipped += 1
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                skipped += 1
                
        print(f"Processed {processed} items, skipped {skipped} items.")
        return items
    except Exception as e:
        print(f"Fatal error in formatter: {e}")
        return []

# Load training samples using updated formatter with better error handling
try:
    # Use the modified parameter list compatible with TTS 0.15.0
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        formatter=debug_formatter, 
        eval_split=True,
        eval_split_max_size=None,  # This is compatible with TTS 0.15.0
        eval_split_size=0.01  # 1% for evaluation
    )
    print(f"Training samples: {len(train_samples)}, Evaluation samples: {len(eval_samples)}")
except TypeError as e:
    print(f"TypeError in load_tts_samples: {e}")
    print("Trying alternative parameter configuration...")
    
    # Try without the 'eval_split_size' parameter
    try:
        train_samples, eval_samples = load_tts_samples(
            dataset_config,
            formatter=debug_formatter, 
            eval_split=True
        )
        print(f"Loading succeeded with simplified parameters.")
        print(f"Training samples: {len(train_samples)}, Evaluation samples: {len(eval_samples)}")
    except Exception as e2:
        print(f"Second attempt failed: {e2}")
        print("Creating empty evaluation set as fallback...")
        
        # As a last resort, get all samples as training and manually split
        all_samples = debug_formatter(dataset_root, tsv_file)
        if len(all_samples) > 0:
            # Manually create train/eval split
            eval_size = max(1, int(len(all_samples) * 0.01))
            train_samples = all_samples[:-eval_size]
            eval_samples = all_samples[-eval_size:]
            print(f"Manually split - Training: {len(train_samples)}, Evaluation: {len(eval_samples)}")
        else:
            print("ERROR: No samples could be loaded! Please check your dataset format.")

# Cell 6: Gender selection and character config
# Gender selection (affects character set)
is_male = False  # Set to True for male voice model

# Define character set based on gender
if is_male:
    characters_config = CharactersConfig(
        pad='<PAD>',
        eos='।',  # Bangla end of sentence
        bos='<BOS>',
        blank='<BLNK>',
        characters="তট৫ভিঐঋখঊড়ইজমএেঘঙসীঢ়হঞ'ঈকণ৬ঁৗশঢঠ\u200c১্২৮দৃঔগও—ছউংবৈঝাযফ\u200dচরষঅৌৎথড়৪ধ০ুূ৩আঃপয়'নলো",
        punctuations="-!,|.? ",
    )
else:
    characters_config = CharactersConfig(
        pad='<PAD>',
        eos='।',
        bos='<BOS>',
        blank='<BLNK>',
        characters="ইগং়'ুঃন১ঝূও'ঊোছপফৈ৮ষযৎঢঈকঠিজ০৬ীটডএঅঋধচে২৩ণউয়ঢ়খলভৗসহ্ড়দথবঔাঞশরৌম—ঐআৃঘঙ\u200cঁ৪৫ত",
        punctuations=".?-!|, ",
    )

# Cell 7: Create audio config
# In TTS 0.15.0, audio config is defined differently
from TTS.tts.models.vits import VitsArgs, VitsAudioConfig

audio_config = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None
)

# Cell 8: Load the original config if available
original_config = None
if os.path.exists(config_path):
    original_config = load_config(config_path)
    print("Loaded original config successfully.")
    
    # In 0.15.0, some config parameters might be different
    # We can extract useful parameters from the original config
    if hasattr(original_config, "characters"):
        characters_config = original_config.characters
        print("Using character config from original model.")

# Cell 9: Create a new config for finetuning
# Note: VitsConfig parameters have changed in 0.15.0
config = VitsConfig(
    model_args=VitsArgs(
        use_d_vector_file=False,
        use_speaker_embedding=False,
        speaker_embedding_channels=0,  # Set to 0 for single speaker
    ),
    audio=audio_config,
    run_name="vits_finetuned",
    batch_size=32,  # Reduce batch size for finetuning
    eval_batch_size=16,
    batch_group_size=0,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,  # Reduced for finetuning
    text_cleaner=None,
    use_phonemes=False,
    compute_input_seq_cache=True,
    print_step=25,
    print_eval=True,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    characters=characters_config,
    test_sentences=[
        "আমার সোনার বাংলা আমি তোমায় ভালোবাসি।",
        "বাংলাদেশ একটি সুন্দর দেশ।",
        # Add your own test sentences here
    ]
)

# Cell 10: Initialize audio processor, tokenizer, and model
ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

# Initialize model (VITS model initialization is different in 0.15.0)
model = Vits(config)

# Cell 11: Load pretrained weights if available
if os.path.exists(model_path):
    print(f"Loading checkpoint from {model_path}")
    try:
        # Newer checkpoint format
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dictionary'}")
        
        # For loading only the model weights (not optimizer state)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            try:
                model.load_state_dict(checkpoint["model"])
                print("Model loaded successfully from 'model' key.")
            except Exception as e:
                print(f"Error loading from 'model' key: {e}")
                print("Trying to load with strict=False...")
                model.load_state_dict(checkpoint["model"], strict=False)
                print("Model loaded with strict=False")
        else:
            try:
                model.load_state_dict(checkpoint)
                print("Model loaded successfully directly from checkpoint.")
            except Exception as e:
                print(f"Error loading directly: {e}")
                print("Trying to load with strict=False...")
                model.load_state_dict(checkpoint, strict=False)
                print("Model loaded with strict=False")
        
        print("Model loaded successfully for finetuning.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error details: {str(e)}")
        print("Will start training from scratch.")
else:
    print("No pretrained model found. Starting from scratch.")

# Cell 12: Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Cell 13: Initialize trainer
# Check the TrainerArgs class to see available parameters (they may have changed in 0.15.0)
print("Available TrainerArgs parameters:")
import inspect
print(inspect.signature(TrainerArgs))

# Create trainer args with fallback options in case of incompatible parameters
try:
    trainer_args = TrainerArgs(
        restore_path=model_path if os.path.exists(model_path) else None,
        # Finetuning parameters
        lr=1e-5,  # Lower learning rate for finetuning
        lr_decay=True,
        use_grad_clip=True,
        grad_clip=1.0,
        early_stopping=True,
        early_stopping_epochs=20,
        save_step=100,  # Save more frequently during finetuning
        epochs=1000,
        # Use mixed precision for faster training
        mixed_precision=True if torch.cuda.is_available() else False,
    )
except TypeError as e:
    print(f"Error with TrainerArgs parameters: {e}")
    print("Trying with minimal parameters...")
    
    # Create with minimal parameters
    trainer_args = TrainerArgs(
        restore_path=model_path if os.path.exists(model_path) else None,
    )
    
    # Then set attributes individually with try/except blocks
    try:
        trainer_args.lr = 1e-5
    except:
        print("Could not set lr")
        
    try:
        trainer_args.lr_decay = True
    except:
        print("Could not set lr_decay")
    
    try:
        trainer_args.grad_clip = 1.0
    except:
        print("Could not set grad_clip")
    
    try:
        trainer_args.early_stopping = True
    except:
        print("Could not set early_stopping")
    
    try:
        trainer_args.early_stopping_epochs = 20
    except:
        print("Could not set early_stopping_epochs")
    
    try:
        trainer_args.save_step = 100
    except:
        print("Could not set save_step")
    
    try:
        trainer_args.epochs = 1000
    except:
        print("Could not set epochs")
    
    try:
        trainer_args.mixed_precision = True if torch.cuda.is_available() else False
    except:
        print("Could not set mixed_precision")

trainer = Trainer(
    trainer_args,
    config, 
    output_path, 
    model=model, 
    train_samples=train_samples, 
    eval_samples=eval_samples
)

# Cell 14: Start training
print("Starting fine-tuning...")
trainer.fit()
print("Fine-tuning completed!")

# Cell 15: Testing the finetuned model (after training is complete)

# This code assumes you have the inference module from the original repository
# If not, you can create a simple inference function:

def test_tts_model(model_path, config_path, text):
    """Test a trained TTS model with a text input"""
    # Load the config
    config = load_config(config_path)
    
    # Initialize the model
    model = Vits.init_from_config(config)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Set the language manager and tokenizer
    tokenizer, _ = TTSTokenizer.init_from_config(config)
    
    # Generate speech
    outputs = model.inference(
        text=text,
        speaker_id=None,
        language_id=None,
        style_wav=None,
        level=1,
        length_scale=1.0,
        noise_scale=0.667,
        noise_scale_w=0.8,
        max_decoder_steps=2000
    )
    
    return outputs["wav"]

# Example usage (uncomment when ready to test)
# model_path = os.path.join(output_path, "best_model.pth")
# config_path = os.path.join(output_path, "config.json")
# text = "বাংলাদেশ একটি সুন্দর দেশ।"
# 
# audio = test_tts_model(model_path, config_path, text)
# 
# # Save the audio
# import soundfile as sf
# sf.write('finetuned_output.wav', audio, 22050)
