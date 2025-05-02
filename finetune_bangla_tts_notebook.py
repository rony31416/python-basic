# Step-by-Step Bangla TTS Finetuning with TTS 0.15.0
# Compatible with Python 3.11.11 and Jupyter Notebook

# Cell 1: Install required dependencies
!pip install TTS==0.15.0 bangla==0.0.2 torch numpy pandas matplotlib

# Cell 2: Import required libraries
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display

# Cell 3: Define paths to your data and model
# Update these paths to match your environment
DATASET_ROOT = "/path/to/your/dataset"  # Root path to your custom dataset
AUDIO_DIR = os.path.join(DATASET_ROOT, "wavs")
TSV_FILE = os.path.join(DATASET_ROOT, "audio_text.tsv")

OUTPUT_PATH = "/path/to/save/finetuned/model"  # Change this to your desired output path
ORIGINAL_MODEL_PATH = "/path/to/original/model.pth"  # Pre-trained model path
ORIGINAL_CONFIG_PATH = "/path/to/original/config.json"  # Original config path

# Gender selection (affects character set)
IS_MALE = False  # Set to True for male voice model

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Cell 4: Define custom formatter for your dataset
def custom_formatter(root_path, meta_file, **kwargs):
    """Formats the TSV file data to TTS format"""
    items = []
    df = pd.read_csv(meta_file, sep='\t', header=None)
    
    # If your TSV has headers, comment out the next line
    df.columns = ['audio_file', 'text']
    
    speaker_name = "custom_speaker"  # You can customize this
    
    for _, row in df.iterrows():
        wav_file = os.path.join(AUDIO_DIR, row['audio_file'])
        text = row['text']
        
        if os.path.exists(wav_file):
            items.append({
                "text": text, 
                "audio_file": wav_file, 
                "speaker_name": speaker_name,
                "root_path": root_path
            })
        else:
            print(f"Warning: File {wav_file} not found, skipping.")
    
    return items

# Cell 5: Check and inspect your dataset
# Load a sample from your TSV to verify format
sample_df = pd.read_csv(TSV_FILE, sep='\t', nrows=5, header=None)
sample_df.columns = ['audio_file', 'text']
print("Sample data from your TSV file:")
display(sample_df)

# Cell 6: Import TTS specific modules
from TTS.config import load_config
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# Cell 7: Create dataset config and load samples
# Create dataset config
dataset_config = BaseDatasetConfig(
    meta_file_train=TSV_FILE,
    path=DATASET_ROOT
)

# Load a sample for testing formatter
try:
    # Try to load a small sample set first to verify
    test_samples = load_tts_samples(
        dataset_config,
        formatter=custom_formatter, 
        eval_split=False,
        max_eval_samples=None
    )
    print(f"Successfully loaded {len(test_samples)} test samples")
    print("Sample item:")
    print(test_samples[0])
except Exception as e:
    print("Error loading test samples:")
    print(str(e))

# Cell 8: Define character sets
# Define character set based on gender
if IS_MALE:
    characters_config = CharactersConfig(
        pad='<PAD>',
        eos='।',  # Bangla end of sentence
        bos='<BOS>',
        blank='<BLNK>',
        phonemes=None,
        characters="তট৫ভিঐঋখঊড়ইজমএেঘঙসীঢ়হঞ'ঈকণ৬ঁৗশঢঠ\u200c১্২৮দৃঔগও—ছউংবৈঝাযফ\u200dচরষঅৌৎথড়৪ধ০ুূ৩আঃপয়'নলো",
        punctuations="-!,|.? ",
    )
else:
    characters_config = CharactersConfig(
        pad='<PAD>',
        eos='।',
        bos='<BOS>',
        blank='<BLNK>',
        phonemes=None,
        characters="ইগং়'ুঃন১ঝূও'ঊোছপফৈ৮ষযৎঢঈকঠিজ০৬ীটডএঅঋধচে২৩ণউয়ঢ়খলভৗসহ্ড়দথবঔাঞশরৌম—ঐআৃঘঙ\u200cঁ৪৫ত",
        punctuations=".?-!|, ",
    )

# Cell 9: Load original config if available
original_config = None
if os.path.exists(ORIGINAL_CONFIG_PATH):
    try:
        original_config = load_config(ORIGINAL_CONFIG_PATH)
        print("Successfully loaded original config")
        # Optionally display some config parameters
        print(f"Original model type: {original_config.model}")
        print(f"Original sample rate: {original_config.audio.sample_rate}")
    except Exception as e:
        print(f"Error loading original config: {str(e)}")
        print("Will create a new config instead")

# Cell 10: Create audio config
audio_config = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None
)

# Cell 11: Create VITS config for finetuning
config = VitsConfig(
    audio=audio_config,
    run_name="vits_finetuned",
    batch_size=16,  # Smaller batch size for finetuning
    eval_batch_size=8,
    batch_group_size=0,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=500,  # Reduced epochs for finetuning
    text_cleaner=None,
    use_phonemes=False,
    compute_input_seq_cache=True,
    print_step=10,
    print_eval=True,
    mixed_precision=True,
    output_path=OUTPUT_PATH,
    datasets=[dataset_config],
    characters=characters_config,
    save_step=50,  # Save more frequently during finetuning
    cudnn_benchmark=True,
    test_sentences=[
        "আমার সোনার বাংলা আমি তোমায় ভালোবাসি।",
        "বাংলাদেশ একটি সুন্দর দেশ।",
    ]
)

# If you're using TTS 0.15.0, you might need to set additional parameters
# These are based on the changes in TTS 0.15.0 vs 0.14.3
config.model_args = original_config.model_args if original_config else {}
config.use_language_embedding = False
config.use_speaker_embedding = False

# Cell 12: Initialize audio processor and tokenizer
ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

# Cell 13: Load actual dataset for training
# Now load the full dataset with train/eval split
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    formatter=custom_formatter, 
    eval_split=True,
    eval_split_size=0.01  # 1% for evaluation
)

print(f"Training samples: {len(train_samples)}")
print(f"Evaluation samples: {len(eval_samples)}")

# Cell 14: Initialize model
model = Vits(config, ap, tokenizer, speaker_manager=None)

# Cell 15: Load pre-trained model weights
if os.path.exists(ORIGINAL_MODEL_PATH):
    print(f"Loading checkpoint from {ORIGINAL_MODEL_PATH}")
    try:
        checkpoint = torch.load(ORIGINAL_MODEL_PATH, map_location=torch.device('cpu'))
        
        # For loading only the model weights (not optimizer state)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            try:
                model.load_state_dict(checkpoint)
            except:
                print("Could not load model directly. Trying with strict=False...")
                model.load_state_dict(checkpoint, strict=False)
        
        print("Model loaded successfully for finetuning.")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Starting from scratch.")
else:
    print("No pretrained model found. Starting from scratch.")

# Cell 16: Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
else:
    print("No GPU available. Training will be slow on CPU.")

# Cell 17: Import and configure Trainer
from TTS.trainer import Trainer, TrainerArgs

# Cell 18: Configure trainer arguments
trainer_args = TrainerArgs(
    continue_path=ORIGINAL_MODEL_PATH if os.path.exists(ORIGINAL_MODEL_PATH) else None,
    restore_path=ORIGINAL_MODEL_PATH if os.path.exists(ORIGINAL_MODEL_PATH) else None,
    # Finetuning specific parameters
    lr=1e-5,  # Lower learning rate for finetuning
    lr_decay=True,
    use_grad_clip=True,
    grad_clip=1.0,
    early_stopping=True,
    early_stopping_epochs=10,
    # TTS 0.15.0 specific args
    max_epochs=500,
    epochs_plot_transfer_weights=100
)

# Cell 19: Initialize trainer
trainer = Trainer(
    trainer_args, 
    config, 
    OUTPUT_PATH, 
    model=model, 
    train_samples=train_samples, 
    eval_samples=eval_samples
)

# Cell 20: Start finetuning
# This cell will take a long time to run
print("Starting fine-tuning...")
trainer.fit()
print("Fine-tuning completed!")

# Cell 21: Test the finetuned model
# This assumes you have the inference module from the original repository
# If not, you'll need to create a simple inference script

def load_model_for_inference(model_path, config_path):
    """Load model for inference"""
    config = load_config(config_path)
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = Vits(config, ap, tokenizer, speaker_manager=None)
    
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model, ap

def generate_speech(model, ap, text, use_griffin_lim=False):
    """Generate speech from text"""
    # Set model to eval mode
    model.eval()
    
    # Generate speech with TTS 0.15.0 syntax
    with torch.no_grad():
        outputs = model.inference(text)
    
    waveform = outputs["waveform"]
    
    # Convert to numpy array
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
    
    # Ensure it's the right shape
    if len(waveform.shape) > 1:
        waveform = waveform.squeeze()
    
    return waveform

# Load and test your finetuned model
best_model_path = os.path.join(OUTPUT_PATH, "best_model.pth")
config_path = os.path.join(OUTPUT_PATH, "config.json")

if os.path.exists(best_model_path) and os.path.exists(config_path):
    model, ap = load_model_for_inference(best_model_path, config_path)
    
    # Generate and play audio from a test sentence
    test_text = "জাতীয় পার্টিকে ২৬টি এবং ১৪ দলের শরিকদের ৬টি আসন ছেড়ে দিয়েছে আওয়ামী লীগ।"
    audio = generate_speech(model, ap, test_text)
    
    # Play the audio
    display(Audio(audio, rate=ap.sample_rate))
    
    # Save the audio to file
    import soundfile as sf
    sf.write('finetuned_test.wav', audio, ap.sample_rate)
    print("Generated and saved test audio!")
else:
    print("Model files not found. Please check if training completed successfully.")