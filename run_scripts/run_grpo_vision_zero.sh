#!/bin/bash
IMAGES_DIR="Vision-Zero-clevr-dataset/output/replacement_images"
SCENES_DIR="Vision-Zero-clevr-dataset/output/replacement_scenes"
MODEL = Qwen/Qwen2.5-VL-7B-Instruct
OUTPUT_BASE_DIR="output/"
RUN_NAME="Qwen2.5-VL-7B-GRPO-Vision-Zero"


cd src/open-r1-multimodal
# Set Python path to include the src directory for module imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export DEBUG_MODE="true"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


export LOG_PATH="./debug_log_$RUN_NAME.txt"

# Interactive training configuration
EPOCH_SIZE=450   # Samples per epoch
NUM_EPOCHS=40    # Total number of training epochs
NUM_PLAYERS=4    # Number of players in each game
NUM_ROUNDS=2     # Number of clue rounds per game

# INTERACTIVE TRAINING MODE PARAMETERS
TRAINING_PHASE="interactive"    # Alternates between decision and clue phases
INTERACTIVE_CYCLE_LENGTH=1      # Number of steps per phase (adjustable parameter)
                                # Total cycle = 2 * INTERACTIVE_CYCLE_LENGTH steps



# Create output directory on local SSD if it doesn't exist

mkdir -p $OUTPUT_BASE_DIR/$RUN_NAME

echo "Starting Interactive CLEVR Spot-the-Difference Training..."
echo "Run Name: $RUN_NAME"
echo "Number of Players: $NUM_PLAYERS"
echo "Number of Rounds: $NUM_ROUNDS"
echo "Epoch Size: $EPOCH_SIZE games per epoch"
echo "Number of Epochs: $NUM_EPOCHS"
echo "CLEVR Images Directory: $IMAGES_DIR"
echo "CLEVR Scenes Directory: $SCENES_DIR"
echo "Training Output Directory: $OUTPUT_BASE_DIR/$RUN_NAME"
echo ""
echo "🔄 TRAINING PHASE: $TRAINING_PHASE"
echo "  - INTERACTIVE MODE: Alternating between decision and clue training"
echo "  - CYCLE LENGTH: $INTERACTIVE_CYCLE_LENGTH steps per phase"
TOTAL_CYCLE_LENGTH=$((INTERACTIVE_CYCLE_LENGTH * 2))
echo "    * Steps 0-$((INTERACTIVE_CYCLE_LENGTH-1)): Decision phase training (clue inference only)"
echo "    * Steps $INTERACTIVE_CYCLE_LENGTH-$((INTERACTIVE_CYCLE_LENGTH*2-1)): Clue phase training (decision inference only)"
echo "    * Steps $((INTERACTIVE_CYCLE_LENGTH*2))-$((INTERACTIVE_CYCLE_LENGTH*3-1)): Decision phase training (clue inference only)"
echo "    * Steps $((INTERACTIVE_CYCLE_LENGTH*3))-$((INTERACTIVE_CYCLE_LENGTH*4-1)): Clue phase training (decision inference only)"
echo "    * Pattern repeats every $TOTAL_CYCLE_LENGTH steps for rapid skill switching"
echo ""
echo "🎯 TRAINING STRATEGY:"
echo "  - Phase 1 (Decision): Focus on voting accuracy and strategic reasoning"
echo "  - Phase 2 (Clue): Focus on content quality and information sharing"
echo "  - Alternating approach allows model to develop complementary skills"
echo "  - Each phase benefits from the other's improvements"
echo ""

# Interactive training with model parallel setup
torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12350" \
    src/open_r1/grpo_jsonl.py \
    --deepspeed local_scripts/zero3_model_parallel.json \
    --output_dir $OUTPUT_BASE_DIR/$RUN_NAME \
    --model_name_or_path MODEL \
    --dataset_name "dynamic_clevr_spotdiff" \
    --use_dynamic_dataset \
    --epoch_size $EPOCH_SIZE \
    --data_generator_type clevr_spotdiff \
    --clevr_images_dir $IMAGES_DIR \
    --clevr_scenes_dir $SCENES_DIR \
    --clevr_num_players $NUM_PLAYERS \
    --clevr_num_rounds $NUM_ROUNDS \
    --training_phase $TRAINING_PHASE \
    --interactive_cycle_length $INTERACTIVE_CYCLE_LENGTH \
    --data_generator_seed 42 \
    --max_anyres_num 6 \
    --max_prompt_length 8000 \
    --max_completion_length 512 \
    --num_generations 8 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --beta 0.04 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate 1e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --run_name $RUN_NAME \
    --save_steps 5 \
    --save_only_model true \
    --reward_funcs clevr_clue_format_with_votes clevr_decision_accuracy \
    --dispatch_batches False \
    --val_split_ratio 0.0 \
    --num_iterations 1

echo "Interactive training completed! Check $OUTPUT_BASE_DIR/$RUN_NAME for results."
echo ""
echo "Training Summary:"
echo "  - Completed $NUM_EPOCHS epochs with interactive phase switching"
echo "  - Cycle length: $INTERACTIVE_CYCLE_LENGTH epochs per phase"
EXPECTED_CYCLES=$((NUM_EPOCHS / (INTERACTIVE_CYCLE_LENGTH * 2)))
echo "  - Expected $EXPECTED_CYCLES complete decision→clue cycles"
echo "  - Model should show improved performance in both phases"
echo "  - Check wandb logs for phase-specific metrics and transitions" 
