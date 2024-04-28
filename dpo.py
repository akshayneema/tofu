import os
import gc, json
import pandas as pd
import torch
import wandb
from scipy import stats
import tqdm as notebook_tqdm
import argparse
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, SFTTrainer
import bitsandbytes as bnb
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset, load_from_disk
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from transformers import TrainerCallback, TrainerState, TrainerControl, Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# Set the environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"
seed=42
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_folder)
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control
    
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def DPO(input_args):
    base_model_id = input_args.model_name_or_path
    # from datetime import datetime
    # current_time = datetime.now()
    # # formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    # formatted_time = current_time.strftime("%d-%B-%Y")
    run_id=input_args.run_id
    # iteration=input_args.iteration
    
    wandb.init(project=input_args.run_id)
    

    print(f"############### RUN ID is {run_id}")
    # print(f"############### Iteration is {iteration}")

    print("############# Model name : ", base_model_id)
    #################################### Tokenizer ##############################################
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side="right",
        add_eos_token=True,
        trust_remote_code=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("############# Tokenizer loaded")
    ####################################### Load Data #########################################
    
    # raw_pref_data_path=input_args.dataset_dir
    # model_cache_dir="/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/rishabh/cache-models"
    
    # filter_flag=input_args.filter
    
    datapath=input_args.data_path
    print(f"############# Loading Dataset from {datapath}")
    
    data=load_dataset("json", data_files=datapath, split="train")
    data=data.train_test_split(test_size=0.1)
    train_dataset,eval_dataset=data['train'],data['test']
    
        
        
    ######################################## Load Model #########################################
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(base_model_id, 
                                                # quantization_config=bnb_config, 
                                                device_map={"": 0},
                                                use_cache=False,)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)

    print("Model Loaded")
    print()

    ####################################### QLoRA setting #########################################
    if input_args.lora_rank:
        config = LoraConfig(
            r=input_args.lora_rank,
            lora_alpha=input_args.lora_alpha, 
            target_modules=find_all_linear_names(model), 
            lora_dropout=input_args.lora_dropout,
            bias="none", 
            task_type="CAUSAL_LM"
        )
    # if input_args.lora_rank != 0:
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    # --gradient_checkpointing True, workers speed up processing,  grad accumulation - 8, 16, 22 (less memory)
    # output_dir = f"/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/rishabh/models/{model_name}-{run_id}-sft"
    # model_name = 'Mistral-7B-Instruct-v0.2'

    output_path = f"{input_args.output_dir}/{run_id}-dpo"
    print(f"#### Output path is {output_path}")  
    # gc.collect()
    # torch.cuda.empty_cache()
    
    model_kwargs = dict(
            revision="main",
            trust_remote_code=False,
            use_flash_attention_2=False,
            use_cache=False,
            device_map={"": 0},
        )
   ####################################### Training Arguments #########################################
    args=transformers.TrainingArguments(
        # model_init_kwargs=model_kwargs,
        output_dir=output_path,
        warmup_steps=1,
        per_device_train_batch_size=input_args.per_device_train_batch_size,
        # per_device_eval_batch_size=input_args.per_device_eval_batch_size,
        gradient_accumulation_steps=input_args.gradient_accumulation_steps,
        gradient_checkpointing=False,
        group_by_length=False,
        num_train_epochs=input_args.num_train_epochs,
        learning_rate=input_args.learning_rate,
        optim="paged_adamw_32bit",
        logging_strategy=input_args.logging_strategy,
        logging_steps=input_args.log_steps,              # When to start reporting loss
        save_strategy=input_args.save_strategy,  
        save_total_limit=2,# Save the model checkpoint every logging step              # Save checkpoints every 100 steps
        report_to=input_args.report_to,           # Comment this out if you don't want to use weights & baises
        dataloader_pin_memory=True,                           
        dataloader_num_workers=4,
        dataloader_prefetch_factor=1,
        logging_first_step=input_args.logging_first_step,
        lr_scheduler_type="cosine",
        seed=42,
        # bf16=True,
        fp16=True,
        fp16_full_eval=True,
        ddp_find_unused_parameters= False,
        # tf32=True,
    )
    ####################################### DPO Training #########################################
    print("################# DPO Training started")
    ### DPO Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=config,
        beta=0.5,
        max_prompt_length=1024,
        max_length=2900
    )
    
    print(f"################# DPO Trainer {trainer}")
    
    trainer.train()

    print("################# Training is done")
    
    merged_path=f'{output_path}/merged_model'
    adapter_path=f'{output_path}/adapters'
    # print(f"#### Saving to f{adapter_path}")  
    # trainer.save_model(output_path)
    if input_args.lora_rank:
        trainer.model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        print(f"######## Saving adapter to {adapter_path}")  
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()
        
        # # adapter_path=f'/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/rishabh/saved-models/Mistral-7B-Instruct-v0.2-2024-03-18_23-52-22-dpo-m2/final_m2'
        # base_model_id=f''
        base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                return_dict=True,
                torch_dtype=torch.float16,
            )
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)

        # # Merge base model with the adapter
        model = PeftModel.from_pretrained(base_model, model_id=f"{adapter_path}")
        model = model.merge_and_unload()
    
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    print(f"################# Saving final model at {merged_path}")

    # Flush memory

    # # Save model and tokenizer
    

    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    # # Flush memory
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"####################### Model saved at : {merged_path}")
    return


if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description="DPO Training Arguments")

    # parser.add_argument("", type=str, default="", help="Model name or path")
    parser.add_argument("--model_name_or_path", type=str, default="/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/rishabh/saved-models/merged_model", help="Model name or path, including Finetuned model")

    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size per device")
    # parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")

    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting destination")
    # parser.add_argument("--run_name", type=str, default="DPO-Training", help="Name of the run")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum Sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs")

    # parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy")
    # parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--run-id", type=str, default="", help="Run id for labelling folders")
    parser.add_argument("--logging_strategy", type=str, default="steps", help="Logging strategy")
    parser.add_argument("--log_steps", type=int, default=500, help="Logging steps")
    parser.add_argument("--logging_first_step", action="store_true", help="Log the first step")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")

    parser.add_argument("--lora_rank", type=int, default=32, help='Rank in LoRA config')
    parser.add_argument("--lora_alpha", type=int, default=16, help='Alpha in LoRA config')
    parser.add_argument("--lora_dropout", type=float, default=0.05, help='Dropout in LoRA config')

    parser.add_argument("--output_dir", type=str, default="./saved-models/no_name-dpo", help="Output directory")
    parser.add_argument("--data-path", type=str, default="/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/rishabh/data/raw_reward_1_2024-03-18_18-28-26", help=" Preference dataset path, must be a json dataset")

    
    # input_args = parser.parse_args([])

    # DPO(input_args)
    input_args = parser.parse_args()

    DPO(input_args)