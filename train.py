from argparse import Namespace
from transformers import (
    AutoConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    set_seed,
    get_scheduler,
    AdamW,
)
from accelerate import Accelerator

from utils import (
    load_dataset,
    model_size,
    create_dataloaders,
    setup_logging,
    log_metrics,
    get_grouped_params,
    evaluate,
)

model_ckpt = "oscar-pt"
org = "coldfir3"
project_name = 'baseline-gpt2'

config = {
    "train_batch_size": 64,
    "valid_batch_size": 64,
    "weight_decay": 0.1,
    "shuffle_buffer": 1000,
    "learning_rate": 5e-4, 
    "lr_scheduler_type": "cosine",
    "num_warmup_steps": 2000,
    "gradient_accumulation_steps": 1,
    "max_train_steps": 150000,
    "valid_size": 1000,
    "max_eval_steps": 1000,
    "seq_length": 256,
    'n_ctx': 256,
    'n_positions': 256,
    'n_embd': 384,
    'n_layer': 6,
    'n_head': 6,
    "seed": 42,
    "save_checkpoint_steps": 1500,
    "mixed_precision": 'bf16',
    } 

args = Namespace(**config)
     
tokenizer = AutoTokenizer.from_pretrained(org + "/" + model_ckpt)
config = AutoConfig.from_pretrained(
    "gpt2", 
    vocab_size=len(tokenizer), 
    bos_token_id=len(tokenizer)-1,
    eos_token_id=len(tokenizer)-1,
    **config
    )
model = AutoModelForCausalLM.from_config(config)

print(f'GPT-2 (nano) size: {model_size(model)/1000**2:.1f}M parameters')

dataset = load_dataset('oscar', "unshuffled_deduplicated_pt", split='train', streaming=True)

train_dataloader, eval_dataloader = create_dataloaders(tokenizer, args)

## training loop
set_seed(args.seed)

# Accelerator
accelerator = Accelerator(mixed_precision=args.mixed_precision)
samples_per_step = accelerator.state.num_processes * args.train_batch_size

# Logging
logger, tb_writer, run_name = setup_logging(project_name, accelerator, args)
logger.info(accelerator.state)

# Load model and tokenizer
# if accelerator.is_main_process:
#     hf_repo = Repository("./", clone_from=project_name, revision=run_name)
# model = AutoModelForCausalLM.from_pretrained("./", gradient_checkpointing=True)
# tokenizer = AutoTokenizer.from_pretrained("./")

# Load dataset and dataloader
train_dataloader, eval_dataloader = create_dataloaders(tokenizer, args)

# Prepare the optimizer and learning rate scheduler
optimizer = AdamW(get_grouped_params(model, args), lr=args.learning_rate)
lr_scheduler = get_scheduler(name=args.lr_scheduler_type, optimizer=optimizer,
                             num_warmup_steps=args.num_warmup_steps,
                             num_training_steps=args.max_train_steps,)

def get_lr():
    return optimizer.param_groups[0]['lr']

# Prepare everything with our `accelerator` (order of args is not important)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader)

# Train model
model.train()
completed_steps = 0
for step, batch in enumerate(train_dataloader, start=1):
    loss = model(batch, labels=batch).loss
    log_metrics(step, {'lr': get_lr(), 'samples': step*samples_per_step,
                       'steps': completed_steps, 'loss/train': loss.item()},
                       logger, accelerator, tb_writer)
    loss = loss / args.gradient_accumulation_steps
    accelerator.backward(loss)
    if step % args.gradient_accumulation_steps == 0:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        completed_steps += 1
    if step % args.save_checkpoint_steps == 0:
        logger.info('Evaluating and saving model checkpoint')
        eval_loss, perplexity = evaluate(model, eval_dataloader, args, accelerator)
        log_metrics(step, {'loss/eval': eval_loss, 'perplexity': perplexity}, logger, accelerator, tb_writer)
        # accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained("./model")
        model.train()
    if completed_steps >= args.max_train_steps:
        break

model.save_pretrained("models/" + model_ckpt, push_to_hub=True, organization=org)