"""Official Unsloth vision SFT configuration, adapted for SROIE."""
from .data import training_examples


def make_trainer(model, processor, train_dataset, config):
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    conversations = training_examples(train_dataset, config.max_image_edge)
    return SFTTrainer(
        model=model,
        train_dataset=conversations,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args=SFTConfig(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            max_steps=config.train_steps,
            learning_rate=config.learning_rate,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=config.seed,
            output_dir=config.output_dir + "/trainer",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
        ),
    )
