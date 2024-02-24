from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from transformers import TrainingArguments
import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# Configure the BitsAndBytesConfig to enable  8-bit quantization with CPU offloading


# Load the model with the custom device map and configuration

from transformers import logging as hf_logging
import logging





class fine_tune_llm():
    def __init__(self):
        self.model_name_or_path = "mistralai/Mistral-7B-Instruct-v0.1"
        self.tokenizer_name = "mistralai/Mistral-7B-v0.1"
        self.output_dir = "./fine_tuned_model"
        self.datset_path =  "lcquad_train_prompt.csv"
        self.test_dataset_path = "lcquad_test.csv"

        #logging.set_verbosity_info()

        #self.dataset = dataset
    
        self.peft_config = LoraConfig(
                                        lora_alpha=16,
                                        lora_dropout=0.1,
                                        r=64,
                                        bias="none",
                                        task_type="CAUSAL_LM" )

        self.args = TrainingArguments(
            output_dir = self.output_dir,
            num_train_epochs=1,
            #max_steps = 100, # comment out this line if you want to train in epochs
            per_device_train_batch_size = 2,
            per_device_eval_batch_size=2,
            warmup_steps = 0.03,
            logging_steps=10,
            save_strategy="epoch",
            #evaluation_strategy="epoch",
            evaluation_strategy="steps",
            #eval_steps=20, # comment out this line if you want to evaluate at the end of each epoch
            learning_rate=2e-4,
            fp16=False,
            lr_scheduler_type='cosine',
            )

    def tokenize_function(self, examples):
        return self.tokenizer(examples["prompt"], padding="max_length", truncation=True)
    
    def load_model(self):
       
        nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_enable_fp32_cpu_offload=True,
                )
        model = AutoModelForCausalLM.from_pretrained(
                    "mistralai/Mistral-7B-Instruct-v0.1",
                    device_map="auto",
                    quantization_config=nf4_config,
                    use_cache=False, )
        
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        model = prepare_model_for_kbit_training(model, self.peft_config)
        model = get_peft_model(model, self.peft_config)

        return model, tokenizer

    def prepare_dataset(self):
        train = pd.read_csv('lcquad_train_prompt.csv')
        test = pd.read_csv('lcquad_test.csv')
        train['question'] = train['question'].astype(str)
        test['question'] = test['question'].astype(str)
        dataset = Dataset.from_pandas(train)
        eval_dataset = dataset.select(range(500))
        train_dataset = dataset.select(range(500, len(dataset)))
        test_dataset = Dataset.from_pandas(test)
        #test_dataset = Dataset.from_pandas(test)
        #self.dataset = self.dataset.map(self.tokenize_function, batched=True)

        return train_dataset, eval_dataset
    
    def logger(self):
        file_handler = logging.FileHandler("fine_tune.log")
        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #file_handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
        hf_logging.set_verbosity_info()
        return 


    

    def train(self,model,dataset,test_dataset, tokenizer):
        self.trainer = SFTTrainer(
            model=model,
            peft_config=self.peft_config,
            max_seq_length=2048, 
            args=self.args,
            train_dataset=dataset,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            packing=True,
            dataset_text_field = "prompt",
            

        )
        self.trainer.train()
        self.trainer.save_model(self.output_dir)


fine_tune_llm = fine_tune_llm()
logger = fine_tune_llm.logger()
logging.info("Logger initiated")

dataset , test = fine_tune_llm.prepare_dataset()
logging.info("dataset prepared and split into train and test sets")
model, tokenizer = fine_tune_llm.load_model()
logging("Model loaded")
print("Dataset prepared")

