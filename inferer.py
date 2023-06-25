import argparse
import torch
from transformers import LlamaTokenizer,LlamaForCausalLM,GenerationConfig
from peft import PeftModel
from prompter import Prompter
import sys

class Inferer:
  def __init__(self, model_path, load_8bit=True):
    self.model_path = model_path
    self.load_8bit = load_8bit
    self.use_lora = False
    if torch.cuda.is_available():
      self.device = "cuda"
  def load_lora_weights(self, lora_weights):
    self.use_lora = True
    self.lora_weights = lora_weights
  def prepare(self):
    self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
    self.model = LlamaForCausalLM.from_pretrained(self.model_path,
                                                  load_in_8bit=self.load_8bit,
                                                  torch_dtype=torch.float16,
                                                  device_map="auto" )
    if self.use_lora:
      print(f"using lora {self.lora_weights}")
      self.model = PeftModel.from_pretrained(
          self.model,
          self.lora_weights,
          torch_dtype=torch.float16,
      )
    # unwind broken decapoda-research config
    self.model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
    self.model.config.bos_token_id = 1
    self.model.config.eos_token_id = 2
    if not self.load_8bit:
        self.model.half()  # seems to fix bugs for some users.
    self.model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
      self.model = torch.compile(self.model)

    self.prompter = Prompter()

  def predict(self, instruction , 
              input = None,
              temperature=0.1,
              top_p=0.75,
              top_k=40,
              num_beams=4,
              max_new_tokens=256,
              **kwargs):
    prompt = self.prompter.generate_prompt(instruction, input)
    inputs = self.tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(self.device)
    generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
    with torch.no_grad():
      generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
      s = generation_output.sequences[0]
      output = self.tokenizer.decode(s)
      return self.prompter.get_response(output)
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description= "Train on a model on a dataset")
  parser.add_argument("--model_path", type=str, default="decapoda-research/llama-7b-hf")
  parser.add_argument("--lora_weights", type=str, default=None)
  parser.add_argument("--prompt", type=str, default="")
  parser.add_argument("--load_8bit", type=bool, default=True)

  args = parser.parse_args()
  model_path = args.model_path
  lora_weights = args.lora_weights
  prompt = args.prompt
  load_8bit = args.load_8bit

  inferer = Inferer(model_path, load_8bit=load_8bit)
  if lora_weights is not None:
    inferer.load_lora_weights(lora_weights)
  inferer.prepare()
  if prompt == "":
    while True:
      prompt = input("prompt: ")
      if prompt == "q!":
        break
      print(inferer.predict(prompt))
  else:
    print(inferer.predict(prompt))