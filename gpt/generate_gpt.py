import torch
import torch.nn as nn

from gpt_model import GPT, ModelConfig
from trainer_gpt import DataConfig


def main(unused_args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data config.
    data_config = DataConfig()
    tokenizer = data_config.tokenizer()
    vocab_size = tokenizer.vocab_size()

    # Build the model.
    model_config = ModelConfig()
    model = GPT(model_config, data_config.cap_sequence_length - 1, vocab_size).to(device)
    # Load the checkpoint.
    checkpoint = torch.load("gpt_best_checkpoint.pt")
    model.load_state_dict(checkpoint)
    # Model in eval.
    model.eval()

    # Generate text.
    prompt = "QUEEN ELIZABETH:\nBut she, your subject,"
    for _ in range(1000):
        print(f"Prompt: {prompt}")
        tokens_list = tokenizer.tokenize(prompt, add_eos=False)
        original_length = len(tokens_list)
        tokens_list += [data_config.pad_token] * (data_config.cap_sequence_length - len(tokens_list) - 1)
        input_tensor = torch.tensor(tokens_list, device=device).unsqueeze(0)
        model_logits = model.forward(input_tensor)
        last_char_logits = model_logits[0, original_length - 1, :]
        last_char_probs = nn.functional.softmax(last_char_logits, dim=-1)
        last_char = torch.argmax(last_char_probs)
        gen_token = tokenizer.detokenize([last_char.item()])
        if gen_token == "<EOS>":
            break
        prompt += tokenizer.detokenize([last_char.item()])
    print(f"Generated text: {prompt}")


if __name__ == "__main__":
    main(None)