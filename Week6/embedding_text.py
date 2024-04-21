import torch
from transformers import AutoModel, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def extract_features(
    text: str,
    model_name: str = "bert-base-uncased",
    tokenizer_name: str = "bert-base-uncased",
):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    model.to(device)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        add_special_tokens=True,
        return_attention_mask=True,
    ).to(device)

    # Shape (768)
    return model(**inputs).last_hidden_state[:, 0, :].squeeze()

# Example
if __name__ == "__main__":

    test = "This is a test"
    print(extract_features(test).shape)
    print(extract_features(test))