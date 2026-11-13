"""One-call workflow regression tests (CPU, no external model downloads)."""

import json
import pytest


def _dataset():
    torch = pytest.importorskip("torch")
    x = torch.tensor([[0., 1.], [1., 0.], [1., 1.], [0., 0.]])
    y = torch.tensor([0, 1, 1, 0])
    return torch, torch.utils.data.TensorDataset(x, y)


def test_one_call_training_and_checkpoint(tmp_path):
    torch, dataset = _dataset()
    from lispyn.workflow import Run, train
    from lispyn.training import load_checkpoint

    model = torch.nn.Linear(2, 2)
    saved = tmp_path / "best.pt"
    log = tmp_path / "history.json"
    result = train(model, dataset, validation=dataset,
                   loss_fn=torch.nn.CrossEntropyLoss(), epochs=2,
                   batch_size=2, lr=0.02, device="cpu", seed=17,
                   metrics="accuracy", save_best=saved, log_path=log)
    assert isinstance(result, Run)
    assert result.model is model
    assert result.device == "cpu"
    assert len(result.history["train_loss"]) == 2
    assert len(result.history["val_accuracy"]) == 2
    assert saved.exists()
    assert json.loads(log.read_text()) == result.history
    assert len(load_checkpoint(model, saved)["train_loss"]) in (1, 2)


def test_freeze_head_and_custom_native_optimizer():
    torch, dataset = _dataset()
    from lispyn.workflow import train

    model = torch.nn.Sequential(torch.nn.Linear(2, 4), torch.nn.ReLU(),
                                torch.nn.Linear(4, 2))
    result = train(model, dataset, loss_fn=torch.nn.CrossEntropyLoss(),
                   head="2", epochs=1, device="cpu", optimizer=torch.optim.SGD,
                   optimizer_kwargs={"momentum": 0.8}, lr=0.04)
    assert result.model[0].weight.requires_grad is False
    assert result.model[2].weight.requires_grad is True
    assert result.optimizer.defaults["momentum"] == 0.8
    assert sum(p.numel() for group in result.optimizer.param_groups
               for p in group["params"]) == 10


def test_existing_batched_iterable_mapping():
    torch, _ = _dataset()
    from lispyn.workflow import train

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(2, 2)

        def forward(self, input_ids, labels):
            logits = self.fc(input_ids)
            return {"loss": torch.nn.functional.cross_entropy(logits, labels),
                    "logits": logits}

    model = Network()
    batches = [{"input_ids": torch.ones(2, 2), "labels": torch.tensor([0, 1])}]
    run = train(model, batches, epochs=2, device="cpu", metrics="accuracy")
    assert len(run.history["train_accuracy"]) == 2


def test_invalid_modes_and_no_trainable_parameters():
    torch, dataset = _dataset()
    from lispyn.workflow import train

    def attempt(**kwargs):
        return train(torch.nn.Linear(2, 2), dataset,
                     loss_fn=torch.nn.CrossEntropyLoss(), device="cpu",
                     **kwargs)

    with pytest.raises(ValueError, match="positive integer"):
        attempt(batch_size=0)
    with pytest.raises(ValueError, match="lr"):
        attempt(lr=0)
    with pytest.raises(ValueError, match="head-only or LoRA"):
        attempt(head="fc", lora_modules=["query"])
    with pytest.raises(ValueError, match="Unknown optimizer"):
        attempt(optimizer="magic")
    with pytest.raises(ValueError, match="instead of optimizer_kwargs"):
        attempt(optimizer_kwargs={"lr": 0.1})
    with pytest.raises(ValueError, match="No trainable parameters"):
        m = torch.nn.Linear(2, 2)
        for p in m.parameters():
            p.requires_grad_(False)
        train(m, dataset, loss_fn=torch.nn.CrossEntropyLoss(), device="cpu")


def test_text_csv_labels_as_integers(tmp_path):
    torch = pytest.importorskip("torch")
    from lispyn.media import text_dataset, token_collate

    file = tmp_path / "samples.csv"
    file.write_text("text,label\nyes,1\nno,0\n")
    dataset = text_dataset(file, label_key="label", label_cast=int)
    assert dataset[0] == {"text": "yes", "label": 1}
    assert dataset[1]["label"] == 0
    def fake_tokenizer(lines, **kwargs):
        return {"input_ids": torch.tensor([[len(t)] for t in lines])}
    batch = token_collate(fake_tokenizer)([dataset[0], dataset[1]])
    assert batch["labels"].tolist() == [1, 0]


def test_auto_device_uses_torch_backend():
    pytest.importorskip("torch")
    from lispyn.workflow import device_auto
    assert device_auto() in {"cpu", "cuda", "mps"}


def test_one_call_lora_with_native_transformers():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    from lispyn.workflow import train
    from lispyn.peft_utils import parameter_counts

    config = transformers.BertConfig(
        vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=4, intermediate_size=32, num_labels=2,
    )
    model = transformers.BertForSequenceClassification(config)
    batch = {
        "input_ids": torch.randint(0, 32, (2, 4)),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "labels": torch.tensor([0, 1]),
    }
    run = train(model, [batch], lora_modules=["query", "value"],
                lora_rank=2, lora_task_type="SEQ_CLS", epochs=1, device="cpu")
    counts = parameter_counts(run.model)
    assert 0 < counts["trainable"] < counts["total"]
    assert len(run.history["train_loss"]) == 1
