; %% [markdown]
; ## One-call LoRA fine-tuning
; Offline miniature BERT: no pretrained model download required.
; %% [code]
(import torch)
(import transformers [BertConfig BertForSequenceClassification])
(import lispyn.workflow [train])
(import lispyn.peft_utils [parameter-counts])

(setv configuration (BertConfig :vocab_size 32 :hidden_size 16
  :num_hidden_layers 1 :num_attention_heads 4
  :intermediate_size 32 :num_labels 2))
(setv model (BertForSequenceClassification configuration))
(setv batch {"input_ids" (torch.randint 0 32 #(2 4))
             "attention_mask" (torch.ones #(2 4) :dtype torch.long)
             "labels" (torch.tensor [0 1])})
(setv run
  (train model [batch]
    :lora-modules ["query" "value"]
    :lora-rank 2
    :lora-task-type "SEQ_CLS"
    :epochs 1
    :device "cpu"))
(print "parameters:" (parameter-counts run.model))
(print "training loss:" (get run.history "train_loss"))
