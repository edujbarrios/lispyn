; %% [markdown]
; ## LoRA fine-tuning for text (offline miniature BERT)
; No model downloads. Replace BertConfig with from_pretrained(...) for real models.
; %% [code]
(import torch)
(import transformers [BertConfig BertForSequenceClassification])
(import lispyn.peft_utils [lora parameter-counts])
(import lispyn.training [fit])

(setv config (BertConfig :vocab_size 64 :hidden_size 32 :num_hidden_layers 1
                        :num_attention_heads 4 :intermediate_size 64 :num_labels 2))
(setv model (BertForSequenceClassification config))
(setv model (lora model ["query" "value"] :rank 4 :task-type "SEQ_CLS"))
(setv batch {"input_ids" (torch.randint 0 64 #(3 8))
             "attention_mask" (torch.ones #(3 8) :dtype torch.long)
             "labels" (torch.tensor [0 1 0])})
(setv optimizer (torch.optim.AdamW
                  (filter (fn [p] p.requires_grad) (.parameters model))
                  :lr 0.001))
(print "LoRA params:" (parameter-counts model))
(print (fit model [batch] optimizer :epochs 1))
